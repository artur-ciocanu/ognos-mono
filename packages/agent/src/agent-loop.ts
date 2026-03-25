/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	EventStream,
	parseStreamingJson,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import {
	type AgentRuntime,
	type AgentRuntimeEvent,
	type AgentRuntimeOptions,
	cloneAssistantMessage,
	createAssistantMessageShell,
	createAssistantUsage,
	getModelMetadata,
	parseToolArguments,
	toRuntimeMessages,
} from "./runtime-bridge.js";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
} from "./types.js";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	runtime?: AgentRuntime,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();
	void resolveAgentLoopStream(
		stream,
		() =>
			runAgentLoop(
				prompts,
				context,
				config,
				async (event) => {
					stream.push(event);
				},
				signal,
				runtime,
			),
		config,
		async (event) => {
			stream.push(event);
		},
		prompts,
		signal,
	);

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	runtime?: AgentRuntime,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();
	void resolveAgentLoopStream(
		stream,
		() =>
			runAgentLoopContinue(
				context,
				config,
				async (event) => {
					stream.push(event);
				},
				signal,
				runtime,
			),
		config,
		async (event) => {
			stream.push(event);
		},
		[],
		signal,
	);

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	runtime?: AgentRuntime,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, runtime);
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	runtime?: AgentRuntime,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, runtime);
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

async function resolveAgentLoopStream(
	stream: EventStream<AgentEvent, AgentMessage[]>,
	run: () => Promise<AgentMessage[]>,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	initialMessages: AgentMessage[],
	signal?: AbortSignal,
): Promise<void> {
	try {
		stream.end(await run());
	} catch (error) {
		const messages = await emitLoopFailure(config, emit, initialMessages, error, signal);
		stream.end(messages);
	}
}

async function emitLoopFailure(
	config: AgentLoopConfig,
	emit: AgentEventSink,
	initialMessages: AgentMessage[],
	error: unknown,
	signal?: AbortSignal,
): Promise<AgentMessage[]> {
	const errorMessage = finalizeAssistantMessage(
		config.model,
		null,
		createAssistantUsage(),
		signal?.aborted ? "aborted" : "error",
		undefined,
		error instanceof Error ? error.message : String(error),
	);

	await emit({ type: "message_start", message: cloneAssistantMessage(errorMessage) });
	await emit({ type: "message_end", message: errorMessage });
	await emit({ type: "turn_end", message: errorMessage, toolResults: [] });

	const messages = [...initialMessages, errorMessage];
	await emit({ type: "agent_end", messages });
	return messages;
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	runtime?: AgentRuntime,
): Promise<void> {
	let firstTurn = true;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				await emit({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			// Process pending messages (inject before next assistant response)
			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					await emit({ type: "message_start", message });
					await emit({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stream assistant response
			const message = await streamAssistantResponse(currentContext, config, signal, emit, runtime);
			newMessages.push(message);

			if (message.stopReason === "error" || message.stopReason === "aborted") {
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return;
			}

			// Check for tool calls
			const toolCalls = message.content.filter((c) => c.type === "toolCall");
			hasMoreToolCalls = toolCalls.length > 0;

			const toolResults: ToolResultMessage[] = [];
			if (hasMoreToolCalls) {
				toolResults.push(...(await executeToolCalls(currentContext, message, config, signal, emit)));

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}
			}

			await emit({ type: "turn_end", message, toolResults });

			pendingMessages = (await config.getSteeringMessages?.()) || [];
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingMessages = followUpMessages;
			continue;
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	runtime?: AgentRuntime,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	const runtimeContext = {
		messages: toRuntimeMessages(llmMessages, context.systemPrompt),
		tools: context.tools,
		rawMessages: llmMessages,
	};

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(getModelMetadata(config.model).provider, config.model) : undefined) ||
		config.apiKey;

	const runtimeInstance = runtime ?? config.runtime;
	if (!runtimeInstance) {
		throw new Error("No runtime configured");
	}

	const runtimeOptions: AgentRuntimeOptions = {
		apiKey: resolvedApiKey,
		maxRetryDelayMs: config.maxRetryDelayMs,
		onPayload: config.onPayload,
		reasoning: config.reasoning,
		sessionId: config.sessionId,
		signal,
		thinkingBudgets: config.thinkingBudgets,
		transport: config.transport,
	};

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;
	let finalUsage = createAssistantUsage();
	const contentIndexById = new Map<string, number>();
	const toolArgumentsById = new Map<string, string>();

	for await (const event of runtimeInstance.stream(config.model, runtimeContext, runtimeOptions)) {
		switch (event.type) {
			case "start":
				partialMessage = createAssistantMessageShell(config.model);
				partialMessage.responseId = event.messageId;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: cloneAssistantMessage(partialMessage) });
				break;

			case "text_start":
				if (!partialMessage) {
					break;
				}
				contentIndexById.set(event.id, partialMessage.content.length);
				partialMessage.content.push({ type: "text", text: "" });
				context.messages[context.messages.length - 1] = partialMessage;
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "text_start",
						contentIndex: partialMessage.content.length - 1,
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "text_delta":
				if (!partialMessage) {
					break;
				}
				updateTextContent(partialMessage, contentIndexById.get(event.id), event.delta);
				context.messages[context.messages.length - 1] = partialMessage;
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "text_delta",
						contentIndex: contentIndexById.get(event.id) ?? 0,
						delta: event.delta,
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "text_end":
				if (!partialMessage) {
					break;
				}
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "text_end",
						contentIndex: contentIndexById.get(event.id) ?? 0,
						content: getTextContent(partialMessage, contentIndexById.get(event.id)),
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "thinking_start":
				if (!partialMessage) {
					break;
				}
				contentIndexById.set(event.id, partialMessage.content.length);
				partialMessage.content.push({ type: "thinking", thinking: "" });
				context.messages[context.messages.length - 1] = partialMessage;
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "thinking_start",
						contentIndex: partialMessage.content.length - 1,
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "thinking_delta":
				if (!partialMessage) {
					break;
				}
				updateThinkingContent(partialMessage, contentIndexById.get(event.id), event.delta);
				context.messages[context.messages.length - 1] = partialMessage;
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "thinking_delta",
						contentIndex: contentIndexById.get(event.id) ?? 0,
						delta: event.delta,
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "thinking_end":
				if (!partialMessage) {
					break;
				}
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "thinking_end",
						contentIndex: contentIndexById.get(event.id) ?? 0,
						content: getThinkingContent(partialMessage, contentIndexById.get(event.id)),
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "toolcall_start":
				if (!partialMessage) {
					break;
				}
				contentIndexById.set(event.id, partialMessage.content.length);
				toolArgumentsById.set(event.id, "");
				partialMessage.content.push({ type: "toolCall", id: event.id, name: event.toolName, arguments: {} });
				context.messages[context.messages.length - 1] = partialMessage;
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "toolcall_start",
						contentIndex: partialMessage.content.length - 1,
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "toolcall_delta":
				if (!partialMessage) {
					break;
				}
				updateToolCallArguments(
					partialMessage,
					contentIndexById.get(event.id),
					toolArgumentsById.get(event.id) ?? "",
					event.delta,
				);
				toolArgumentsById.set(event.id, `${toolArgumentsById.get(event.id) ?? ""}${event.delta}`);
				context.messages[context.messages.length - 1] = partialMessage;
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "toolcall_delta",
						contentIndex: contentIndexById.get(event.id) ?? 0,
						delta: event.delta,
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "toolcall_end":
				if (!partialMessage) {
					break;
				}
				await emit({
					type: "message_update",
					assistantMessageEvent: {
						type: "toolcall_end",
						contentIndex: contentIndexById.get(event.id) ?? 0,
						toolCall: getToolCallContent(partialMessage, contentIndexById.get(event.id)),
						partial: cloneAssistantMessage(partialMessage),
					},
					message: cloneAssistantMessage(partialMessage),
				});
				break;

			case "done": {
				finalUsage = createAssistantUsage(event.usage);
				const finalMessage = finalizeAssistantMessage(
					config.model,
					partialMessage,
					finalUsage,
					normalizeStopReason(event.reason, inferStopReason(partialMessage)),
					event.messageId,
				);
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: cloneAssistantMessage(finalMessage) });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}

			case "error": {
				const finalMessage = finalizeAssistantMessage(
					config.model,
					partialMessage,
					finalUsage,
					signal?.aborted ? "aborted" : "error",
					undefined,
					event.error.message,
				);
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
					await emit({ type: "message_start", message: cloneAssistantMessage(finalMessage) });
				}
				await emit({ type: "message_end", message: finalMessage });
				return finalMessage;
			}
		}
	}

	const finalMessage = finalizeAssistantMessage(
		config.model,
		partialMessage,
		finalUsage,
		normalizeStopReason(undefined, inferStopReason(partialMessage)),
		partialMessage?.responseId,
	);
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: cloneAssistantMessage(finalMessage) });
	}
	await emit({ type: "message_end", message: finalMessage });
	return finalMessage;
}

function updateTextContent(message: AssistantMessage, contentIndex: number | undefined, delta: string): void {
	const content = contentIndex === undefined ? undefined : message.content[contentIndex];
	if (content?.type === "text") {
		content.text += delta;
	}
}

function getTextContent(message: AssistantMessage, contentIndex: number | undefined): string {
	const content = contentIndex === undefined ? undefined : message.content[contentIndex];
	return content?.type === "text" ? content.text : "";
}

function updateThinkingContent(message: AssistantMessage, contentIndex: number | undefined, delta: string): void {
	const content = contentIndex === undefined ? undefined : message.content[contentIndex];
	if (content?.type === "thinking") {
		content.thinking += delta;
	}
}

function getThinkingContent(message: AssistantMessage, contentIndex: number | undefined): string {
	const content = contentIndex === undefined ? undefined : message.content[contentIndex];
	return content?.type === "thinking" ? content.thinking : "";
}

function updateToolCallArguments(
	message: AssistantMessage,
	contentIndex: number | undefined,
	previousDelta: string,
	delta: string,
): void {
	const content = contentIndex === undefined ? undefined : message.content[contentIndex];
	if (content?.type === "toolCall") {
		content.arguments = parseToolArguments(`${previousDelta}${delta}`);
	}
}

function getToolCallContent(message: AssistantMessage, contentIndex: number | undefined): AgentToolCall {
	const content = contentIndex === undefined ? undefined : message.content[contentIndex];
	if (content?.type === "toolCall") {
		return {
			...content,
			arguments: parseStreamingJson(JSON.stringify(content.arguments)) as Record<string, unknown>,
		};
	}

	return {
		type: "toolCall",
		id: "missing-tool-call",
		name: "missing-tool-call",
		arguments: {},
	};
}

function inferStopReason(
	message: AssistantMessage | null,
): NonNullable<Extract<AgentRuntimeEvent, { type: "done" }>["reason"]> {
	if (message?.content.some((content) => content.type === "toolCall")) {
		return "toolUse";
	}

	return "stop";
}

function normalizeStopReason(
	stopReason: Extract<AgentRuntimeEvent, { type: "done" }>["reason"],
	fallback: NonNullable<Extract<AgentRuntimeEvent, { type: "done" }>["reason"]>,
): AssistantMessage["stopReason"] {
	return stopReason ?? fallback;
}

function finalizeAssistantMessage(
	model: AgentLoopConfig["model"],
	partialMessage: AssistantMessage | null,
	usage: AssistantMessage["usage"],
	stopReason: AssistantMessage["stopReason"],
	responseId?: string,
	errorMessage?: string,
): AssistantMessage {
	const metadata = getModelMetadata(model);
	const finalMessage = partialMessage ? cloneAssistantMessage(partialMessage) : createAssistantMessageShell(model);
	finalMessage.api = metadata.api;
	finalMessage.provider = metadata.provider;
	finalMessage.model = model.id;
	finalMessage.responseId = responseId ?? partialMessage?.responseId;
	finalMessage.usage = usage;
	finalMessage.stopReason = stopReason;
	finalMessage.errorMessage = errorMessage;
	finalMessage.timestamp = Date.now();
	return finalMessage;
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	if (config.toolExecution === "sequential") {
		return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
		} else {
			const executed = await executePreparedToolCall(preparation, signal, emit);
			results.push(
				await finalizeExecutedToolCall(
					currentContext,
					assistantMessage,
					preparation,
					executed,
					config,
					signal,
					emit,
				),
			);
		}
	}

	return results;
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage[]> {
	const results: ToolResultMessage[] = [];
	const runnableCalls: PreparedToolCall[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal);
		if (preparation.kind === "immediate") {
			results.push(await emitToolCallOutcome(toolCall, preparation.result, preparation.isError, emit));
		} else {
			runnableCalls.push(preparation);
		}
	}

	const runningCalls = runnableCalls.map((prepared) => ({
		prepared,
		execution: executePreparedToolCall(prepared, signal, emit),
	}));

	for (const running of runningCalls) {
		const executed = await running.execution;
		results.push(
			await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				running.prepared,
				executed,
				config,
				signal,
				emit,
			),
		);
	}

	return results;
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<unknown>;
	isError: boolean;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<unknown>;
	isError: boolean;
};

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
		};
	}

	try {
		const validatedArgs = validateToolArguments(tool, toolCall);
		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(
				{
					assistantMessage,
					toolCall,
					args: validatedArgs,
					context: currentContext,
				},
				signal,
			);
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
				};
			}
		}
		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ToolResultMessage> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		const afterResult = await config.afterToolCall(
			{
				assistantMessage,
				toolCall: prepared.toolCall,
				args: prepared.args,
				result,
				isError,
				context: currentContext,
			},
			signal,
		);
		if (afterResult) {
			result = {
				content: afterResult.content ?? result.content,
				details: afterResult.details ?? result.details,
			};
			isError = afterResult.isError ?? isError;
		}
	}

	return await emitToolCallOutcome(prepared.toolCall, result, isError, emit);
}

function createErrorToolResult(message: string): AgentToolResult<unknown> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolCallOutcome(
	toolCall: AgentToolCall,
	result: AgentToolResult<unknown>,
	isError: boolean,
	emit: AgentEventSink,
): Promise<ToolResultMessage> {
	await emit({
		type: "tool_execution_end",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		result,
		isError,
	});

	const toolResultMessage: ToolResultMessage = {
		role: "toolResult",
		toolCallId: toolCall.id,
		toolName: toolCall.name,
		content: result.content,
		details: result.details,
		isError,
		timestamp: Date.now(),
	};

	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
	return toolResultMessage;
}
