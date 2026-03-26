import type { EventStream } from "@mariozechner/pi-ai";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEvent,
	type Context,
	type ImageContent,
	type Message,
	type Model,
	type Provider,
	parseStreamingJson,
	type StopReason,
	streamSimple,
	type TextContent,
	type ThinkingBudgets,
	type Tool,
	type ToolCall,
	type Transport,
	type UserMessage,
} from "@mariozechner/pi-ai";
import type { ModelHandle, RuntimeMessage, RuntimeUsage } from "@mariozechner/pi-llm-runtime";

export interface AgentRuntimeContext {
	messages: RuntimeMessage[];
	tools?: Tool[];
	rawMessages?: Message[];
}

export interface AgentRuntimeOptions {
	signal?: AbortSignal;
	apiKey?: string;
	maxTokens?: number;
	reasoning?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
	sessionId?: string;
	onPayload?: (payload: unknown, model: ModelHandle) => unknown | undefined | Promise<unknown | undefined>;
	thinkingBudgets?: ThinkingBudgets;
	transport?: Transport;
	maxRetryDelayMs?: number;
}

export type AgentRuntimeEvent =
	| { type: "start"; messageId?: string }
	| { type: "text_start"; id: string }
	| { type: "text_delta"; id: string; delta: string }
	| { type: "text_end"; id: string }
	| { type: "thinking_start"; id: string }
	| { type: "thinking_delta"; id: string; delta: string }
	| { type: "thinking_end"; id: string }
	| { type: "toolcall_start"; id: string; toolName: string }
	| { type: "toolcall_delta"; id: string; delta: string }
	| { type: "toolcall_end"; id: string }
	| {
			type: "done";
			messageId?: string;
			usage: RuntimeUsage;
			reason?: Extract<StopReason, "stop" | "length" | "toolUse">;
	  }
	| { type: "error"; error: Error };

export interface AgentRuntime {
	readonly configured?: boolean;
	stream(
		model: ModelHandle,
		context: AgentRuntimeContext,
		options?: AgentRuntimeOptions,
	): AsyncIterable<AgentRuntimeEvent>;
}

interface PiModelMetadata {
	api: Api;
	provider: Provider;
	reasoning: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

function isPiModel(value: unknown): value is Model<Api> {
	return (
		isRecord(value) &&
		typeof value.id === "string" &&
		typeof value.api === "string" &&
		typeof value.provider === "string" &&
		typeof value.reasoning === "boolean"
	);
}

function serializeMessageContent(content: AssistantMessage["content"][number]): string {
	switch (content.type) {
		case "text":
			return content.text;
		case "thinking":
			return content.thinking;
		case "toolCall":
			return `${content.name}(${JSON.stringify(content.arguments)})`;
	}
}

function serializeUserPart(content: TextContent | ImageContent): string {
	if (content.type === "text") {
		return content.text;
	}

	return `[image:${content.mimeType}]`;
}

function toRuntimeText(message: Message): string {
	if (message.role === "user") {
		return serializeUserContent(message);
	}

	if (message.role === "assistant") {
		return message.content.map(serializeMessageContent).join("\n");
	}

	return message.content.map(serializeUserPart).join("\n");
}

function serializeUserContent(message: UserMessage): string {
	if (typeof message.content === "string") {
		return message.content;
	}

	return message.content.map(serializeUserPart).join("\n");
}

function toRuntimeMessage(message: Message): RuntimeMessage {
	switch (message.role) {
		case "user":
			return {
				role: "user",
				content: toRuntimeText(message),
			};
		case "assistant":
			return {
				role: "assistant",
				content: toRuntimeText(message),
			};
		case "toolResult":
			return {
				role: "tool",
				content: toRuntimeText(message),
				toolCallId: message.toolCallId,
				toolName: message.toolName,
			};
	}
}

export function toRuntimeMessages(messages: Message[], systemPrompt?: string): RuntimeMessage[] {
	const runtimeMessages = messages.map(toRuntimeMessage);
	if (!systemPrompt) {
		return runtimeMessages;
	}

	return [{ role: "system", content: systemPrompt }, ...runtimeMessages];
}

function splitSystemPrompt(messages: RuntimeMessage[]): { systemPrompt: string; messages: RuntimeMessage[] } {
	const remaining: RuntimeMessage[] = [];
	const systemParts: string[] = [];

	for (const message of messages) {
		if (message.role === "system") {
			systemParts.push(message.content);
			continue;
		}
		remaining.push(message);
	}

	return {
		systemPrompt: systemParts.join("\n\n"),
		messages: remaining,
	};
}

export function createPlaceholderModelHandle(): ModelHandle {
	return {
		id: "runtime-unconfigured",
		displayName: "Unconfigured Model",
		family: "runtime",
		capabilities: {
			tools: true,
			images: true,
			streaming: true,
		},
		raw: undefined,
	};
}

export function toModelHandle(model: Model<Api>): ModelHandle {
	return {
		id: model.id,
		displayName: model.name,
		family: model.provider,
		capabilities: {
			tools: true,
			images: model.input.includes("image"),
			streaming: true,
			thinking: model.reasoning,
		},
		raw: model,
	};
}

export function getModelMetadata(model: ModelHandle): PiModelMetadata {
	if (isPiModel(model.raw)) {
		return {
			api: model.raw.api,
			provider: model.raw.provider,
			reasoning: model.raw.reasoning,
		};
	}

	return {
		api: "runtime",
		provider: model.family ?? "runtime",
		reasoning: model.capabilities.thinking === true,
	};
}

function createRuntimeUsageFromPi(message: AssistantMessage): RuntimeUsage {
	return {
		inputTokens: message.usage.input,
		outputTokens: message.usage.output,
		cachedInputTokens: message.usage.cacheRead,
		estimatedCost: message.usage.cost.total,
	};
}

function mapEvent(streamEvent: AssistantMessageEvent): AgentRuntimeEvent[] {
	switch (streamEvent.type) {
		case "start":
			return [{ type: "start", messageId: streamEvent.partial.responseId }];
		case "text_start":
			return [{ type: "text_start", id: `text-${streamEvent.contentIndex}` }];
		case "text_delta":
			return [{ type: "text_delta", id: `text-${streamEvent.contentIndex}`, delta: streamEvent.delta }];
		case "text_end":
			return [{ type: "text_end", id: `text-${streamEvent.contentIndex}` }];
		case "thinking_start":
			return [{ type: "thinking_start", id: `thinking-${streamEvent.contentIndex}` }];
		case "thinking_delta":
			return [{ type: "thinking_delta", id: `thinking-${streamEvent.contentIndex}`, delta: streamEvent.delta }];
		case "thinking_end":
			return [{ type: "thinking_end", id: `thinking-${streamEvent.contentIndex}` }];
		case "toolcall_start": {
			const content = streamEvent.partial.content[streamEvent.contentIndex];
			return [
				{
					type: "toolcall_start",
					id: content?.type === "toolCall" ? content.id : `tool-${streamEvent.contentIndex}`,
					toolName: content?.type === "toolCall" ? content.name : "tool",
				},
			];
		}
		case "toolcall_delta": {
			const content = streamEvent.partial.content[streamEvent.contentIndex];
			return [
				{
					type: "toolcall_delta",
					id: content?.type === "toolCall" ? content.id : `tool-${streamEvent.contentIndex}`,
					delta: streamEvent.delta,
				},
			];
		}
		case "toolcall_end":
			return [{ type: "toolcall_end", id: streamEvent.toolCall.id }];
		case "done":
			return [
				{
					type: "done",
					messageId: streamEvent.message.responseId,
					usage: createRuntimeUsageFromPi(streamEvent.message),
					reason: streamEvent.reason,
				},
			];
		case "error":
			return [
				{
					type: "error",
					error: new Error(streamEvent.error.errorMessage || "Runtime stream failed"),
				},
			];
	}
}

async function* streamFromPiAi(
	stream: EventStream<AssistantMessageEvent, AssistantMessage>,
): AsyncIterable<AgentRuntimeEvent> {
	for await (const event of stream) {
		for (const runtimeEvent of mapEvent(event)) {
			yield runtimeEvent;
		}
	}
}

export function createUnconfiguredRuntime(
	message = "No runtime configured. Pass `runtime` to Agent or AgentLoopConfig.",
): AgentRuntime {
	return {
		configured: false,
		async *stream() {
			yield {
				type: "error",
				error: new Error(message),
			};
		},
	};
}

export function createPiAiCompatRuntime(): AgentRuntime {
	return {
		configured: true,
		stream(model, context, options = {}) {
			const piModel = isPiModel(model.raw) ? model.raw : undefined;
			if (!piModel) {
				return (async function* (): AsyncIterable<AgentRuntimeEvent> {
					yield {
						type: "error",
						error: new Error(
							"Pi-ai compatibility runtime requires ModelHandle.raw to contain a pi-ai model. Use a llm-runtime implementation or pass a compatible handle explicitly.",
						),
					};
				})();
			}

			const { systemPrompt } = splitSystemPrompt(context.messages);
			const piContext: Context = {
				systemPrompt,
				messages: context.rawMessages ?? [],
				tools: context.tools,
			};

			const stream = streamSimple(piModel, piContext, {
				apiKey: options.apiKey,
				maxTokens: options.maxTokens,
				maxRetryDelayMs: options.maxRetryDelayMs,
				onPayload: options.onPayload ? (payload) => options.onPayload?.(payload, model) : undefined,
				reasoning: options.reasoning === "off" ? undefined : options.reasoning,
				sessionId: options.sessionId,
				signal: options.signal,
				thinkingBudgets: options.thinkingBudgets,
				transport: options.transport,
			});

			return streamFromPiAi(stream);
		},
	};
}

export function createAssistantUsage(usage?: RuntimeUsage): AssistantMessage["usage"] {
	const input = usage?.inputTokens ?? 0;
	const output = usage?.outputTokens ?? 0;
	const cacheRead = usage?.cachedInputTokens ?? 0;
	const total = usage?.estimatedCost ?? 0;

	return {
		input,
		output,
		cacheRead,
		cacheWrite: 0,
		totalTokens: input + output + cacheRead,
		cost: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			total,
		},
	};
}

export function createAssistantMessageShell(model: ModelHandle): AssistantMessage {
	const metadata = getModelMetadata(model);
	return {
		role: "assistant",
		content: [],
		api: metadata.api,
		provider: metadata.provider,
		model: model.id,
		usage: createAssistantUsage(),
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function cloneToolCall(content: ToolCall): ToolCall {
	return {
		...content,
		arguments: parseStreamingJson(JSON.stringify(content.arguments)) as Record<string, unknown>,
	};
}

function cloneAssistantContent(content: AssistantMessage["content"][number]): AssistantMessage["content"][number] {
	switch (content.type) {
		case "text":
			return { ...content };
		case "thinking":
			return { ...content };
		case "toolCall":
			return cloneToolCall(content);
	}
}

export function cloneAssistantMessage(message: AssistantMessage): AssistantMessage {
	return {
		...message,
		content: message.content.map(cloneAssistantContent),
		usage: {
			...message.usage,
			cost: { ...message.usage.cost },
		},
	};
}

export function parseToolArguments(delta: string): Record<string, unknown> {
	const parsed = parseStreamingJson(delta);
	if (isRecord(parsed)) {
		return parsed;
	}

	return {};
}
