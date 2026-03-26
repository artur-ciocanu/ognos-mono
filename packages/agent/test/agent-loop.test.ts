import type { AssistantMessage, Message, UserMessage } from "@mariozechner/pi-ai";
import { Type } from "@sinclair/typebox";
import type { ModelHandle, RuntimeUsage } from "pi-baml-runtime";
import { describe, expect, it } from "vitest";
import { agentLoop, agentLoopContinue } from "../src/agent-loop.js";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentRuntime,
	AgentRuntimeEvent,
	AgentTool,
} from "../src/types.js";

type RuntimeDoneReason = NonNullable<Extract<AgentRuntimeEvent, { type: "done" }>["reason"]>;

function createUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createModel(): ModelHandle {
	return {
		id: "mock",
		displayName: "mock",
		capabilities: {
			tools: true,
			images: true,
			streaming: true,
		},
		raw: {
			api: "openai-responses",
			provider: "openai",
		},
	};
}

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function createRuntimeUsage(): RuntimeUsage {
	return {
		inputTokens: 0,
		outputTokens: 0,
		cachedInputTokens: 0,
	};
}

function createUserMessage(text: string): UserMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

function hasRole(value: AgentMessage, role: string): value is AgentMessage & { role: string } {
	return "role" in value && value.role === role;
}

// Simple identity converter for tests - just passes through standard messages
function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
}

function createRuntime(eventsFactory: () => AgentRuntimeEvent[]): AgentRuntime {
	return {
		async *stream() {
			for (const event of eventsFactory()) {
				yield event;
			}
		},
	};
}

function createRuntimeFromMessages(messages: AssistantMessage[]): AgentRuntime {
	let callIndex = 0;

	return createRuntime(() => {
		const message = messages[Math.min(callIndex, messages.length - 1)];
		callIndex++;

		const events: AgentRuntimeEvent[] = [{ type: "start" }];
		let textIndex = 0;
		let thinkingIndex = 0;
		for (const content of message.content) {
			switch (content.type) {
				case "text":
					events.push({ type: "text_start", id: `text-${textIndex}` });
					if (content.text.length > 0) {
						events.push({ type: "text_delta", id: `text-${textIndex}`, delta: content.text });
					}
					events.push({ type: "text_end", id: `text-${textIndex}` });
					textIndex++;
					break;
				case "thinking":
					events.push({ type: "thinking_start", id: `thinking-${thinkingIndex}` });
					if (content.thinking.length > 0) {
						events.push({
							type: "thinking_delta",
							id: `thinking-${thinkingIndex}`,
							delta: content.thinking,
						});
					}
					events.push({ type: "thinking_end", id: `thinking-${thinkingIndex}` });
					thinkingIndex++;
					break;
				case "toolCall": {
					const serializedArguments = JSON.stringify(content.arguments);
					events.push({ type: "toolcall_start", id: content.id, toolName: content.name });
					if (serializedArguments.length > 0) {
						events.push({ type: "toolcall_delta", id: content.id, delta: serializedArguments });
					}
					events.push({ type: "toolcall_end", id: content.id });
					break;
				}
			}
		}

		events.push({
			type: "done",
			usage: createRuntimeUsage(),
			reason: normalizeRuntimeDoneReason(message.stopReason),
		});
		return events;
	});
}

function normalizeRuntimeDoneReason(stopReason: AssistantMessage["stopReason"]): RuntimeDoneReason {
	if (stopReason === "error" || stopReason === "aborted") {
		return "stop";
	}

	return stopReason;
}

describe("agentLoop with AgentMessage", () => {
	it("should emit events with AgentMessage types", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("Hello");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			runtime: createRuntime(() => [
				{ type: "start" },
				{ type: "text_start", id: "text-0" },
				{ type: "text_delta", id: "text-0", delta: "Hi there!" },
				{ type: "text_end", id: "text-0" },
				{ type: "done", usage: createRuntimeUsage(), reason: "stop" },
			]),
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config);

		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();

		// Should have user message and assistant message
		expect(messages.length).toBe(2);
		expect(messages[0].role).toBe("user");
		expect(messages[1].role).toBe("assistant");

		// Verify event sequence
		const eventTypes = events.map((e) => e.type);
		expect(eventTypes).toContain("agent_start");
		expect(eventTypes).toContain("turn_start");
		expect(eventTypes).toContain("message_start");
		expect(eventTypes).toContain("message_end");
		expect(eventTypes).toContain("turn_end");
		expect(eventTypes).toContain("agent_end");
	});

	it("should handle custom message types via convertToLlm", async () => {
		// Create a custom message type
		interface CustomNotification {
			role: "notification";
			text: string;
			timestamp: number;
		}

		const notification: CustomNotification = {
			role: "notification",
			text: "This is a notification",
			timestamp: Date.now(),
		};

		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [notification as unknown as AgentMessage], // Custom message in context
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("Hello");

		let convertedMessages: Message[] = [];
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				// Filter out notifications, convert rest
				convertedMessages = messages
					.filter((m) => (m as { role: string }).role !== "notification")
					.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
				return convertedMessages;
			},
			runtime: createRuntimeFromMessages([createAssistantMessage([{ type: "text", text: "Response" }])]),
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config);

		for await (const event of stream) {
			events.push(event);
		}

		// The notification should have been filtered out in convertToLlm
		expect(convertedMessages.length).toBe(1); // Only user message
		expect(convertedMessages[0].role).toBe("user");
	});

	it("should apply transformContext before convertToLlm", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [
				createUserMessage("old message 1"),
				createAssistantMessage([{ type: "text", text: "old response 1" }]),
				createUserMessage("old message 2"),
				createAssistantMessage([{ type: "text", text: "old response 2" }]),
			],
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("new message");

		let transformedMessages: AgentMessage[] = [];
		let convertedMessages: Message[] = [];

		const config: AgentLoopConfig = {
			model: createModel(),
			transformContext: async (messages) => {
				// Keep only last 2 messages (prune old ones)
				transformedMessages = messages.slice(-2);
				return transformedMessages;
			},
			convertToLlm: (messages) => {
				convertedMessages = messages.filter(
					(m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult",
				) as Message[];
				return convertedMessages;
			},
			runtime: createRuntimeFromMessages([createAssistantMessage([{ type: "text", text: "Response" }])]),
		};

		const stream = agentLoop([userPrompt], context, config);

		for await (const _ of stream) {
			// consume
		}

		// transformContext should have been called first, keeping only last 2
		expect(transformedMessages.length).toBe(2);
		// Then convertToLlm receives the pruned messages
		expect(convertedMessages.length).toBe(2);
	});

	it("should prepend the system prompt to runtime messages", async () => {
		const runtimeMessages: Array<{ role: string; content: string }> = [];
		const context: AgentContext = {
			systemPrompt: "System contract prompt",
			messages: [],
			tools: [],
		};

		const stream = agentLoop([createUserMessage("Hello")], context, {
			model: createModel(),
			convertToLlm: identityConverter,
			runtime: {
				async *stream(_model, runtimeContext) {
					for (const message of runtimeContext.messages) {
						runtimeMessages.push({ role: message.role, content: message.content });
					}
					yield { type: "start" };
					yield { type: "done", usage: createRuntimeUsage(), reason: "stop" };
				},
			},
		});

		for await (const _event of stream) {
			// consume
		}

		expect(runtimeMessages[0]).toEqual({ role: "system", content: "System contract prompt" });
		expect(runtimeMessages[1]).toEqual({ role: "user", content: "Hello" });
	});

	it("should handle tool calls and results", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo something");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			runtime: createRuntimeFromMessages([
				createAssistantMessage(
					[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
					"toolUse",
				),
				createAssistantMessage([{ type: "text", text: "done" }]),
			]),
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config);

		for await (const event of stream) {
			events.push(event);
		}

		// Tool should have been executed
		expect(executed).toEqual(["hello"]);

		// Should have tool execution events
		const toolStart = events.find((e) => e.type === "tool_execution_start");
		const toolEnd = events.find((e) => e.type === "tool_execution_end");
		expect(toolStart).toBeDefined();
		expect(toolEnd).toBeDefined();
		if (toolEnd?.type === "tool_execution_end") {
			expect(toolEnd.isError).toBe(false);
		}
	});

	it("should execute tool calls in parallel and emit tool results in source order", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let firstResolved = false;
		let parallelObserved = false;
		let releaseFirst: (() => void) | undefined;
		const firstDone = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});

		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				if (params.value === "first") {
					await firstDone;
					firstResolved = true;
				}
				if (params.value === "second" && !firstResolved) {
					parallelObserved = true;
				}
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo both");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
		};

		const runtime = createRuntimeFromMessages([
			createAssistantMessage(
				[
					{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
					{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
				],
				"toolUse",
			),
			createAssistantMessage([{ type: "text", text: "done" }]),
		]);
		setTimeout(() => releaseFirst?.(), 20);
		const stream = agentLoop([userPrompt], context, { ...config, runtime });

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolResultIds = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message.toolCallId];
		});

		expect(parallelObserved).toBe(true);
		expect(toolResultIds).toEqual(["tool-1", "tool-2"]);
	});

	it("should inject queued messages after all tool calls complete", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `ok:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("start");
		const queuedUserMessage: AgentMessage = createUserMessage("interrupt");

		let queuedDelivered = false;
		let callIndex = 0;
		let sawInterruptInContext = false;

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "sequential",
			getSteeringMessages: async () => {
				// Return steering message after tool execution has started.
				if (executed.length >= 1 && !queuedDelivered) {
					queuedDelivered = true;
					return [queuedUserMessage];
				}
				return [];
			},
			runtime: {
				async *stream(_model, ctx) {
					const currentCallIndex = callIndex;
					callIndex++;

					// Check if interrupt message is in context on second call
					if (currentCallIndex === 1) {
						sawInterruptInContext = ctx.messages.some((message) => {
							return message.role === "user" && message.content === "interrupt";
						});
					}

					if (currentCallIndex === 0) {
						yield* createRuntimeFromMessages([
							createAssistantMessage(
								[
									{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
									{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
								],
								"toolUse",
							),
						]).stream(createModel(), ctx);
					} else {
						yield* createRuntimeFromMessages([createAssistantMessage([{ type: "text", text: "done" }])]).stream(
							createModel(),
							ctx,
						);
					}
				},
			},
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config);

		for await (const event of stream) {
			events.push(event);
		}

		// Both tools should execute before steering is injected
		expect(executed).toEqual(["first", "second"]);

		const toolEnds = events.filter(
			(e): e is Extract<AgentEvent, { type: "tool_execution_end" }> => e.type === "tool_execution_end",
		);
		expect(toolEnds.length).toBe(2);
		expect(toolEnds[0].isError).toBe(false);
		expect(toolEnds[1].isError).toBe(false);

		// Queued message should appear in events after both tool result messages
		const eventSequence = events.flatMap((event) => {
			if (event.type !== "message_start") return [];
			if (event.message.role === "toolResult") return [`tool:${event.message.toolCallId}`];
			if (event.message.role === "user" && typeof event.message.content === "string") {
				return [event.message.content];
			}
			return [];
		});
		expect(eventSequence).toContain("interrupt");
		expect(eventSequence.indexOf("tool:tool-1")).toBeLessThan(eventSequence.indexOf("interrupt"));
		expect(eventSequence.indexOf("tool:tool-2")).toBeLessThan(eventSequence.indexOf("interrupt"));

		// Interrupt message should be in context when second LLM call is made
		expect(sawInterruptInContext).toBe(true);
	});
});

describe("agentLoopContinue with AgentMessage", () => {
	it("should throw when context has no messages", () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		expect(() => agentLoopContinue(context, config)).toThrow("Cannot continue: no messages in context");
	});

	it("should continue from existing context without emitting user message events", async () => {
		const userMessage: AgentMessage = createUserMessage("Hello");

		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [userMessage],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			runtime: createRuntimeFromMessages([createAssistantMessage([{ type: "text", text: "Response" }])]),
		};

		const events: AgentEvent[] = [];
		const stream = agentLoopContinue(context, config);

		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();

		// Should only return the new assistant message (not the existing user message)
		expect(messages.length).toBe(1);
		expect(messages[0].role).toBe("assistant");

		// Should NOT have user message events (that's the key difference from agentLoop)
		const messageEndEvents = events.filter((e) => e.type === "message_end");
		expect(messageEndEvents.length).toBe(1);
		expect(messageEndEvents[0]?.type === "message_end" ? messageEndEvents[0].message.role : undefined).toBe(
			"assistant",
		);
	});

	it("should allow custom message types as last message (caller responsibility)", async () => {
		// Custom message that will be converted to user message by convertToLlm
		interface CustomMessage {
			role: "custom";
			text: string;
			timestamp: number;
		}

		const customMessage: CustomMessage = {
			role: "custom",
			text: "Hook content",
			timestamp: Date.now(),
		};
		const isCustomMessage = (value: AgentMessage): value is AgentMessage & CustomMessage => {
			return hasRole(value, "custom") && "text" in value && "timestamp" in value;
		};

		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [customMessage as unknown as AgentMessage],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				// Convert custom to user message
				return messages
					.map((m) => {
						if (isCustomMessage(m)) {
							return {
								role: "user" as const,
								content: m.text,
								timestamp: m.timestamp,
							};
						}
						return m;
					})
					.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
			},
			runtime: createRuntimeFromMessages([
				createAssistantMessage([{ type: "text", text: "Response to custom message" }]),
			]),
		};

		// Should not throw - the custom message will be converted to user message
		const stream = agentLoopContinue(context, config);

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		expect(messages.length).toBe(1);
		expect(messages[0].role).toBe("assistant");
	});

	it("should end with an assistant error message when runtime throws before streaming", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [],
			tools: [],
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([createUserMessage("Hello")], context, {
			model: createModel(),
			convertToLlm: identityConverter,
			runtime: createRuntime(() => {
				throw new Error("runtime failed early");
			}),
		});

		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		expect(messages).toHaveLength(2);
		expect(messages[1]?.role).toBe("assistant");
		if (messages[1]?.role === "assistant") {
			expect(messages[1].stopReason).toBe("error");
			expect(messages[1].errorMessage).toBe("runtime failed early");
		}
		expect(events[events.length - 1]?.type).toBe("agent_end");
	});

	it("should end continue streams with an assistant error message when runtime throws before streaming", async () => {
		const context: AgentContext = {
			systemPrompt: "You are helpful.",
			messages: [createUserMessage("Hello")],
			tools: [],
		};

		const stream = agentLoopContinue(context, {
			model: createModel(),
			convertToLlm: identityConverter,
			runtime: createRuntime(() => {
				throw new Error("continue failed early");
			}),
		});

		for await (const _event of stream) {
			// consume
		}

		const messages = await stream.result();
		expect(messages).toHaveLength(1);
		expect(messages[0]?.role).toBe("assistant");
		if (messages[0]?.role === "assistant") {
			expect(messages[0].stopReason).toBe("error");
			expect(messages[0].errorMessage).toBe("continue failed early");
		}
	});
});
