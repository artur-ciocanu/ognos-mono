import { type AssistantMessage, getModel } from "@mariozechner/pi-ai";
import { Type } from "@sinclair/typebox";
import { describe, expect, it } from "vitest";
import {
	Agent,
	type AgentRuntime,
	type AgentRuntimeEvent,
	type AgentTool,
	createPiAiCompatRuntime,
	toModelHandle,
} from "../src/index.js";

function createAssistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function createRuntime(...responses: AssistantMessage[]): AgentRuntime {
	let index = 0;

	return {
		async *stream(_model, _context, options) {
			const message = responses[Math.min(index, responses.length - 1)];
			index++;

			yield { type: "start" } satisfies AgentRuntimeEvent;
			yield { type: "text_start", id: "text-0" } satisfies AgentRuntimeEvent;
			if (message.content[0]?.type === "text" && message.content[0].text.length > 0) {
				yield {
					type: "text_delta",
					id: "text-0",
					delta: message.content[0].text,
				} satisfies AgentRuntimeEvent;
			}

			while (!options?.signal?.aborted && message.content[0]?.type === "text" && message.content[0].text === "") {
				await new Promise((resolve) => setTimeout(resolve, 5));
			}

			if (options?.signal?.aborted) {
				yield { type: "error", error: new Error("Aborted") } satisfies AgentRuntimeEvent;
				return;
			}

			yield { type: "text_end", id: "text-0" } satisfies AgentRuntimeEvent;
			yield {
				type: "done",
				usage: {
					inputTokens: 0,
					outputTokens: 0,
					cachedInputTokens: 0,
				},
				reason: "stop",
			} satisfies AgentRuntimeEvent;
		},
	};
}

describe("Agent", () => {
	it("should create an agent instance with default state", () => {
		const agent = new Agent();

		expect(agent.state).toBeDefined();
		expect(agent.state.systemPrompt).toBe("");
		expect(agent.state.model).toBeDefined();
		expect(agent.state.thinkingLevel).toBe("off");
		expect(agent.state.tools).toEqual([]);
		expect(agent.state.messages).toEqual([]);
		expect(agent.state.isStreaming).toBe(false);
		expect(agent.state.streamMessage).toBe(null);
		expect(agent.state.pendingToolCalls).toEqual(new Set());
		expect(agent.state.error).toBeUndefined();
	});

	it("should fail clearly when execution starts without a runtime", async () => {
		const agent = new Agent();

		await expect(agent.prompt("Hello")).resolves.toBeUndefined();
		expect(agent.state.error).toBe("No runtime configured. Pass `runtime` to Agent or AgentLoopConfig.");

		const lastMessage = agent.state.messages[agent.state.messages.length - 1];
		expect(lastMessage?.role).toBe("assistant");
		if (lastMessage?.role === "assistant") {
			expect(lastMessage.stopReason).toBe("error");
			expect(lastMessage.errorMessage).toBe("No runtime configured. Pass `runtime` to Agent or AgentLoopConfig.");
		}
	});

	it("should create an agent instance with custom initial state", () => {
		const customModel = toModelHandle(getModel("openai", "gpt-4o-mini"));
		const agent = new Agent({
			initialState: {
				systemPrompt: "You are a helpful assistant.",
				model: customModel,
				thinkingLevel: "low",
			},
		});

		expect(agent.state.systemPrompt).toBe("You are a helpful assistant.");
		expect(agent.state.model).toBe(customModel);
		expect(agent.state.thinkingLevel).toBe("low");
	});

	it("should subscribe to events", () => {
		const agent = new Agent();

		let eventCount = 0;
		const unsubscribe = agent.subscribe((_event) => {
			eventCount++;
		});

		// No initial event on subscribe
		expect(eventCount).toBe(0);

		// State mutators don't emit events
		agent.setSystemPrompt("Test prompt");
		expect(eventCount).toBe(0);
		expect(agent.state.systemPrompt).toBe("Test prompt");

		// Unsubscribe should work
		unsubscribe();
		agent.setSystemPrompt("Another prompt");
		expect(eventCount).toBe(0); // Should not increase
	});

	it("should update state with mutators", () => {
		const agent = new Agent();

		// Test setSystemPrompt
		agent.setSystemPrompt("Custom prompt");
		expect(agent.state.systemPrompt).toBe("Custom prompt");

		// Test setModel
		const newModel = toModelHandle(getModel("google", "gemini-2.5-flash"));
		agent.setModel(newModel);
		expect(agent.state.model).toBe(newModel);

		// Test setThinkingLevel
		agent.setThinkingLevel("high");
		expect(agent.state.thinkingLevel).toBe("high");

		// Test setTools
		const tools: AgentTool[] = [
			{
				name: "test",
				label: "Test",
				description: "test tool",
				parameters: Type.Object({}),
				async execute() {
					return {
						content: [{ type: "text", text: "ok" }],
						details: {},
					};
				},
			},
		];
		agent.setTools(tools);
		expect(agent.state.tools).toBe(tools);

		// Test replaceMessages
		const messages = [{ role: "user" as const, content: "Hello", timestamp: Date.now() }];
		agent.replaceMessages(messages);
		expect(agent.state.messages).toEqual(messages);
		expect(agent.state.messages).not.toBe(messages); // Should be a copy

		// Test appendMessage
		const newMessage = createAssistantMessage("Hi");
		agent.appendMessage(newMessage);
		expect(agent.state.messages).toHaveLength(2);
		expect(agent.state.messages[1]).toBe(newMessage);

		// Test clearMessages
		agent.clearMessages();
		expect(agent.state.messages).toEqual([]);
	});

	it("should support steering message queue", async () => {
		const agent = new Agent();

		const message = { role: "user" as const, content: "Steering message", timestamp: Date.now() };
		agent.steer(message);

		// The message is queued but not yet in state.messages
		expect(agent.state.messages).not.toContainEqual(message);
	});

	it("should support follow-up message queue", async () => {
		const agent = new Agent();

		const message = { role: "user" as const, content: "Follow-up message", timestamp: Date.now() };
		agent.followUp(message);

		// The message is queued but not yet in state.messages
		expect(agent.state.messages).not.toContainEqual(message);
	});

	it("should handle abort controller", () => {
		const agent = new Agent();

		// Should not throw even if nothing is running
		expect(() => agent.abort()).not.toThrow();
	});

	it("should throw when prompt() called while streaming", async () => {
		let abortSignal: AbortSignal | undefined;
		const agent = new Agent({
			runtime: {
				async *stream(_model, _context, options) {
					abortSignal = options?.signal;
					yield { type: "start" } satisfies AgentRuntimeEvent;
					yield { type: "text_start", id: "text-0" } satisfies AgentRuntimeEvent;
					while (!abortSignal?.aborted) {
						await new Promise((resolve) => setTimeout(resolve, 5));
					}
					yield { type: "error", error: new Error("Aborted") } satisfies AgentRuntimeEvent;
				},
			},
		});

		// Start first prompt (don't await, it will block until abort)
		const firstPrompt = agent.prompt("First message");

		// Wait a tick for isStreaming to be set
		await new Promise((resolve) => setTimeout(resolve, 10));
		expect(agent.state.isStreaming).toBe(true);

		// Second prompt should reject
		await expect(agent.prompt("Second message")).rejects.toThrow(
			"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
		);

		// Cleanup - abort to stop the stream
		agent.abort();
		await firstPrompt.catch(() => {}); // Ignore abort error
	});

	it("should throw when continue() called while streaming", async () => {
		let abortSignal: AbortSignal | undefined;
		const agent = new Agent({
			runtime: {
				async *stream(_model, _context, options) {
					abortSignal = options?.signal;
					yield { type: "start" } satisfies AgentRuntimeEvent;
					yield { type: "text_start", id: "text-0" } satisfies AgentRuntimeEvent;
					while (!abortSignal?.aborted) {
						await new Promise((resolve) => setTimeout(resolve, 5));
					}
					yield { type: "error", error: new Error("Aborted") } satisfies AgentRuntimeEvent;
				},
			},
		});

		// Start first prompt
		const firstPrompt = agent.prompt("First message");
		await new Promise((resolve) => setTimeout(resolve, 10));
		expect(agent.state.isStreaming).toBe(true);

		// continue() should reject
		await expect(agent.continue()).rejects.toThrow(
			"Agent is already processing. Wait for completion before continuing.",
		);

		// Cleanup
		agent.abort();
		await firstPrompt.catch(() => {});
	});

	it("continue() should process queued follow-up messages after an assistant turn", async () => {
		const agent = new Agent({
			runtime: createRuntime(createAssistantMessage("Processed")),
		});

		agent.replaceMessages([
			{
				role: "user",
				content: [{ type: "text", text: "Initial" }],
				timestamp: Date.now() - 10,
			},
			createAssistantMessage("Initial response"),
		]);

		agent.followUp({
			role: "user",
			content: [{ type: "text", text: "Queued follow-up" }],
			timestamp: Date.now(),
		});

		await expect(agent.continue()).resolves.toBeUndefined();

		const hasQueuedFollowUp = agent.state.messages.some((message) => {
			if (message.role !== "user") return false;
			if (typeof message.content === "string") return message.content === "Queued follow-up";
			return message.content.some((part) => part.type === "text" && part.text === "Queued follow-up");
		});

		expect(hasQueuedFollowUp).toBe(true);
		expect(agent.state.messages[agent.state.messages.length - 1].role).toBe("assistant");
	});

	it("continue() should keep one-at-a-time steering semantics from assistant tail", async () => {
		let responseCount = 0;
		const agent = new Agent({
			runtime: {
				async *stream() {
					responseCount++;
					yield* createRuntime(createAssistantMessage(`Processed ${responseCount}`)).stream(
						toModelHandle(getModel("openai", "gpt-4o-mini")),
						{ messages: [] },
					);
				},
			},
		});

		agent.replaceMessages([
			{
				role: "user",
				content: [{ type: "text", text: "Initial" }],
				timestamp: Date.now() - 10,
			},
			createAssistantMessage("Initial response"),
		]);

		agent.steer({
			role: "user",
			content: [{ type: "text", text: "Steering 1" }],
			timestamp: Date.now(),
		});
		agent.steer({
			role: "user",
			content: [{ type: "text", text: "Steering 2" }],
			timestamp: Date.now() + 1,
		});

		await expect(agent.continue()).resolves.toBeUndefined();

		const recentMessages = agent.state.messages.slice(-4);
		expect(recentMessages.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
		expect(responseCount).toBe(2);
	});

	it("forwards sessionId to runtime options", async () => {
		let receivedSessionId: string | undefined;
		const agent = new Agent({
			sessionId: "session-abc",
			runtime: {
				async *stream(_model, _context, options) {
					receivedSessionId = options?.sessionId;
					yield* createRuntime(createAssistantMessage("ok")).stream(
						toModelHandle(getModel("openai", "gpt-4o-mini")),
						{ messages: [] },
					);
				},
			},
		});

		await agent.prompt("hello");
		expect(receivedSessionId).toBe("session-abc");

		// Test setter
		agent.sessionId = "session-def";
		expect(agent.sessionId).toBe("session-def");

		await agent.prompt("hello again");
		expect(receivedSessionId).toBe("session-def");
	});

	it("supports explicit pi-ai compatibility runtime when requested", async () => {
		const agent = new Agent({
			runtime: createPiAiCompatRuntime(),
			initialState: {
				model: toModelHandle(getModel("google", "gemini-2.5-flash")),
			},
		});

		expect(agent.runtime).toBeDefined();
	});
});
