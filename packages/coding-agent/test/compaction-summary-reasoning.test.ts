import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	Agent,
	type AgentMessage,
	type AgentRuntime,
	type AgentRuntimeEvent,
	toModelHandle,
} from "@mariozechner/pi-agent-core";
import type { AssistantMessage, AssistantMessageEvent, Model } from "@mariozechner/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import { generateSummary } from "../src/core/compaction/index.js";
import { executeSummary } from "../src/core/compaction/summary-runtime.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { createTestResourceLoader } from "./utilities.js";

const { streamSimpleMock } = vi.hoisted(() => ({
	streamSimpleMock: vi.fn(),
}));

vi.mock("@mariozechner/pi-ai", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@mariozechner/pi-ai")>();
	return {
		...actual,
		streamSimple: streamSimpleMock,
	};
});

function createModel(reasoning: boolean): Model<"anthropic-messages"> {
	return {
		id: reasoning ? "reasoning-model" : "non-reasoning-model",
		name: reasoning ? "Reasoning Model" : "Non-reasoning Model",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "https://api.anthropic.com",
		reasoning,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 8192,
	};
}

function createCompatAssistant(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "reasoning-model",
		usage: {
			input: 10,
			output: 10,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 20,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

async function* createSummaryRuntimeStream(text: string): AsyncIterable<AgentRuntimeEvent> {
	yield { type: "start", messageId: "summary-message" };
	yield { type: "text_start", id: "text-0" };
	yield { type: "text_delta", id: "text-0", delta: text };
	yield {
		type: "done",
		messageId: "summary-message",
		usage: { inputTokens: 10, outputTokens: 10, cachedInputTokens: 0, estimatedCost: 0 },
		reason: "stop",
	};
}

async function* createCompatStream(text: string): AsyncIterable<AssistantMessageEvent> {
	const message = createCompatAssistant(text);
	yield { type: "start", partial: message };
	yield { type: "text_start", contentIndex: 0, partial: message };
	yield {
		type: "text_delta",
		contentIndex: 0,
		delta: text,
		partial: {
			...message,
			content: [{ type: "text", text }],
		},
	};
	yield {
		type: "text_end",
		contentIndex: 0,
		content: text,
		partial: {
			...message,
			content: [{ type: "text", text }],
		},
	};
	yield {
		type: "done",
		reason: "stop",
		message: {
			...message,
			content: [{ type: "text", text }],
		},
	};
}

const messages: AgentMessage[] = [{ role: "user", content: "Summarize this.", timestamp: Date.now() }];

function createAssistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "non-reasoning-model",
		usage: {
			input: 20,
			output: 10,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 30,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

describe("compaction summary runtime execution", () => {
	beforeEach(() => {
		streamSimpleMock.mockReset();
	});

	it("uses the active runtime when provided and forwards reasoning/maxTokens", async () => {
		const runtimeStreamMock = vi.fn((..._args: Parameters<AgentRuntime["stream"]>) =>
			createSummaryRuntimeStream("## Goal\nRuntime summary"),
		);
		const runtime: AgentRuntime = {
			stream: runtimeStreamMock,
		};

		const result = await generateSummary(
			messages,
			createModel(true),
			2000,
			"test-key",
			undefined,
			undefined,
			undefined,
			runtime,
		);

		expect(result).toBe("## Goal\nRuntime summary");
		expect(runtimeStreamMock).toHaveBeenCalledTimes(1);
		expect(runtimeStreamMock.mock.calls[0][2]).toMatchObject({
			reasoning: "high",
			apiKey: "test-key",
			maxTokens: 1600,
		});
		expect(streamSimpleMock).not.toHaveBeenCalled();
	});

	it("falls back to the compat runtime and forwards maxTokens without spurious reasoning", async () => {
		streamSimpleMock.mockImplementation(() => createCompatStream("## Goal\nCompat summary"));

		const result = await executeSummary({
			model: createModel(false),
			systemPrompt: "Summarize",
			messages,
			apiKey: "test-key",
			maxTokens: 1600,
		});

		expect(result).toEqual({ status: "ok", text: "## Goal\nCompat summary" });
		expect(streamSimpleMock).toHaveBeenCalledTimes(1);
		expect(streamSimpleMock.mock.calls[0][2]).toMatchObject({
			apiKey: "test-key",
			maxTokens: 1600,
			reasoning: undefined,
		});
	});

	it("uses compat fallback for default AgentSession runtimes that are still unconfigured", async () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-compaction-runtime-"));
		const model = createModel(false);
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: toModelHandle(model),
				systemPrompt: "You are concise.",
				tools: [],
			},
		});
		const sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		settingsManager.applyOverrides({ compaction: { keepRecentTokens: 1 } });
		const authStorage = AuthStorage.inMemory();
		authStorage.setRuntimeApiKey(model.provider, "test-key");
		const modelRegistry = new ModelRegistry(authStorage, tempDir);
		const session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRegistry,
			resourceLoader: createTestResourceLoader(),
		});

		try {
			session.subscribe(() => {});
			sessionManager.appendMessage({ role: "user", content: "First", timestamp: Date.now() });
			sessionManager.appendMessage(createAssistantMessage("First reply"));
			sessionManager.appendMessage({ role: "user", content: "Second", timestamp: Date.now() });
			sessionManager.appendMessage(createAssistantMessage("Second reply"));
			agent.replaceMessages(sessionManager.buildSessionContext().messages);

			streamSimpleMock.mockImplementation(() => createCompatStream("## Goal\nSession compat summary"));

			const result = await session.compact();

			expect(result.summary).toContain("Session compat summary");
			expect(streamSimpleMock.mock.calls.length).toBeGreaterThan(0);
		} finally {
			session.dispose();
			rmSync(tempDir, { recursive: true, force: true });
		}
	});
});
