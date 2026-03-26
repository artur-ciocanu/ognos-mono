import { Agent, type ThinkingLevel, toModelHandle } from "@mariozechner/pi-agent-core";
import { getModel } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import { AgentSession } from "../src/core/agent-session.js";
import { AuthStorage } from "../src/core/auth-storage.js";
import { toCodingAgentModelHandle } from "../src/core/model-handle.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";
import { createTestResourceLoader } from "./utilities.js";

const reasoningModel = getModel("anthropic", "claude-sonnet-4-5")!;
const nonReasoningModel = getModel("anthropic", "claude-3-5-haiku-latest")!;

function createSession({
	thinkingLevel = "high",
	defaultThinkingLevel = thinkingLevel,
	scopedModels,
}: {
	thinkingLevel?: ThinkingLevel;
	defaultThinkingLevel?: ThinkingLevel;
	scopedModels?: Array<{ model: ReturnType<typeof toCodingAgentModelHandle>; thinkingLevel?: ThinkingLevel }>;
} = {}) {
	const settingsManager = SettingsManager.inMemory({ defaultThinkingLevel });
	const sessionManager = SessionManager.inMemory();
	const authStorage = AuthStorage.inMemory();
	authStorage.setRuntimeApiKey("anthropic", "test-key");
	const session = new AgentSession({
		agent: new Agent({
			getApiKey: () => "test-key",
			initialState: {
				model: toModelHandle(reasoningModel),
				systemPrompt: "You are a helpful assistant.",
				tools: [],
				thinkingLevel,
			},
		}),
		sessionManager,
		settingsManager,
		cwd: process.cwd(),
		modelRegistry: new ModelRegistry(authStorage, undefined),
		resourceLoader: createTestResourceLoader(),
		scopedModels,
	});

	return { session, sessionManager, settingsManager };
}

describe("AgentSession model switching", () => {
	it("preserves the saved thinking preference through non-reasoning models", async () => {
		const { session, sessionManager, settingsManager } = createSession({
			scopedModels: [
				{ model: toCodingAgentModelHandle(reasoningModel) },
				{ model: toCodingAgentModelHandle(nonReasoningModel) },
			],
		});

		try {
			await session.setModel(nonReasoningModel);
			expect(session.thinkingLevel).toBe("off");
			expect(settingsManager.getDefaultThinkingLevel()).toBe("high");

			await session.setModel(reasoningModel);
			expect(session.thinkingLevel).toBe("high");

			await session.cycleModel();
			expect(session.thinkingLevel).toBe("off");
			expect(settingsManager.getDefaultThinkingLevel()).toBe("high");

			await session.cycleModel();
			expect(session.thinkingLevel).toBe("high");
			expect(settingsManager.getDefaultThinkingLevel()).toBe("high");
			expect(
				sessionManager
					.getEntries()
					.filter((entry) => entry.type === "thinking_level_change")
					.map((entry) => entry.thinkingLevel),
			).toEqual(["off", "high", "off", "high"]);
		} finally {
			session.dispose();
		}
	});

	it("normalizes plain pi-ai models before persisting setModel selections", async () => {
		const { session, sessionManager, settingsManager } = createSession();

		try {
			await session.setModel(nonReasoningModel);

			const modelChanges = sessionManager.getEntries().filter((entry) => entry.type === "model_change");
			expect(modelChanges).toHaveLength(1);
			expect(modelChanges[0]?.modelHandleId).toBeTruthy();
			expect(modelChanges[0]?.modelId).toBe("claude-3-5-haiku-latest");
			expect(settingsManager.getDefaultModelHandleId()).toBeTruthy();
			expect(settingsManager.getDefaultProvider()).toBe("anthropic");
		} finally {
			session.dispose();
		}
	});
});
