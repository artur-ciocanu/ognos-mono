import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.js";
import { toCodingAgentModelHandle } from "../src/core/model-handle.js";
import { ModelRegistry } from "../src/core/model-registry.js";
import { createAgentSession } from "../src/core/sdk.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";

const testModel = {
	id: "claude-sonnet-4-5",
	name: "Claude Sonnet 4.5",
	api: "anthropic-messages",
	provider: "anthropic",
	baseUrl: "https://api.anthropic.com",
	reasoning: true,
	input: ["text"],
	cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
	contextWindow: 200000,
	maxTokens: 8192,
} as const;

describe("createAgentSession session manager defaults", () => {
	let tempDir: string;
	let cwd: string;
	let agentDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-sdk-session-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		cwd = join(tempDir, "project");
		agentDir = join(tempDir, "agent");
		mkdirSync(cwd, { recursive: true });
		mkdirSync(agentDir, { recursive: true });
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	it("uses agentDir for the default persisted session path", async () => {
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model: testModel,
		});

		const safePath = `--${cwd.replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`;
		const expectedSessionDir = join(agentDir, "sessions", safePath);
		const sessionDir = session.sessionManager.getSessionDir();
		const sessionFile = session.sessionManager.getSessionFile();

		expect(sessionDir).toBe(expectedSessionDir);
		expect(sessionFile?.startsWith(`${expectedSessionDir}/`)).toBe(true);

		session.dispose();
	});

	it("keeps an explicit sessionManager override", async () => {
		const sessionManager = SessionManager.inMemory(cwd);
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model: testModel,
			sessionManager,
		});

		expect(session.sessionManager).toBe(sessionManager);
		expect(session.sessionManager.isPersisted()).toBe(false);

		session.dispose();
	});

	it("accepts plain pi-ai models and adapts them to coding-agent handles", async () => {
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model: testModel,
		});

		expect(session.model?.id).toBe("claude-sonnet-4-5");
		expect(session.model?.provider).toBe("anthropic");
		expect(session.model?.modelHandleId).toBeTruthy();
		expect(session.model?.raw.id).toBe("claude-sonnet-4-5");

		session.dispose();
	});

	it("falls back to the active model auth provider when request provider has no direct key", async () => {
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		authStorage.setRuntimeApiKey("anthropic", "runtime-key");
		const aliasedModel = {
			...toCodingAgentModelHandle(testModel, "anthropic"),
			provider: "openrouter",
			modelHandleId: "pi-model:test-openrouter-claude-sonnet-4-5",
			raw: { ...testModel, provider: "openrouter" },
		};

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			authStorage,
			model: aliasedModel,
		});

		await expect(session.agent.getApiKey?.("openrouter")).resolves.toBe("runtime-key");

		session.dispose();
	});

	it("prefers the request model auth provider when both auth and request provider keys exist", async () => {
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		authStorage.setRuntimeApiKey("anthropic", "anthropic-key");
		authStorage.setRuntimeApiKey("openrouter", "openrouter-key");
		const aliasedModel = {
			...toCodingAgentModelHandle(testModel, "anthropic"),
			provider: "openrouter",
			modelHandleId: "pi-model:test-openrouter-auth-provider-wins",
			raw: { ...testModel, provider: "openrouter" },
		};

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			authStorage,
			model: aliasedModel,
		});

		await expect(session.agent.getApiKey?.("openrouter", aliasedModel)).resolves.toBe("anthropic-key");

		session.dispose();
	});

	it("uses the in-flight request model for credential lookup when agent state changed", async () => {
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		authStorage.setRuntimeApiKey("anthropic", "anthropic-key");
		const requestModel = {
			...toCodingAgentModelHandle(testModel, "anthropic"),
			provider: "openrouter",
			modelHandleId: "pi-model:test-openrouter-request-model",
			raw: { ...testModel, provider: "openrouter" },
		};
		const staleStateModel = {
			...toCodingAgentModelHandle(testModel, "bedrock"),
			provider: "openrouter",
			modelHandleId: "pi-model:test-openrouter-stale-model",
			raw: { ...testModel, provider: "openrouter" },
		};

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			authStorage,
			model: requestModel,
		});
		session.agent.setModel(staleStateModel);

		await expect(session.agent.getApiKey?.("openrouter", requestModel)).resolves.toBe("anthropic-key");

		session.dispose();
	});

	it("uses the in-flight request model for missing-key guidance when agent state changed", async () => {
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		const requestModel = {
			...toCodingAgentModelHandle(testModel, "anthropic"),
			provider: "openrouter",
			modelHandleId: "pi-model:test-openrouter-request-guidance",
			raw: { ...testModel, provider: "openrouter" },
		};
		const staleStateModel = {
			...toCodingAgentModelHandle(testModel, "bedrock"),
			provider: "openrouter",
			modelHandleId: "pi-model:test-openrouter-stale-guidance",
			raw: { ...testModel, provider: "openrouter" },
		};

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			authStorage,
			model: requestModel,
		});
		session.agent.setModel(staleStateModel);

		await expect(session.agent.getApiKey?.("openrouter", requestModel)).rejects.toThrow(
			"No API key found for \"anthropic\". Set an API key environment variable or run '/login anthropic'.",
		);

		session.dispose();
	});

	it("uses a configured runtime for normal SDK prompt execution", async () => {
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model: testModel,
		});

		expect(session.agent.runtime.configured).toBe(true);

		await expect(session.prompt("hello")).rejects.toThrow("No API key found for anthropic.");

		session.dispose();
	});

	it("starts new scoped sessions on the first scoped model instead of an out-of-scope saved default", async () => {
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		const modelRegistry = new ModelRegistry(authStorage, join(agentDir, "models.json"));
		modelRegistry.registerProvider("demo-provider", {
			baseUrl: "https://provider.test/v1",
			apiKey: "TEST_KEY",
			api: "openai-completions",
			models: [
				{
					id: "default-model",
					name: "Default Model",
					reasoning: false,
					input: ["text"],
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: 128000,
					maxTokens: 4096,
				},
				{
					id: "scoped-model",
					name: "Scoped Model",
					reasoning: false,
					input: ["text"],
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: 128000,
					maxTokens: 4096,
				},
			],
		});
		const settingsManager = SettingsManager.create(cwd, agentDir);
		settingsManager.setDefaultModelAndProvider("demo-provider", "default-model");

		const scopedModel = modelRegistry.find("demo-provider", "scoped-model");
		expect(scopedModel).toBeDefined();

		const { session } = await createAgentSession({
			cwd,
			agentDir,
			authStorage,
			modelRegistry,
			settingsManager,
			scopedModels: [{ model: scopedModel! }],
		});

		expect(session.model?.provider).toBe("demo-provider");
		expect(session.model?.id).toBe("scoped-model");
		expect(session.scopedModels.map(({ model }) => model.id)).toEqual(["scoped-model"]);

		session.dispose();
	});
});
