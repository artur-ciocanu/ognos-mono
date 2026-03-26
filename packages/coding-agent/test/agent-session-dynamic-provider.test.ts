import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentRuntime } from "@mariozechner/pi-agent-core";
import { getModel, type Model } from "@mariozechner/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.js";
import { DefaultResourceLoader } from "../src/core/resource-loader.js";
import type { ExtensionFactory } from "../src/core/sdk.js";
import { createAgentSession } from "../src/core/sdk.js";
import { SessionManager } from "../src/core/session-manager.js";
import { SettingsManager } from "../src/core/settings-manager.js";

describe("AgentSession dynamic provider registration", () => {
	let tempDir: string;
	let agentDir: string;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-dynamic-provider-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		agentDir = join(tempDir, "agent");
		mkdirSync(agentDir, { recursive: true });
	});

	afterEach(() => {
		if (tempDir && existsSync(tempDir)) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	async function createSession(extensionFactories: ExtensionFactory[]) {
		const settingsManager = SettingsManager.create(tempDir, agentDir);
		const sessionManager = SessionManager.inMemory();
		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		authStorage.setRuntimeApiKey("anthropic", "test-key");
		const resourceLoader = new DefaultResourceLoader({
			cwd: tempDir,
			agentDir,
			settingsManager,
			extensionFactories,
		});
		await resourceLoader.reload();

		const { session } = await createAgentSession({
			cwd: tempDir,
			agentDir,
			model: getModel("anthropic", "claude-sonnet-4-5")!,
			settingsManager,
			sessionManager,
			authStorage,
			resourceLoader,
		});

		return session;
	}

	async function syncAnthropicSessionModel(session: Awaited<ReturnType<typeof createSession>>) {
		const model = session.modelRegistry.find("anthropic", "claude-sonnet-4-5");
		if (!model) {
			throw new Error("Expected anthropic/claude-sonnet-4-5 to be available in model registry");
		}
		await session.setModel(model);
		return model;
	}

	async function capturePromptBaseUrl(
		session: Awaited<ReturnType<typeof createSession>>,
	): Promise<string | undefined> {
		let baseUrl: string | undefined;
		session.agent.runtime = {
			configured: true,
			async *stream(model) {
				baseUrl = (model.raw as Model<any> | undefined)?.baseUrl;
				yield { type: "error", error: new Error("stop") };
			},
		} satisfies AgentRuntime;
		await session.prompt("hello");
		return baseUrl;
	}

	it("applies top-level registerProvider overrides to the active model", async () => {
		const session = await createSession([
			(pi) => {
				pi.registerProvider("anthropic", { baseUrl: "http://localhost:8080/top-level" });
			},
		]);

		await syncAnthropicSessionModel(session);
		expect(session.model?.raw.baseUrl).toBe("http://localhost:8080/top-level");
		expect(await capturePromptBaseUrl(session)).toBe("http://localhost:8080/top-level");

		session.dispose();
	});

	it("applies session_start registerProvider overrides to the active model", async () => {
		const session = await createSession([
			(pi) => {
				pi.on("session_start", () => {
					pi.registerProvider("anthropic", { baseUrl: "http://localhost:8080/session-start" });
				});
			},
		]);

		await session.bindExtensions({});
		await syncAnthropicSessionModel(session);

		expect(session.model?.raw.baseUrl).toBe("http://localhost:8080/session-start");
		expect(await capturePromptBaseUrl(session)).toBe("http://localhost:8080/session-start");

		session.dispose();
	});

	it("applies command-time registerProvider overrides without reload", async () => {
		const session = await createSession([
			(pi) => {
				pi.registerCommand("use-proxy", {
					description: "Use proxy",
					handler: async () => {
						pi.registerProvider("anthropic", { baseUrl: "http://localhost:8080/command" });
					},
				});
			},
		]);

		await session.bindExtensions({});
		await session.prompt("/use-proxy");
		await syncAnthropicSessionModel(session);

		expect(session.model?.raw.baseUrl).toBe("http://localhost:8080/command");
		expect(await capturePromptBaseUrl(session)).toBe("http://localhost:8080/command");

		session.dispose();
	});
});
