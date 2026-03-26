import { join } from "node:path";
import { Agent, type AgentMessage, createPiAiCompatRuntime, type ThinkingLevel } from "@mariozechner/pi-agent-core";
import type { Message } from "@mariozechner/pi-ai";
import type { ModelHandle } from "@mariozechner/pi-llm-runtime";
import { getAgentDir, getDocsPath } from "../config.js";
import { AgentSession } from "./agent-session.js";
import { AuthStorage } from "./auth-storage.js";
import { DEFAULT_THINKING_LEVEL } from "./defaults.js";
import type { ExtensionRunner, LoadExtensionsResult, ToolDefinition } from "./extensions/index.js";
import { convertToLlm } from "./messages.js";
import {
	type CodingAgentModelHandle,
	type CompatiblePiModel,
	formatPersistedModelReference,
	getAuthProvider,
	normalizeModelHandle,
	toPersistedModelReference,
} from "./model-handle.js";
import { ModelRegistry } from "./model-registry.js";
import { findInitialModel } from "./model-resolver.js";
import type { ResourceLoader } from "./resource-loader.js";
import { DefaultResourceLoader } from "./resource-loader.js";
import { getDefaultSessionDir, SessionManager } from "./session-manager.js";
import { SettingsManager } from "./settings-manager.js";
import { time } from "./timings.js";
import {
	allTools,
	bashTool,
	codingTools,
	createBashTool,
	createCodingTools,
	createEditTool,
	createFindTool,
	createGrepTool,
	createLsTool,
	createReadOnlyTools,
	createReadTool,
	createWriteTool,
	editTool,
	findTool,
	grepTool,
	lsTool,
	readOnlyTools,
	readTool,
	type Tool,
	type ToolName,
	withFileMutationQueue,
	writeTool,
} from "./tools/index.js";

export interface CreateAgentSessionOptions {
	/** Working directory for project-local discovery. Default: process.cwd() */
	cwd?: string;
	/** Global config directory. Default: ~/.pi/agent */
	agentDir?: string;

	/** Auth storage for credentials. Default: AuthStorage.create(agentDir/auth.json) */
	authStorage?: AuthStorage;
	/** Model registry. Default: new ModelRegistry(authStorage, agentDir/models.json) */
	modelRegistry?: ModelRegistry;

	/** Model to use. Default: from settings, else the deterministically ranked available fallback model */
	model?: CodingAgentModelHandle | CompatiblePiModel;
	/** Thinking level. Default: from settings, else 'medium' (clamped to model capabilities) */
	thinkingLevel?: ThinkingLevel;
	/** Models available for cycling (Ctrl+P in interactive mode) */
	scopedModels?: Array<{ model: CodingAgentModelHandle | CompatiblePiModel; thinkingLevel?: ThinkingLevel }>;

	/** Built-in tools to use. Default: codingTools [read, bash, edit, write] */
	tools?: Tool[];
	/** Custom tools to register (in addition to built-in tools). */
	customTools?: ToolDefinition[];

	/** Resource loader. When omitted, DefaultResourceLoader is used. */
	resourceLoader?: ResourceLoader;

	/** Session manager. Default: SessionManager.create(cwd) */
	sessionManager?: SessionManager;

	/** Settings manager. Default: SettingsManager.create(cwd, agentDir) */
	settingsManager?: SettingsManager;
}

/** Result from createAgentSession */
export interface CreateAgentSessionResult {
	/** The created session */
	session: AgentSession;
	/** Extensions result (for UI context setup in interactive mode) */
	extensionsResult: LoadExtensionsResult;
	/** Warning if session was restored with a different model than saved */
	modelFallbackMessage?: string;
}

// Re-exports

export type {
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
	ExtensionFactory,
	SlashCommandInfo,
	SlashCommandSource,
	ToolDefinition,
} from "./extensions/index.js";
export type { PromptTemplate } from "./prompt-templates.js";
export type { Skill } from "./skills.js";
export type { Tool } from "./tools/index.js";

export {
	// Pre-built tools (use process.cwd())
	readTool,
	bashTool,
	editTool,
	writeTool,
	grepTool,
	findTool,
	lsTool,
	codingTools,
	readOnlyTools,
	allTools as allBuiltInTools,
	withFileMutationQueue,
	// Tool factories (for custom cwd)
	createCodingTools,
	createReadOnlyTools,
	createReadTool,
	createBashTool,
	createEditTool,
	createWriteTool,
	createGrepTool,
	createFindTool,
	createLsTool,
};

// Helper Functions

function getDefaultAgentDir(): string {
	return getAgentDir();
}

function getRequestProvider(model: ModelHandle | undefined): string | undefined {
	if (!model || typeof model !== "object") {
		return undefined;
	}
	if ("provider" in model && typeof model.provider === "string") {
		return model.provider;
	}
	return typeof model.family === "string" ? model.family : undefined;
}

function getRequestAuthProvider(model: ModelHandle | undefined): string | undefined {
	if (!model || typeof model !== "object") {
		return undefined;
	}
	if ("authProvider" in model && typeof model.authProvider === "string") {
		return model.authProvider;
	}
	return getRequestProvider(model);
}

/**
 * Create an AgentSession with the specified options.
 *
 * @example
 * ```typescript
 * // Minimal - uses defaults
 * const { session } = await createAgentSession();
 *
 * // With explicit model
 * import { getModel } from '@mariozechner/pi-ai';
 * const { session } = await createAgentSession({
 *   model: getModel('anthropic', 'claude-opus-4-5'),
 *   thinkingLevel: 'high',
 * });
 *
 * // Continue previous session
 * const { session, modelFallbackMessage } = await createAgentSession({
 *   continueSession: true,
 * });
 *
 * // Full control
 * const loader = new DefaultResourceLoader({
 *   cwd: process.cwd(),
 *   agentDir: getAgentDir(),
 *   settingsManager: SettingsManager.create(),
 * });
 * await loader.reload();
 * const { session } = await createAgentSession({
 *   model: myModel,
 *   tools: [readTool, bashTool],
 *   resourceLoader: loader,
 *   sessionManager: SessionManager.inMemory(),
 * });
 * ```
 */
export async function createAgentSession(options: CreateAgentSessionOptions = {}): Promise<CreateAgentSessionResult> {
	const cwd = options.cwd ?? process.cwd();
	const agentDir = options.agentDir ?? getDefaultAgentDir();
	let resourceLoader = options.resourceLoader;

	// Use provided or create AuthStorage and ModelRegistry
	const authPath = options.agentDir ? join(agentDir, "auth.json") : undefined;
	const modelsPath = options.agentDir ? join(agentDir, "models.json") : undefined;
	const authStorage = options.authStorage ?? AuthStorage.create(authPath);
	const modelRegistry = options.modelRegistry ?? new ModelRegistry(authStorage, modelsPath);

	const settingsManager = options.settingsManager ?? SettingsManager.create(cwd, agentDir);
	const sessionManager = options.sessionManager ?? SessionManager.create(cwd, getDefaultSessionDir(cwd, agentDir));

	if (!resourceLoader) {
		resourceLoader = new DefaultResourceLoader({ cwd, agentDir, settingsManager });
		await resourceLoader.reload();
		time("resourceLoader.reload");
	}

	// Check if session has existing data to restore
	const existingSession = sessionManager.buildSessionContext();
	const hasExistingSession = existingSession.messages.length > 0;
	const hasThinkingEntry = sessionManager.getBranch().some((entry) => entry.type === "thinking_level_change");

	let model = normalizeModelHandle(options.model);
	let modelFallbackMessage: string | undefined;
	let initialThinkingLevel = options.thinkingLevel;
	const scopedModels = options.scopedModels?.map((scoped) => ({
		model: normalizeModelHandle(scoped.model)!,
		thinkingLevel: scoped.thinkingLevel,
	}));

	// If session has data, try to restore model from it
	if (!model && hasExistingSession && existingSession.model) {
		const restoredModel = modelRegistry.resolveStoredModel(existingSession.model);
		if (restoredModel && (await modelRegistry.getApiKey(restoredModel))) {
			model = restoredModel;
		}
		if (!model) {
			modelFallbackMessage = `Could not restore model ${formatPersistedModelReference(existingSession.model)}`;
		}
	}

	// If still no model, use findInitialModel (checks settings defaults, then deterministic fallback ordering)
	if (!model) {
		const result = await findInitialModel({
			scopedModels: scopedModels ?? [],
			isContinuing: hasExistingSession,
			defaultModelReference: settingsManager.getDefaultModelSelection(),
			defaultModelHandleId: settingsManager.getDefaultModelHandleId(),
			defaultProvider: settingsManager.getDefaultProvider(),
			defaultModelId: settingsManager.getDefaultModel(),
			defaultThinkingLevel: settingsManager.getDefaultThinkingLevel(),
			modelRegistry,
		});
		model = result.model;
		if (initialThinkingLevel === undefined) {
			initialThinkingLevel = result.thinkingLevel;
		}
		if (!model) {
			modelFallbackMessage = `No models available. Use /login or set an API key environment variable. See ${join(getDocsPath(), "providers.md")}. Then use /model to select a model.`;
		} else if (modelFallbackMessage) {
			modelFallbackMessage += `. Using ${model.provider}/${model.id}`;
		}
	}

	let thinkingLevel = initialThinkingLevel;

	// If session has data, restore thinking level from it
	if (thinkingLevel === undefined && hasExistingSession) {
		thinkingLevel = hasThinkingEntry
			? (existingSession.thinkingLevel as ThinkingLevel)
			: (settingsManager.getDefaultThinkingLevel() ?? DEFAULT_THINKING_LEVEL);
	}

	// Fall back to settings default
	if (thinkingLevel === undefined) {
		thinkingLevel = settingsManager.getDefaultThinkingLevel() ?? DEFAULT_THINKING_LEVEL;
	}

	// Clamp to model capabilities
	if (!model || !model.reasoning) {
		thinkingLevel = "off";
	}

	const defaultActiveToolNames: ToolName[] = ["read", "bash", "edit", "write"];
	const initialActiveToolNames: ToolName[] = options.tools
		? options.tools.map((t) => t.name).filter((n): n is ToolName => n in allTools)
		: defaultActiveToolNames;

	let agent: Agent;

	// Create convertToLlm wrapper that filters images if blockImages is enabled (defense-in-depth)
	const convertToLlmWithBlockImages = (messages: AgentMessage[]): Message[] => {
		const converted = convertToLlm(messages);
		// Check setting dynamically so mid-session changes take effect
		if (!settingsManager.getBlockImages()) {
			return converted;
		}
		// Filter out ImageContent from all messages, replacing with text placeholder
		return converted.map((msg) => {
			if (msg.role === "user" || msg.role === "toolResult") {
				const content = msg.content;
				if (Array.isArray(content)) {
					const hasImages = content.some((c) => c.type === "image");
					if (hasImages) {
						const filteredContent = content
							.map((c) =>
								c.type === "image" ? { type: "text" as const, text: "Image reading is disabled." } : c,
							)
							.filter(
								(c, i, arr) =>
									// Dedupe consecutive "Image reading is disabled." texts
									!(
										c.type === "text" &&
										c.text === "Image reading is disabled." &&
										i > 0 &&
										arr[i - 1].type === "text" &&
										(arr[i - 1] as { type: "text"; text: string }).text === "Image reading is disabled."
									),
							);
						return { ...msg, content: filteredContent };
					}
				}
			}
			return msg;
		});
	};

	const extensionRunnerRef: { current?: ExtensionRunner } = {};

	agent = new Agent({
		initialState: {
			systemPrompt: "",
			model,
			thinkingLevel,
			tools: [],
		},
		convertToLlm: convertToLlmWithBlockImages,
		onPayload: async (payload, _model) => {
			const runner = extensionRunnerRef.current;
			if (!runner?.hasHandlers("before_provider_request")) {
				return payload;
			}
			return runner.emitBeforeProviderRequest(payload);
		},
		sessionId: sessionManager.getSessionId(),
		transformContext: async (messages) => {
			const runner = extensionRunnerRef.current;
			if (!runner) return messages;
			return runner.emitContext(messages);
		},
		runtime: createPiAiCompatRuntime(),
		steeringMode: settingsManager.getSteeringMode(),
		followUpMode: settingsManager.getFollowUpMode(),
		transport: settingsManager.getTransport(),
		thinkingBudgets: settingsManager.getThinkingBudgets(),
		maxRetryDelayMs: settingsManager.getRetrySettings().maxDelayMs,
		getApiKey: async (provider, requestModelHandle) => {
			// Use the provider argument from the in-flight request;
			// fall back to agent.state.model only for direct callers without request context.
			const currentModel = normalizeModelHandle(agent.state.model);
			const resolvedProvider = provider || getRequestProvider(requestModelHandle) || currentModel?.provider;
			const authProvider = getRequestAuthProvider(requestModelHandle) ?? getAuthProvider(currentModel);
			const providersToTry = [
				...new Set([authProvider, resolvedProvider].filter((value): value is string => !!value)),
			];
			if (providersToTry.length === 0) {
				throw new Error("No model selected");
			}
			let key: string | undefined;
			for (const providerId of providersToTry) {
				key = await modelRegistry.getApiKeyForProvider(providerId);
				if (key) {
					break;
				}
			}
			if (!key) {
				const authModel = normalizeModelHandle(requestModelHandle) ?? currentModel;
				const isOAuth = authModel ? modelRegistry.isUsingOAuth(authModel) : false;
				const credentialProvider = authProvider ?? resolvedProvider;
				if (isOAuth) {
					throw new Error(
						`Authentication failed for "${credentialProvider}". ` +
							`Credentials may have expired or network is unavailable. ` +
							`Run '/login ${credentialProvider}' to re-authenticate.`,
					);
				}
				throw new Error(
					`No API key found for "${credentialProvider}". ` +
						`Set an API key environment variable or run '/login ${credentialProvider}'.`,
				);
			}
			return key;
		},
	});

	// Restore messages if session has existing data
	if (hasExistingSession) {
		agent.replaceMessages(existingSession.messages);
		if (!hasThinkingEntry) {
			sessionManager.appendThinkingLevelChange(thinkingLevel);
		}
	} else {
		// Save initial model and thinking level for new sessions so they can be restored on resume
		if (model) {
			sessionManager.appendModelChange(toPersistedModelReference(model));
		}
		sessionManager.appendThinkingLevelChange(thinkingLevel);
	}

	const session = new AgentSession({
		agent,
		sessionManager,
		settingsManager,
		cwd,
		scopedModels,
		resourceLoader,
		customTools: options.customTools,
		modelRegistry,
		initialActiveToolNames,
		extensionRunnerRef,
	});
	const extensionsResult = resourceLoader.getExtensions();

	return {
		session,
		extensionsResult,
		modelFallbackMessage,
	};
}
