import { createHash } from "node:crypto";
import type { AgentState } from "@mariozechner/pi-agent-core";
import type { Api, AssistantMessage, Model } from "@mariozechner/pi-ai";

const MODEL_HANDLE_PREFIX = "pi-model:";
const LEGACY_MODEL_HANDLE_PREFIX = "handle://";
const OVERFLOW_PATTERNS = [
	/prompt is too long/i,
	/input is too long for requested model/i,
	/exceeds the context window/i,
	/input token count.*exceeds the maximum/i,
	/maximum prompt length is \d+/i,
	/reduce the length of the messages/i,
	/maximum context length is \d+ tokens/i,
	/exceeds the limit of \d+/i,
	/exceeds the available context size/i,
	/greater than the context length/i,
	/context window exceeds limit/i,
	/exceeded model token limit/i,
	/too large for model with \d+ maximum context length/i,
	/model_context_window_exceeded/i,
	/context[_ ]length[_ ]exceeded/i,
	/too many tokens/i,
	/token limit exceeded/i,
];

export type BaseModelHandle = AgentState["model"];

export interface PersistedModelReference {
	modelHandleId?: string;
	authProvider?: string;
	modelId: string;
	provider?: string;
}

export type CompatiblePiModel = Omit<Model<Api>, "input"> & {
	input: readonly ("text" | "image")[];
};

export interface CodingAgentModelHandle extends BaseModelHandle, Model<Api> {
	modelHandleId: string;
	authProvider: string;
	raw: Model<Api>;
}

function createOpaqueModelHandleId(parts: Record<string, string>): string {
	const payload = JSON.stringify(parts);
	const digest = createHash("sha256").update(payload).digest("hex");
	return `${MODEL_HANDLE_PREFIX}${digest}`;
}

export function createPersistentModelHandleId(authProvider: string, provider: string, modelId: string): string {
	return createOpaqueModelHandleId({
		authProvider,
		provider,
		modelId,
	});
}

export function parseLegacyPersistentModelHandleId(modelHandleId: string): PersistedModelReference | undefined {
	if (!modelHandleId.startsWith(MODEL_HANDLE_PREFIX) && !modelHandleId.startsWith(LEGACY_MODEL_HANDLE_PREFIX)) {
		return undefined;
	}

	const prefix = modelHandleId.startsWith(MODEL_HANDLE_PREFIX) ? MODEL_HANDLE_PREFIX : LEGACY_MODEL_HANDLE_PREFIX;
	const encoded = modelHandleId.slice(prefix.length);
	const separatorIndex = encoded.indexOf(":");
	if (separatorIndex === -1) {
		return undefined;
	}

	const authProvider = decodeURIComponent(encoded.slice(0, separatorIndex));
	const modelId = decodeURIComponent(encoded.slice(separatorIndex + 1));
	return {
		modelHandleId,
		authProvider,
		modelId,
		provider: authProvider,
	};
}

export function toCodingAgentModelHandle(
	model: CompatiblePiModel,
	authProvider = model.provider,
): CodingAgentModelHandle {
	const raw = {
		...model,
		input: [...model.input],
	} satisfies Model<Api>;

	return {
		...raw,
		id: raw.id,
		displayName: raw.name,
		family: raw.provider,
		capabilities: {
			tools: true,
			images: raw.input.includes("image"),
			streaming: true,
			thinking: raw.reasoning,
		},
		modelHandleId: createPersistentModelHandleId(authProvider, raw.provider, raw.id),
		authProvider,
		raw,
	};
}

export function cloneCodingAgentModelHandle(
	baseModel: CodingAgentModelHandle,
	overrides: Partial<Model<Api>> & Pick<Model<Api>, "id">,
): CodingAgentModelHandle {
	const raw = {
		...baseModel.raw,
		...overrides,
		id: overrides.id,
		name: overrides.name ?? overrides.id,
	} satisfies Model<Api>;

	return toCodingAgentModelHandle(raw, baseModel.authProvider);
}

export function isCodingAgentModelHandle(model: unknown): model is CodingAgentModelHandle {
	return (
		typeof model === "object" &&
		model !== null &&
		"modelHandleId" in model &&
		typeof (model as { modelHandleId?: unknown }).modelHandleId === "string" &&
		"authProvider" in model &&
		typeof (model as { authProvider?: unknown }).authProvider === "string"
	);
}

function isPiAiModel(model: unknown): model is Model<Api> {
	return (
		typeof model === "object" &&
		model !== null &&
		"id" in model &&
		typeof (model as { id?: unknown }).id === "string" &&
		"provider" in model &&
		typeof (model as { provider?: unknown }).provider === "string" &&
		"api" in model &&
		typeof (model as { api?: unknown }).api === "string"
	);
}

export function getAuthProvider(
	model: Pick<PersistedModelReference, "authProvider" | "provider"> | undefined,
): string | undefined {
	return model?.authProvider ?? model?.provider;
}

export function areModelHandlesEqual(
	left: { id?: string; provider?: string; modelHandleId?: string } | undefined,
	right: { id?: string; provider?: string; modelHandleId?: string } | undefined,
): boolean {
	if (!left || !right) {
		return left === right;
	}

	return left.modelHandleId === right.modelHandleId || (left.provider === right.provider && left.id === right.id);
}

export function supportsXhighThinking(model: { id?: string } | undefined): boolean {
	if (!model) {
		return false;
	}

	return (
		model.id?.includes("gpt-5.2") === true ||
		model.id?.includes("gpt-5.3") === true ||
		model.id?.includes("gpt-5.4") === true ||
		model.id?.includes("opus-4-6") === true ||
		model.id?.includes("opus-4.6") === true
	);
}

export function isAssistantContextOverflow(message: AssistantMessage, contextWindow?: number): boolean {
	if (message.stopReason === "error" && message.errorMessage) {
		if (OVERFLOW_PATTERNS.some((pattern) => pattern.test(message.errorMessage!))) {
			return true;
		}

		if (/^4(00|13)\s*(status code)?\s*\(no body\)/i.test(message.errorMessage)) {
			return true;
		}
	}

	if (contextWindow && message.stopReason === "stop") {
		const inputTokens = message.usage.input + message.usage.cacheRead;
		if (inputTokens > contextWindow) {
			return true;
		}
	}

	return false;
}

export function normalizeModelHandle(
	model: BaseModelHandle | CompatiblePiModel | undefined,
): CodingAgentModelHandle | undefined {
	if (!model) {
		return undefined;
	}

	if (isCodingAgentModelHandle(model)) {
		return model;
	}
	if (isPiAiModel(model)) {
		return toCodingAgentModelHandle(model);
	}
	if (typeof model === "object" && model !== null && "raw" in model && isPiAiModel((model as { raw?: unknown }).raw)) {
		return toCodingAgentModelHandle((model as { raw: Model<Api> }).raw);
	}

	return undefined;
}

export function toPersistedModelReference(model: CodingAgentModelHandle): PersistedModelReference {
	return {
		modelHandleId: model.modelHandleId,
		authProvider: model.authProvider,
		modelId: model.id,
		provider: model.provider,
	};
}

export function formatPersistedModelReference(model: PersistedModelReference): string {
	if (model.provider && model.modelId) {
		return `${model.provider}/${model.modelId}`;
	}
	if (model.authProvider && model.modelId) {
		return `${model.authProvider}/${model.modelId}`;
	}
	return model.modelId;
}
