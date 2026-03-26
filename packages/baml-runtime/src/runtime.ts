import type { BamlClient, BamlStreamResponse } from "./baml-client.js";
import type { ModelHandle } from "./model-handle.js";
import type { RuntimeEvent } from "./stream-adapter.js";
import { adaptBamlStream } from "./stream-adapter.js";

export interface RuntimeToolDefinition {
	name: string;
	description?: string;
	inputSchema?: unknown;
}

export interface RuntimeMessage {
	role: "system" | "user" | "assistant" | "tool";
	content: string;
	toolCallId?: string;
	toolName?: string;
}

export interface RuntimeContext {
	messages: readonly RuntimeMessage[];
	tools?: readonly RuntimeToolDefinition[];
}

export interface RuntimeOptions {
	signal?: AbortSignal;
}

export interface LlmRuntime {
	stream(model: ModelHandle, context: RuntimeContext, options?: RuntimeOptions): AsyncIterable<RuntimeEvent>;
}

function toError(error: unknown): Error {
	if (error instanceof Error) {
		return error;
	}

	return new Error(typeof error === "string" ? error : "Unknown runtime error");
}

export class BamlRuntime implements LlmRuntime {
	readonly #client: BamlClient;

	constructor(client: BamlClient) {
		this.#client = client;
	}

	async *stream(
		model: ModelHandle,
		context: RuntimeContext,
		options: RuntimeOptions = {},
	): AsyncIterable<RuntimeEvent> {
		let response: BamlStreamResponse;
		try {
			response = await this.#client.stream({
				modelId: model.id,
				messages: context.messages,
				tools: context.tools,
				signal: options.signal,
			});
		} catch (error) {
			yield {
				type: "error",
				error: toError(error),
			};
			return;
		}

		yield* adaptBamlStream(response.stream, { collector: response.collector });
	}
}
