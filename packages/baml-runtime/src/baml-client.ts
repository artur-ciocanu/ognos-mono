import type { ModelCapabilities } from "./model-handle.js";

export interface BamlModelDefinition {
	id: string;
	displayName: string;
	family?: string;
	capabilities: ModelCapabilities;
	raw?: unknown;
	isDefault?: boolean;
}

export interface BamlCollectorUsageData {
	inputTokens?: number;
	outputTokens?: number;
	cachedInputTokens?: number;
}

export interface BamlCollectorData {
	usage?: BamlCollectorUsageData;
	raw?: unknown;
}

export type BamlStreamChunk =
	| { type: "message_start"; messageId?: string }
	| { type: "text_start"; id: string }
	| { type: "text_delta"; id: string; delta: string }
	| { type: "text_end"; id: string }
	| { type: "tool_call_start"; id: string; toolName: string }
	| { type: "tool_call_delta"; id: string; delta: string }
	| { type: "tool_call_end"; id: string }
	| { type: "message_end"; messageId?: string };

export interface BamlStreamResponse {
	stream: AsyncIterable<BamlStreamChunk>;
	collector?: BamlCollectorData;
}

export interface BamlRuntimeRequest {
	modelId: string;
	messages: readonly {
		role: string;
		content: string;
		toolCallId?: string;
		toolName?: string;
	}[];
	tools?: readonly {
		name: string;
		description?: string;
		inputSchema?: unknown;
	}[];
	signal?: AbortSignal;
}

export interface BamlClient {
	listModels(): Promise<BamlModelDefinition[]>;
	stream(request: BamlRuntimeRequest): Promise<BamlStreamResponse>;
}
