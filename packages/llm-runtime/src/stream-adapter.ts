import type { BamlCollectorData, BamlStreamChunk } from "./baml-client.js";
import type { RuntimeUsage } from "./usage.js";
import { extractRuntimeUsage } from "./usage.js";

export type RuntimeEvent =
	| { type: "start"; messageId?: string }
	| { type: "text_start"; id: string }
	| { type: "text_delta"; id: string; delta: string }
	| { type: "text_end"; id: string }
	| { type: "toolcall_start"; id: string; toolName: string }
	| { type: "toolcall_delta"; id: string; delta: string }
	| { type: "toolcall_end"; id: string }
	| { type: "done"; messageId?: string; usage: RuntimeUsage }
	| { type: "error"; error: Error };

export interface AdaptBamlStreamOptions {
	collector?: BamlCollectorData;
}

function toError(error: unknown): Error {
	if (error instanceof Error) {
		return error;
	}

	return new Error(typeof error === "string" ? error : "Unknown stream error");
}

export async function* adaptBamlStream(
	stream: AsyncIterable<BamlStreamChunk>,
	options: AdaptBamlStreamOptions = {},
): AsyncIterable<RuntimeEvent> {
	let lastMessageId: string | undefined;

	try {
		for await (const chunk of stream) {
			switch (chunk.type) {
				case "message_start":
					lastMessageId = chunk.messageId;
					yield { type: "start", messageId: chunk.messageId };
					break;
				case "text_start":
					yield { type: "text_start", id: chunk.id };
					break;
				case "text_delta":
					yield { type: "text_delta", id: chunk.id, delta: chunk.delta };
					break;
				case "text_end":
					yield { type: "text_end", id: chunk.id };
					break;
				case "tool_call_start":
					yield { type: "toolcall_start", id: chunk.id, toolName: chunk.toolName };
					break;
				case "tool_call_delta":
					yield { type: "toolcall_delta", id: chunk.id, delta: chunk.delta };
					break;
				case "tool_call_end":
					yield { type: "toolcall_end", id: chunk.id };
					break;
				case "message_end":
					lastMessageId = chunk.messageId ?? lastMessageId;
					yield {
						type: "done",
						messageId: chunk.messageId ?? lastMessageId,
						usage: extractRuntimeUsage(options.collector),
					};
					break;
			}
		}
	} catch (error) {
		yield {
			type: "error",
			error: toError(error),
		};
	}
}
