import { describe, expect, test } from "vitest";
import type { BamlCollectorData, BamlStreamChunk } from "../src/baml-client.js";
import { adaptBamlStream } from "../src/stream-adapter.js";

async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
	const items: T[] = [];
	for await (const item of stream) {
		items.push(item);
	}
	return items;
}

describe("adaptBamlStream", () => {
	test("normalizes text and tool-call chunks into runtime events", async () => {
		const collector: BamlCollectorData = {
			usage: {
				inputTokens: 120,
				outputTokens: 45,
				cachedInputTokens: 10,
			},
		};
		const chunks: BamlStreamChunk[] = [
			{ type: "message_start", messageId: "msg-1" },
			{ type: "text_start", id: "text-1" },
			{ type: "text_delta", id: "text-1", delta: "Hello" },
			{ type: "text_delta", id: "text-1", delta: " world" },
			{ type: "text_end", id: "text-1" },
			{ type: "tool_call_start", id: "tool-1", toolName: "read_file" },
			{ type: "tool_call_delta", id: "tool-1", delta: '{"path":"README.md"}' },
			{ type: "tool_call_end", id: "tool-1" },
			{ type: "message_end", messageId: "msg-1" },
		];

		const events = await collect(
			adaptBamlStream(
				(async function* () {
					for (const chunk of chunks) {
						yield chunk;
					}
				})(),
				{ collector },
			),
		);

		expect(events).toEqual([
			{ type: "start", messageId: "msg-1" },
			{ type: "text_start", id: "text-1" },
			{ type: "text_delta", id: "text-1", delta: "Hello" },
			{ type: "text_delta", id: "text-1", delta: " world" },
			{ type: "text_end", id: "text-1" },
			{ type: "toolcall_start", id: "tool-1", toolName: "read_file" },
			{ type: "toolcall_delta", id: "tool-1", delta: '{"path":"README.md"}' },
			{ type: "toolcall_end", id: "tool-1" },
			{
				type: "done",
				messageId: "msg-1",
				usage: {
					inputTokens: 120,
					outputTokens: 45,
					cachedInputTokens: 10,
				},
			},
		]);
	});

	test("emits a terminal error event when the source stream throws", async () => {
		const source = (async function* (): AsyncGenerator<BamlStreamChunk> {
			yield { type: "message_start", messageId: "msg-err" };
			throw new Error("adapter exploded");
		})();

		const events = await collect(adaptBamlStream(source));

		expect(events).toEqual([
			{ type: "start", messageId: "msg-err" },
			{
				type: "error",
				error: expect.objectContaining({
					message: "adapter exploded",
				}),
			},
		]);
	});
});
