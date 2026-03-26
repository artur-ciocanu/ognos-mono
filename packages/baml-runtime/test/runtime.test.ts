import { describe, expect, test, vi } from "vitest";
import type { BamlClient, BamlRuntimeRequest } from "../src/baml-client.js";
import type { ModelHandle } from "../src/model-handle.js";
import { BamlRuntime } from "../src/runtime.js";

async function collect<T>(stream: AsyncIterable<T>): Promise<T[]> {
	const items: T[] = [];
	for await (const item of stream) {
		items.push(item);
	}
	return items;
}

const model: ModelHandle = {
	id: "claude-sonnet",
	displayName: "Claude Sonnet",
	family: "claude",
	capabilities: {
		tools: true,
		images: true,
		streaming: true,
	},
	raw: { provider: "baml", model: "claude-sonnet" },
};

describe("BamlRuntime", () => {
	test("maps runtime requests into client.stream()", async () => {
		const requests: BamlRuntimeRequest[] = [];
		const client: BamlClient = {
			listModels: vi.fn(async () => []),
			stream: vi.fn(async (request) => {
				requests.push(request);
				return {
					stream: (async function* () {
						yield { type: "message_start", messageId: "msg-1" } as const;
						yield { type: "text_start", id: "text-1" } as const;
						yield { type: "text_delta", id: "text-1", delta: "hello" } as const;
						yield { type: "text_end", id: "text-1" } as const;
						yield { type: "message_end", messageId: "msg-1" } as const;
					})(),
				};
			}),
		};
		const runtime = new BamlRuntime(client);
		const signal = new AbortController().signal;

		const events = await collect(
			runtime.stream(
				model,
				{
					messages: [
						{ role: "system", content: "system prompt" },
						{ role: "user", content: "hello" },
					],
					tools: [
						{
							name: "read_file",
							description: "Read a file from disk",
							inputSchema: { type: "object" },
						},
					],
				},
				{ signal },
			),
		);

		expect(requests).toEqual([
			{
				modelId: "claude-sonnet",
				messages: [
					{ role: "system", content: "system prompt" },
					{ role: "user", content: "hello" },
				],
				tools: [
					{
						name: "read_file",
						description: "Read a file from disk",
						inputSchema: { type: "object" },
					},
				],
				signal,
			},
		]);
		expect(events).toEqual([
			{ type: "start", messageId: "msg-1" },
			{ type: "text_start", id: "text-1" },
			{ type: "text_delta", id: "text-1", delta: "hello" },
			{ type: "text_end", id: "text-1" },
			{
				type: "done",
				messageId: "msg-1",
				usage: {
					inputTokens: 0,
					outputTokens: 0,
					cachedInputTokens: 0,
				},
			},
		]);
	});

	test("normalizes startup failures into a terminal error event", async () => {
		const client: BamlClient = {
			listModels: vi.fn(async () => []),
			stream: vi.fn(async () => {
				throw new Error("startup failure");
			}),
		};
		const runtime = new BamlRuntime(client);

		const events = await collect(
			runtime.stream(model, {
				messages: [{ role: "user", content: "hello" }],
			}),
		);

		expect(events).toEqual([
			{
				type: "error",
				error: expect.objectContaining({
					message: "startup failure",
				}),
			},
		]);
	});
});
