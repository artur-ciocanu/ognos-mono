import { describe, expect, test } from "vitest";
import { StaticModelCatalog } from "../src/model-catalog.js";
import type { ModelHandle } from "../src/model-handle.js";

const models: ModelHandle[] = [
	{
		id: "claude-sonnet",
		displayName: "Claude Sonnet",
		family: "claude",
		capabilities: {
			tools: true,
			images: true,
			streaming: true,
			thinking: true,
		},
		raw: { provider: "baml", model: "claude-sonnet" },
	},
	{
		id: "gpt-4.1-mini",
		displayName: "GPT-4.1 Mini",
		family: "gpt",
		capabilities: {
			tools: true,
			images: false,
			streaming: true,
		},
		raw: { provider: "baml", model: "gpt-4.1-mini" },
	},
];

describe("StaticModelCatalog", () => {
	test("lists models in declared order", async () => {
		const catalog = new StaticModelCatalog(models, { defaultModelId: "gpt-4.1-mini" });

		await expect(catalog.list()).resolves.toEqual(models);
	});

	test("resolves known model ids", async () => {
		const catalog = new StaticModelCatalog(models, { defaultModelId: "gpt-4.1-mini" });

		await expect(catalog.resolve("claude-sonnet")).resolves.toEqual(models[0]);
		await expect(catalog.resolve("missing")).resolves.toBeUndefined();
	});

	test("returns configured default when present", async () => {
		const catalog = new StaticModelCatalog(models, { defaultModelId: "gpt-4.1-mini" });

		await expect(catalog.getDefault()).resolves.toEqual(models[1]);
	});

	test("falls back to the first model when configured default is missing", async () => {
		const catalog = new StaticModelCatalog(models, { defaultModelId: "missing" });

		await expect(catalog.getDefault()).resolves.toEqual(models[0]);
	});

	test("returns undefined default when catalog is empty", async () => {
		const catalog = new StaticModelCatalog([], { defaultModelId: "missing" });

		await expect(catalog.getDefault()).resolves.toBeUndefined();
	});

	test("returns cloned handles so consumer mutation does not affect future reads", async () => {
		const catalog = new StaticModelCatalog(models, { defaultModelId: "gpt-4.1-mini" });

		const listedModels = await catalog.list();
		listedModels[0].displayName = "Mutated";
		listedModels[0].capabilities.tools = false;
		(listedModels[0].raw as { provider: string }).provider = "changed";

		await expect(catalog.resolve("claude-sonnet")).resolves.toEqual({
			id: "claude-sonnet",
			displayName: "Claude Sonnet",
			family: "claude",
			capabilities: {
				tools: true,
				images: true,
				streaming: true,
				thinking: true,
			},
			raw: { provider: "baml", model: "claude-sonnet" },
		});
	});
});
