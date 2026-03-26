import { describe, expect, test } from "vitest";
import type { BamlCollectorData } from "../src/baml-client.js";
import { extractRuntimeUsage } from "../src/usage.js";

describe("extractRuntimeUsage", () => {
	test("extracts normalized token counts from collector usage", () => {
		const collector: BamlCollectorData = {
			usage: {
				inputTokens: 1000,
				outputTokens: 250,
				cachedInputTokens: 400,
			},
		};

		expect(extractRuntimeUsage(collector)).toEqual({
			inputTokens: 1000,
			outputTokens: 250,
			cachedInputTokens: 400,
		});
	});

	test("derives estimated cost when model pricing is available", () => {
		const collector: BamlCollectorData = {
			usage: {
				inputTokens: 2000,
				outputTokens: 500,
				cachedInputTokens: 1000,
			},
		};

		expect(
			extractRuntimeUsage(collector, {
				inputPerMillion: 2,
				outputPerMillion: 8,
				cachedInputPerMillion: 0.5,
			}),
		).toEqual({
			inputTokens: 2000,
			outputTokens: 500,
			cachedInputTokens: 1000,
			estimatedCost: 0.0085,
		});
	});

	test("defaults missing usage buckets to zero", () => {
		const collector: BamlCollectorData = {
			usage: {},
		};

		expect(extractRuntimeUsage(collector)).toEqual({
			inputTokens: 0,
			outputTokens: 0,
			cachedInputTokens: 0,
		});
	});

	test("returns zeroed usage when collector data is absent", () => {
		expect(extractRuntimeUsage(undefined)).toEqual({
			inputTokens: 0,
			outputTokens: 0,
			cachedInputTokens: 0,
		});
	});
});
