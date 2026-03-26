import type { BamlCollectorData } from "./baml-client.js";

export interface RuntimeUsage {
	inputTokens: number;
	outputTokens: number;
	cachedInputTokens: number;
	estimatedCost?: number;
}

export interface RuntimeModelPricing {
	inputPerMillion: number;
	outputPerMillion: number;
	cachedInputPerMillion?: number;
}

function normalizeTokenCount(value: number | undefined): number {
	return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function extractRuntimeUsage(collector?: BamlCollectorData, pricing?: RuntimeModelPricing): RuntimeUsage {
	const inputTokens = normalizeTokenCount(collector?.usage?.inputTokens);
	const outputTokens = normalizeTokenCount(collector?.usage?.outputTokens);
	const cachedInputTokens = normalizeTokenCount(collector?.usage?.cachedInputTokens);

	if (!pricing) {
		return {
			inputTokens,
			outputTokens,
			cachedInputTokens,
		};
	}

	const estimatedCost =
		inputTokens * (pricing.inputPerMillion / 1_000_000) +
		outputTokens * (pricing.outputPerMillion / 1_000_000) +
		cachedInputTokens * ((pricing.cachedInputPerMillion ?? 0) / 1_000_000);

	return {
		inputTokens,
		outputTokens,
		cachedInputTokens,
		estimatedCost,
	};
}
