import type { AgentMessage, AgentRuntime } from "@mariozechner/pi-agent-core";
import { createPiAiCompatRuntime, toModelHandle, toRuntimeMessages } from "@mariozechner/pi-agent-core";
import type { Message, Model } from "@mariozechner/pi-ai";
import { convertToLlm } from "../messages.js";

export type SummaryExecutionResult =
	| { status: "ok"; text: string }
	| { status: "aborted" }
	| { status: "error"; error: string };

export interface SummaryExecutionOptions {
	model: Model<any>;
	systemPrompt: string;
	messages: AgentMessage[];
	apiKey: string;
	maxTokens: number;
	signal?: AbortSignal;
	reasoning?: "off" | "minimal" | "low" | "medium" | "high" | "xhigh";
	runtime?: AgentRuntime;
}

function buildSummaryPromptMessages(messages: AgentMessage[]): Message[] {
	return convertToLlm(messages);
}

export async function executeSummary(options: SummaryExecutionOptions): Promise<SummaryExecutionResult> {
	const runtime = options.runtime?.configured === false ? undefined : options.runtime;
	const activeRuntime = runtime ?? createPiAiCompatRuntime();
	const llmMessages = buildSummaryPromptMessages(options.messages);
	const textById = new Map<string, string>();
	const orderedTextIds: string[] = [];
	const runtimeOptions = {
		apiKey: options.apiKey,
		maxTokens: options.maxTokens,
		signal: options.signal,
		...(options.reasoning ? { reasoning: options.reasoning } : {}),
	};

	try {
		for await (const event of activeRuntime.stream(
			toModelHandle(options.model),
			{
				messages: toRuntimeMessages(llmMessages, options.systemPrompt),
				rawMessages: llmMessages,
			},
			runtimeOptions,
		)) {
			switch (event.type) {
				case "text_start":
					orderedTextIds.push(event.id);
					textById.set(event.id, "");
					break;
				case "text_delta":
					textById.set(event.id, `${textById.get(event.id) ?? ""}${event.delta}`);
					break;
				case "done":
					return {
						status: "ok",
						text: orderedTextIds.map((id) => textById.get(id) ?? "").join("\n"),
					};
				case "error":
					if (options.signal?.aborted) {
						return { status: "aborted" };
					}
					return { status: "error", error: event.error.message || "Runtime stream failed" };
			}
		}
	} catch (error) {
		if (options.signal?.aborted) {
			return { status: "aborted" };
		}
		return {
			status: "error",
			error: error instanceof Error ? error.message : "Runtime stream failed",
		};
	}

	if (options.signal?.aborted) {
		return { status: "aborted" };
	}

	return {
		status: "ok",
		text: orderedTextIds.map((id) => textById.get(id) ?? "").join("\n"),
	};
}
