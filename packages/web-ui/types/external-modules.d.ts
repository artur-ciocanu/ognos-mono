declare module "@mariozechner/pi-ai" {
	import type { TSchema, TString } from "@sinclair/typebox";

	export type Api = string;
	export type Provider = string;
	export type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh";

	export interface TextContent {
		type: "text";
		text: string;
	}

	export interface ThinkingContent {
		type: "thinking";
		thinking: string;
	}

	export interface ImageContent {
		type: "image";
		data: string;
		mimeType: string;
	}

	export interface ToolCall {
		type: "toolCall";
		id: string;
		name: string;
		arguments: Record<string, unknown>;
	}

	export interface UsageCost {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total?: number;
	}

	export interface Usage {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		totalTokens: number;
		cost: UsageCost;
	}

	export type StopReason = "stop" | "length" | "toolUse" | "error" | "aborted";

	export interface UserMessage {
		role: "user";
		content: string | (TextContent | ImageContent)[];
		timestamp: number;
	}

	export interface AssistantMessage {
		role: "assistant";
		content: (TextContent | ThinkingContent | ToolCall)[];
		api: Api;
		provider: Provider;
		model: string;
		usage: Usage;
		stopReason: StopReason;
		errorMessage?: string;
		timestamp: number;
	}

	export interface ToolResultMessage<TDetails = unknown> {
		role: "toolResult";
		toolCallId: string;
		toolName: string;
		content: (TextContent | ImageContent)[];
		details?: TDetails;
		isError: boolean;
		timestamp: number;
	}

	export type Message = UserMessage | AssistantMessage | ToolResultMessage;

	export interface Tool<TParameters extends TSchema = TSchema> {
		name: string;
		description: string;
		parameters: TParameters;
	}

	export interface Context {
		systemPrompt?: string;
		messages: Message[];
		tools?: Tool[];
	}

	export interface StreamOptions {
		temperature?: number;
		maxTokens?: number;
		signal?: AbortSignal;
		apiKey?: string;
		headers?: Record<string, string>;
		sessionId?: string;
	}

	export interface SimpleStreamOptions extends StreamOptions {
		reasoning?: ThinkingLevel;
		thinkingBudgets?: Partial<Record<Exclude<ThinkingLevel, "xhigh">, number>>;
	}

	export interface Model<TApi extends Api = Api> {
		id: string;
		name: string;
		api: TApi;
		provider: Provider;
		baseUrl: string;
		reasoning: boolean;
		input: ("text" | "image")[];
		cost: UsageCost;
		contextWindow: number;
		maxTokens: number;
		headers?: Record<string, string>;
	}

	export type StreamSimpleFn = (
		model: Model<Api>,
		context: Context,
		options?: SimpleStreamOptions,
	) => unknown | Promise<unknown>;

	export const streamSimple: StreamSimpleFn;

	export function complete(
		model: Model<Api>,
		context: Context,
		options?: SimpleStreamOptions,
	): Promise<AssistantMessage>;

	export function getProviders(): string[];
	export function getModels(provider: string): Model<Api>[];
	export function getModel(provider: string, modelId: string): Model<Api> | undefined;
	export function modelsAreEqual(
		a: Model<Api> | null | undefined,
		b: Model<Api> | null | undefined,
	): boolean;

	export function StringEnum(
		values: readonly string[],
		options?: { default?: string },
	): TSchema & { enum?: readonly string[] };

	export function StringEnum<TValue extends string>(
		values: readonly TValue[],
		options?: { default?: TValue; description?: string },
	): TString;
}

declare module "@mariozechner/pi-agent-core" {
	import type {
		Context,
		ImageContent,
		Message,
		Model,
		SimpleStreamOptions,
		TextContent,
		Tool,
		ToolResultMessage,
	} from "@mariozechner/pi-ai";
	import type { Static, TSchema } from "@sinclair/typebox";

	export interface AgentRuntimeContext {
		messages: Array<{ role: string; content: string; toolCallId?: string; toolName?: string }>;
		tools?: Tool[];
		rawMessages?: Message[];
	}

	export interface AgentRuntimeOptions {
		signal?: AbortSignal;
		apiKey?: string;
		maxTokens?: number;
		reasoning?: ThinkingLevel;
		sessionId?: string;
		onPayload?: (payload: unknown, model: unknown) => unknown | undefined | Promise<unknown | undefined>;
	}

	export interface AgentRuntime {
		readonly configured?: boolean;
		stream(model: unknown, context: AgentRuntimeContext, options?: AgentRuntimeOptions): AsyncIterable<unknown>;
	}

	export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

	export interface CustomAgentMessages {}

	export type AgentMessage = Message | CustomAgentMessages[keyof CustomAgentMessages];

	export interface AgentToolResult<TDetails = unknown> {
		content: (TextContent | ImageContent)[];
		details: TDetails;
	}

	export type AgentToolUpdateCallback<TDetails = unknown> = (partialResult: AgentToolResult<TDetails>) => void;

	export interface AgentTool<TParameters extends TSchema = TSchema, TDetails = unknown> extends Tool<TParameters> {
		label: string;
		execute(
			toolCallId: string,
			params: Static<TParameters>,
			signal?: AbortSignal,
			onUpdate?: AgentToolUpdateCallback<TDetails>,
		): Promise<AgentToolResult<TDetails>>;
	}

	export interface AgentState {
		systemPrompt: string;
		model: Model;
		thinkingLevel: ThinkingLevel;
		tools: AgentTool[];
		messages: AgentMessage[];
		isStreaming: boolean;
		streamMessage: AgentMessage | null;
		pendingToolCalls: Set<string>;
		error?: string;
	}

	export type AgentEvent =
		| { type: "agent_start" }
		| { type: "agent_end"; messages: AgentMessage[] }
		| { type: "turn_start" }
		| { type: "turn_end"; message: AgentMessage; toolResults: ToolResultMessage[] }
		| { type: "message_start"; message: AgentMessage }
		| { type: "message_update"; message: AgentMessage }
		| { type: "message_end"; message: AgentMessage }
		| { type: "tool_execution_start"; toolCallId: string; toolName: string; args: unknown }
		| { type: "tool_execution_update"; toolCallId: string; toolName: string; args: unknown; partialResult: unknown }
		| { type: "tool_execution_end"; toolCallId: string; toolName: string; result: unknown; isError: boolean }
		| { type: "state-update"; state: AgentState };

	export interface AgentOptions {
		initialState?: Partial<AgentState>;
		convertToLlm?: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	}

	export class Agent {
		constructor(options?: AgentOptions);
		state: AgentState;
		runtime: AgentRuntime;
		getApiKey?: (provider: string, model?: Model) => Promise<string | undefined> | string | undefined;
		subscribe(listener: (event: AgentEvent) => void | Promise<void>): () => void;
		setModel(model: Model): void;
		setThinkingLevel(level: ThinkingLevel): void;
		setTools(tools: AgentTool[]): void;
		appendMessage(message: AgentMessage): void;
		prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
		prompt(input: string, images?: ImageContent[]): Promise<void>;
		steer(message: AgentMessage): void;
		abort(): void;
	}
}
