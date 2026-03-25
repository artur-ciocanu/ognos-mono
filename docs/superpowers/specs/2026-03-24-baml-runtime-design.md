# BAML Runtime Migration Design

Date: 2026-03-24
Status: Draft approved in chat
Scope: Replace `@mariozechner/pi-ai` as the LLM runtime with a BAML-backed runtime boundary.

## Summary

Pi should stop owning provider SDK integrations directly. Instead, a new local runtime package should use BAML as the sole LLM execution layer. `@mariozechner/pi-agent-core` and `@mariozechner/pi-coding-agent` should depend only on a provider-agnostic runtime contract and a typed local model handle.

This is a hard cutover. There will be no feature flag, no dual-runtime mode, and no long-term compatibility layer that keeps `pi-ai` as the execution boundary.

## Goals

- Make BAML the only LLM runtime boundary.
- Remove provider-specific branching from `coding-agent` and, as much as possible, from `pi-agent-core`.
- Make agent code operate on opaque typed model handles instead of `provider + modelId`.
- Keep the core agent loop behavior intact for streaming text and tool execution.
- Use API keys or ambient cloud credentials only for v1.

## Non-Goals

- Preserve existing OAuth or subscription login flows in v1.
- Preserve current cross-provider handoff semantics.
- Preserve exact provider-specific usage, cache, reasoning, or cost accounting parity.
- Preserve provider as a first-class architectural concept in agent code.
- Introduce a dual runtime or migration feature flag.

## Current Problems

Today the runtime stack is:

`coding-agent` -> `pi-agent-core` -> `pi-ai` -> provider SDKs

This creates several forms of coupling:

- `coding-agent` has provider-aware model defaults and CLI resolution.
- Sessions persist provider-aware identities.
- `pi-ai` owns provider auth lookup, model catalogs, transport implementations, usage normalization, and message replay behavior.
- Provider-specific concerns bleed upward into packages that should be concerned only with agent behavior and UI.

## Proposed Architecture

Introduce a new local package, referred to here as `llm-runtime`, backed by BAML.

Target stack:

`coding-agent` -> `pi-agent-core` -> `llm-runtime` -> BAML

### Runtime Boundary

`llm-runtime` owns:

- BAML client configuration
- model discovery and local model metadata
- request execution
- stream adaptation into pi-compatible runtime events
- usage extraction from BAML collector data
- optional auth input wiring for API keys or ambient credentials

`pi-agent-core` owns:

- agent state
- turn loop
- tool execution
- context transformation
- event fan-out to consumers

`coding-agent` owns:

- UI and CLI model selection
- session persistence
- settings
- display metadata

Neither `coding-agent` nor `pi-agent-core` should branch on provider names as a required part of normal operation.

## Canonical Model Identity

The canonical model identity is a local `ModelHandle`, not `provider + modelId`.

Proposed shape:

```ts
interface ModelHandle {
  id: string;
  displayName: string;
  family?: string;
  capabilities: {
    tools: boolean;
    images: boolean;
    streaming: boolean;
    thinking?: boolean;
  };
  raw: unknown;
}
```

Notes:

- `id` is the stable persisted identity used in sessions and settings.
- `displayName` is for selectors and status UI.
- `family` is optional and only for grouping or filtering in UX.
- `raw` is runtime-private BAML-specific data kept behind the boundary.

## Runtime Interfaces

The agent packages should depend on narrow local interfaces rather than BAML directly.

### Model Catalog

```ts
interface ModelCatalog {
  list(): Promise<ModelHandle[]>;
  resolve(id: string): Promise<ModelHandle | undefined>;
  getDefault(): Promise<ModelHandle | undefined>;
}
```

### LLM Runtime

```ts
interface LlmRuntime {
  stream(
    model: ModelHandle,
    context: RuntimeContext,
    options?: RuntimeOptions,
  ): RuntimeEventStream;
}
```

`complete()` is optional. If it simplifies internal implementation it can exist, but the primary contract should be stream-first.

## Normalized Event Model

The event model should remain close to what `pi-agent-core` already consumes so the migration mostly changes the producer side.

Required events:

- `start`
- `text_start`
- `text_delta`
- `text_end`
- `toolcall_start`
- `toolcall_delta`
- `toolcall_end`
- `done`
- `error`

Optional events:

- `thinking_start`
- `thinking_delta`
- `thinking_end`

Thinking is explicitly optional in v1. If BAML cannot expose it cleanly, pi should operate without normalized thinking blocks.

## Usage and Cost

The runtime should extract usage from BAML collector data where available.

V1 guarantees:

- input tokens
- output tokens
- cached input tokens when exposed
- latency or timing metadata if useful

V1 does not guarantee:

- exact provider-specific reasoning token counts
- exact cache write accounting parity
- exact cost parity with current `pi-ai` data structures

If a cost number is shown in the UI, it should be derived from local model metadata and available token counts, not treated as authoritative unless the required inputs are reliably available.

## Authentication

V1 auth posture is intentionally narrow:

- API keys
- ambient cloud credentials that BAML already supports directly

Explicitly deferred:

- Anthropic OAuth subscription flow
- OpenAI Codex subscription flow
- GitHub Copilot login flow
- Gemini CLI login flow
- any other login/refresh system that requires custom token orchestration outside normal BAML config

This keeps the first migration focused on the runtime boundary rather than recreating provider-specific auth behavior.

## Session and Persistence Changes

Current session logic assumes a provider-aware model identity. That should change to a stored opaque model handle id plus display metadata.

At minimum:

- new sessions store `modelHandleId`
- settings store `defaultModelHandleId`
- restore logic resolves the opaque id through `ModelCatalog`
- fallback logic operates on available `ModelHandle`s, not provider-specific defaults

There is no requirement to preserve exact backwards compatibility for old session model metadata if doing so complicates the cutover. A best-effort migration path is acceptable.

## Coding Agent Changes

`coding-agent` should stop importing provider-centric types such as `KnownProvider` as part of model selection logic.

Expected changes:

- remove provider-default tables
- replace `provider/model` parsing logic with model-handle lookup or search
- redesign CLI and `/model` UX around model handles and display metadata
- remove provider-specific docs and setup paths that only existed because `pi-ai` owned auth/runtime concerns

Provider names may still appear in labels when surfaced by BAML metadata, but only as informational display.

## Agent Core Changes

`pi-agent-core` should no longer depend on `@mariozechner/pi-ai` types as its runtime contract.

Expected changes:

- replace `Model<Api>` with `ModelHandle`
- replace `streamFn` dependency on `pi-ai` stream semantics with the local `LlmRuntime`
- keep tool execution and turn-loop behavior intact
- keep the context model aligned to what the runtime layer can map into BAML prompts and tool calls

## Migration Strategy

This is a staged internal migration with a single public runtime path.

1. Create `llm-runtime` and define local interfaces.
2. Build the BAML-backed implementation for model discovery, invocation, streaming adaptation, and usage extraction.
3. Migrate `pi-agent-core` to the new runtime interfaces.
4. Migrate `coding-agent` model selection, persistence, and restore logic to `ModelHandle`.
5. Remove `pi-ai` runtime usage from the active path.
6. Delete obsolete provider-aware code and docs once parity for the chosen v1 scope is reached.

## Risks

### Streaming Tool Semantics

BAML may not expose partial tool-call structure with the same fidelity as `pi-ai`. The adapter may need to reconstruct tool-call events from BAML stream updates, or the runtime may need to reduce granularity.

### Session Replay Assumptions

Some current serialized assistant metadata is shaped around `pi-ai` provider semantics. Those assumptions may need to be loosened or dropped.

### Model Discovery UX

Without provider as the primary grouping axis, model selectors and defaults need a new grouping and fallback strategy.

### Usage Fidelity

If BAML does not surface all token categories, total cost and usage displays may become less exact than they are today for some providers.

## Recommendation

Proceed with a hard cutover to a BAML-backed runtime package, API-key-first auth, opaque model handles, and a reduced normalized event model that keeps text streaming and tool calls as the primary contract.

Do not spend the initial migration trying to preserve OAuth/subscription paths or provider-specific reasoning fidelity. Those can be evaluated later as optional adapters only if they fit cleanly behind the new runtime boundary.
