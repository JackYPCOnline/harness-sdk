/**
 * Hide tool specs — filters which tool specs the model sees on each call.
 *
 * Tool specs are a per-call projection of the tool registry, rebuilt for every model call and
 * carried on `InvokeModelContext.toolSpecs`. The registry is never touched: a hidden spec is
 * absent from one call's projection, and the next invocation recomputes the view.
 *
 * @internal
 */

import { logger } from '../../../logging/logger.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { KeywordSearchStrategy } from '../../../storage/search/index.js'
import { STRUCTURED_OUTPUT_TOOL_NAME } from '../../../tools/structured-output-tool.js'
import { TextBlock } from '../../../types/messages.js'
import { RETRIEVAL_TOOL_NAME } from '../../retrieval-tool.js'
import { BaseHideStrategy } from './base.js'
import type { InvokeModelContext } from '../../../middleware/stages.js'
import type { SearchStrategy } from '../../../storage/search/index.js'
import type { ToolSpec } from '../../../tools/types.js'
import type { LocalAgent } from '../../../types/agent.js'
import type { Message } from '../../../types/messages.js'
import type { ContextStrategy } from '../../types.js'
import type { HideConditions } from './base.js'

/**
 * Target for `Hide.toolSpecs` — which specs are candidates for hiding.
 *
 * - `"toolSpecs"` — every tool spec on the call
 * - `string[]` — `toolSpec::*` (every spec) or `toolSpec::<name>` entries name the candidates;
 *   a `!toolSpec::<name>` entry is pinned: always visible, never a candidate, and does not consume `keep`
 *
 * @internal
 */
export type HideToolSpecsTarget = 'toolSpecs' | string[]

/**
 * What the model sees when search fails or returns no usable match and there is no previous
 * selection to carry forward.
 *
 * - `"all"` — every candidate
 * - `"none"` — only pinned and protected tools
 *
 * @internal
 */
export type HideFailurePolicy = 'all' | 'none'

/**
 * Configuration for `Hide.toolSpecs`.
 *
 * @internal
 */
export interface HideToolSpecsConfig {
  /**
   * Ranks specs by relevance to the latest user message. Runs over a per-invocation
   * `InMemoryStorage` index of the candidate specs. Defaults to `KeywordSearchStrategy`.
   */
  search?: SearchStrategy<InMemoryStorage>
  /** How many candidates to keep visible. Defaults to 10. */
  keep?: number
  /**
   * Tool names the model never sees, regardless of search, `count`, or pinning. Tools the
   * ContextManager and agent loop depend on (`structured_output`, `retrieve_context`) are
   * never hidden, even if listed here.
   */
  alwaysHide?: readonly string[]
  /** What to show when no selection can be made. Defaults to `"all"`. */
  onFailure?: HideFailurePolicy
}

const DEFAULT_KEEP = 10
const TOOL_SPEC_PREFIX = 'toolSpec::'
const TOOL_SPEC_WILDCARD = `${TOOL_SPEC_PREFIX}*`

/** Tools the ContextManager or agent loop depends on; never hidden. */
const PROTECTED_TOOLS: ReadonlySet<string> = new Set([STRUCTURED_OUTPUT_TOOL_NAME, RETRIEVAL_TOOL_NAME])

/** Per-invocation selection. */
class ToolSpecsState {
  constructor(readonly selected: ReadonlySet<string>) {}
}

/**
 * Selects once per invocation and reuses the selection through the tool loop, so a stable
 * selection is a byte-identical tool prefix from call to call.
 *
 * @internal
 */
export class HideToolSpecsStrategy extends BaseHideStrategy<ToolSpecsState> {
  readonly name = 'hide:toolSpecs'

  private readonly _target: HideToolSpecsTarget
  private readonly _config: HideToolSpecsConfig
  private readonly _search: SearchStrategy<InMemoryStorage>
  private readonly _keep: number
  private readonly _alwaysHide: ReadonlySet<string>
  private readonly _onFailure: HideFailurePolicy
  /** Candidate names from `toolSpec::<name>` entries; undefined means every spec is a candidate. */
  private readonly _candidates: ReadonlySet<string> | undefined
  /** Names from `!toolSpec::<name>` entries; always visible, never candidates. */
  private readonly _pinned: ReadonlySet<string>
  /** Each agent's last search-backed selection, carried forward when a follow-up turn has no matches. */
  private readonly _previous = new WeakMap<LocalAgent, ReadonlySet<string>>()

  constructor(target: HideToolSpecsTarget, config?: HideToolSpecsConfig, conditions?: HideConditions) {
    super(conditions)
    if (Array.isArray(target) && target.length === 0) {
      throw new Error('Empty array target matches nothing — provide at least one target')
    }
    if (config?.keep !== undefined && (!Number.isInteger(config.keep) || config.keep < 1)) {
      throw new Error(`keep must be a positive integer, got ${config.keep}`)
    }
    this._target = target
    this._config = config ?? {}
    this._search = config?.search ?? KeywordSearchStrategy
    this._keep = config?.keep ?? DEFAULT_KEEP
    this._alwaysHide = new Set(config?.alwaysHide ?? [])
    this._onFailure = config?.onFailure ?? 'all'
    const { candidates, pinned } = resolveTarget(target)
    this._candidates = candidates
    this._pinned = pinned
  }

  when(conditions: HideConditions): ContextStrategy {
    return new HideToolSpecsStrategy(this._target, this._config, conditions)
  }

  protected async _transform(context: InvokeModelContext): Promise<InvokeModelContext> {
    if (context.toolChoice !== undefined) return context

    const catalog = context.toolSpecs
    const shown = catalog.filter((spec) => PROTECTED_TOOLS.has(spec.name) || !this._alwaysHide.has(spec.name))
    const eligible = shown.filter((spec) => this._isEligible(spec))
    const gated = this._count !== undefined && eligible.length < this._count
    if (gated || eligible.length === 0) {
      return shown.length === catalog.length ? context : this._emit(context, catalog, shown)
    }

    const state = this._getState(context.invocationState) ?? (await this._open(context, eligible))
    const visible = shown.filter((spec) => !this._isEligible(spec) || state.selected.has(spec.name))
    return this._emit(context, catalog, visible)
  }

  private async _open(context: InvokeModelContext, eligible: readonly ToolSpec[]): Promise<ToolSpecsState> {
    const state = await this._select(context, eligible)
    this._setState(context.invocationState, state)
    return state
  }

  private async _emit(
    context: InvokeModelContext,
    catalog: readonly ToolSpec[],
    visible: ToolSpec[]
  ): Promise<InvokeModelContext> {
    logger.debug(
      `strategy=<${this.name}>, catalog=<${catalog.length}>, visible=<${visible.length}> | tool specs filtered`
    )
    const projectedInputTokens = await this._correctProjection(context, catalog, visible)
    return {
      ...context,
      toolSpecs: visible,
      ...(projectedInputTokens !== undefined && { projectedInputTokens }),
    }
  }

  /**
   * Open a selection for this invocation. Falls back, in order, to the previous invocation's
   * selection (no matches) and then to `onFailure` (nothing to carry, or search failed).
   */
  private async _select(context: InvokeModelContext, eligible: readonly ToolSpec[]): Promise<ToolSpecsState> {
    const eligibleNames = new Set(eligible.map((spec) => spec.name))

    try {
      const selected = await this._rank(eligible, queryFromMessages(context.messages))
      if (selected.size > 0) {
        this._previous.set(context.agent, selected)
        return new ToolSpecsState(selected)
      }

      const carried = this._carryForward(context.agent, eligibleNames)
      if (carried) {
        logger.debug(`strategy=<${this.name}> | no matches, carrying previous selection forward`)
        return new ToolSpecsState(carried)
      }
      logger.debug(`strategy=<${this.name}>, onFailure=<${this._onFailure}> | no matches and nothing to carry forward`)
      return this._fallback(eligibleNames)
    } catch (error) {
      logger.warn(`strategy=<${this.name}>, onFailure=<${this._onFailure}>, error=<${error}> | search failed`)
      return this._fallback(eligibleNames)
    }
  }

  /**
   * Rank the candidates and return the top `keep` names. Storage keys are path-normalized and
   * tool names are not, so names are URI-encoded into keys and mapped back from results.
   */
  private async _rank(eligible: readonly ToolSpec[], query: string): Promise<Set<string>> {
    const index = new InMemoryStorage(this._search)
    const encoder = new TextEncoder()
    const nameByKey = new Map<string, string>()
    for (const spec of eligible) {
      const key = indexKey(spec.name)
      nameByKey.set(key, spec.name)
      await index.write(key, encoder.encode(searchableText(spec)))
    }

    const selected = new Set<string>()
    for (const result of await this._search.search(index, query)) {
      const name = nameByKey.get(result.key)
      if (name !== undefined) selected.add(name)
      if (selected.size >= this._keep) break
    }
    return selected
  }

  private _fallback(eligibleNames: ReadonlySet<string>): ToolSpecsState {
    return new ToolSpecsState(this._onFailure === 'all' ? eligibleNames : new Set())
  }

  /** The agent's previous selection, intersected with what is eligible now; undefined if nothing survives. */
  private _carryForward(agent: LocalAgent, eligibleNames: ReadonlySet<string>): ReadonlySet<string> | undefined {
    const previous = this._previous.get(agent)
    if (previous === undefined) return undefined
    const carried = new Set([...previous].filter((name) => eligibleNames.has(name)))
    return carried.size > 0 ? carried : undefined
  }

  /**
   * The loop projects input tokens against the full catalog before input middleware runs, but only
   * on a cold start. Warm calls derive the projection from the previous call's actual usage, which
   * already excluded hidden specs, so subtracting again would double-count.
   */
  private async _correctProjection(
    context: InvokeModelContext,
    catalog: readonly ToolSpec[],
    visible: readonly ToolSpec[]
  ): Promise<number | undefined> {
    if (context.projectedInputTokens === undefined || visible.length === catalog.length) {
      return context.projectedInputTokens
    }
    if (hasUsageBaseline(context.messages)) return context.projectedInputTokens

    const visibleNames = new Set(visible.map((spec) => spec.name))
    const removed = catalog.filter((spec) => !visibleNames.has(spec.name))
    try {
      const removedTokens = await context.model.countTokens([], { toolSpecs: removed })
      return Math.max(0, context.projectedInputTokens - removedTokens)
    } catch (error) {
      logger.debug(`strategy=<${this.name}>, error=<${error}> | token recount failed, keeping projection`)
      return context.projectedInputTokens
    }
  }

  /** A candidate for hiding: in the target, not pinned, not a protected tool. */
  private _isEligible(spec: ToolSpec): boolean {
    if (PROTECTED_TOOLS.has(spec.name) || this._pinned.has(spec.name)) return false
    return this._candidates === undefined || this._candidates.has(spec.name)
  }

  protected _isState(value: unknown): value is ToolSpecsState {
    return value instanceof ToolSpecsState
  }
}

/** A storage key for a tool name; plain identifiers pass through unchanged, `/`, `\`, and `.` do not. */
function indexKey(name: string): string {
  return encodeURIComponent(name).replaceAll('.', '%2E')
}

/** The searchable text of a spec: name, description, and input-property names and descriptions. */
function searchableText(spec: ToolSpec): string {
  const properties = spec.inputSchema?.properties
  const propertyText =
    properties && typeof properties === 'object'
      ? Object.entries(properties)
          .map(([key, value]) => {
            const description =
              value && typeof value === 'object' && 'description' in value ? String(value.description) : ''
            return `${key} ${description}`
          })
          .join(' ')
      : ''
  return `${spec.name} ${spec.description} ${propertyText}`
}

/**
 * Parses a target into candidate and pinned names. `toolSpec::*`, or no plain entries at all,
 * makes every spec a candidate.
 */
function resolveTarget(target: HideToolSpecsTarget): {
  candidates: ReadonlySet<string> | undefined
  pinned: ReadonlySet<string>
} {
  const pinned = new Set<string>()
  if (!Array.isArray(target)) return { candidates: undefined, pinned }

  const candidates = new Set<string>()
  let wildcard = false
  for (const entry of target) {
    const isPin = entry.startsWith('!')
    const name = isPin ? entry.slice(1) : entry
    if (!name.startsWith(TOOL_SPEC_PREFIX)) {
      throw new Error(
        `Hide targets must be '${TOOL_SPEC_PREFIX}<name>', '${TOOL_SPEC_WILDCARD}', or '!${TOOL_SPEC_PREFIX}<name>', got '${entry}'`
      )
    }
    if (isPin) pinned.add(name.slice(TOOL_SPEC_PREFIX.length))
    else if (entry === TOOL_SPEC_WILDCARD) wildcard = true
    else candidates.add(name.slice(TOOL_SPEC_PREFIX.length))
  }
  return { candidates: wildcard || candidates.size === 0 ? undefined : candidates, pinned }
}

/** Latest user text is the query; tool-result-only user turns are skipped. */
function queryFromMessages(messages: readonly Message[]): string {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!
    if (message.role !== 'user') continue
    const text = message.content
      .filter((block): block is TextBlock => block instanceof TextBlock)
      .map((block) => block.text)
      .join(' ')
      .trim()
    if (text.length > 0) return text
  }
  return ''
}

/** True once an assistant message carries usage; the loop then projects from that baseline. */
function hasUsageBaseline(messages: readonly Message[]): boolean {
  return messages.some((message) => message.role === 'assistant' && message.metadata?.usage !== undefined)
}
