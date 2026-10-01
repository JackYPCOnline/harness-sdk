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
import { resolveToolFilter } from '../offload/base.js'
import { BaseHideStrategy } from './base.js'
import type { InvokeModelContext } from '../../../middleware/stages.js'
import type { SearchStrategy } from '../../../storage/search/index.js'
import type { ToolSpec } from '../../../tools/types.js'
import type { Message } from '../../../types/messages.js'
import type { ContextStrategy } from '../../types.js'
import type { HideConditions } from './base.js'

/**
 * Target for `Hide.toolSpecs`.
 *
 * - `"toolSpecs"` — every tool spec on the call
 * - `string[]` — `['toolSpec::*', '!toolSpec::ask_user']`; a `!` entry stays visible without consuming `keep`
 *
 * @internal
 */
export type HideToolSpecsTarget = 'toolSpecs' | string[]

/**
 * Configuration for `Hide.toolSpecs`.
 *
 * @internal
 */
export interface HideToolSpecsConfig {
  /**
   * Ranks specs by relevance to the latest user message. Runs over a per-invocation
   * `InMemoryStorage` index of the eligible specs. Defaults to `KeywordSearchStrategy`.
   */
  search?: SearchStrategy<InMemoryStorage>
  /** How many specs to keep visible, not counting exclusions. Defaults to 10. */
  keep?: number
}

const DEFAULT_KEEP = 10
const TOOL_SPEC_PREFIX = 'toolSpec::'
const TOOL_PREFIX = 'tool::'

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
  private readonly _search: SearchStrategy<InMemoryStorage>
  private readonly _keep: number
  private readonly _excluded: ReadonlySet<string>
  /** The last selection this instance made, carried forward when a follow-up turn has no matches. */
  private _previous: ReadonlySet<string> | undefined

  constructor(target: HideToolSpecsTarget, config?: HideToolSpecsConfig, conditions?: HideConditions) {
    super(conditions)
    if (Array.isArray(target) && target.length === 0) {
      throw new Error('Empty array target matches nothing — provide at least one target')
    }
    if (config?.keep !== undefined && (!Number.isInteger(config.keep) || config.keep < 1)) {
      throw new Error(`keep must be a positive integer, got ${config.keep}`)
    }
    this._target = target
    this._search = config?.search ?? KeywordSearchStrategy
    this._keep = config?.keep ?? DEFAULT_KEEP
    this._excluded = resolveExcludedSpecs(target)
  }

  when(conditions: HideConditions): ContextStrategy {
    return new HideToolSpecsStrategy(this._target, { search: this._search, keep: this._keep }, conditions)
  }

  protected async _transform(context: InvokeModelContext): Promise<InvokeModelContext> {
    if (context.toolChoice !== undefined) return context

    const catalog = context.toolSpecs
    const eligible = catalog.filter((spec) => this._isEligible(spec))
    if (this._count !== undefined && eligible.length < this._count) return context

    let state = this._getState(context.invocationState)
    if (state === undefined) {
      state = await this._select(context, eligible)
      this._setState(context.invocationState, state)
      this._previous = state.selected
    }

    const visible = catalog.filter((spec) => !this._isEligible(spec) || state!.selected.has(spec.name))
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
   * selection (no matches) and then to every eligible spec (nothing to carry, or search failed).
   */
  private async _select(context: InvokeModelContext, eligible: readonly ToolSpec[]): Promise<ToolSpecsState> {
    const eligibleNames = new Set(eligible.map((spec) => spec.name))
    const query = queryFromMessages(context.messages)

    try {
      const index = new InMemoryStorage(this._search)
      const encoder = new TextEncoder()
      for (const spec of eligible) {
        await index.write(spec.name, encoder.encode(searchableText(spec)))
      }
      const results = await this._search.search(index, query)

      const selected = new Set<string>()
      for (const result of results) {
        if (eligibleNames.has(result.key)) selected.add(result.key)
        if (selected.size >= this._keep) break
      }
      if (selected.size > 0) return new ToolSpecsState(selected)

      const carried = this._carryForward(eligibleNames)
      if (carried) {
        logger.debug(`strategy=<${this.name}> | no matches, carrying previous selection forward`)
        return new ToolSpecsState(carried)
      }
      logger.warn(`strategy=<${this.name}> | no matches and nothing to carry forward, showing every tool`)
      return new ToolSpecsState(eligibleNames)
    } catch (error) {
      logger.warn(`strategy=<${this.name}>, error=<${error}> | search failed, showing every tool`)
      return new ToolSpecsState(eligibleNames)
    }
  }

  /** The previous selection, intersected with what is eligible now; undefined if nothing survives. */
  private _carryForward(eligibleNames: ReadonlySet<string>): ReadonlySet<string> | undefined {
    if (this._previous === undefined) return undefined
    const carried = new Set([...this._previous].filter((name) => eligibleNames.has(name)))
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

  /** Excluded specs and protected tools are never hidden. */
  private _isEligible(spec: ToolSpec): boolean {
    return !PROTECTED_TOOLS.has(spec.name) && !this._excluded.has(spec.name)
  }

  protected _isState(value: unknown): value is ToolSpecsState {
    return value instanceof ToolSpecsState
  }
}

/**
 * The searchable text of a spec: name, description, and input-property names and descriptions.
 *
 * @param spec - The tool spec
 * @returns Space-joined searchable text
 * @internal
 */
export function searchableText(spec: ToolSpec): string {
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

/** Parses `!toolSpec::name` entries into the set of specs that stay visible without consuming `keep`. */
function resolveExcludedSpecs(target: HideToolSpecsTarget): ReadonlySet<string> {
  if (!Array.isArray(target)) return new Set()
  const normalized = target.map((entry) =>
    entry.startsWith(`!${TOOL_SPEC_PREFIX}`) ? `!${TOOL_PREFIX}${entry.slice(TOOL_SPEC_PREFIX.length + 1)}` : entry
  )
  for (const entry of normalized.filter((entry) => !entry.startsWith('!'))) {
    if (entry !== `${TOOL_SPEC_PREFIX}*`) {
      throw new Error(`Hide targets must be '${TOOL_SPEC_PREFIX}*' or '!${TOOL_SPEC_PREFIX}<name>', got '${entry}'`)
    }
  }
  return resolveToolFilter(normalized.filter((entry) => entry.startsWith('!'))).exclude ?? new Set()
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
