/**
 * Hide strategy — filters which tool specs the model sees on each call.
 *
 * Tool specs are a per-call projection of the tool registry, rebuilt for every model call and
 * carried on `InvokeModelContext.toolSpecs`, not on `ContextState.messages`. `Hide` therefore
 * registers an `InvokeModelStage.Input` handler from `init()` and is a no-op in the message
 * pipeline. The registry is never touched: a hidden spec is absent from one call's projection,
 * and the next invocation recomputes the view.
 *
 * @experimental
 */

import { AfterInvocationEvent, BeforeInvocationEvent } from '../../hooks/events.js'
import { HookOrder } from '../../hooks/types.js'
import { logger } from '../../logging/logger.js'
import { InvokeModelStage } from '../../middleware/stages.js'
import { STRUCTURED_OUTPUT_TOOL_NAME } from '../../tools/structured-output-tool.js'
import { TextBlock } from '../../types/messages.js'
import { LexicalSearch } from '../tool-search.js'
import { resolveToolFilter } from './offload/base.js'
import type { InvokeModelContext } from '../../middleware/stages.js'
import type { ToolSpec } from '../../tools/types.js'
import type { InvocationState, LocalAgent } from '../../types/agent.js'
import type { Message } from '../../types/messages.js'
import type { ContextState, ContextStrategy } from '../types.js'
import type { ToolSearchCandidate, ToolSearchStrategy } from '../tool-search.js'

/**
 * Target for hide operations.
 *
 * - `"toolSpecs"` — every tool spec on the call
 * - `string[]` — specific tool specs, namespaced with `toolSpec::` (e.g. `['toolSpec::*', '!toolSpec::ask_user']`);
 *   prefix with `!` to keep a spec visible without consuming `keep`
 *
 * @experimental
 */
export type HideTarget = 'toolSpecs' | string[]

/**
 * Configuration for a hide strategy.
 *
 * @experimental
 */
export interface HideConfig {
  /** Relevance ranking. Defaults to `LexicalSearch`. */
  search?: ToolSearchStrategy
  /** How many specs to keep visible, not counting exclusions. Defaults to 10. */
  keep?: number
}

/**
 * Conditions that determine when a hide strategy fires.
 *
 * `Hide` fires on catalog size, not on token utilization: `count` is the number of eligible
 * specs after exclusions. Message conditions (`threshold`, `utilization`, `preserveRecent`)
 * do not apply to tool specs.
 *
 * @experimental
 */
export interface HideConditions {
  /** Fire only when at least this many eligible specs are on the call. */
  count?: number
}

/**
 * Intermediate builder result that allows chaining `.when()` conditions.
 * Also implements `ContextStrategy` directly so it can be used without `.when()`.
 */
export interface HideStrategyBuilder extends ContextStrategy {
  /** Add conditions that determine when this strategy fires. */
  when(conditions: HideConditions): ContextStrategy
}

const DEFAULT_KEEP = 10
const TOOL_SPEC_PREFIX = 'toolSpec::'
const STATE_KEY_PREFIX = 'strands:hideToolSpecs'

/** Per-invocation selection, keyed into `invocationState` so concurrent invocations never share one. */
class HideState {
  constructor(readonly selected: ReadonlySet<string>) {}
}

/**
 * Filters `InvokeModelContext.toolSpecs` once per invocation and reuses the selection through the
 * tool loop, so a stable selection is a byte-identical tool prefix from call to call.
 *
 * @internal
 */
export class HideStrategy implements HideStrategyBuilder {
  readonly name = 'hide:toolSpecs'

  private readonly _target: HideTarget
  private readonly _search: ToolSearchStrategy
  private readonly _keep: number
  private readonly _count: number | undefined
  private readonly _excluded: ReadonlySet<string>

  constructor(target: HideTarget, config?: HideConfig, conditions?: HideConditions) {
    if (Array.isArray(target) && target.length === 0) {
      throw new Error('Empty array target matches nothing — provide at least one target')
    }
    if (config?.keep !== undefined && (!Number.isInteger(config.keep) || config.keep < 1)) {
      throw new Error(`keep must be a positive integer, got ${config.keep}`)
    }
    if (conditions?.count !== undefined && (!Number.isInteger(conditions.count) || conditions.count < 0)) {
      throw new Error(`count must be a non-negative integer, got ${conditions.count}`)
    }

    this._target = target
    this._search = config?.search ?? new LexicalSearch()
    this._keep = config?.keep ?? DEFAULT_KEEP
    this._count = conditions?.count
    this._excluded = resolveExcludedSpecs(target)
  }

  when(conditions: HideConditions): ContextStrategy {
    return new HideStrategy(this._target, { search: this._search, keep: this._keep }, conditions)
  }

  init(agent: LocalAgent): void {
    agent.addMiddleware(InvokeModelStage.Input, (context) => this._applyToModelInput(context))
    agent.addHook(AfterInvocationEvent, (event) => this._clearState(event.invocationState), {
      order: HookOrder.SDK_LAST,
    })
    agent.addHook(BeforeInvocationEvent, (event) => this._clearState(event.invocationState), {
      order: HookOrder.SDK_FIRST,
    })
  }

  /** Tool specs are not in the message pipeline. */
  async apply(_context: ContextState): Promise<boolean> {
    return false
  }

  private async _applyToModelInput(context: InvokeModelContext): Promise<InvokeModelContext> {
    if (context.toolChoice !== undefined) return context

    const catalog = context.toolSpecs
    const eligible = catalog.filter((spec) => this._isEligible(spec))
    if (this._count !== undefined && eligible.length < this._count) return context

    let state = this._getState(context.invocationState)
    if (state === undefined) {
      state = await this._select(context, eligible)
      context.invocationState[this._stateKey] = state
    }

    const selected = state.selected
    const visible = catalog.filter((spec) => !this._isEligible(spec) || selected.has(spec.name))
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
   * The loop projects input tokens against the full catalog before input middleware runs.
   * Subtract the specs this call removed so downstream consumers see the filtered size.
   */
  private async _correctProjection(
    context: InvokeModelContext,
    catalog: readonly ToolSpec[],
    visible: readonly ToolSpec[]
  ): Promise<number | undefined> {
    if (context.projectedInputTokens === undefined || visible.length === catalog.length) {
      return context.projectedInputTokens
    }
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

  /** Open a selection for this invocation. Fails open to every eligible spec. */
  private async _select(context: InvokeModelContext, eligible: readonly ToolSpec[]): Promise<HideState> {
    const eligibleNames = new Set(eligible.map((spec) => spec.name))
    const query = queryFromMessages(context.messages)
    const candidates = eligible.map(toCandidate)

    try {
      const matches = await this._search.search(query, candidates, this._keep)
      const selected = new Set<string>()
      for (const match of matches) {
        if (eligibleNames.has(match.id)) selected.add(match.id)
        if (selected.size >= this._keep) break
      }
      if (selected.size === 0) {
        logger.warn(`strategy=<${this.name}>, search=<${this._search.name}> | no valid matches, showing every tool`)
        return new HideState(eligibleNames)
      }
      return new HideState(selected)
    } catch (error) {
      logger.warn(
        `strategy=<${this.name}>, search=<${this._search.name}>, error=<${error}> | search failed, showing every tool`
      )
      return new HideState(eligibleNames)
    }
  }

  /** Excluded specs and the structured-output tool are never hidden. */
  private _isEligible(spec: ToolSpec): boolean {
    if (spec.name === STRUCTURED_OUTPUT_TOOL_NAME) return false
    return !this._excluded.has(spec.name)
  }

  private _getState(invocationState: InvocationState): HideState | undefined {
    const value = invocationState[this._stateKey]
    return value instanceof HideState ? value : undefined
  }

  private _clearState(invocationState: InvocationState): void {
    if (this._getState(invocationState) !== undefined) delete invocationState[this._stateKey]
  }

  /** One key per strategy instance, so two `Hide` entries in one list keep separate selections. */
  private get _stateKey(): string {
    return `${STATE_KEY_PREFIX}:${objectId(this).toString(16)}`
  }
}

/** Parses `!toolSpec::name` entries into the set of specs that stay visible without consuming `keep`. */
function resolveExcludedSpecs(target: HideTarget): ReadonlySet<string> {
  if (!Array.isArray(target)) return new Set()
  const normalized = target.map((entry) =>
    entry.startsWith(`!${TOOL_SPEC_PREFIX}`) ? `!tool::${entry.slice(TOOL_SPEC_PREFIX.length + 1)}` : entry
  )
  const wildcardOnly = normalized.filter((entry) => !entry.startsWith('!'))
  for (const entry of wildcardOnly) {
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

function toCandidate(spec: ToolSpec): ToolSearchCandidate {
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
  return { id: spec.name, text: `${spec.name} ${spec.description} ${propertyText}` }
}

const objectIds = new WeakMap<object, number>()
let nextObjectId = 1

function objectId(value: object): number {
  let id = objectIds.get(value)
  if (id === undefined) {
    id = nextObjectId++
    objectIds.set(value, id)
  }
  return id
}

/**
 * Builder for hide strategies — filters which tool specs the model sees.
 *
 * @example
 * ```typescript
 * // Keep the 10 most relevant specs once the catalog has 20 or more
 * Hide('toolSpecs', { keep: 10 }).when({ count: 20 })
 * // Keep ask_user and finish visible without consuming keep
 * Hide(['toolSpec::*', '!toolSpec::ask_user', '!toolSpec::finish'], { keep: 15 }).when({ count: 20 })
 * ```
 *
 * @param target - Which specs are eligible for hiding
 * @param config - Search strategy and keep count
 * @returns A strategy that can be used directly or refined with `.when()`
 * @throws Error if `target` is an empty array, `keep` is not a positive integer, or a target entry is malformed
 */
export function Hide(target: HideTarget, config?: HideConfig): HideStrategyBuilder {
  return new HideStrategy(target, config)
}
