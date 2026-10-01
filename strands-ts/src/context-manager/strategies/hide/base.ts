/**
 * Base hide strategy and shared infrastructure.
 *
 * Hide strategies operate on the per-call model input (`InvokeModelContext`), not on
 * `ContextState.messages`. They register an `InvokeModelStage.Input` handler from `init()`
 * and are a no-op in the message pipeline.
 *
 * Not exported from the package barrel while the API is experimental.
 *
 * @internal
 */

import { AfterInvocationEvent, BeforeInvocationEvent } from '../../../hooks/events.js'
import { HookOrder } from '../../../hooks/types.js'
import { InvokeModelStage } from '../../../middleware/stages.js'
import type { InvokeModelContext } from '../../../middleware/stages.js'
import type { InvocationState, LocalAgent } from '../../../types/agent.js'
import type { ContextState, ContextStrategy } from '../../types.js'

/**
 * Conditions that determine when a hide strategy fires.
 *
 * Hide strategies fire on catalog size, not on token utilization. Message conditions
 * (`threshold`, `utilization`, `preserveRecent`) do not apply.
 *
 * @internal
 */
export interface HideConditions {
  /** Fire only when at least this many eligible items are on the call. */
  count?: number
}

/**
 * Intermediate builder result that allows chaining `.when()` conditions.
 * Also implements `ContextStrategy` directly so it can be used without `.when()`.
 *
 * @internal
 */
export interface HideStrategyBuilder extends ContextStrategy {
  /** Add conditions that determine when this strategy fires. */
  when(conditions: HideConditions): ContextStrategy
}

const STATE_KEY_PREFIX = 'strands'
let nextInstanceId = 1

/**
 * Shared hide logic: middleware registration, per-invocation state, and the `count` gate.
 * Subclasses implement `_transform` to filter one field of the model input.
 *
 * @internal
 */
export abstract class BaseHideStrategy<TState extends object> implements HideStrategyBuilder {
  abstract readonly name: string

  protected readonly _count: number | undefined
  private readonly _instanceId = nextInstanceId++

  constructor(conditions?: HideConditions) {
    if (conditions?.count !== undefined && (!Number.isInteger(conditions.count) || conditions.count < 0)) {
      throw new Error(`count must be a non-negative integer, got ${conditions.count}`)
    }
    this._count = conditions?.count
  }

  abstract when(conditions: HideConditions): ContextStrategy

  init(agent: LocalAgent): void {
    agent.addMiddleware(InvokeModelStage.Input, (context) => this._transform(context))
    agent.addHook(AfterInvocationEvent, (event) => this._clearState(event.invocationState), {
      order: HookOrder.SDK_LAST,
    })
    agent.addHook(BeforeInvocationEvent, (event) => this._clearState(event.invocationState), {
      order: HookOrder.SDK_FIRST,
    })
  }

  /** Hide strategies do not touch the message pipeline. */
  async apply(_context: ContextState): Promise<boolean> {
    return false
  }

  /** Filter the model input for one call. */
  protected abstract _transform(context: InvokeModelContext): Promise<InvokeModelContext>

  /** Narrow a stored value to this strategy's state type; foreign values are ignored. */
  protected abstract _isState(value: unknown): value is TState

  protected _getState(invocationState: InvocationState): TState | undefined {
    const value = invocationState[this._stateKey]
    return this._isState(value) ? value : undefined
  }

  protected _setState(invocationState: InvocationState, state: TState): void {
    invocationState[this._stateKey] = state
  }

  private _clearState(invocationState: InvocationState): void {
    if (this._getState(invocationState) !== undefined) delete invocationState[this._stateKey]
  }

  /** One key per strategy instance, so two hide entries in one list keep separate state. */
  private get _stateKey(): string {
    return `${STATE_KEY_PREFIX}:${this.name}:${this._instanceId}`
  }
}
