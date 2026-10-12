import type { InterruptStateData } from '../../interrupt.js'
import type { BackgroundTaskFailureType, BackgroundTaskStatus } from '../types.js'

/**
 * Options and callbacks for bounded in-process task execution.
 *
 * The engine is generic over its payload: `TInput` is whatever a consumer needs to execute a task
 * and `TResult` is what a completed task produces. The background tool plugin instantiates it with
 * tool calls and tool results; other consumers supply their own types and never see tool shapes.
 *
 * @internal
 */
export interface InProcessTaskEngineOptions<TInput, TResult> {
  /** Maximum number of task executions that may run concurrently. */
  readonly maxConcurrency: number
  /** Per-execution timeout in milliseconds, or `Infinity` to disable timeouts. */
  readonly timeout: number
  /** Runs one task execution and returns whether it completed, requires input, or failed. */
  readonly execute: (context: InProcessTaskExecutionContext<TInput>) => Promise<InProcessTaskExecutionOutcome<TResult>>
  /** Receives committed task updates without blocking or failing the engine. */
  readonly onTaskUpdated: (task: InProcessTaskRecord<TInput, TResult>) => void
}

/** Context supplied to one in-process task execution. @internal */
export interface InProcessTaskExecutionContext<TInput> {
  /** Identifier of the logical task. */
  readonly taskId: string
  /** The payload the task was submitted with. */
  readonly input: TInput
  /** Interrupt state supplied when resuming a task that requires input. */
  readonly state?: InterruptStateData
  /** Signal aborted when this execution should stop. */
  readonly cancelSignal: AbortSignal
}

/** Outcome returned by one in-process task execution. @internal */
export type InProcessTaskExecutionOutcome<TResult> =
  | {
      readonly status: 'completed'
      readonly result: TResult
    }
  | {
      readonly status: 'input_required'
      readonly state: InterruptStateData
    }
  | {
      readonly status: 'failed'
      readonly failure: {
        readonly type: BackgroundTaskFailureType
        readonly message: string
      }
      readonly result?: TResult
    }

/**
 * In-process task record.
 *
 * `input` and `result` belong to the consumer and are shared by reference; the engine never copies them.
 *
 * @internal
 */
export interface InProcessTaskRecord<TInput, TResult> {
  /** Identifier of the logical task. */
  readonly taskId: string
  /** The payload the task was submitted with. */
  readonly input: TInput
  /** Current task lifecycle status. */
  status: BackgroundTaskStatus
  /** Interrupt state, required while the task needs input and retained while resuming. */
  state?: InterruptStateData
  /** Execution result, required while completed and optional while failed. */
  result?: TResult
  /** Failure details recorded when the task fails. */
  failure?: {
    readonly type: BackgroundTaskFailureType
    readonly message: string
  }
  /** ISO-8601 creation timestamp. */
  readonly createdAt: string
  /** ISO-8601 timestamp of the latest transition. */
  lastUpdatedAt: string
}
