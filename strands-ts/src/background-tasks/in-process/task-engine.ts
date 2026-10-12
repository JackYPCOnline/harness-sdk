import type { InterruptStateData } from '../../interrupt.js'
import { TaskScheduler } from './scheduler.js'
import type { TaskEngineOptions, TaskRecord } from './task-types.js'

export { TaskNotFoundError } from './scheduler.js'

/**
 * Runs and tracks bounded tasks with consumer-defined input and result types.
 *
 * This facade owns no scheduling behavior; it exposes the generic contract over the same private
 * scheduler used by the compatibility-preserving tool-task engine.
 *
 * @internal
 */
export class TaskEngine<TInput, TResult> {
  private readonly _scheduler: TaskScheduler<TInput, TResult>

  /**
   * Creates a generic in-process task engine.
   *
   * @param options - Execution limits and callbacks supplied by the consumer.
   * @throws TypeError if the concurrency or timeout configuration is invalid.
   */
  constructor(options: TaskEngineOptions<TInput, TResult>) {
    this._scheduler = new TaskScheduler(options)
  }

  /** Submits a task for execution. */
  submit(input: TInput): TaskRecord<TInput, TResult> {
    return this._scheduler.submit(input)
  }

  /** Gets a task by ID. */
  get(taskId: string): TaskRecord<TInput, TResult> | undefined {
    return this._scheduler.get(taskId)
  }

  /** Lists the tasks tracked by this engine. */
  list(): readonly TaskRecord<TInput, TResult>[] {
    return this._scheduler.list()
  }

  /** Removes a terminal task. */
  remove(taskId: string): void {
    this._scheduler.remove(taskId)
  }

  /** Cancels a non-terminal task and aborts its active execution. */
  cancel(taskId: string, options: { readonly reason: string }): TaskRecord<TInput, TResult> {
    return this._scheduler.cancel(taskId, options)
  }

  /** Waits until no tasks are queued or executing. */
  async waitForIdle(options?: { readonly cancelSignal?: AbortSignal }): Promise<void> {
    await this._scheduler.waitForIdle(options)
  }

  /** Applies input to a task with `input_required` status. */
  resume(
    taskId: string,
    update: (state: InterruptStateData) => {
      readonly state: InterruptStateData
      readonly ready: boolean
    }
  ): TaskRecord<TInput, TResult> {
    return this._scheduler.resume(taskId, update)
  }
}
