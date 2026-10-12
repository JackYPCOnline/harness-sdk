import { expect } from 'vitest'
import { InProcessTaskEngine } from '../engine.js'
import type { InterruptStateData } from '../../../interrupt.js'
import type { ToolResultBlockData } from '../../../types/messages.js'
import type { InProcessTaskEngineOptions, InProcessTaskRecord } from '../types.js'

/** Engine instantiation used by the engine tests: a string input and a tool result. */
export type TestTaskRecord = InProcessTaskRecord<string, ToolResultBlockData>
export type TestEngineOptions = InProcessTaskEngineOptions<string, ToolResultBlockData>

type ExpectedTaskFields = Pick<TestTaskRecord, 'status'> & Partial<Pick<TestTaskRecord, 'state' | 'result' | 'failure'>>

export function createEngine(
  execute: TestEngineOptions['execute'],
  options: Partial<TestEngineOptions> = {}
): InProcessTaskEngine<string, ToolResultBlockData> {
  return new InProcessTaskEngine({
    maxConcurrency: 2,
    timeout: 1_000,
    execute,
    onTaskUpdated: () => undefined,
    ...options,
  })
}

export function createAdmission(value: string): string {
  return value
}

export function createResult(value: string): ToolResultBlockData {
  return {
    toolUseId: value,
    status: 'success',
    content: [{ text: value }],
  }
}

export function createState(value: string): InterruptStateData {
  return {
    interrupts: {
      [value]: {
        id: value,
        name: value,
      },
    },
    activated: true,
  }
}

export function getStateValue(state: InterruptStateData): string {
  return Object.keys(state.interrupts)[0]!
}

export function deferred<Value>(): { promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

export function expectTask(actual: TestTaskRecord | undefined, task: TestTaskRecord, fields: ExpectedTaskFields): void {
  expect(actual).toEqual({
    taskId: task.taskId,
    input: task.input,
    ...fields,
    createdAt: task.createdAt,
    lastUpdatedAt: expect.any(String),
  })
}
