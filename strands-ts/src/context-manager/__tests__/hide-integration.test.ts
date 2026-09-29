import { describe, it, expect } from 'vitest'
import { Agent } from '../../agent/agent.js'
import { ContextManager } from '../context-manager.js'
import { Hide } from '../strategies/hide.js'
import { StaticSearch } from '../tool-search.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import { TextBlock } from '../../types/messages.js'
import type { Message } from '../../types/messages.js'
import type { ModelStreamEvent } from '../../models/streaming.js'
import type { StreamOptions } from '../../models/model.js'

/** Records the tool specs each stream() call receives, which is what reaches the provider. */
class RecordingModel extends MockMessageModel {
  readonly seenToolSpecs: string[][] = []

  override async *stream(messages: Message[], options?: StreamOptions): AsyncGenerator<ModelStreamEvent> {
    this.seenToolSpecs.push((options?.toolSpecs ?? []).map((spec) => spec.name))
    yield* super.stream(messages, options)
  }
}

describe('Hide through ContextManager', () => {
  it('filters the specs the model receives and leaves the registry intact', async () => {
    const model = new RecordingModel().addTurn(new TextBlock('done'))
    const tools = ['alpha', 'beta', 'gamma'].map((name) => createMockTool(name, () => 'ok'))
    const agent = new Agent({
      model,
      tools,
      printer: false,
      contextManager: new ContextManager({
        strategies: [Hide('toolSpecs', { search: new StaticSearch(['beta']), keep: 1 })],
        stash: false,
      }),
    })

    await agent.invoke('hello')

    expect(model.seenToolSpecs).toEqual([['beta']])
    expect(agent.tools.map((tool) => tool.name).sort()).toEqual(['alpha', 'beta', 'gamma'])
  })

  it('leaves the catalog unchanged when Hide is not configured', async () => {
    const model = new RecordingModel().addTurn(new TextBlock('done'))
    const tools = ['alpha', 'beta'].map((name) => createMockTool(name, () => 'ok'))
    const agent = new Agent({
      model,
      tools,
      printer: false,
      contextManager: new ContextManager({ stash: false }),
    })

    await agent.invoke('hello')

    expect(model.seenToolSpecs).toEqual([['alpha', 'beta']])
  })
})
