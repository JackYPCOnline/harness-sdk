import { describe, it, expect } from 'vitest'
import { Agent } from '../../agent/agent.js'
import { ContextManager } from '../context-manager.js'
import { RETRIEVAL_TOOL_NAME } from '../retrieval-tool.js'
import { Hide } from '../strategies/hide/index.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import { TextBlock } from '../../types/messages.js'
import type { InMemoryStorage } from '../../storage/in-memory-storage.js'
import type { SearchStrategy } from '../../storage/search/index.js'
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

/** A search strategy that returns fixed keys best-first, ignoring the query and the index. */
function staticSearch(keys: string[]): SearchStrategy<InMemoryStorage> {
  return { search: async () => keys.map((key, index) => ({ key, score: keys.length - index })) }
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
        strategies: [Hide.toolSpecs('toolSpecs', { search: staticSearch(['beta']), keep: 1 })],
        stash: false,
      }),
    })

    await agent.invoke('hello')

    expect(model.seenToolSpecs).toEqual([['beta']])
    expect(agent.tools.map((tool) => tool.name).sort()).toEqual(['alpha', 'beta', 'gamma'])
  })

  it('keeps the retrieval tool visible alongside an offload preset', async () => {
    const model = new RecordingModel().addTurn(new TextBlock('done'))
    const tools = ['alpha', 'beta', 'gamma'].map((name) => createMockTool(name, () => 'ok'))
    const agent = new Agent({
      model,
      tools,
      printer: false,
      contextManager: new ContextManager({
        strategies: ['largeToolOffloading', Hide.toolSpecs('toolSpecs', { search: staticSearch(['beta']), keep: 1 })],
      }),
    })

    await agent.invoke('hello')

    expect(model.seenToolSpecs).toEqual([['beta', RETRIEVAL_TOOL_NAME]])
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
