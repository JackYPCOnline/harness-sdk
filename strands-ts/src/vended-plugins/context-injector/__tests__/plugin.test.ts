import { describe, it, expect, vi } from 'vitest'
import { ContextInjector } from '../plugin.js'
import { InjectedBlock, TextBlock } from '../../../types/messages.js'
import { BeforeModelCallEvent } from '../../../hooks/events.js'
import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { Agent } from '../../../agent/agent.js'
import { MockMessageModel } from '../../../__fixtures__/mock-message-model.js'
import type { StreamOptions } from '../../../models/model.js'

describe('ContextInjector', () => {
  describe('plugin interface', () => {
    it('defaults to the strands:context-injector name', () => {
      expect(new ContextInjector({ renderContent: async () => 'x' }).name).toBe('strands:context-injector')
    })

    it('honors a custom name (so multiple injectors can be told apart)', () => {
      expect(new ContextInjector({ name: 'now', renderContent: async () => 'x' }).name).toBe('now')
    })

    it('registers a BeforeModelCallEvent hook on initAgent', () => {
      const agent = createMockAgent()
      new ContextInjector({ renderContent: async () => 'x' }).initAgent(agent)

      expect(agent.trackedHooks.map((hook) => hook.eventType)).toStrictEqual([BeforeModelCallEvent])
    })
  })

  describe('with an agent', () => {
    it('writes renderContent() text into the prompt as an InjectedBlock', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const agent = new Agent({
        model,
        printer: false,
        plugins: [new ContextInjector({ renderContent: async () => '<ctx>INJECTED</ctx>' })],
      })

      await agent.invoke('what is the weather?')

      expect(agent.messages[0]!.content).toStrictEqual([
        new TextBlock('what is the weather?'),
        new InjectedBlock({ text: '<ctx>INJECTED</ctx>' }),
      ])
    })

    it('leaves the cache point on the last block, since the injected block no longer changes', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'Hello' })
      const seen: number[] = []
      const original = model.stream.bind(model)
      vi.spyOn(model, 'stream').mockImplementation(async function* (messages, options?: StreamOptions) {
        seen.push(options?.dynamicTrailingBlocks ?? 0)
        yield* original(messages, options)
      })
      const agent = new Agent({
        model,
        printer: false,
        plugins: [new ContextInjector({ renderContent: async () => '<ctx>INJECTED</ctx>' })],
      })

      await agent.invoke('what is the weather?')

      expect(seen).toStrictEqual([0])
    })
  })
})
