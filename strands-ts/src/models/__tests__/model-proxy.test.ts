import { describe, it, expect, vi } from 'vitest'
import { ModelProxy } from '../model-proxy.js'
import type { StreamAggregatedResult } from '../model.js'
import { type ModelStreamEvent } from '../streaming.js'
import { InvocationContextImpl } from '../../agent/invocation-context.js'
import { anyTrackingId } from '../../__fixtures__/message-helpers.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { foldIntoLastUserMessage } from '../../injection/message-injection.js'
import { logger } from '../../logging/logger.js'
import { warnOnce } from '../../logging/warn-once.js'
import { ContextBlock, Message, TextBlock, ToolResultBlock, type ContentBlock } from '../../types/messages.js'

// warnOnce logs each message once per process, so a real call can't be asserted in more than one test.
vi.mock('../../logging/warn-once.js', () => ({
  warnOnce: vi.fn(),
}))

function userMessage(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

/** Drains a streamAggregated generator, collecting every yielded event and its final result. */
async function drain(
  stream: AsyncGenerator<ModelStreamEvent | ContentBlock, StreamAggregatedResult, undefined>
): Promise<{ events: Array<ModelStreamEvent | ContentBlock>; result: StreamAggregatedResult }> {
  const events: Array<ModelStreamEvent | ContentBlock> = []
  let next = await stream.next()
  while (!next.done) {
    events.push(next.value)
    next = await stream.next()
  }
  return { events, result: next.value }
}

describe('ModelProxy', () => {
  describe('usage recording', () => {
    it('accumulates a clean call usage into the invocation context total', async () => {
      const model = new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'hi' },
        { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } }
      )
      const invocationContext = new InvocationContextImpl()

      await drain(new ModelProxy(model).streamAggregated([userMessage('go')], undefined, invocationContext))

      expect(invocationContext.usage).toEqual({ turnsTaken: 0, inputTokens: 10, outputTokens: 5, totalTokens: 15 })
    })

    it('accumulates across calls that share one invocation context', async () => {
      const invocationContext = new InvocationContextImpl()
      const first = new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'one' },
        { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } }
      )
      const second = new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'two' },
        { usage: { inputTokens: 20, outputTokens: 10, totalTokens: 30 } }
      )

      await drain(new ModelProxy(first).streamAggregated([userMessage('go')], undefined, invocationContext))
      await drain(new ModelProxy(second).streamAggregated([userMessage('again')], undefined, invocationContext))

      expect(invocationContext.usage).toEqual({ turnsTaken: 0, inputTokens: 30, outputTokens: 15, totalTokens: 45 })
    })

    it('records nothing when the model call throws', async () => {
      const model = new MockMessageModel().addTurn(new Error('boom'))
      const invocationContext = new InvocationContextImpl()

      await expect(
        drain(new ModelProxy(model).streamAggregated([userMessage('go')], undefined, invocationContext))
      ).rejects.toThrow('boom')

      expect(invocationContext.usage).toEqual(new InvocationContextImpl().usage)
    })

    it('records nothing when the model reports no usage', async () => {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'hi' })
      const invocationContext = new InvocationContextImpl()

      await drain(new ModelProxy(model).streamAggregated([userMessage('go')], undefined, invocationContext))

      expect(invocationContext.usage).toEqual(new InvocationContextImpl().usage)
    })

    it('is a no-op when called without an invocation context', async () => {
      const model = new MockMessageModel().addTurn(
        { type: 'textBlock', text: 'hi' },
        { usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 } }
      )

      const { result } = await drain(new ModelProxy(model).streamAggregated([userMessage('go')]))

      expect(result.metadata?.usage).toEqual({ inputTokens: 10, outputTokens: 5, totalTokens: 15 })
    })
  })

  describe('pass-through', () => {
    it('yields the same events as the underlying model', async () => {
      const build = (): MockMessageModel =>
        new MockMessageModel().addTurn(
          { type: 'textBlock', text: 'hello' },
          { usage: { inputTokens: 1, outputTokens: 2, totalTokens: 3 } }
        )

      const direct = await drain(build().streamAggregated([userMessage('go')]))
      const proxied = await drain(
        new ModelProxy(build()).streamAggregated([userMessage('go')], undefined, new InvocationContextImpl())
      )

      expect(proxied.events).toEqual(direct.events)
      expect(proxied.result).toEqual({
        ...direct.result,
        message: { ...direct.result.message, trackingId: anyTrackingId },
      })
    })
  })

  describe('context blocks', () => {
    /** Sends the messages through the proxy and returns the messages the model received. */
    async function sent(messages: Message[]): Promise<Message[]> {
      const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'ok' })
      const streamSpy = vi.spyOn(model, 'stream')
      await drain(new ModelProxy(model).streamAggregated(messages))
      return streamSpy.mock.calls[0]![0]
    }

    it.each([
      ['a prompt', [new TextBlock('Refactor the tokenizer.')], 'reminder'],
      [
        'a tool result turn',
        [new ToolResultBlock({ toolUseId: 'tool-1', status: 'success', content: [new TextBlock('done')] })],
        'reminder',
      ],
      ['an empty user message', [], 'reminder'],
      ['text that starts with a newline', [new TextBlock('Refactor the tokenizer.')], '\nreminder'],
    ])('sends the text the injectors fold in today, for %s', async (_name, content: ContentBlock[], text) => {
      const { messages: folded } = foldIntoLastUserMessage(
        [new Message({ role: 'user', content, trackingId: 'msg-1' })],
        text
      )

      const messages = [
        new Message({ role: 'user', content: [...content, new ContextBlock({ text })], trackingId: 'msg-1' }),
      ]

      expect(await sent(messages)).toEqual(folded)
    })

    it('sends each of several context blocks as its own text block, in place', async () => {
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('Refactor the tokenizer.'),
            new ContextBlock({ text: 'one' }),
            new ContextBlock({ text: 'two' }),
          ],
          trackingId: 'msg-1',
          metadata: { custom: { source: 'test' } },
        }),
      ]

      expect(await sent(messages)).toEqual([
        new Message({
          role: 'user',
          content: [new TextBlock('Refactor the tokenizer.'), new TextBlock('\n\none'), new TextBlock('\n\ntwo')],
          trackingId: 'msg-1',
          metadata: { custom: { source: 'test' } },
        }),
      ])
    })

    it('separates text that follows a context block, without changing how the block is sent', async () => {
      const messages = [
        new Message({
          role: 'user',
          content: [
            new TextBlock('Refactor.'),
            new ContextBlock({ text: 'reminder' }),
            new TextBlock('Also the docs.'),
          ],
          trackingId: 'msg-1',
        }),
        new Message({
          role: 'user',
          content: [new ContextBlock({ text: 'first' }), new TextBlock('Then the docs.')],
          trackingId: 'msg-2',
        }),
      ]

      expect(await sent(messages)).toEqual([
        new Message({
          role: 'user',
          content: [new TextBlock('Refactor.'), new TextBlock('\n\nreminder'), new TextBlock('\n\nAlso the docs.')],
          trackingId: 'msg-1',
        }),
        new Message({
          role: 'user',
          content: [new TextBlock('first'), new TextBlock('\n\nThen the docs.')],
          trackingId: 'msg-2',
        }),
      ])
    })

    it('sends a context block in an earlier user message too', async () => {
      const answer = new Message({ role: 'assistant', content: [new TextBlock('Done.')], trackingId: 'msg-2' })
      const latest = userMessage('Now the docs.')
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('Refactor.'), new ContextBlock({ text: 'reminder' })],
          trackingId: 'msg-1',
        }),
        answer,
        latest,
      ]

      expect(await sent(messages)).toEqual([
        new Message({
          role: 'user',
          content: [new TextBlock('Refactor.'), new TextBlock('\n\nreminder')],
          trackingId: 'msg-1',
        }),
        answer,
        latest,
      ])
    })

    it('leaves out a context block outside a user message, with a warning', async () => {
      vi.mocked(warnOnce).mockClear()
      const prompt = userMessage('go')
      const messages = [
        prompt,
        new Message({
          role: 'assistant',
          content: [new TextBlock('Hi'), new ContextBlock({ text: 'reminder' }), new TextBlock('Bye')],
          trackingId: 'msg-2',
        }),
      ]

      expect(await sent(messages)).toEqual([
        prompt,
        new Message({ role: 'assistant', content: [new TextBlock('Hi'), new TextBlock('Bye')], trackingId: 'msg-2' }),
      ])
      expect(vi.mocked(warnOnce).mock.calls).toEqual([
        [logger, 'role=<assistant> | context block outside a user message | skipping it'],
      ])
    })

    it('does not change the messages it was given', async () => {
      const messages = [
        new Message({
          role: 'user',
          content: [new TextBlock('Refactor the tokenizer.'), new ContextBlock({ text: 'reminder' })],
        }),
      ]

      await sent(messages)

      expect(messages[0]!.content).toEqual([
        new TextBlock('Refactor the tokenizer.'),
        new ContextBlock({ text: 'reminder' }),
      ])
    })

    it('passes messages without a context block through as they are', async () => {
      const messages = [userMessage('go')]

      expect(await sent(messages)).toBe(messages)
    })
  })
})
