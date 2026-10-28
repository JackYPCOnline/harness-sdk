import { describe, it, expect, vi } from 'vitest'
import { InjectedBlock, Message, TextBlock, ToolResultBlock } from '../../types/messages.js'
import type { MessageData } from '../../types/messages.js'
import { addInjection, isUserTurn, resolveTrigger } from '../message-injection.js'
import type { InjectionContext, InjectionOptions } from '../types.js'
import { Agent } from '../../agent/agent.js'
import {
  AfterModelCallEvent,
  BeforeModelCallEvent,
  MessageAddedEvent,
  MessageUpdatedEvent,
} from '../../hooks/events.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import { logger } from '../../logging/logger.js'
import type { Tool } from '../../tools/tool.js'

const user = (text: string) => new Message({ role: 'user', content: [new TextBlock(text)] })
const assistant = (text: string) => new Message({ role: 'assistant', content: [new TextBlock(text)] })
const toolResult = () =>
  new Message({
    role: 'user',
    content: [new ToolResultBlock({ toolUseId: 't1', status: 'success', content: [new TextBlock('done')] })],
  })

// resolveTrigger predicates take an InjectionContext; tests only exercise `messages`, so a minimal bag suffices.
const injectionCtx = (messages: MessageData[]) => ({ messages }) as unknown as InjectionContext

describe('isUserTurn', () => {
  it('is true when the last message is a plain user ask', () => {
    expect(isUserTurn([assistant('prior').toJSON(), user('ask').toJSON()])).toBe(true)
  })

  it('is false on a tool-result turn', () => {
    expect(isUserTurn([user('task').toJSON(), assistant('a').toJSON(), toolResult().toJSON()])).toBe(false)
  })

  it('is false when the last message is from the assistant', () => {
    expect(isUserTurn([user('ask').toJSON(), assistant('reply').toJSON()])).toBe(false)
  })

  it('is false for an empty conversation', () => {
    expect(isUserTurn([])).toBe(false)
  })
})

describe('resolveTrigger', () => {
  it('defaults to userTurn', () => {
    const trigger = resolveTrigger(undefined)
    expect(trigger(injectionCtx([user('ask').toJSON()]))).toBe(true)
    expect(trigger(injectionCtx([user('ask').toJSON(), assistant('reply').toJSON()]))).toBe(false)
  })

  it("'userTurn' uses isUserTurn", () => {
    const trigger = resolveTrigger('userTurn')
    expect(trigger(injectionCtx([user('task').toJSON(), assistant('a').toJSON(), toolResult().toJSON()]))).toBe(false)
  })

  it("'everyTurn' always fires", () => {
    const trigger = resolveTrigger('everyTurn')
    expect(trigger(injectionCtx([]))).toBe(true)
  })

  it('passes a predicate through', () => {
    const trigger = resolveTrigger((context) => context.messages.length >= 2)
    expect(trigger(injectionCtx([user('a').toJSON()]))).toBe(false)
    expect(trigger(injectionCtx([user('a').toJSON(), assistant('b').toJSON()]))).toBe(true)
  })

  it('fails open when the predicate throws', () => {
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const trigger = resolveTrigger(() => {
      throw new Error('boom')
    })
    expect(trigger(injectionCtx([user('ask').toJSON()]))).toBe(false)
    expect(warn).toHaveBeenCalled()
    warn.mockRestore()
  })
})

describe('addInjection', () => {
  /** An agent with one injection and a record of the messages each model call received. */
  function setup(model: MockMessageModel, options: InjectionOptions, tools: Tool[] = []) {
    const agent = new Agent({ model, printer: false, tools })
    addInjection(agent, options)
    const requests: MessageData[][] = []
    const originalStream = model.stream.bind(model)
    vi.spyOn(model, 'stream').mockImplementation(async function* (messages, streamOptions) {
      requests.push(messages.map((message) => message.toJSON()))
      yield* originalStream(messages, streamOptions)
    })
    return { agent, requests }
  }

  /** Every request starts with the previous request, byte for byte. */
  function expectEachRequestExtendsThePrevious(requests: MessageData[][]): void {
    for (let index = 1; index < requests.length; index++) {
      const previous = requests[index - 1]!
      expect(requests[index]!.slice(0, previous.length)).toStrictEqual(previous)
    }
  }

  it('writes the text into the latest user message as an InjectedBlock, sent to the model as text', async () => {
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'ok' })
    const { agent, requests } = setup(model, { renderContent: async () => '<memory>dark mode</memory>' })

    await agent.invoke('what theme?')

    expect(agent.messages[0]!.content).toStrictEqual([
      new TextBlock('what theme?'),
      new InjectedBlock({ text: '<memory>dark mode</memory>' }),
    ])
    expect(requests[0]![0]!.content).toStrictEqual([
      { text: 'what theme?' },
      { text: '\n\n<memory>dark mode</memory>' },
    ])
  })

  it('reports the write with MessageUpdatedEvent, not a second MessageAddedEvent', async () => {
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'ok' })
    const { agent } = setup(model, { renderContent: async () => 'reminder' })
    const added: Message[] = []
    const updated: MessageUpdatedEvent[] = []
    agent.addHook(MessageAddedEvent, (event) => {
      added.push(event.message)
    })
    agent.addHook(MessageUpdatedEvent, (event) => {
      updated.push(event)
    })

    await agent.invoke('ask')

    expect(added.map((message) => message.role)).toStrictEqual(['user', 'assistant'])
    expect(updated).toHaveLength(1)
    expect(updated[0]!.trackingId).toBe(agent.messages[0]!.trackingId)
    expect(updated[0]!.message).toBe(agent.messages[0])
  })

  it("keeps every request a prefix of the next across a tool loop with 'everyTurn'", async () => {
    let todo = 'v1'
    const tool = createMockTool('todo_write', () => {
      todo = 'v2'
      return new ToolResultBlock({ toolUseId: 'tool-1', status: 'success', content: [new TextBlock('saved')] })
    })
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'todo_write', toolUseId: 'tool-1', input: {} })
      .addTurn({ type: 'textBlock', text: 'done' })
    const { agent, requests } = setup(
      model,
      { trigger: 'everyTurn', renderContent: async () => `<todos>${todo}</todos>` },
      [tool]
    )

    await agent.invoke('plan the work')

    expect(requests).toHaveLength(2)
    expectEachRequestExtendsThePrevious(requests)
    // The tool result stays first in its message; the new list follows it.
    expect(agent.messages[2]!.content).toStrictEqual([
      expect.objectContaining({ type: 'toolResultBlock' }),
      new InjectedBlock({ text: '<todos>v2</todos>' }),
    ])
  })

  it('writes the same text only once, and new text again', async () => {
    let text = 'cwd: /repo'
    const model = new MockMessageModel()
      .addTurn({ type: 'textBlock', text: 'one' })
      .addTurn({ type: 'textBlock', text: 'two' })
      .addTurn({ type: 'textBlock', text: 'three' })
    const { agent, requests } = setup(model, { renderContent: async () => text })

    await agent.invoke('first')
    await agent.invoke('second')
    text = 'cwd: /other'
    await agent.invoke('third')

    const injected = agent.messages.flatMap((message) =>
      message.content.filter((block) => block.type === 'injectedBlock')
    )
    expect(injected).toStrictEqual([
      new InjectedBlock({ text: 'cwd: /repo' }),
      new InjectedBlock({ text: 'cwd: /other' }),
    ])
    expect(agent.messages[2]!.content).toStrictEqual([new TextBlock('second')])
    expectEachRequestExtendsThePrevious(requests)
  })

  it('writes into a message once, so a retry resends the same request even when the text changes', async () => {
    let renders = 0
    const model = new MockMessageModel().addTurn(new Error('throttled')).addTurn({ type: 'textBlock', text: 'ok' })
    const { agent, requests } = setup(model, { renderContent: async () => `<now>${++renders}</now>` })
    agent.addHook(AfterModelCallEvent, (event) => {
      if (event.error) event.retry = true
    })

    await agent.invoke('ask')

    expect(agent.messages[0]!.content).toStrictEqual([
      new TextBlock('ask'),
      new InjectedBlock({ text: '<now>1</now>' }),
    ])
    expect(requests[1]).toStrictEqual(requests[0])
    expect(renders).toBe(1)
  })

  it('writes nothing when another hook cancels the model call', async () => {
    const renderContent = vi.fn(async () => 'reminder')
    const { agent } = setup(new MockMessageModel(), { renderContent })
    agent.addHook(BeforeModelCallEvent, (event) => {
      event.cancel = 'denied'
    })

    await agent.invoke('ask')

    expect(agent.messages[0]!.content).toStrictEqual([new TextBlock('ask')])
    expect(renderContent).not.toHaveBeenCalled()
  })

  it('writes into a user message another BeforeModelCallEvent hook adds', async () => {
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'ok' })
    const { agent } = setup(model, { renderContent: async () => 'reminder' })
    agent.addHook(BeforeModelCallEvent, (event) => {
      event.agent.messages.push(new Message({ role: 'user', content: [new TextBlock('[guide] be brief')] }))
    })

    await agent.invoke('ask')

    expect(agent.messages.slice(0, 2).map((message) => message.content)).toStrictEqual([
      [new TextBlock('ask')],
      [new TextBlock('[guide] be brief'), new InjectedBlock({ text: 'reminder' })],
    ])
  })

  it("does not render on a tool-result turn with the default 'userTurn' trigger", async () => {
    const tool = createMockTool(
      'lookup',
      () => new ToolResultBlock({ toolUseId: 'tool-1', status: 'success', content: [new TextBlock('found')] })
    )
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'lookup', toolUseId: 'tool-1', input: {} })
      .addTurn({ type: 'textBlock', text: 'done' })
    const renderContent = vi.fn(async () => 'reminder')
    const { agent, requests } = setup(model, { renderContent }, [tool])

    await agent.invoke('ask')

    expect(renderContent).toHaveBeenCalledTimes(1)
    expectEachRequestExtendsThePrevious(requests)
  })

  it('passes the conversation as data, appState and the agent to renderContent', async () => {
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'ok' })
    let received: InjectionContext | undefined
    const { agent } = setup(model, {
      renderContent: async (context) => {
        received = context
        return undefined
      },
    })

    await agent.invoke('ask')

    expect(received!.messages.map((message) => message.role)).toStrictEqual(['user'])
    expect(received!.appState).toBe(agent.appState)
    expect(received!.agent).toBe(agent)
  })

  it.each([
    ['returns empty text', async () => '   '],
    [
      'throws',
      async () => {
        throw new Error('boom')
      },
    ],
  ])('leaves the conversation unchanged when renderContent %s', async (_name, renderContent) => {
    using _warn = vi.spyOn(logger, 'warn').mockImplementation(() => {})
    const model = new MockMessageModel().addTurn({ type: 'textBlock', text: 'ok' })
    const { agent } = setup(model, { renderContent })

    await agent.invoke('ask')

    expect(agent.messages[0]!.content).toStrictEqual([new TextBlock('ask')])
  })
})
