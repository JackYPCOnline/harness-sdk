import { BeforeModelCallEvent } from '../hooks/events.js'
import { HookOrder } from '../hooks/types.js'
import { InjectedBlock } from '../types/messages.js'
import type { ContentBlock, Message, MessageData } from '../types/messages.js'
import type { InvocationContext } from '../agent/invocation-context.js'
import type { InvocationState, LocalAgent } from '../types/agent.js'
import { logger } from '../logging/logger.js'
import { warnOnce } from '../logging/warn-once.js'
import { normalizeError } from '../errors.js'
import type { InjectionOptions, InjectionTrigger, InjectionContext } from './types.js'

/**
 * Writes {@link InjectionOptions.renderContent}'s text into the latest user message before each model
 * call, as an {@link InjectedBlock} that stays in the conversation.
 *
 * The write happens in a `BeforeModelCallEvent` hook that runs after the SDK's other hooks, so it reaches
 * that model call and lands in any user message those hooks added. It only ever touches the latest message
 * while it is a user message, which the model has not answered yet. Every later request therefore starts
 * with the earlier one unchanged.
 *
 * An injection writes into a message at most once, so a retry, a forced structured output call or a resume
 * sends the same request again. Across turns, text equal to its last write is not written again while that
 * block is still in the conversation. Anything that skips — a cancelled call, the trigger not firing,
 * `renderContent` returning empty, or any callback throwing — leaves the conversation unchanged so the
 * model call proceeds (fail open).
 *
 * @param agent - The agent to inject into
 * @param opts - The trigger and `renderContent` callback
 * @internal Delivery primitive. Reach injection through `ContextInjector` or `MemoryManager`.
 */
export function addInjection(agent: LocalAgent, opts: InjectionOptions): void {
  const trigger = resolveTrigger(opts.trigger)
  let lastWrite: { trackingId: string; text: string } | undefined

  agent.addHook(
    BeforeModelCallEvent,
    async (event) => {
      const target = event.agent
      const latest = target.messages.at(-1)
      if (event.cancel || !latest || latest.role !== 'user') {
        return
      }
      // A retry, forced structured output call or resume re-enters a message this injection already wrote into.
      if (lastWrite?.trackingId === latest.trackingId && holdsInjectedText([latest], lastWrite.text)) {
        return
      }

      const injectionContext: InjectionContext = {
        messages: target.messages.map((message) => message.toJSON()),
        appState: target.appState,
        agent: target,
      }
      if (!trigger(injectionContext)) {
        return
      }

      let text: string | undefined
      try {
        text = await opts.renderContent(injectionContext)
      } catch (error) {
        logger.warn(`reason=<${normalizeError(error).message}> | injection renderContent threw; skipping injection`)
        return
      }
      if (
        !text?.trim() ||
        holdsInjectedText([latest], text) ||
        (text === lastWrite?.text && holdsInjectedText(target.messages, text))
      ) {
        return
      }

      // Recorded before the write, so a MessageUpdatedEvent listener that throws cannot cause a second copy.
      lastWrite = { trackingId: latest.trackingId, text }
      await writeToLatestUserMessage(
        target,
        [new InjectedBlock({ text })],
        event.invocationState,
        event.invocationContext
      )
    },
    { order: HookOrder.SDK_LAST }
  )
}

/**
 * Resolves an {@link InjectionTrigger} name or predicate into a single gate predicate over the
 * {@link InjectionContext}.
 *
 * `'userTurn'` maps to {@link isUserTurn} (over `ctx.messages`); `'everyTurn'` to an always-true gate;
 * a user-supplied predicate is wrapped so that a throw fails open (logs and skips injection rather than
 * aborting the model call).
 *
 * @param trigger - An {@link InjectionTrigger} name, a predicate, or `undefined` (defaults to `'userTurn'`)
 * @returns A predicate that, given the {@link InjectionContext}, returns whether to inject this call
 * @internal Delivery primitive. Reach injection through `ContextInjector` or `MemoryManager`.
 */
export function resolveTrigger(
  trigger: InjectionTrigger | ((context: InjectionContext) => boolean) | undefined
): (context: InjectionContext) => boolean {
  if (trigger === undefined || trigger === 'userTurn') {
    return (context) => isUserTurn(context.messages)
  }
  if (trigger === 'everyTurn') {
    return () => true
  }
  const predicate = trigger
  return (context) => {
    try {
      return predicate(context)
    } catch (error) {
      logger.warn(`reason=<${normalizeError(error).message}> | injection trigger threw; skipping injection`)
      return false
    }
  }
}

/**
 * Whether the latest message is a fresh user ask: a `user` message carrying no tool result. This is
 * the `'userTurn'` policy — it distinguishes a new chat ask from an autonomous tool-result turn.
 *
 * @param messages - The current conversation, as data
 * @returns `true` when the latest message is a plain user ask, otherwise `false`
 * @internal Delivery primitive. Reach injection through `ContextInjector` or `MemoryManager`.
 */
export function isUserTurn(messages: MessageData[]): boolean {
  const last = messages[messages.length - 1]
  return !!last && last.role === 'user' && !last.content.some((block) => 'toolResult' in block)
}

/** The agent's write into the latest user message, which fires `MessageUpdatedEvent`. */
interface LatestUserMessageWriter {
  _appendToLatestUserMessage(
    blocks: readonly ContentBlock[],
    invocationState: InvocationState,
    invocationContext?: InvocationContext
  ): Promise<Message | undefined>
}

async function writeToLatestUserMessage(
  agent: LocalAgent,
  blocks: ContentBlock[],
  invocationState: InvocationState,
  invocationContext: InvocationContext | undefined
): Promise<void> {
  const writer = agent as Partial<LatestUserMessageWriter>
  if (typeof writer._appendToLatestUserMessage !== 'function') {
    warnOnce(logger, 'agent does not support writing injected blocks | skipping injection')
    return
  }
  await writer._appendToLatestUserMessage(blocks, invocationState, invocationContext)
}

function holdsInjectedText(messages: Message[], text: string): boolean {
  return messages.some((message) =>
    message.content.some((block) => block.type === 'injectedBlock' && block.injected.text === text)
  )
}
