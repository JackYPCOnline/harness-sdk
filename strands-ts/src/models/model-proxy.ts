import { unwrapInvocationContext, type InvocationContext } from '../agent/invocation-context.js'
import { logger } from '../logging/logger.js'
import { warnOnce } from '../logging/warn-once.js'
import { Message, TextBlock, type ContentBlock } from '../types/messages.js'
import type { Model, StreamAggregatedResult, StreamOptions } from './model.js'
import type { ModelStreamEvent } from './streaming.js'

/**
 * SDK-internal wrapper that sends a model call's messages in the form models accept and
 * adds the call's token usage to its request. The agent loop and every auxiliary call
 * route their model calls through it. The {@link InvocationContext} is passed per call
 * rather than on {@link StreamOptions}, so providers never see request state. Usage is
 * added only when the call returns normally and the provider reported it; a thrown error
 * passes through and adds nothing.
 *
 * @internal
 */
export class ModelProxy {
  private readonly _model: Model

  constructor(model: Model) {
    this._model = model
  }

  /**
   * Delegates to the wrapped model's `streamAggregated` with every {@link ContextBlock}
   * sent as text, passing every event through unchanged, then adds the call's usage to
   * the request.
   *
   * @param messages - The conversation history to send to the model
   * @param options - Optional streaming configuration
   * @param invocationContext - The request to add the usage to, or `undefined` to add it nowhere
   * @returns Async generator mirroring {@link Model.streamAggregated}
   */
  async *streamAggregated(
    messages: Message[],
    options?: StreamOptions,
    invocationContext?: InvocationContext
  ): AsyncGenerator<ModelStreamEvent | ContentBlock, StreamAggregatedResult, undefined> {
    const result = yield* this._model.streamAggregated(renderContextBlocks(messages), options)
    this._record(result, invocationContext)
    return result
  }

  /** Adds a completed call's reported usage to the request. */
  private _record(result: StreamAggregatedResult, invocationContext?: InvocationContext): void {
    const usage = result?.metadata?.usage
    const context = unwrapInvocationContext(invocationContext)
    if (usage !== undefined && context !== undefined) {
      context.recordUsage(usage)
    }
  }
}

/**
 * Returns the messages as the model receives them. A context block in a user message becomes a
 * text block in its place, and a context block in any other message is left out with a warning.
 * Messages without a context block are returned as they are, and the input is never changed.
 *
 * @param messages - The messages of a model call
 * @returns The messages to send
 */
function renderContextBlocks(messages: Message[]): Message[] {
  if (!messages.some(hasContextBlock)) return messages
  return messages.map((message) => {
    if (!hasContextBlock(message)) return message
    const content: ContentBlock[] = []
    for (const block of message.content) {
      if (block.type !== 'contextBlock') {
        content.push(block)
      } else if (message.role !== 'user') {
        warnOnce(logger, `role=<${message.role}> | context block outside a user message | skipping it`)
      } else {
        // The separator foldIntoLastUserMessage uses: some providers join adjacent text blocks,
        // which would run this text onto the content before it.
        const separator = content.length > 0 && !block.text.startsWith('\n') ? '\n\n' : ''
        content.push(new TextBlock(`${separator}${block.text}`))
      }
    }
    return new Message({
      role: message.role,
      content,
      trackingId: message.trackingId,
      ...(message.metadata !== undefined && { metadata: message.metadata }),
    })
  })
}

function hasContextBlock(message: Message): boolean {
  return message.content.some((block) => block.type === 'contextBlock')
}
