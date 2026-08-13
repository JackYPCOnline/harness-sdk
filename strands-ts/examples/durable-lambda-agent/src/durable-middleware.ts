import { ExecuteToolStage, InvokeModelStage, Message, ToolResultBlock } from '@strands-agents/sdk'

import type { DurableContext, DurableLoggingContext } from '@aws/durable-execution-sdk-js'
import type {
  Agent,
  AgentStreamEvent,
  ExecuteToolContext,
  ExecuteToolResult,
  InvokeModelContext,
  InvokeModelResult,
  MessageData,
} from '@strands-agents/sdk'

interface RegisterOptions {
  crashOnSecondCycle?: boolean
}

/**
 * Checkpoints model and tool middleware stages with Lambda durable execution.
 *
 * @param agent - Agent whose model and tool stages should be checkpointed.
 * @param context - Durable Lambda invocation context.
 * @param options - Optional failure injection used by the recovery demo.
 * @returns A function that removes both middleware handlers.
 */
export function registerDurableMiddleware(
  agent: Agent,
  context: DurableContext,
  options: RegisterOptions = {}
): () => void {
  let modelCallIndex = 0
  let loggingContext: DurableLoggingContext | undefined

  context.configureLogger({
    customLogger: {
      log: (level, ...parameters) => console.log(`[${level}]`, ...parameters),
      info: (...parameters) => console.info(...parameters),
      warn: (...parameters) => console.warn(...parameters),
      error: (...parameters) => console.error(...parameters),
      debug: (...parameters) => console.debug(...parameters),
      configureDurableLoggingContext: (nextLoggingContext) => {
        loggingContext = nextLoggingContext
      },
    },
    modeAware: false,
  })

  const removeModelMiddleware = agent.addMiddleware(
    InvokeModelStage,
    async function* (
      stageContext: InvokeModelContext,
      next
    ): AsyncGenerator<AgentStreamEvent, InvokeModelResult, undefined> {
      const cycleIndex = modelCallIndex
      modelCallIndex += 1
      const stepName = `invoke-model:cycle-${cycleIndex}`
      const shouldCrash = options.crashOnSecondCycle === true && cycleIndex === 1
      const events: AgentStreamEvent[] = []

      const persisted = await context.step<{ message: MessageData; stopReason: string }>(
        stepName,
        async (stepContext) => {
          const attempt = loggingContext?.getDurableLogData().attempt ?? 1
          if (shouldCrash && attempt === 1) {
            stepContext.logger.warn(`step=<${stepName}>, attempt=<${attempt}> | injecting failure`)
            throw new Error('Injected failure during the second model cycle')
          }

          const result = await drainGenerator(
            () => next(stageContext),
            (event) => events.push(event)
          )
          return {
            message: result.result.message.toJSON(),
            stopReason: result.result.stopReason,
          }
        },
        shouldCrash
          ? {
              retryStrategy: (_error, attemptCount) => ({
                shouldRetry: attemptCount < 2,
                delay: { seconds: 1 },
              }),
            }
          : undefined
      )

      yield* events
      return {
        result: {
          message: Message.fromMessageData(persisted.message),
          stopReason: persisted.stopReason as InvokeModelResult['result']['stopReason'],
        },
      }
    }
  )

  const removeToolMiddleware = agent.addMiddleware(
    ExecuteToolStage,
    async function* (
      stageContext: ExecuteToolContext,
      next
    ): AsyncGenerator<AgentStreamEvent, ExecuteToolResult, undefined> {
      const stepName = `tool:${stageContext.toolUse.name}:${stageContext.toolUse.toolUseId}`
      const events: AgentStreamEvent[] = []
      const persisted = await context.step<ReturnType<ToolResultBlock['toJSON']>>(stepName, async () => {
        const result = await drainGenerator(
          () => next(stageContext),
          (event) => events.push(event)
        )
        return result.result.toJSON()
      })

      yield* events
      return { result: ToolResultBlock.fromJSON(persisted) }
    }
  )

  return () => {
    removeToolMiddleware()
    removeModelMiddleware()
  }
}

async function drainGenerator<TEvent, TResult>(
  generatorFactory: () => AsyncGenerator<TEvent, TResult, undefined>,
  onEvent: (event: TEvent) => void
): Promise<TResult> {
  const generator = generatorFactory()
  let nextResult = await generator.next()
  while (!nextResult.done) {
    onEvent(nextResult.value)
    nextResult = await generator.next()
  }
  return nextResult.value
}
