import { withDurableExecution } from '@aws/durable-execution-sdk-js'
import { Agent, BedrockModel } from '@strands-agents/sdk'

import { registerDurableMiddleware } from './durable-middleware.js'
import { buildTools } from './tools.js'

import type { DurableContext } from '@aws/durable-execution-sdk-js'

interface AgentEvent {
  prompt?: string
  simulateRestart?: boolean
  crashOnSecondCycle?: boolean
}

interface AgentOutput {
  stopReason: string
  text: string
}

const DEFAULT_MODEL_ID = 'global.anthropic.claude-sonnet-4-6'
const DEFAULT_PROMPT = 'Plan my trip to Seattle.'
const SYSTEM_PROMPT = [
  'You are a trip planner. The user will name a city.',
  'Call get_weather with that city, then call book_flight to that city.',
  'After both tools succeed, respond with one short sentence.',
].join(' ')

async function handler(event: AgentEvent, context: DurableContext): Promise<AgentOutput> {
  const prompt = event.prompt ?? DEFAULT_PROMPT
  context.logger.info('agent invocation started', { prompt })

  const agent = new Agent({
    model: new BedrockModel({ modelId: process.env.BEDROCK_MODEL_ID ?? DEFAULT_MODEL_ID }),
    tools: buildTools(),
    systemPrompt: SYSTEM_PROMPT,
    printer: false,
    toolExecutor: 'sequential',
  })

  registerDurableMiddleware(agent, context, {
    crashOnSecondCycle: event.crashOnSecondCycle === true,
  })

  const result = await agent.invoke(prompt)
  if (event.simulateRestart === true) {
    await context.wait('replay-trigger', { seconds: 1 })
  }

  context.logger.info('agent invocation completed', { stopReason: result.stopReason })
  return { stopReason: result.stopReason, text: String(result.lastMessage ?? '') }
}

export const lambdaHandler = withDurableExecution(handler)
