import { describe, it, expect, vi } from 'vitest'
import { Hide, HideStrategy } from '../hide.js'
import { LexicalSearch, StaticSearch } from '../../tool-search.js'
import { InvokeModelStage } from '../../../middleware/stages.js'
import { AfterInvocationEvent, BeforeInvocationEvent } from '../../../hooks/events.js'
import { STRUCTURED_OUTPUT_TOOL_NAME } from '../../../tools/structured-output-tool.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import { createMockAgent, invokeTrackedHook } from '../../../__fixtures__/agent-helpers.js'
import type { MockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { InvokeModelContext } from '../../../middleware/stages.js'
import type { ToolSpec } from '../../../tools/types.js'
import type { InvocationState } from '../../../types/agent.js'
import type { ToolSearchStrategy } from '../../tool-search.js'

type InputHandler = (context: InvokeModelContext) => InvokeModelContext | Promise<InvokeModelContext>

/** Attach a strategy to a mock agent and capture the Input handler it registers. */
function attach(strategy: HideStrategy): { agent: MockAgent; handler: InputHandler } {
  let handler: InputHandler | undefined
  const agent = createMockAgent({
    extra: {
      addMiddleware: ((stage: unknown, registered: InputHandler) => {
        if (stage === InvokeModelStage.Input) handler = registered
        return () => {}
      }) as never,
    },
  })
  strategy.init(agent)
  if (!handler) throw new Error('strategy did not register an Input handler')
  return { agent, handler }
}

function spec(name: string, description = '', properties?: Record<string, { description?: string }>): ToolSpec {
  return {
    name,
    description,
    inputSchema: { type: 'object', properties: properties ?? {} },
  }
}

function user(text: string): Message {
  return new Message({ role: 'user', content: [new TextBlock(text)] })
}

function toolResultOnly(): Message {
  return new Message({
    role: 'user',
    content: [new ToolResultBlock({ toolUseId: 't1', status: 'success', content: [new TextBlock('done')] })],
  })
}

function context(
  agent: MockAgent,
  toolSpecs: ToolSpec[],
  overrides?: Partial<Pick<InvokeModelContext, 'messages' | 'toolChoice' | 'invocationState'>>
): InvokeModelContext {
  return {
    agent,
    model: agent.model,
    messages: overrides?.messages ?? [user('search the billing records')],
    toolSpecs,
    invocationState: overrides?.invocationState ?? {},
    ...(overrides?.toolChoice !== undefined && { toolChoice: overrides.toolChoice }),
  }
}

const catalog = [
  spec('billing_search', 'Search billing records'),
  spec('billing_summary', 'Summarize a billing account'),
  spec('shipping_search', 'Search shipments'),
  spec('shipping_track', 'Track a shipment'),
  spec('ask_user', 'Ask the user a question'),
]

const names = (specs: readonly ToolSpec[]): string[] => specs.map((entry) => entry.name)

describe('Hide', () => {
  describe('construction', () => {
    it('returns a strategy usable without when()', () => {
      const strategy = Hide('toolSpecs')
      expect(strategy.name).toBe('hide:toolSpecs')
      expect(typeof strategy.apply).toBe('function')
    })

    it('when() returns a new strategy and leaves the original unchanged', () => {
      const base = Hide('toolSpecs', { keep: 2 })
      const gated = base.when({ count: 20 })
      expect(gated).not.toBe(base)
      expect(gated.name).toBe('hide:toolSpecs')
    })

    it('throws for an empty array target', () => {
      expect(() => Hide([])).toThrow('Empty array target')
    })

    it('throws for a non-positive keep', () => {
      expect(() => Hide('toolSpecs', { keep: 0 })).toThrow('keep must be a positive integer')
    })

    it('throws for a negative count', () => {
      expect(() => Hide('toolSpecs').when({ count: -1 })).toThrow('count must be a non-negative integer')
    })

    it('throws for an include entry other than the wildcard', () => {
      expect(() => Hide(['toolSpec::billing_search'])).toThrow("must be 'toolSpec::*'")
    })
  })

  describe('message pipeline', () => {
    it('apply() is a no-op', async () => {
      const strategy = Hide('toolSpecs') as HideStrategy
      const agent = createMockAgent()
      const acted = await strategy.apply({ messages: agent.messages, agent, utilization: 0 })
      expect(acted).toBe(false)
    })
  })

  describe('init', () => {
    it('registers an Input handler and both invocation-boundary hooks', () => {
      const strategy = Hide('toolSpecs') as HideStrategy
      const { agent } = attach(strategy)
      const eventTypes = agent.trackedHooks.map((hook) => hook.eventType)
      expect(eventTypes).toEqual([AfterInvocationEvent, BeforeInvocationEvent])
    })
  })

  describe('selection', () => {
    it('keeps the top matches in catalog order', async () => {
      const strategy = Hide('toolSpecs', { keep: 2 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search', 'billing_summary'])
    })

    it('emits in catalog order even when search ranks differently', async () => {
      const search = new StaticSearch(['shipping_track', 'billing_search'])
      const strategy = Hide('toolSpecs', { search, keep: 2 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search', 'shipping_track'])
    })

    it('reuses the selection for later calls in the same invocation', async () => {
      const search: ToolSearchStrategy = { name: 'spy', search: vi.fn(async () => [{ id: 'billing_search' }]) }
      const strategy = Hide('toolSpecs', { search, keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const invocationState: InvocationState = {}

      const first = await handler(context(agent, catalog, { invocationState }))
      const second = await handler(context(agent, catalog, { invocationState, messages: [toolResultOnly()] }))

      expect(names(first.toolSpecs)).toEqual(['billing_search'])
      expect(names(second.toolSpecs)).toEqual(['billing_search'])
      expect(search.search).toHaveBeenCalledTimes(1)
    })

    it('intersects a stored selection with the current catalog', async () => {
      const search = new StaticSearch(['billing_search', 'shipping_track'])
      const strategy = Hide('toolSpecs', { search, keep: 2 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const invocationState: InvocationState = {}

      await handler(context(agent, catalog, { invocationState }))
      const shrunk = catalog.filter((entry) => entry.name !== 'shipping_track')
      const result = await handler(context(agent, shrunk, { invocationState }))

      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })

    it('keeps separate selections for concurrent invocations', async () => {
      const search: ToolSearchStrategy = {
        name: 'by-query',
        search: async (query) => [{ id: query.includes('billing') ? 'billing_search' : 'shipping_search' }],
      }
      const strategy = Hide('toolSpecs', { search, keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)

      const billing = await handler(context(agent, catalog, { invocationState: {} }))
      const shipping = await handler(
        context(agent, catalog, { invocationState: {}, messages: [user('track my shipping order')] })
      )

      expect(names(billing.toolSpecs)).toEqual(['billing_search'])
      expect(names(shipping.toolSpecs)).toEqual(['shipping_search'])
    })

    it('derives the query from the latest user text, skipping tool-result-only turns', async () => {
      const search: ToolSearchStrategy = { name: 'spy', search: vi.fn(async () => [{ id: 'billing_search' }]) }
      const strategy = Hide('toolSpecs', { search }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const messages = [user('first'), user('  track the shipment  '), toolResultOnly()]

      await handler(context(agent, catalog, { messages }))

      expect(search.search).toHaveBeenCalledWith('track the shipment', expect.any(Array), 10)
    })

    it('renders name, description, and input-property descriptions into candidate text', async () => {
      const search: ToolSearchStrategy = { name: 'spy', search: vi.fn(async () => []) }
      const strategy = Hide('toolSpecs', { search }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const withProps = [spec('lookup', 'Find a record', { account_id: { description: 'Customer account id' } })]

      await handler(context(agent, withProps))

      const candidates = (search.search as ReturnType<typeof vi.fn>).mock.calls[0]![1] as { text: string }[]
      expect(candidates[0]!.text).toBe('lookup Find a record account_id Customer account id')
    })
  })

  describe('exclusions', () => {
    it('keeps excluded specs visible without consuming keep', async () => {
      const strategy = Hide(['toolSpec::*', '!toolSpec::ask_user'], { keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search', 'ask_user'])
    })

    it('always keeps the structured-output tool', async () => {
      const strategy = Hide('toolSpecs', { keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const withStructured = [...catalog, spec(STRUCTURED_OUTPUT_TOOL_NAME, 'Return structured output')]
      const result = await handler(context(agent, withStructured))
      expect(names(result.toolSpecs)).toEqual(['billing_search', STRUCTURED_OUTPUT_TOOL_NAME])
    })
  })

  describe('bypass', () => {
    it('passes the catalog through when toolChoice is set', async () => {
      const strategy = Hide('toolSpecs', { keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const input = context(agent, catalog, { toolChoice: { any: {} } })
      const result = await handler(input)
      expect(result).toBe(input)
    })

    it('passes the catalog through when count is not met', async () => {
      const strategy = Hide('toolSpecs', { keep: 1 }).when({ count: 20 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const input = context(agent, catalog)
      const result = await handler(input)
      expect(result).toBe(input)
    })

    it('counts eligible specs, not the whole catalog', async () => {
      const strategy = Hide(['toolSpec::*', '!toolSpec::ask_user'], { keep: 1 }).when({ count: 5 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const input = context(agent, catalog)
      const result = await handler(input)
      expect(result).toBe(input)
    })
  })

  describe('fail-open', () => {
    it('shows every eligible spec when search throws', async () => {
      const search: ToolSearchStrategy = {
        name: 'broken',
        search: async () => {
          throw new Error('boom')
        },
      }
      const strategy = Hide('toolSpecs', { search, keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(names(catalog))
    })

    it('shows every eligible spec when search returns nothing', async () => {
      const strategy = Hide('toolSpecs', { search: new StaticSearch([]), keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(names(catalog))
    })

    it('ignores ids that are not in the catalog', async () => {
      const strategy = Hide('toolSpecs', {
        search: new StaticSearch(['ghost', 'billing_search']),
        keep: 2,
      }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })
  })

  describe('invocation state', () => {
    it('clears the selection on AfterInvocationEvent', async () => {
      const search: ToolSearchStrategy = { name: 'spy', search: vi.fn(async () => [{ id: 'billing_search' }]) }
      const strategy = Hide('toolSpecs', { search, keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const invocationState: InvocationState = {}

      await handler(context(agent, catalog, { invocationState }))
      await invokeTrackedHook(agent, new AfterInvocationEvent({ agent, invocationState }))
      await handler(context(agent, catalog, { invocationState }))

      expect(search.search).toHaveBeenCalledTimes(2)
      expect(Object.keys(invocationState)).toHaveLength(1)
    })

    it('ignores a foreign value under its key', async () => {
      const strategy = Hide('toolSpecs', { search: new StaticSearch(['billing_search']), keep: 1 }) as HideStrategy
      const { agent, handler } = attach(strategy)
      const invocationState: InvocationState = {}

      await handler(context(agent, catalog, { invocationState }))
      const [key] = Object.keys(invocationState)
      invocationState[key!] = 'not a HideState'
      const result = await handler(context(agent, catalog, { invocationState }))

      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })

    it('uses a distinct key per strategy instance', async () => {
      const first = Hide('toolSpecs', { search: new StaticSearch(['billing_search']), keep: 1 }) as HideStrategy
      const second = Hide('toolSpecs', { search: new StaticSearch(['shipping_track']), keep: 1 }) as HideStrategy
      const { agent, handler: firstHandler } = attach(first)
      const { handler: secondHandler } = attach(second)
      const invocationState: InvocationState = {}

      await firstHandler(context(agent, catalog, { invocationState }))
      await secondHandler(context(agent, catalog, { invocationState }))

      expect(Object.keys(invocationState)).toHaveLength(2)
    })
  })
})

describe('LexicalSearch', () => {
  const search = new LexicalSearch()
  const candidates = [
    { id: 'billing_search', text: 'billing_search Search billing records' },
    { id: 'shipping_search', text: 'shipping_search Search shipments by tracking number' },
    { id: 'ask_user', text: 'ask_user Ask the user a question' },
  ]

  it('ranks by overlap and weights name terms higher', async () => {
    const matches = await search.search('search shipping', candidates, 3)
    expect(matches.map((match) => match.id)).toEqual(['shipping_search', 'billing_search'])
  })

  it('breaks ties by candidate order', async () => {
    const matches = await search.search('search', candidates, 3)
    expect(matches.map((match) => match.id)).toEqual(['billing_search', 'shipping_search'])
  })

  it('returns nothing for an empty query', async () => {
    expect(await search.search('   ', candidates, 3)).toEqual([])
  })

  it('respects the limit', async () => {
    const matches = await search.search('search', candidates, 1)
    expect(matches).toHaveLength(1)
  })
})

describe('StaticSearch', () => {
  it('returns its ids in order up to the limit', async () => {
    const search = new StaticSearch(['a', 'b', 'c'])
    const matches = await search.search('ignored', [], 2)
    expect(matches.map((match) => match.id)).toEqual(['a', 'b'])
  })
})
