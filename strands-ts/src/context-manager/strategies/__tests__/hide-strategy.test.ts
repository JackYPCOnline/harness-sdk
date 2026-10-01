import { describe, it, expect, vi } from 'vitest'
import { Hide, HideToolSpecsStrategy, searchableText } from '../hide/index.js'
import { InvokeModelStage } from '../../../middleware/stages.js'
import { AfterInvocationEvent, BeforeInvocationEvent } from '../../../hooks/events.js'
import { InMemoryStorage } from '../../../storage/in-memory-storage.js'
import { STRUCTURED_OUTPUT_TOOL_NAME } from '../../../tools/structured-output-tool.js'
import { RETRIEVAL_TOOL_NAME } from '../../retrieval-tool.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import { createMockAgent, invokeTrackedHook } from '../../../__fixtures__/agent-helpers.js'
import type { MockAgent } from '../../../__fixtures__/agent-helpers.js'
import type { InvokeModelContext } from '../../../middleware/stages.js'
import type { SearchStrategy } from '../../../storage/search/index.js'
import type { ToolSpec } from '../../../tools/types.js'
import type { InvocationState } from '../../../types/agent.js'

type InputHandler = (context: InvokeModelContext) => InvokeModelContext | Promise<InvokeModelContext>

/** Attach a strategy to a mock agent and capture the Input handler it registers. */
function attach(strategy: HideToolSpecsStrategy): { agent: MockAgent; handler: InputHandler } {
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

/** A search strategy that returns fixed keys best-first, ignoring the query and the index. */
function staticSearch(keys: string[]): SearchStrategy<InMemoryStorage> {
  return { search: async () => keys.map((key, index) => ({ key, score: keys.length - index })) }
}

/** A search strategy whose `search` is a spy returning fixed keys. */
function spySearch(keys: string[]): SearchStrategy<InMemoryStorage> & { search: ReturnType<typeof vi.fn> } {
  return { search: vi.fn(async () => keys.map((key, index) => ({ key, score: keys.length - index }))) }
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

function assistantWithUsage(): Message {
  return new Message({
    role: 'assistant',
    content: [new TextBlock('working')],
    metadata: { usage: { inputTokens: 500, outputTokens: 10, totalTokens: 510 } },
  })
}

/** A model whose countTokens charges a flat rate per tool spec. */
function countingModel(tokensPerSpec = 25): InvokeModelContext['model'] {
  return {
    countTokens: vi.fn(async (_messages: Message[], options?: { toolSpecs?: readonly ToolSpec[] }) => {
      return (options?.toolSpecs?.length ?? 0) * tokensPerSpec
    }),
  } as unknown as InvokeModelContext['model']
}

function context(
  agent: MockAgent,
  toolSpecs: ToolSpec[],
  overrides?: Partial<Pick<InvokeModelContext, 'messages' | 'toolChoice' | 'invocationState' | 'model'>>
): InvokeModelContext {
  return {
    agent,
    model: overrides?.model ?? countingModel(),
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

const toolSpecs = (...args: Parameters<typeof Hide.toolSpecs>): HideToolSpecsStrategy =>
  Hide.toolSpecs(...args) as HideToolSpecsStrategy

describe('Hide.toolSpecs', () => {
  describe('construction', () => {
    it('defaults the target to every tool spec', () => {
      const strategy = Hide.toolSpecs()
      expect(strategy.name).toBe('hide:toolSpecs')
      expect(typeof strategy.apply).toBe('function')
    })

    it('when() returns a new strategy and leaves the original unchanged', () => {
      const base = Hide.toolSpecs('toolSpecs', { keep: 2 })
      const gated = base.when({ count: 20 })
      expect(gated).not.toBe(base)
      expect(gated.name).toBe('hide:toolSpecs')
    })

    it('when() carries the full config across', async () => {
      const search = staticSearch(['shipping_track'])
      const config = { search, keep: 1, alwaysHide: ['billing_search'], onFailure: 'none' as const }
      const gated = toolSpecs(['toolSpec::*', '!toolSpec::ask_user'], config).when({ count: 0 })
      const { agent, handler } = attach(gated as HideToolSpecsStrategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['shipping_track', 'ask_user'])
    })

    it('throws for an empty array target', () => {
      expect(() => Hide.toolSpecs([])).toThrow('Empty array target')
    })

    it('throws for a non-positive keep', () => {
      expect(() => Hide.toolSpecs('toolSpecs', { keep: 0 })).toThrow('keep must be a positive integer')
    })

    it('throws for a negative count', () => {
      expect(() => Hide.toolSpecs().when({ count: -1 })).toThrow('count must be a non-negative integer')
    })

    it('throws for an entry without the toolSpec prefix', () => {
      expect(() => Hide.toolSpecs(['tool::billing_search'])).toThrow("must be 'toolSpec::<name>'")
      expect(() => Hide.toolSpecs(['!billing_search'])).toThrow("must be 'toolSpec::<name>'")
    })
  })

  describe('message pipeline', () => {
    it('apply() is a no-op', async () => {
      const agent = createMockAgent()
      const acted = await toolSpecs().apply({ messages: agent.messages, agent, utilization: 0 })
      expect(acted).toBe(false)
    })
  })

  describe('init', () => {
    it('registers an Input handler and both invocation-boundary hooks', () => {
      const { agent } = attach(toolSpecs())
      const eventTypes = agent.trackedHooks.map((hook) => hook.eventType)
      expect(eventTypes).toEqual([AfterInvocationEvent, BeforeInvocationEvent])
    })
  })

  describe('selection', () => {
    it('keeps the top keyword matches in catalog order by default', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { keep: 2 }))
      const result = await handler(context(agent, catalog, { messages: [user('summarize billing records')] }))
      expect(names(result.toolSpecs)).toEqual(['billing_search', 'billing_summary'])
    })

    it('emits in catalog order even when search ranks differently', async () => {
      const search = staticSearch(['shipping_track', 'billing_search'])
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 2 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search', 'shipping_track'])
    })

    it('reuses the selection for later calls in the same invocation', async () => {
      const search = spySearch(['billing_search'])
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const invocationState: InvocationState = {}
      const first = await handler(context(agent, catalog, { invocationState }))
      const second = await handler(context(agent, catalog, { invocationState, messages: [toolResultOnly()] }))
      expect(names(first.toolSpecs)).toEqual(['billing_search'])
      expect(names(second.toolSpecs)).toEqual(['billing_search'])
      expect(search.search).toHaveBeenCalledTimes(1)
    })

    it('intersects a stored selection with the current catalog', async () => {
      const search = staticSearch(['billing_search', 'shipping_track'])
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 2 }))
      const invocationState: InvocationState = {}
      await handler(context(agent, catalog, { invocationState }))
      const shrunk = catalog.filter((entry) => entry.name !== 'shipping_track')
      const result = await handler(context(agent, shrunk, { invocationState }))
      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })

    it('keeps separate selections for concurrent invocations', async () => {
      const search: SearchStrategy<InMemoryStorage> = {
        search: async (_storage, query) => [
          { key: query.includes('billing') ? 'billing_search' : 'shipping_search', score: 1 },
        ],
      }
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const billing = await handler(context(agent, catalog, { invocationState: {} }))
      const shipping = await handler(
        context(agent, catalog, { invocationState: {}, messages: [user('track my shipping order')] })
      )
      expect(names(billing.toolSpecs)).toEqual(['billing_search'])
      expect(names(shipping.toolSpecs)).toEqual(['shipping_search'])
    })

    it('derives the query from the latest user text, skipping tool-result-only turns', async () => {
      const search = spySearch(['billing_search'])
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search }))
      const messages = [user('first'), user('  track the shipment  '), toolResultOnly()]
      await handler(context(agent, catalog, { messages }))
      expect(search.search).toHaveBeenCalledWith(expect.any(InMemoryStorage), 'track the shipment')
    })

    it('indexes only the eligible specs, keyed by tool name', async () => {
      let indexed: string[] = []
      let billingText = ''
      const search: SearchStrategy<InMemoryStorage> = {
        search: async (storage) => {
          indexed = await storage.list('')
          billingText = new TextDecoder().decode((await storage.read('billing_search')) ?? new Uint8Array())
          return []
        },
      }
      const { agent, handler } = attach(toolSpecs(['toolSpec::*', '!toolSpec::ask_user'], { search }))
      await handler(context(agent, catalog))
      expect(indexed).toEqual(['billing_search', 'billing_summary', 'shipping_search', 'shipping_track'])
      expect(billingText).toBe(searchableText(catalog[0]!))
    })
  })

  describe('target', () => {
    it('keeps pinned specs visible without consuming keep', async () => {
      const { agent, handler } = attach(toolSpecs(['toolSpec::*', '!toolSpec::ask_user'], { keep: 1 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search', 'ask_user'])
    })

    it('treats a pin-only target as every spec', async () => {
      const { agent, handler } = attach(toolSpecs(['!toolSpec::ask_user'], { keep: 1 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search', 'ask_user'])
    })

    it('narrows the candidates to named specs and leaves the rest visible', async () => {
      const search = staticSearch(['billing_summary'])
      const target = ['toolSpec::billing_search', 'toolSpec::billing_summary']
      const { agent, handler } = attach(toolSpecs(target, { search, keep: 1 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_summary', 'shipping_search', 'shipping_track', 'ask_user'])
    })

    it('lets the wildcard override named candidates', async () => {
      const search = staticSearch(['shipping_track'])
      const { agent, handler } = attach(toolSpecs(['toolSpec::billing_search', 'toolSpec::*'], { search, keep: 1 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['shipping_track'])
    })

    it('always keeps the structured-output and retrieval tools', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { keep: 1 }))
      const withProtected = [
        ...catalog,
        spec(STRUCTURED_OUTPUT_TOOL_NAME, 'Return structured output'),
        spec(RETRIEVAL_TOOL_NAME, 'Retrieve offloaded content'),
      ]
      const result = await handler(context(agent, withProtected))
      expect(names(result.toolSpecs)).toEqual(['billing_search', STRUCTURED_OUTPUT_TOOL_NAME, RETRIEVAL_TOOL_NAME])
    })
  })

  describe('alwaysHide', () => {
    it('removes the named specs before selection', async () => {
      const search = staticSearch(['billing_search', 'ask_user'])
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 2, alwaysHide: ['billing_search'] }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['ask_user'])
    })

    it('wins over a pin', async () => {
      const target = ['toolSpec::*', '!toolSpec::ask_user']
      const { agent, handler } = attach(toolSpecs(target, { keep: 1, alwaysHide: ['ask_user'] }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })

    it('applies even when count is not met', async () => {
      const strategy = toolSpecs('toolSpecs', { keep: 1, alwaysHide: ['ask_user'] }).when({ count: 20 })
      const { agent, handler } = attach(strategy as HideToolSpecsStrategy)
      const model = countingModel(25)
      const result = await handler({ ...context(agent, catalog, { model }), projectedInputTokens: 1000 })
      expect(names(result.toolSpecs)).toEqual([
        'billing_search',
        'billing_summary',
        'shipping_search',
        'shipping_track',
      ])
      expect(result.projectedInputTokens).toBe(975)
    })

    it('does not count toward the count gate', async () => {
      const strategy = toolSpecs('toolSpecs', { keep: 1, alwaysHide: ['ask_user'] }).when({ count: 5 })
      const { agent, handler } = attach(strategy as HideToolSpecsStrategy)
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual([
        'billing_search',
        'billing_summary',
        'shipping_search',
        'shipping_track',
      ])
    })
  })

  describe('onFailure', () => {
    const broken: SearchStrategy<InMemoryStorage> = {
      search: async () => {
        throw new Error('boom')
      },
    }

    it('"none" shows only pinned and protected tools when search throws', async () => {
      const target = ['toolSpec::*', '!toolSpec::ask_user']
      const { agent, handler } = attach(toolSpecs(target, { search: broken, onFailure: 'none' }))
      const withProtected = [...catalog, spec(RETRIEVAL_TOOL_NAME, 'Retrieve offloaded content')]
      const result = await handler(context(agent, withProtected))
      expect(names(result.toolSpecs)).toEqual(['ask_user', RETRIEVAL_TOOL_NAME])
    })

    it('"none" shows only pinned tools when nothing matches and nothing can be carried', async () => {
      const target = ['toolSpec::*', '!toolSpec::ask_user']
      const { agent, handler } = attach(toolSpecs(target, { search: staticSearch([]), onFailure: 'none' }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['ask_user'])
    })

    it('"none" still carries a previous selection forward', async () => {
      const search: SearchStrategy<InMemoryStorage> = {
        search: async (_storage, query) => (query === 'billing' ? [{ key: 'billing_search', score: 1 }] : []),
      }
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1, onFailure: 'none' }))
      await handler(context(agent, catalog, { invocationState: {}, messages: [user('billing')] }))
      const result = await handler(context(agent, catalog, { invocationState: {}, messages: [user('thanks')] }))
      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })
  })

  describe('bypass', () => {
    it('passes the catalog through when toolChoice is set', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { keep: 1 }))
      const input = context(agent, catalog, { toolChoice: { any: {} } })
      const result = await handler(input)
      expect(result).toBe(input)
    })

    it('passes the catalog through when count is not met', async () => {
      const { agent, handler } = attach(
        toolSpecs('toolSpecs', { keep: 1 }).when({ count: 20 }) as HideToolSpecsStrategy
      )
      const input = context(agent, catalog)
      const result = await handler(input)
      expect(result).toBe(input)
    })

    it('counts eligible specs, not the whole catalog', async () => {
      const strategy = toolSpecs(['toolSpec::*', '!toolSpec::ask_user'], { keep: 1 }).when({ count: 5 })
      const { agent, handler } = attach(strategy as HideToolSpecsStrategy)
      const input = context(agent, catalog)
      const result = await handler(input)
      expect(result).toBe(input)
    })
  })

  describe('no matches', () => {
    it('carries the previous invocation selection forward', async () => {
      const search: SearchStrategy<InMemoryStorage> = {
        search: async (_storage, query) => (query === 'billing' ? [{ key: 'billing_search', score: 1 }] : []),
      }
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      await handler(context(agent, catalog, { invocationState: {}, messages: [user('billing')] }))
      const result = await handler(context(agent, catalog, { invocationState: {}, messages: [user('thanks')] }))
      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })

    it('shows every spec when there is nothing to carry forward', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search: staticSearch([]), keep: 1 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(names(catalog))
    })

    it('shows every spec when the carried selection is no longer in the catalog', async () => {
      const search: SearchStrategy<InMemoryStorage> = {
        search: async (_storage, query) => (query === 'billing' ? [{ key: 'billing_search', score: 1 }] : []),
      }
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      await handler(context(agent, catalog, { invocationState: {}, messages: [user('billing')] }))
      const shrunk = catalog.filter((entry) => entry.name !== 'billing_search')
      const result = await handler(context(agent, shrunk, { invocationState: {}, messages: [user('thanks')] }))
      expect(names(result.toolSpecs)).toEqual(names(shrunk))
    })
  })

  describe('fail-open', () => {
    it('shows every eligible spec when search throws', async () => {
      const search: SearchStrategy<InMemoryStorage> = {
        search: async () => {
          throw new Error('boom')
        },
      }
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(names(catalog))
    })

    it('ignores keys that are not in the catalog', async () => {
      const search = staticSearch(['ghost', 'billing_search'])
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 2 }))
      const result = await handler(context(agent, catalog))
      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })
  })

  describe('token projection', () => {
    const search = staticSearch(['billing_search'])

    it('subtracts the removed specs on a cold start', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const model = countingModel(25)
      const result = await handler({ ...context(agent, catalog, { model }), projectedInputTokens: 1000 })
      expect(result.projectedInputTokens).toBe(900)
      expect(model.countTokens).toHaveBeenCalledWith([], {
        toolSpecs: catalog.filter((entry) => entry.name !== 'billing_search'),
      })
    })

    it('keeps the projection once an assistant message carries usage', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const model = countingModel(25)
      const messages = [user('search the billing records'), assistantWithUsage(), toolResultOnly()]
      const result = await handler({ ...context(agent, catalog, { model, messages }), projectedInputTokens: 1000 })
      expect(result.projectedInputTokens).toBe(1000)
      expect(model.countTokens).not.toHaveBeenCalled()
    })

    it('leaves the projection alone when nothing was hidden', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search: staticSearch([]), keep: 1 }))
      const model = countingModel()
      const result = await handler({ ...context(agent, catalog, { model }), projectedInputTokens: 1000 })
      expect(result.projectedInputTokens).toBe(1000)
      expect(model.countTokens).not.toHaveBeenCalled()
    })

    it('does not add a projection when the loop supplied none', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const result = await handler(context(agent, catalog))
      expect('projectedInputTokens' in result).toBe(false)
    })

    it('keeps the original projection when the recount throws', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const model = {
        countTokens: vi.fn(async () => {
          throw new Error('count failed')
        }),
      } as unknown as InvokeModelContext['model']
      const result = await handler({ ...context(agent, catalog, { model }), projectedInputTokens: 1000 })
      expect(result.projectedInputTokens).toBe(1000)
    })
  })

  describe('invocation state', () => {
    it('clears the selection on AfterInvocationEvent', async () => {
      const search = spySearch(['billing_search'])
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search, keep: 1 }))
      const invocationState: InvocationState = {}
      await handler(context(agent, catalog, { invocationState }))
      await invokeTrackedHook(agent, new AfterInvocationEvent({ agent, invocationState }))
      await handler(context(agent, catalog, { invocationState }))
      expect(search.search).toHaveBeenCalledTimes(2)
      expect(Object.keys(invocationState)).toHaveLength(1)
    })

    it('ignores a foreign value under its key', async () => {
      const { agent, handler } = attach(toolSpecs('toolSpecs', { search: staticSearch(['billing_search']), keep: 1 }))
      const invocationState: InvocationState = {}
      await handler(context(agent, catalog, { invocationState }))
      const [key] = Object.keys(invocationState)
      invocationState[key!] = 'not a selection'
      const result = await handler(context(agent, catalog, { invocationState }))
      expect(names(result.toolSpecs)).toEqual(['billing_search'])
    })

    it('uses a distinct key per strategy instance', async () => {
      const first = toolSpecs('toolSpecs', { search: staticSearch(['billing_search']), keep: 1 })
      const second = toolSpecs('toolSpecs', { search: staticSearch(['shipping_track']), keep: 1 })
      const { agent, handler: firstHandler } = attach(first)
      const { handler: secondHandler } = attach(second)
      const invocationState: InvocationState = {}
      await firstHandler(context(agent, catalog, { invocationState }))
      await secondHandler(context(agent, catalog, { invocationState }))
      expect(Object.keys(invocationState)).toHaveLength(2)
    })
  })
})

describe('searchableText', () => {
  it('joins name, description, and input-property names and descriptions', () => {
    const withProps = spec('lookup', 'Find a record', { account_id: { description: 'Customer account id' } })
    expect(searchableText(withProps)).toBe('lookup Find a record account_id Customer account id')
  })
})
