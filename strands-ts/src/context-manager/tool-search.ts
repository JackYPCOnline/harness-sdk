/**
 * Tool search — ranks tool specs by relevance to a query.
 *
 * This is the relevance contract behind `Hide('toolSpecs')`. It ranks specs that are
 * already in memory, which is why it is a separate contract from the storage package's
 * `SearchStrategy` (that one ranks stored keys for a `Storage`).
 *
 * Not exported from the package barrel while the API is experimental.
 *
 * @internal
 */

import type { ToolSpec } from '../tools/types.js'

/**
 * One ranked result.
 *
 * @internal
 */
export interface ToolSearchMatch {
  /** The matched tool's name. */
  readonly name: string
  /** Informational score. The contract is result order, best-first; score semantics are strategy-specific. */
  readonly score?: number
}

/**
 * Ranks tool specs for a query, best-first.
 *
 * @internal
 */
export interface ToolSearchStrategy {
  /** Stable identifier for logging and observability. */
  readonly name: string
  /**
   * Return up to `limit` matches, best-first. Names not present in `specs` are ignored by the caller.
   *
   * @param query - Text derived from the conversation, or an explicit search string
   * @param specs - The eligible tool specs for this decision
   * @param limit - Maximum number of matches to return
   */
  search(query: string, specs: readonly ToolSpec[], limit: number): Promise<readonly ToolSearchMatch[]>
}

const NAME_TERM_WEIGHT = 3

/**
 * Tokenizes text into lowercase alphanumeric terms.
 *
 * @param text - Text to tokenize
 * @returns Set of unique terms
 * @internal
 */
export function tokenize(text: string): Set<string> {
  const terms = text.toLowerCase().match(/[a-z0-9]+/g) ?? []
  return new Set(terms)
}

/**
 * The searchable text of a spec: name, description, and input-property names and descriptions.
 *
 * @param spec - The tool spec
 * @returns Space-joined searchable text
 * @internal
 */
export function searchableText(spec: ToolSpec): string {
  const properties = spec.inputSchema?.properties
  const propertyText =
    properties && typeof properties === 'object'
      ? Object.entries(properties)
          .map(([key, value]) => {
            const description =
              value && typeof value === 'object' && 'description' in value ? String(value.description) : ''
            return `${key} ${description}`
          })
          .join(' ')
      : ''
  return `${spec.name} ${spec.description} ${propertyText}`
}

/**
 * Deterministic in-process ranking by token overlap between the query and each spec.
 * Terms that appear in the tool name count more than terms that appear only in the description.
 * Ties break by spec order, so the same input always yields the same ranking.
 *
 * @internal
 */
export class LexicalSearch implements ToolSearchStrategy {
  readonly name = 'toolSearch:lexical'

  async search(query: string, specs: readonly ToolSpec[], limit: number): Promise<readonly ToolSearchMatch[]> {
    const queryTerms = tokenize(query)
    if (queryTerms.size === 0) return []

    const scored: { index: number; name: string; score: number }[] = []
    for (const [index, spec] of specs.entries()) {
      const nameTerms = tokenize(spec.name)
      const textTerms = tokenize(searchableText(spec))
      let score = 0
      for (const term of queryTerms) {
        if (nameTerms.has(term)) score += NAME_TERM_WEIGHT
        else if (textTerms.has(term)) score += 1
      }
      if (score > 0) scored.push({ index, name: spec.name, score })
    }

    scored.sort((left, right) => right.score - left.score || left.index - right.index)
    return scored.slice(0, limit).map(({ name, score }) => ({ name, score }))
  }
}

/**
 * Returns a fixed list of names regardless of query. A testing seam and a way to pin selection.
 *
 * @internal
 */
export class StaticSearch implements ToolSearchStrategy {
  readonly name = 'toolSearch:static'
  private readonly _names: readonly string[]

  /**
   * @param names - Tool names to return, in this order
   */
  constructor(names: readonly string[]) {
    this._names = [...names]
  }

  async search(_query: string, _specs: readonly ToolSpec[], limit: number): Promise<readonly ToolSearchMatch[]> {
    return this._names.slice(0, limit).map((name) => ({ name }))
  }
}
