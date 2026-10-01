/**
 * Tool search — ranks in-memory tool specs by relevance to a query.
 *
 * Tool specs are a per-call projection of the registry, never stored, so this is a separate
 * contract from the storage package's `SearchStrategy`, which ranks keys held by a `Storage`.
 * The result shape mirrors `StorageSearchResult` with `name` in place of `key`.
 *
 * @internal
 */

import { STOP_WORDS, tokenOverlapScore, tokenize } from '../../../storage/search/keyword.js'
import type { ToolSpec } from '../../../tools/types.js'

/**
 * A ranked match from a {@link ToolSearchStrategy}.
 *
 * @internal
 */
export interface ToolSearchResult {
  /** Name of the matched tool spec. */
  name: string
  /** Relevance score; higher is more relevant. */
  score: number
}

/**
 * Ranks tool specs by relevance to a query. Implementations may be lexical, an LLM judge, or an
 * adapter over a persistent index; `Hide` keeps the first `limit` results that name a candidate.
 *
 * @internal
 */
export interface ToolSearchStrategy {
  /**
   * Ranks `candidates` against `query`.
   *
   * @param query - The latest user text
   * @param candidates - The specs eligible for selection, in catalog order
   * @param limit - How many results the caller will use
   * @returns Matches ranked best-first, at most `limit`
   */
  search(query: string, candidates: readonly ToolSpec[], limit: number): Promise<ToolSearchResult[]>
}

const NAME_WEIGHT = 3

/**
 * Keyword tool search: distinct content words of the query that appear in a spec, with name hits
 * weighted over description and input-property hits. Names are split on `_ - . : /` and camelCase
 * so `get_weather` matches "weather". Ties keep candidate order. No external dependencies.
 *
 * @internal
 */
export const KeywordToolSearch: ToolSearchStrategy = {
  async search(query: string, candidates: readonly ToolSpec[], limit: number): Promise<ToolSearchResult[]> {
    const queryTokens = contentTokens(query)
    if (queryTokens.size === 0) return []

    const scored: ToolSearchResult[] = []
    for (const spec of candidates) {
      const nameScore = tokenOverlapScore(queryTokens, splitIdentifier(spec.name))
      const bodyScore = tokenOverlapScore(queryTokens, bodyText(spec))
      const score = nameScore * NAME_WEIGHT + bodyScore
      if (score > 0) scored.push({ name: spec.name, score })
    }
    scored.sort((left, right) => right.score - left.score)
    return scored.slice(0, limit)
  },
}

/**
 * The query's content words: tokens that are not stop words or single characters.
 *
 * @param query - Raw query text
 * @returns Lowercased content tokens
 * @internal
 */
export function contentTokens(query: string): Set<string> {
  const tokens = new Set<string>()
  for (const token of tokenize(query)) {
    if (token.length > 1 && !STOP_WORDS.has(token)) tokens.add(token)
  }
  return tokens
}

/** Breaks an identifier into words: `get_weather`, `get-weather`, `getWeather`, `mcp::get.weather`. */
function splitIdentifier(identifier: string): string {
  return identifier.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_\-.:/]+/g, ' ')
}

/** Description plus input-property names and descriptions. */
function bodyText(spec: ToolSpec): string {
  const properties = spec.inputSchema?.properties
  if (!properties || typeof properties !== 'object') return spec.description
  const propertyText = Object.entries(properties)
    .map(([key, value]) => {
      const description = value && typeof value === 'object' && 'description' in value ? String(value.description) : ''
      return `${splitIdentifier(key)} ${description}`
    })
    .join(' ')
  return `${spec.description} ${propertyText}`
}
