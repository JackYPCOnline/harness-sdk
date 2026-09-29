/**
 * Tool search — ranks in-memory tool candidates by relevance to a query.
 *
 * This is the relevance contract behind `Hide('toolSpecs')`. It ranks candidates that
 * are never stored, which is why it is a separate contract from the storage package's
 * `SearchStrategy` (that one ranks stored keys for a `Storage`).
 *
 * @experimental
 */

/**
 * One searchable tool.
 *
 * @experimental
 */
export interface ToolSearchCandidate {
  /** The tool's registered name. */
  readonly id: string
  /** Searchable text: name, description, and input-property descriptions. */
  readonly text: string
}

/**
 * One ranked result.
 *
 * @experimental
 */
export interface ToolSearchMatch {
  /** The candidate's id. */
  readonly id: string
  /** Informational score. The contract is result order, best-first; score semantics are strategy-specific. */
  readonly score?: number
}

/**
 * Ranks tool candidates for a query, best-first.
 *
 * @experimental
 */
export interface ToolSearchStrategy {
  /** Stable identifier for logging and observability. */
  readonly name: string
  /**
   * Return up to `limit` matches, best-first. Ids not present in `candidates` are ignored by the caller.
   *
   * @param query - Text derived from the conversation, or an explicit search string
   * @param candidates - The eligible tools for this decision
   * @param limit - Maximum number of matches to return
   */
  search(query: string, candidates: readonly ToolSearchCandidate[], limit: number): Promise<readonly ToolSearchMatch[]>
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
 * Deterministic in-process ranking by token overlap between the query and each candidate.
 * Terms that appear in the tool name count more than terms that appear only in the description.
 * Ties break by candidate order, so the same input always yields the same ranking.
 *
 * @experimental
 */
export class LexicalSearch implements ToolSearchStrategy {
  readonly name = 'toolSearch:lexical'

  async search(
    query: string,
    candidates: readonly ToolSearchCandidate[],
    limit: number
  ): Promise<readonly ToolSearchMatch[]> {
    const queryTerms = tokenize(query)
    if (queryTerms.size === 0) return []

    const scored: { index: number; id: string; score: number }[] = []
    for (const [index, candidate] of candidates.entries()) {
      const nameTerms = tokenize(candidate.id)
      const textTerms = tokenize(candidate.text)
      let score = 0
      for (const term of queryTerms) {
        if (nameTerms.has(term)) score += NAME_TERM_WEIGHT
        else if (textTerms.has(term)) score += 1
      }
      if (score > 0) scored.push({ index, id: candidate.id, score })
    }

    scored.sort((left, right) => right.score - left.score || left.index - right.index)
    return scored.slice(0, limit).map(({ id, score }) => ({ id, score }))
  }
}

/**
 * Returns a fixed list of ids regardless of query. A testing seam and a way to pin selection.
 *
 * @experimental
 */
export class StaticSearch implements ToolSearchStrategy {
  readonly name = 'toolSearch:static'
  private readonly _ids: readonly string[]

  /**
   * @param ids - Ids to return, in this order
   */
  constructor(ids: readonly string[]) {
    this._ids = [...ids]
  }

  async search(
    _query: string,
    _candidates: readonly ToolSearchCandidate[],
    limit: number
  ): Promise<readonly ToolSearchMatch[]> {
    return this._ids.slice(0, limit).map((id) => ({ id }))
  }
}
