import type { Storage } from '../storage/storage.js'
import type { SearchStrategy } from '../storage/search/types.js'

/**
 * A search strategy that returns fixed keys best-first, ignoring the query and the storage.
 *
 * @param keys - Keys to return, in rank order
 * @returns A SearchStrategy usable with any storage backend
 */
export function createStaticSearch<S extends Storage = Storage>(keys: string[]): SearchStrategy<S> {
  return { search: async () => keys.map((key, index) => ({ key, score: keys.length - index })) }
}
