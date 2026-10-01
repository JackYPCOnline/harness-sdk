/**
 * Hide strategies — filter what the model sees on each call without touching durable state.
 *
 * Not exported from the package barrel while the API is experimental.
 *
 * @internal
 */

import type { HideStrategyBuilder } from './base.js'
import type { HideToolSpecsConfig, HideToolSpecsTarget } from './tool-specs.js'
import { HideToolSpecsStrategy } from './tool-specs.js'

export type { HideConditions, HideStrategyBuilder } from './base.js'
export type { HideToolSpecsConfig, HideToolSpecsTarget } from './tool-specs.js'
export { HideToolSpecsStrategy, searchableText } from './tool-specs.js'

/**
 * Hide strategy builder namespace.
 *
 * - `Hide.toolSpecs(target, config)` — show the model only the tool specs relevant to the current turn
 */
interface HideNamespace {
  /** Show the model only the tool specs relevant to the current turn. */
  toolSpecs(target?: HideToolSpecsTarget, config?: HideToolSpecsConfig): HideStrategyBuilder
}

/**
 * Builder for hide strategies — filters the per-call model input.
 *
 * @example
 * ```typescript
 * // Keep the 10 most relevant specs once the catalog has 20 or more
 * Hide.toolSpecs().when({ count: 20 })
 * // Keep ask_user and finish visible without consuming keep
 * Hide.toolSpecs(['toolSpec::*', '!toolSpec::ask_user', '!toolSpec::finish'], { keep: 15 }).when({ count: 20 })
 * // Rank with a different storage search strategy
 * Hide.toolSpecs('toolSpecs', { search: mySearchStrategy })
 * ```
 */
export const Hide: HideNamespace = {
  toolSpecs(target: HideToolSpecsTarget = 'toolSpecs', config?: HideToolSpecsConfig): HideStrategyBuilder {
    return new HideToolSpecsStrategy(target, config)
  },
}
