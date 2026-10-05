/**
 * The `Hide` namespace: the entry point users type to add a hide strategy.
 *
 * @internal
 */

import { HideToolSpecsStrategy } from './tool-specs.js'
import type { HideStrategyBuilder } from './base.js'
import type { HideToolSpecsConfig, HideToolSpecsTarget } from './tool-specs.js'

/**
 * Hide strategy builder namespace.
 *
 * - `Hide.toolSpecs(target, config)` — show the model only the tool specs relevant to the current turn
 */
interface HideNamespace {
  /** Show the model only the tool specs relevant to the current turn, over every spec. */
  toolSpecs(config?: HideToolSpecsConfig): HideStrategyBuilder
  /** Show the model only the tool specs relevant to the current turn, over the targeted specs. */
  toolSpecs(target: HideToolSpecsTarget, config?: HideToolSpecsConfig): HideStrategyBuilder
}

/**
 * Builder for hide strategies — filters the per-call model input.
 *
 * @example
 * ```typescript
 * // Keep the 10 most relevant specs once the catalog has 20 or more
 * Hide.toolSpecs().when({ count: 20 })
 * // Same, keeping 5
 * Hide.toolSpecs({ keep: 5 }).when({ count: 20 })
 * // Pin ask_user and finish: always visible, outside keep and count (count sees 18 of 20 tools here)
 * Hide.toolSpecs(['toolSpec::*', '!toolSpec::ask_user', '!toolSpec::finish'], { keep: 15 }).when({ count: 18 })
 * // Only the billing tools are candidates; everything else stays visible
 * Hide.toolSpecs(['toolSpec::billing_search', 'toolSpec::billing_summary'], { keep: 1 })
 * // Never show debug_dump, and show only pinned tools if search fails
 * Hide.toolSpecs({ alwaysHide: ['debug_dump'], onFailure: 'none' })
 * // Rank with a custom ToolSearchStrategy (an LLM judge, embeddings, ...)
 * Hide.toolSpecs({ search: myToolSearch })
 * ```
 */
export const Hide: HideNamespace = {
  toolSpecs(
    targetOrConfig: HideToolSpecsTarget | HideToolSpecsConfig = 'toolSpecs',
    config?: HideToolSpecsConfig
  ): HideStrategyBuilder {
    // The two overloads are told apart at runtime so an untyped caller cannot lose the config:
    // a target is the string 'toolSpecs' or an array; a config is a plain object.
    if (typeof targetOrConfig === 'object' && !Array.isArray(targetOrConfig)) {
      return new HideToolSpecsStrategy('toolSpecs', targetOrConfig)
    }
    return new HideToolSpecsStrategy(targetOrConfig, config)
  },
}
