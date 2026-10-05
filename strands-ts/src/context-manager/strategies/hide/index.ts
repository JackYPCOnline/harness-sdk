/**
 * Hide strategies — filter what the model sees on each call without touching durable state.
 *
 * Not exported from the package barrel while the API is experimental.
 *
 * @internal
 */

export { Hide } from './hide.js'
export type { HideConditions, HideStrategyBuilder } from './base.js'
export type { HideFailurePolicy, HideToolSpecsConfig, HideToolSpecsTarget } from './tool-specs.js'
export type { ToolSearchOptions, ToolSearchResult, ToolSearchStrategy } from './tool-search.js'
export { KeywordToolSearch } from './tool-search.js'
