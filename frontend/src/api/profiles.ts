import type { MetricProfile } from './types'

/**
 * Profiles the GUI offers for selection: everything whose numbers have been checked beyond the implementation
 * itself. A profile still pending cross-validation is computed and stored by the service, listed by the CLI and
 * the API, and stays out of the GUI's choices until the protocol record says it agrees with an independent backend.
 */
const isOffered = (profile: MetricProfile): boolean => profile.validation !== 'pending_cross_validation'

export const offeredProfiles = (profiles: readonly MetricProfile[] | undefined): MetricProfile[] => (profiles ?? []).filter(isOffered)
