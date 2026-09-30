import type { ArenaBudget, ArenaRow } from '@/api/arena'

export interface ArenaPoint {
  id: string
  row: ArenaRow
  budget: ArenaBudget
  parameters: number
  ops: number
  evm: number
  aclr: number
  evmStd: number
  aclrStd: number
  eligible: boolean
}

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length
const std = (values: number[]) => values.length < 2 ? 0 : Math.sqrt(values.reduce((sum, n) => sum + (n - mean(values)) ** 2, 0) / (values.length - 1))
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
export const isArenaExcluded = (backbone: string): boolean => backbone === 'ilc_dpd'

/** Points always use actual configuration costs; test seeds are averaged, never selected. */
export function configurationPoints(rows: ArenaRow[], protocol: string, condition?: string): ArenaPoint[] {
  return rows.flatMap(row => {
    if (isArenaExcluded(row.backbone) || row.status !== 'succeeded' || row.protocol_sha256 !== protocol) return []
    return (row.budgets ?? []).flatMap(budget => {
      if (!budget.available || !finite(budget.parameters) || budget.parameters <= 0 || !finite(budget.ops) || budget.ops <= 0) return []
      const samples: { evm: number; aclr: number; seed: number }[] = []
      for (const item of row.cases ?? []) {
        if (item.budget !== budget.budget || (condition && item.condition_id !== condition) || !Array.isArray(item.judges)) continue
        for (const observation of item.judges) {
          if (!observation || typeof observation !== 'object' || Array.isArray(observation)) continue
          if (finite(observation.evm_db) && finite(observation.aclr_l_db) && finite(observation.aclr_r_db) && finite(item.seed))
            samples.push({ evm: observation.evm_db, aclr: Math.max(observation.aclr_l_db, observation.aclr_r_db), seed: item.seed })
        }
      }
      // Per-seed means keep multi-condition boards from inventing seed variability.
      const seeds = [...new Set(samples.map(s => s.seed))]
      const evms = seeds.map(seed => mean(samples.filter(s => s.seed === seed).map(s => s.evm)))
      const aclrs = seeds.map(seed => mean(samples.filter(s => s.seed === seed).map(s => s.aclr)))
      const evm = evms.length ? mean(evms) : condition ? null : budget.metrics?.evm_db
      const aclr = aclrs.length ? mean(aclrs) : condition ? null : budget.metrics?.aclr_db
      if (!finite(evm) || !finite(aclr)) return []
      return [{ id: `${row.entry_id}:${budget.budget}`, row, budget, parameters: budget.parameters, ops: budget.ops,
        evm, aclr, evmStd: std(evms), aclrStd: std(aclrs),
        eligible: !!row.eligible && !!budget.qualified && finite(budget.quality_db) && budget.quality_db > 0 }]
    })
  })
}

/** Both coordinates are minimized. Equal points are nondominated; ties in one coordinate are not. */
export function paretoFront(points: ArenaPoint[], cost: 'parameters' | 'ops', metric: 'evm' | 'aclr'): ArenaPoint[] {
  const valid = points.filter(p => p.eligible)
  return valid.filter(p => !valid.some(q => q[cost] <= p[cost] && q[metric] <= p[metric] && (q[cost] < p[cost] || q[metric] < p[metric])))
    .sort((a, b) => a[cost] - b[cost] || a[metric] - b[metric] || a.id.localeCompare(b.id))
}

export function configurationScore(point: ArenaPoint, ranking: string): number | null {
  if (!point.eligible) return null
  const cap = /^budget-(\d+)$/.exec(ranking)
  if (cap && point.parameters > Number(cap[1])) return null
  const value = ranking === 'linearization' ? point.budget.quality_db
    : ranking === 'parameter_efficiency' ? point.budget.parameter_efficiency_db
    : ranking === 'arithmetic_efficiency' ? point.budget.arithmetic_efficiency_db
    : ranking === 'evm' ? point.budget.metrics?.evm_improvement_db
    : ranking === 'aclr' ? point.budget.metrics?.aclr_improvement_db : point.budget.score
  return finite(value) ? value : null
}
