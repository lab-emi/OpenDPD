import { expect, test } from 'vitest'
import type { ArenaRow } from '@/api/arena'
import { configurationPoints, configurationScore, paretoFront, type ArenaPoint } from './arenaPareto'

const pointFixture = (id: string, parameters: number, ops: number, evm: number, aclr: number, eligible = true): ArenaPoint => ({ id, parameters, ops, evm, aclr, eligible, evmStd: 0, aclrStd: 0, row: {} as ArenaRow, budget: { budget: parameters, score: 8, available: true, qualified: true, expected_cases: 3, completed_cases: 3 } })

test('each projection has its own front and preserves exact ties', () => {
  const points = [pointFixture('a', 250, 2000, -30, -40), pointFixture('b', 500, 1000, -35, -38), pointFixture('c', 1000, 3000, -32, -45), pointFixture('tie', 250, 2000, -30, -40), pointFixture('invalid', 1, 1, -100, -100, false)]
  expect(paretoFront(points, 'parameters', 'evm').map(p => p.id)).toEqual(['a', 'tie', 'b'])
  expect(paretoFront(points, 'parameters', 'aclr').map(p => p.id)).toEqual(['a', 'tie', 'c'])
  expect(paretoFront(points, 'ops', 'evm').map(p => p.id)).toEqual(['b'])
})

test('equal costs retain the better metric and equal metrics retain the cheaper cost', () => {
  expect(paretoFront([pointFixture('a', 200, 400, -30, -40), pointFixture('b', 200, 400, -31, -40), pointFixture('c', 300, 600, -31, -40)], 'parameters', 'evm').map(p => p.id)).toEqual(['b'])
})

test('budget rankings include smaller configurations and preserve negative FoM', () => {
  const point = pointFixture('small', 247, 504, -30, -40)
  point.budget.score = -2
  expect(configurationScore(point, 'budget-1000')).toBe(-2)
  expect(configurationScore(point, 'budget-200')).toBeNull()
})

test('plots use output ACLR and seed means, with protocol and condition isolation', () => {
  const row = { entry_id: 'a', status: 'succeeded', protocol_sha256: 'current', eligible: true,
    budgets: [{ budget: 500, available: true, qualified: true, quality_db: 10, parameters: 442, ops: 900 }],
    cases: [0, 1, 2].map(seed => ({ budget: 500, condition_id: 'A', seed, judges: [{ evm_db: -30 - seed, aclr_l_db: -40 - seed, aclr_r_db: -45, aer_l_db: -100, aer_r_db: -110 }] })) } as unknown as ArenaRow
  const points = configurationPoints([row], 'current', 'A')
  expect(points).toHaveLength(1)
  expect(points[0]).toMatchObject({ evm: -31, aclr: -41, evmStd: 1, aclrStd: 1 })
  expect(configurationPoints([row], 'old')).toEqual([])
  expect(configurationPoints([row], 'current', 'B')).toEqual([])
  expect(configurationPoints([{ ...row, status: 'failed' }], 'current')).toEqual([])
  expect(configurationPoints([{ ...row, backbone: 'ilc_dpd' }], 'current')).toEqual([])
})
