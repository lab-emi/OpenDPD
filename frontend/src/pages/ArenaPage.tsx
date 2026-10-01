import EmojiEventsOutlinedIcon from '@mui/icons-material/EmojiEventsOutlined'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RefreshIcon from '@mui/icons-material/Refresh'
import { Alert, Box, Button, ButtonBase, Checkbox, Chip, FormControlLabel, LinearProgress, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Tooltip, Typography } from '@mui/material'
import { useState } from 'react'
import { Link as RouterLink, useLocation, useSearchParams } from 'react-router'
import { API } from '@/api/client'
import { useArena, useArenaBoard, useArenaSubmissions, useSubmitArena, type ArenaBoard, type ArenaBudget, type ArenaCatalog, type ArenaRanking, type ArenaRow } from '@/api/arena'
import { BackbonePicker, type BackboneEntry } from '@/components/BackbonePicker'
import { ArenaConfigurationTable, ArenaTradeoffs } from '@/components/ArenaTradeoffs'
import { DownloadLink } from '@/components/DownloadLink'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { formatDateTime, formatNumber, message, t, type MessageKey } from '@/i18n'
import { useStudioColors } from '@/theme'
import { isArenaExcluded } from '@/utils/arenaPareto'

const number = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : formatNumber(value, { maximumFractionDigits: digits, minimumFractionDigits: digits })
const integer = (value: number | null | undefined) => value == null ? '—' : formatNumber(value)
const evidence = (type: ArenaBoard['evidence_type']) => t(type === 'synthetic_simulation' ? 'arena.synthetic' : 'arena.measured')
const boardLink = (id: string, page = '') => `/arena${page}?board=${encodeURIComponent(id)}`
const executionLabel = (mode?: string) => t(mode === 'offline_overlap_200_100' ? 'arena.executionOffline' : mode === 'streaming_stateful' ? 'arena.executionStreaming' : 'arena.executionUnavailable')
const RANKING_KEYS: Record<string, [MessageKey, MessageKey]> = {
  overall: ['arena.ranking.overall', 'arena.rankingHelp.overall'],
  linearization: ['arena.ranking.linearization', 'arena.rankingHelp.linearization'],
  parameter_efficiency: ['arena.ranking.parameterEfficiency', 'arena.rankingHelp.parameterEfficiency'],
  arithmetic_efficiency: ['arena.ranking.arithmeticEfficiency', 'arena.rankingHelp.arithmeticEfficiency'],
  evm: ['arena.ranking.evm', 'arena.rankingHelp.evm'],
  aclr: ['arena.ranking.aer', 'arena.rankingHelp.aer'],
}
const budgetOf = (ranking: ArenaRanking) => /^budget-(\d+)$/.exec(ranking.ranking_id)?.[1]
function rankingText(ranking: ArenaRanking): [string, string] {
  const keys = RANKING_KEYS[ranking.ranking_id], budget = budgetOf(ranking)
  if (keys) return [t(keys[0]), t(keys[1])]
  if (budget) return [t('arena.ranking.budget', { budget: formatNumber(Number(budget)) }), t('arena.rankingHelp.budget', { budget: formatNumber(Number(budget)) })]
  return [ranking.title, ranking.description]
}
/** The preset that distinguishes one sweep point: a hidden size or polynomial orders. */
function configuration(point: ArenaBudget) {
  const values = point.model_parameters ?? {}
  return Object.entries(values).filter(([key, value]) => typeof value === 'number' && !['rcond', 'thx', 'thh', 'num_layers', 'iterations', 'fit_samples', 'peak_factor', 'learning_gain', 'target_nmse_db', 'backtracking_steps', 'min_improvement_db'].includes(key))
    .map(([key, value]) => `${key === 'hidden_size' ? 'H' : key}=${String(value)}`).join(' · ') || '—'
}

export function ArenaPage() {
  const catalog = useArena()
  const { pathname } = useLocation()
  const [params, setParams] = useSearchParams()
  if (catalog.isPending) return <LoadingState />
  if (catalog.isError) return <ErrorState error={catalog.error} onRetry={() => void catalog.refetch()} />
  const board = catalog.data.protocol.boards.find(b => b.board_id === params.get('board')) ?? catalog.data.protocol.boards[0]!
  const page = pathname === '/arena/submit' ? 'submit' : pathname === '/arena/rules' ? 'rules' : 'rank'
  return <Stack spacing={2} sx={{ maxWidth: 1600, mx: 'auto' }} data-testid="arena-page" data-view={page}>
    <Stack direction="row" useFlexGap sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap' }}>
      <Box><Typography variant="h1">{t('arena.title')} · {t(`arena.${page}`)}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: .75 }}>{t('arena.intro')}</Typography></Box>
      <Chip variant="outlined" size="small" label={catalog.data.protocol.protocol_id} />
    </Stack>
    <Alert severity="info">{t('arena.evidenceNotice')}</Alert>
    {catalog.data.protocol.boards.length > 1 && <BoardPicker boards={catalog.data.protocol.boards} selected={board.board_id} onSelect={id => setParams(params.get('ranking') ? { board: id, ranking: params.get('ranking')! } : { board: id })} />}
    {page === 'rank' ? <Ranking key={board.board_id} board={board} catalog={catalog.data} /> : page === 'submit' ? <Submit catalog={catalog.data} board={board} /> : <Rules catalog={catalog.data} board={board} />}
  </Stack>
}

function BoardPicker({ boards, selected, onSelect }: { boards: ArenaBoard[]; selected: string; onSelect: (id: string) => void }) {
  const colors = useStudioColors()
  return <Box role="group" aria-label={t('arena.boards')} sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0,1fr)', md: 'repeat(2,minmax(0,1fr))', xl: `repeat(${boards.length},minmax(0,1fr))` }, gap: 1.25 }}>
    {boards.map((board, index) => <ButtonBase key={board.board_id} onClick={() => onSelect(board.board_id)} aria-pressed={selected === board.board_id} sx={{ display: 'block', textAlign: 'left', p: 2, border: 1, borderRadius: 1.5, borderColor: selected === board.board_id ? 'primary.main' : 'divider', bgcolor: selected === board.board_id ? colors.selected : 'background.paper', '&:hover': { borderColor: 'primary.main' }, '&.Mui-focusVisible': { outline: '3px solid', outlineColor: 'primary.main', outlineOffset: 2 } }}>
      <Typography variant="caption" color="text.secondary">0{index + 1} · {evidence(board.evidence_type)}</Typography>
      <Typography sx={{ fontWeight: 750, fontSize: 17, mt: .5 }}>{message(board.title)}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: .75 }}>{message(board.description)}</Typography>
    </ButtonBase>)}
  </Box>
}

function Ranking({ board, catalog }: { board: ArenaBoard; catalog: ArenaCatalog }) {
  const result = useArenaBoard(board.board_id)
  const [params, setParams] = useSearchParams()
  const rankings = catalog.protocol.rankings
  const selected = rankings.find(item => item.ranking_id === params.get('ranking')) ?? rankings[0]!
  const [, help] = rankingText(selected)
  const rows = result.data?.rows.filter(row => row.board_id === board.board_id && !isArenaExcluded(row.backbone)) ?? []
  return <Stack spacing={2}>
    <Paper sx={{ p: 2 }}><Stack spacing={1.5}>
      <Stack direction="row" useFlexGap sx={{ gap: 1, alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <Box><Typography variant="h2">{message(board.title)}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: .5 }}>{t('arena.rankHelp')}</Typography></Box>
        <Stack direction="row" spacing={1}><Button startIcon={<RefreshIcon />} onClick={() => void result.refetch()} disabled={result.isFetching}>{t('arena.refresh')}</Button><Button component={RouterLink} to={boardLink(board.board_id, '/submit')} variant="contained">{t('arena.submit')}</Button></Stack>
      </Stack>
      {result.isPending ? <LoadingState /> : result.isError ? <ErrorState error={result.error} onRetry={() => void result.refetch()} /> : <>
        <Stack direction="row" useFlexGap sx={{ gap: 1, flexWrap: 'wrap' }}>
          <Chip size="small" label={evidence(board.evidence_type)} />
          <Chip size="small" variant="outlined" label={`${t('arena.coverage')}: ${result.data.coverage.evaluated} / ${result.data.coverage.expected}`} />
          <Chip size="small" variant="outlined" label={t('arena.sweepBudgets', { budgets: catalog.protocol.budgets.map(value => formatNumber(value)).join(' · ') })} />
          <Button component={RouterLink} to={boardLink(board.board_id, '/rules')} size="small">{t('arena.rules')}</Button>
        </Stack>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ alignItems: { md: 'center' } }}>
          <TextField select size="small" label={t('arena.ranking')} value={selected.ranking_id} onChange={event => setParams({ board: board.board_id, ranking: event.target.value })} sx={{ minWidth: 280 }}>
            {rankings.map(item => <MenuItem key={item.ranking_id} value={item.ranking_id}>{rankingText(item)[0]}</MenuItem>)}
          </TextField>
          <Typography variant="body2" color="text.secondary" sx={{ flex: 1 }}>{help}</Typography>
        </Stack>
        <Typography variant="body2" color="text.secondary">{t('arena.coverageHelp')}</Typography>
        {board.conditions.map(condition => {
          const pa = catalog.protocol.pa_models?.[condition]
          return pa && <Typography variant="body2" key={condition}>{condition} · {t('arena.paModelDetails', { model: pa.model, parameters: formatNumber(pa.parameters), validation: number(pa.validation_nmse_db), test: number(pa.test_nmse_db) })}</Typography>
        })}
        <Typography variant="body2" color="text.secondary">{t('arena.cohortHelp')}</Typography>
        {!!result.data.coverage.missing?.length && <Alert severity="warning">{t('arena.notAvailable')}: {result.data.coverage.missing.join(', ')}</Alert>}
        {rows.length ? <>
          {(['official', 'workspace'] as const).map(origin => <ResultGroup key={origin} rows={rows.filter(row => row.origin === origin)} origin={origin} protocolSha={result.data.protocol_sha256} ranking={selected} budgets={catalog.protocol.budgets} />)}
        </> : <Typography sx={{ py: 3 }}>{t('arena.noEntries')}</Typography>}
        <DownloadLink href={`${API}/arena/boards/${encodeURIComponent(board.board_id)}`} download>{t('arena.download')}</DownloadLink>
      </>}
    </Stack></Paper>
  </Stack>
}

function ResultGroup({ rows, origin, protocolSha, ranking, budgets }: { rows: ArenaRow[]; origin: ArenaRow['origin']; protocolSha: string; ranking: ArenaRanking; budgets: number[] }) {
  if (!rows.length) return null
  const cohorts = new Map<string, ArenaRow[]>()
  for (const row of rows) cohorts.set(row.execution_semantics ?? '', [...(cohorts.get(row.execution_semantics ?? '') ?? []), row])
  const position = (row: ArenaRow) => row.rankings?.[ranking.ranking_id]?.rank ?? Number.POSITIVE_INFINITY
  const heading = t(origin === 'official' ? 'arena.official' : 'arena.workspace')
  return <Stack spacing={1}>
    <Typography variant="h3" sx={{ mt: 1 }}>{heading}</Typography>
    {/* The offline cohort is the main board: keep it first whichever cohort holds the first-sorted row. */}
    {[...cohorts.entries()].sort(([a], [b]) => Number(a !== 'offline_overlap_200_100') - Number(b !== 'offline_overlap_200_100') || a.localeCompare(b)).map(([cohort, entries]) => <Box key={cohort}>
    <Typography variant="body2" sx={{ fontWeight: 600, mt: .5, mb: 1 }}>{executionLabel(cohort)} · {rankingText(ranking)[0]}</Typography>
    <ArenaConfigurationTable rows={entries} protocol={protocolSha} ranking={ranking.ranking_id} />
    <ArenaTradeoffs rows={entries} protocol={protocolSha} />
    <Box component="details"><Typography component="summary" variant="h3" sx={{ cursor: 'pointer', mb: 1 }}>{t('arena.backboneSummaries')}</Typography>
    <TableContainer role="region" tabIndex={0} aria-label={`${heading} · ${executionLabel(cohort)}`}>
      <Table size="small" aria-label={heading} sx={{ minWidth: 1080, '& .MuiTableCell-root': { px: 1 } }}>
        <TableHead><TableRow>
          <TableCell>{t('arena.rank')}</TableCell><TableCell>{t('arena.entry')}</TableCell>
          <TableCell align="right">{t('arena.scoreColumn')}</TableCell>
          <TableCell align="right"><Tooltip describeChild title={t('arena.qualityGate')}><span tabIndex={0}>{t('arena.bestQuality')}</span></Tooltip></TableCell>
          <TableCell align="right"><Tooltip describeChild title={t('arena.evmHelp')}><span tabIndex={0}>{t('arena.evm')}</span></Tooltip></TableCell>
          <TableCell align="right"><Tooltip describeChild title={t('arena.aerHelp')}><span tabIndex={0}>{t('arena.aclr')}</span></Tooltip></TableCell>
          {budgets.map(budget => <TableCell key={budget} align="right"><Tooltip describeChild title={t('arena.sweepQualityHelp')}><span tabIndex={0}>{t('arena.budgetColumn', { budget: formatNumber(budget) })}</span></Tooltip></TableCell>)}
          <TableCell align="right"><Tooltip describeChild title={t('arena.opsHelp')}><span tabIndex={0}>{t('arena.opsPerParameter')}</span></Tooltip></TableCell>
          <TableCell>{t('arena.status')}</TableCell><TableCell>{t('arena.details')}</TableCell>
        </TableRow></TableHead>
        <TableBody>{[...entries].sort((a, b) => position(a) - position(b)).map(row => <ResultRow key={row.entry_id} row={row} protocolSha={protocolSha} ranking={ranking} budgets={budgets} />)}</TableBody>
      </Table>
    </TableContainer>
    </Box>
    </Box>)}
  </Stack>
}

function ResultRow({ row, protocolSha, ranking, budgets }: { row: ArenaRow; protocolSha: string; ranking: ArenaRanking; budgets: number[] }) {
  const [open, setOpen] = useState(false)
  const complete = row.status === 'succeeded'
  const sameProtocol = row.protocol_sha256 === protocolSha
  const entry = row.rankings?.[ranking.ranking_id]
  const ranked = complete && sameProtocol && row.eligible && entry?.rank != null && entry.score != null
  const eligible = complete && sameProtocol && row.eligible
  const detailsId = `arena-result-${row.entry_id}`
  const displayName = row.display_name
  const point = (budget: number) => row.budgets?.find(item => item.budget === budget)
  return <>
    <TableRow hover>
      <TableCell>{ranked ? <Stack direction="row" spacing={.5} sx={{ alignItems: 'center' }}>{entry.rank === 1 && <EmojiEventsOutlinedIcon color="primary" sx={{ fontSize: 17 }} />}<span>{entry.rank}</span></Stack> : '—'}</TableCell>
      <TableCell><Typography variant="body2" sx={{ fontWeight: 650 }}>{displayName}</Typography><Typography variant="caption" color="text.secondary">{row.backbone}</Typography></TableCell>
      <TableCell align="right" sx={{ fontWeight: 700, color: ranked ? 'primary.main' : 'text.secondary' }}>{ranked ? number(entry.score) : '—'}</TableCell>
      <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>{number(row.quality_db)}{row.best_budget != null && <Typography component="span" variant="caption" color="text.secondary"> @ ≤{formatNumber(row.best_budget)}</Typography>}</TableCell>
      <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>{row.metrics?.evm_pct == null ? '—' : <Tooltip describeChild title={t('arena.baselineValue', { value: `${number(row.metrics.baseline_evm_pct)} % (${number(row.metrics.baseline_evm_db, 1)} dB)` })}><span tabIndex={0}>{number(row.metrics.evm_pct)} %<Typography component="span" variant="caption" color="text.secondary"> {number(row.metrics.evm_db, 1)} dB</Typography></span></Tooltip>}</TableCell>
      <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>{row.metrics?.aclr_db == null ? '—' : <Tooltip describeChild title={t('arena.baselineValue', { value: `${number(row.metrics.baseline_aclr_db, 1)} dBc` })}><span tabIndex={0}>{number(row.metrics.aclr_db, 1)} dBc</span></Tooltip>}</TableCell>
      {budgets.map(budget => { const item = point(budget); return <TableCell key={budget} align="right" sx={{ whiteSpace: 'nowrap', color: item?.qualified ? 'text.primary' : 'text.secondary' }}>
        {!item?.available ? <Tooltip describeChild title={t('arena.unavailableBudget')}><span tabIndex={0}>—</span></Tooltip> : item.qualified ? number(item.quality_db) : <Tooltip describeChild title={[t('arena.unqualified'), ...(item.reasons ?? []).map(message)].join(' · ')}><span tabIndex={0}>({number(item.quality_db)})</span></Tooltip>}
      </TableCell> })}
      <TableCell align="right">{number(row.ops_per_parameter)}</TableCell>
      <TableCell><Chip size="small" variant="outlined" color={eligible ? 'success' : row.status === 'failed' ? 'error' : 'default'} label={eligible ? t('arena.ranked') : complete ? t('arena.unranked') : t(`status.${row.status}`)} /><Typography component="div" variant="caption" color="text.secondary">{t('arena.qualifiedBudgets', { qualified: row.qualified_budgets ?? 0, total: budgets.length })}</Typography></TableCell>
      <TableCell><Button size="small" onClick={() => setOpen(!open)} aria-expanded={open} aria-controls={open ? detailsId : undefined} aria-label={`${t('arena.details')}: ${displayName}`}>{t('arena.details')}</Button></TableCell>
    </TableRow>
    {open && <TableRow><TableCell colSpan={9 + budgets.length} sx={{ p: 2, bgcolor: 'action.hover' }}>{/* width 0 + minWidth 100%: wide evidence tables scroll inside the cell and never widen the leaderboard */}<Stack id={detailsId} spacing={1.25} sx={{ width: 0, minWidth: '100%' }}>
      <Typography variant="h3">{displayName} · {t('arena.provenance')}</Typography>
      <Stack direction="row" useFlexGap sx={{ gap: 1, flexWrap: 'wrap' }}><Chip size="small" label={evidence(row.evidence_type)} /><Chip size="small" label={`${t('arena.seeds')}: ${row.seeds?.join(', ') || '—'}`} /><Chip size="small" label={`${t('arena.conditions')}: ${row.completed_cases} / ${row.expected_cases}`} /></Stack>
      <Typography variant="body2" color="text.secondary">{t('arena.qualityGate')}</Typography>
      {!sameProtocol && <Alert severity="warning">{t('arena.protocolMismatch')}</Alert>}
      {row.eligibility_reasons?.map(reason => <Alert severity="warning" key={reason}>{message(reason)}</Alert>)}
      {row.error && <Alert severity={row.status === 'failed' ? 'error' : 'warning'}>{message(row.error)}</Alert>}
      <SweepTable row={row} />
      <RankingSummary row={row} />
      <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>{t('arena.protocol')} SHA256: {row.protocol_sha256}</Typography>
      <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>{executionLabel(row.execution_semantics)} · ID: {row.entry_id}{row.created_at ? ` · ${formatDateTime(row.created_at)}` : ''}</Typography>
      <CaseMetrics cases={row.cases ?? []} />
      <Box component="details"><Typography component="summary" variant="body2" sx={{ cursor: 'pointer' }}>{t('arena.rawEvidence')}</Typography><Box component="pre" sx={{ mt: 1, mb: 0, fontSize: 12, maxHeight: 280, overflow: 'auto', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify({ execution_semantics: row.execution_semantics, rankings: row.rankings, budgets: row.budgets, provenance: row.provenance, cases: row.cases }, null, 2)}</Box></Box>
    </Stack></TableCell></TableRow>}
  </>
}

function SweepTable({ row }: { row: ArenaRow }) {
  const points = row.budgets ?? []
  if (!points.length) return null
  return <Stack spacing={1}>
    <Typography variant="h3">{t('arena.sweepDetails')}</Typography>
    <Typography variant="body2" color="text.secondary">{t('arena.sweepHelp')}</Typography>
    <TableContainer tabIndex={0} role="region" aria-label={t('arena.sweepDetails')}><Table size="small" aria-label={t('arena.sweepDetails')} sx={{ minWidth: 1220 }}>
      <TableHead><TableRow>
        <TableCell>{t('arena.budget')}</TableCell><TableCell>{t('arena.configuration')}</TableCell>
        {(['arena.parameters', 'arena.mul', 'arena.add', 'arena.ops', 'arena.quality', 'arena.qualityConservative', 'arena.evm', 'arena.evmImprovement', 'arena.aclr', 'arena.aerImprovement', 'arena.parameterEfficiency', 'arena.arithmeticEfficiency', 'arena.budgetScore'] as const).map(key => <TableCell key={key} align="right">{t(key)}</TableCell>)}
        <TableCell>{t('arena.status')}</TableCell>
      </TableRow></TableHead>
      <TableBody>{points.map(item => <TableRow key={item.budget}>
        <TableCell sx={{ whiteSpace: 'nowrap' }}>≤ {formatNumber(item.budget)}</TableCell>
        <TableCell sx={{ whiteSpace: 'nowrap' }}>{item.available ? configuration(item) : '—'}</TableCell>
        <TableCell align="right">{integer(item.parameters)}</TableCell><TableCell align="right">{integer(item.mul)}</TableCell>
        <TableCell align="right">{integer(item.add)}</TableCell><TableCell align="right">{integer(item.ops)}</TableCell>
        <TableCell align="right">{number(item.quality_db)}</TableCell><TableCell align="right">{number(item.quality_conservative_db)}</TableCell>
        <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>{item.metrics?.evm_pct == null ? '—' : `${number(item.metrics.evm_pct)} %`}</TableCell><TableCell align="right">{number(item.metrics?.evm_improvement_db)}</TableCell>
        <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>{item.metrics?.aclr_db == null ? '—' : `${number(item.metrics.aclr_db, 1)} dBc`}</TableCell><TableCell align="right">{number(item.metrics?.aclr_improvement_db)}</TableCell>
        <TableCell align="right">{number(item.parameter_efficiency_db)}</TableCell><TableCell align="right">{number(item.arithmetic_efficiency_db)}</TableCell>
        <TableCell align="right" sx={{ fontWeight: 650 }}>{number(item.score)}</TableCell>
        <TableCell>{!item.available ? t('arena.unavailableBudget') : item.qualified ? (item.quality_db != null && item.quality_db > 0 ? t('arena.qualified') : t('arena.noQualityGain')) : <Tooltip describeChild title={(item.reasons ?? []).map(message).join(' · ')}><span tabIndex={0}>{t('arena.unqualified')}</span></Tooltip>}</TableCell>
      </TableRow>)}</TableBody>
    </Table></TableContainer>
    {points.filter(item => item.available && item.operation_items?.length).map(item => <Box component="details" key={item.budget}>
      <Typography component="summary" variant="body2" sx={{ cursor: 'pointer' }}>{t('arena.operationBreakdown', { budget: formatNumber(item.budget) })}</Typography>
      <TableContainer tabIndex={0} role="region" aria-label={t('arena.operationBreakdown', { budget: formatNumber(item.budget) })}><Table size="small" sx={{ minWidth: 640, mt: 1 }}>
        <TableHead><TableRow><TableCell>{t('arena.component')}</TableCell><TableCell align="right">MUL</TableCell><TableCell align="right">ADD</TableCell><TableCell>{t('arena.nonlinear')}</TableCell></TableRow></TableHead>
        <TableBody>
          {(item.operation_items ?? []).map((part, index) => { const functions = part.nonlinear && typeof part.nonlinear === 'object' && !Array.isArray(part.nonlinear) ? Object.entries(part.nonlinear) : []
            return <TableRow key={index}><TableCell>{String(part.component ?? '')}</TableCell><TableCell align="right">{integer(Number(part.mul))}</TableCell><TableCell align="right">{integer(Number(part.add))}</TableCell><TableCell>{functions.map(([name, count]) => `${name} × ${String(count)}`).join(' · ') || '—'}</TableCell></TableRow> })}
          <TableRow><TableCell sx={{ fontWeight: 650 }}>{t('arena.nonlinearEquivalent')}</TableCell><TableCell align="right">{integer(item.nonlinear_mul)}</TableCell><TableCell align="right">{integer(item.nonlinear_add)}</TableCell><TableCell>{Object.entries(item.nonlinear ?? {}).map(([name, count]) => `${name} × ${count}`).join(' · ') || '—'}</TableCell></TableRow>
          <TableRow><TableCell sx={{ fontWeight: 650 }}>{t('arena.total')}</TableCell><TableCell align="right" sx={{ fontWeight: 650 }}>{integer(item.mul)}</TableCell><TableCell align="right" sx={{ fontWeight: 650 }}>{integer(item.add)}</TableCell><TableCell sx={{ fontWeight: 650 }}>{t('arena.ops')}: {integer(item.ops)}</TableCell></TableRow>
        </TableBody>
      </Table></TableContainer>
    </Box>)}
  </Stack>
}

function RankingSummary({ row }: { row: ArenaRow }) {
  const entries = Object.entries(row.rankings ?? {}).filter(([, value]) => value.score != null)
  if (!entries.length) return null
  return <Stack direction="row" useFlexGap sx={{ gap: 1, flexWrap: 'wrap' }}>{entries.map(([id, value]) => <Chip key={id} size="small" variant="outlined"
    label={`${rankingText({ ranking_id: id, title: id, description: '', unit: 'dB' })[0]}: ${number(value.score)}${value.rank != null ? ` · #${value.rank}` : ''}`} />)}</Stack>
}

function CaseMetrics({ cases }: { cases: NonNullable<ArenaRow['cases']> }) {
  const object = (value: unknown): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
  const numeric = (value: unknown) => typeof value === 'number' ? number(value) : '—'
  const text = (value: unknown) => typeof value === 'string' || typeof value === 'number' ? String(value) : '—'
  const observations = cases.flatMap(item => Array.isArray(item.judges) ? item.judges.map(value => ({ budget: item.budget, condition: item.condition_id, seed: item.seed, judge: object(value) })) : [])
  if (!observations.length) return null
  const identity = ({ budget, condition, seed, judge }: typeof observations[number]) => <><TableCell sx={{ whiteSpace: 'nowrap' }}>{typeof budget === 'number' ? `≤ ${formatNumber(budget)}` : '—'}</TableCell><TableCell>{text(condition)}</TableCell><TableCell>{text(seed)}</TableCell><TableCell>{text(judge.judge_id ?? judge.judge)}</TableCell></>
  const head = <><TableCell>{t('arena.budget')}</TableCell><TableCell>{t('arena.conditions')}</TableCell><TableCell>{t('arena.seeds')}</TableCell><TableCell>{t('arena.judge')}</TableCell></>
  return <Box component="details"><Typography component="summary" variant="body2" sx={{ cursor: 'pointer' }}>{t('arena.caseMetrics')} ({observations.length})</Typography><Stack spacing={1} sx={{ mt: 1 }}>
    <Typography variant="body2" color="text.secondary">{t('arena.baselineToDpd')}</Typography>
    <TableContainer tabIndex={0} role="region" aria-label={t('arena.caseMetrics')} sx={{ maxHeight: 420 }}><Table size="small" stickyHeader aria-label={t('arena.caseMetrics')} sx={{ minWidth: 1080 }}>
      <TableHead><TableRow>{head}<TableCell align="right"><Tooltip describeChild title={t('arena.evmHelp')}><span tabIndex={0}>EVM (dB)</span></Tooltip></TableCell><TableCell align="right"><Tooltip describeChild title={t('arena.aerHelp')}><span tabIndex={0}>ACLR L (dBc)</span></Tooltip></TableCell><TableCell align="right"><Tooltip describeChild title={t('arena.aerHelp')}><span tabIndex={0}>ACLR R (dBc)</span></Tooltip></TableCell><TableCell align="right"><Tooltip describeChild title={t('arena.nmseHelp')}><span tabIndex={0}>NMSE (dB)</span></Tooltip></TableCell><TableCell align="right">{t('arena.powerError')}</TableCell></TableRow></TableHead>
      <TableBody>{observations.map((item, index) => <TableRow key={index}>
        {identity(item)}
        {['evm_db', 'aclr_l_db', 'aclr_r_db', 'nmse_db'].map(key => <TableCell align="right" key={key} sx={{ whiteSpace: 'nowrap' }}>{numeric(item.judge[`baseline_${key}`])} → {numeric(item.judge[key])}</TableCell>)}
        <TableCell align="right">{numeric(item.judge.power_error_db)}</TableCell>
      </TableRow>)}</TableBody>
    </Table></TableContainer>
    <Typography variant="h3" sx={{ pt: 1 }}>{t('arena.rawAclrDiagnostics')}</Typography>
    <Typography variant="body2" color="text.secondary">{t('arena.rawAclrHelp')}</Typography>
    <TableContainer tabIndex={0} role="region" aria-label={t('arena.rawAclrDiagnostics')} sx={{ maxHeight: 420 }}><Table size="small" stickyHeader aria-label={t('arena.rawAclrDiagnostics')} sx={{ minWidth: 980 }}>
      <TableHead><TableRow>{head}{['L', 'R'].map(side => <TableCell key={side} align="right">{t('arena.rawOutputAclr')} {side} (dBc)</TableCell>)}{['L', 'R'].map(side => <TableCell key={side} align="right">{t('arena.idealReferenceAclr')} {side} (dBc)</TableCell>)}</TableRow></TableHead>
      <TableBody>{observations.map((item, index) => <TableRow key={index}>
        {identity(item)}
        {['aclr_l_db', 'aclr_r_db'].map(key => <TableCell align="right" key={key} sx={{ whiteSpace: 'nowrap' }}>{numeric(item.judge[`baseline_${key}`])} → {numeric(item.judge[key])}</TableCell>)}
        {['reference_aclr_l_db', 'reference_aclr_r_db'].map(key => <TableCell align="right" key={key}>{numeric(item.judge[key])}</TableCell>)}
      </TableRow>)}</TableBody>
    </Table></TableContainer>
  </Stack></Box>
}

function Submit({ catalog, board }: { catalog: ArenaCatalog; board: ArenaBoard }) {
  const backbones = catalog.backbones.filter(item => !isArenaExcluded(item.key))
  const [backbone, setBackbone] = useState(backbones.find(item => item.key === 'gru')?.key ?? backbones[0]?.key ?? '')
  const [custom, setCustom] = useState<BackboneEntry | null>(null)
  const [name, setName] = useState('')
  const [accepted, setAccepted] = useState('')
  const submit = useSubmitArena()
  const submissions = useArenaSubmissions()
  const modelInfo = backbones.find(item => item.key === backbone)
  const modelName = custom?.name ?? modelInfo?.display_name ?? backbone
  const seeds = modelInfo?.deterministic && !custom ? catalog.protocol.seeds.slice(0, 1) : catalog.protocol.seeds
  const displayName = name.trim() || `${modelName} · ${board.title}`.slice(0, 80)
  const validName = displayName.length > 0 && displayName.length <= 80 && [...displayName].every(char => char.charCodeAt(0) >= 32)
  const currentAgreement = `${board.board_id}:${catalog.protocol.protocol_sha256}`
  const ready = catalog.submissions_available && !submit.isPending && (!!custom || !!modelInfo) && validName && accepted === currentAgreement
  return <Stack spacing={2}>
    <Paper sx={{ p: { xs: 1.5, md: 2 } }}><Stack spacing={2}>
      <Box><Typography variant="h2">{t('arena.submit')}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: .75 }}>{t('arena.submitHelp')}</Typography></Box>
      {!catalog.submissions_available && <Alert severity="info">{message(catalog.submission_unavailable_reason) || t('arena.notAvailable')}</Alert>}
      <Box component="fieldset" disabled={submit.isPending || !catalog.submissions_available} sx={{ p: 0, m: 0, border: 0, minWidth: 0 }}><Stack spacing={2}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
          <TextField select label={t('arena.model')} value={custom?.backbone_id ?? backbone} onChange={event => { setBackbone(event.target.value); setCustom(null) }} sx={{ flex: 1 }}>
            {backbones.map(item => <MenuItem key={item.key} value={item.key}>{item.display_name} · {item.family}</MenuItem>)}
            {custom && <MenuItem value={custom.backbone_id}>{custom.name}</MenuItem>}
          </TextField>
          <TextField label={t('arena.displayName')} value={name} placeholder={displayName} onChange={event => setName(event.target.value)} error={!validName} slotProps={{ htmlInput: { maxLength: 80 } }} sx={{ flex: 1 }} />
        </Stack>
        {catalog.submissions_available && <BackbonePicker selected={custom} onSelect={entry => { setCustom(entry); if (entry) setBackbone('user_template'); else setBackbone(backbones[0]?.key ?? '') }} />}
        <Paper variant="outlined" sx={{ p: 1.5 }}><Stack spacing={1}>
          <Typography variant="body2" sx={{ fontWeight: 650 }}>{message(board.title)} · {evidence(board.evidence_type)}</Typography>
          <Typography variant="body2">{t('arena.seeds')}: {seeds.join(', ')} · {t('arena.conditions')}: {board.conditions.join(', ')}</Typography>
          <Typography variant="body2">{t('arena.sweepBudgets', { budgets: catalog.protocol.budgets.map(value => formatNumber(value)).join(' · ') })} · {t('arena.sweepSubmitHelp')}</Typography>
          {modelInfo?.deterministic && !custom ? <Typography variant="body2">{t('arena.deterministicBudget')}</Typography> : <Typography variant="body2">{t('arena.fullTrainingBudget', { epochs: String(catalog.protocol.training.epochs ?? '—'), batch: String(catalog.protocol.training.batch_size ?? '—') })}</Typography>}
          <Button component={RouterLink} to={boardLink(board.board_id, '/rules')} sx={{ alignSelf: 'flex-start' }}>{t('arena.rules')} · {catalog.protocol.protocol_id}</Button>
          <FormControlLabel control={<Checkbox checked={accepted === currentAgreement} onChange={event => setAccepted(event.target.checked ? currentAgreement : '')} />} label={t('arena.accept')} />
        </Stack></Paper>
        <Button variant="contained" startIcon={<PlayArrowIcon />} disabled={!ready} onClick={() => submit.mutate({ board_id: board.board_id, backbone: custom ? 'user_template' : backbone, ...(custom ? { backbone_id: custom.backbone_id } : {}), display_name: displayName, accepted_protocol_sha256: catalog.protocol.protocol_sha256 })} sx={{ alignSelf: 'flex-start' }}>{t(submit.isPending ? 'arena.launching' : 'arena.launch')}</Button>
      </Stack></Box>
      {submit.isPending && <LinearProgress aria-label={t('arena.launching')} />}
      {submit.isError && <ErrorState error={submit.error} />}
      {submit.isSuccess && <Alert severity="success">{t(`status.${submit.data.status}`)} · {submit.data.submission_id}</Alert>}
    </Stack></Paper>
    <Paper sx={{ p: 2 }}><Stack spacing={1.5}>
      <Stack direction="row" sx={{ justifyContent: 'space-between', gap: 1 }}><Typography variant="h2">{t('arena.submissions')}</Typography><Button startIcon={<RefreshIcon />} disabled={submissions.isFetching} onClick={() => void submissions.refetch()}>{t('arena.refresh')}</Button></Stack>
      {submissions.isPending ? <LoadingState /> : submissions.isError ? <ErrorState error={submissions.error} onRetry={() => void submissions.refetch()} /> : !submissions.data.length ? <Typography color="text.secondary">{t('arena.noSubmissions')}</Typography> : submissions.data.map(item => <Paper variant="outlined" key={item.submission_id} sx={{ p: 1.5 }}><Stack spacing={1}>
        <Stack direction="row" useFlexGap sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}><Typography sx={{ fontWeight: 650 }}>{item.request.display_name}</Typography><Chip size="small" label={t(`status.${item.status}`)} color={item.status === 'failed' ? 'error' : item.status === 'succeeded' ? 'success' : 'default'} /></Stack>
        <Typography variant="caption" color="text.secondary">{item.request.backbone} · {item.request.board_id}{item.created_at ? ` · ${formatDateTime(item.created_at)}` : ''}</Typography>
        {['queued', 'running'].includes(item.status) && <LinearProgress aria-label={t('arena.progress')} />}
        {item.progress && <Typography variant="body2" role="status">{message(item.progress.message) || item.progress.phase}{!catalog.backbones.find(option => option.key === item.request.backbone)?.deterministic && item.progress.epoch != null && item.progress.epochs != null ? ` · ${item.progress.epoch} / ${item.progress.epochs}` : ''}{item.progress.completed_cases != null && item.progress.expected_cases != null ? ` · ${t('arena.conditions')}: ${item.progress.completed_cases} / ${item.progress.expected_cases}` : ''}</Typography>}
        {item.protocol_sha256 !== catalog.protocol.protocol_sha256 && <Alert severity="info">{t('arena.protocolMismatch')}</Alert>}
        {item.error && <Alert severity="error">{message(item.error)}</Alert>}
        <Stack direction="row" useFlexGap sx={{ gap: 1, flexWrap: 'wrap' }}>{item.protocol_sha256 === catalog.protocol.protocol_sha256 && <Button component={RouterLink} to={boardLink(item.request.board_id)} size="small">{t('arena.rank')}</Button>}<DownloadLink href={`${API}/arena/submissions/${encodeURIComponent(item.submission_id)}`} download>{t('arena.download')}</DownloadLink></Stack>
      </Stack></Paper>)}
    </Stack></Paper>
  </Stack>
}

function Rules({ catalog, board }: { catalog: ArenaCatalog; board: ArenaBoard }) {
  const protocol = catalog.protocol
  return <Stack spacing={2}>
    <Paper sx={{ p: 2 }}><Stack spacing={1.5}>
      <Typography variant="h2">{t('arena.protocol')} · {protocol.protocol_id}</Typography>
      <Typography>{message(protocol.description)}</Typography>
      <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>SHA256: {protocol.protocol_sha256}</Typography>
      <Typography variant="h3">{t('arena.scope')}</Typography>
      <Typography variant="body2">{message(catalog.scope_note)}</Typography>
      <Chip label={evidence(board.evidence_type)} sx={{ alignSelf: 'flex-start' }} />
      <Typography variant="body2">{board.dataset} · {board.conditions.join(', ')}</Typography>
      <Typography variant="h3">{t('arena.scoring')}</Typography>
      <Box component="pre" sx={{ p: 1.5, m: 0, borderRadius: 1, bgcolor: 'action.hover', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: 13 }}>{message(protocol.score_formula)}</Box>
      <Typography variant="body2" color="text.secondary">{t('arena.rankHelp')}</Typography>
      <Typography variant="body2">{t('arena.qualityGate')}</Typography>
      <Typography variant="body2">{t('arena.evmHelp')}</Typography>
      <Typography variant="body2">{t('arena.aerHelp')}</Typography>
      <Typography variant="h3">{t('arena.rankings')}</Typography>
      {protocol.rankings.map(item => { const [title, help] = rankingText(item); return <Typography variant="body2" key={item.ranking_id}><Box component="span" sx={{ fontWeight: 650 }}>{title}</Box> — {help}</Typography> })}
    </Stack></Paper>
    <CostModel catalog={catalog} />
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0,1fr)', md: 'repeat(2,minmax(0,1fr))' }, gap: 1.5 }}>
      {protocol.rules.map((rule, index) => <Paper key={`${index}-${rule.title}`} sx={{ p: 2 }}><Typography variant="caption" color="primary">{String(index + 1).padStart(2, '0')}</Typography><Typography variant="h3" sx={{ mt: .5 }}>{message(rule.title)}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{message(rule.description)}</Typography></Paper>)}
    </Box>
    <Paper sx={{ p: 2 }}><Stack spacing={1.5}>
      <Typography variant="h2">{t('arena.details')}</Typography>
      <Typography variant="body2">{t('arena.seeds')}: {protocol.seeds.join(', ')}</Typography>
      <Box component="pre" sx={{ m: 0, fontSize: 12, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{JSON.stringify(protocol.training, null, 2)}</Box>
      <DownloadLink href={`${API}/arena`} download>{t('arena.download')}</DownloadLink>
    </Stack></Paper>
  </Stack>
}

function CostModel({ catalog }: { catalog: ArenaCatalog }) {
  const prices = catalog.protocol.cost_model.nonlinear_cost
  const rows = prices && typeof prices === 'object' && !Array.isArray(prices) ? Object.entries(prices) : []
  const count = (value: unknown, key: 'mul' | 'add') => { const item = value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>)[key] : null; return typeof item === 'number' ? item : null }
  return <Paper sx={{ p: 2 }}><Stack spacing={1.5}>
    <Typography variant="h2">{t('arena.costModel')}</Typography>
    <Typography variant="body2">{t('arena.costModelHelp')}</Typography>
    <Typography variant="body2" color="text.secondary">{t('arena.opsHelp')}</Typography>
    <Typography variant="h3">{t('arena.nonlinearPrice')}</Typography>
    <Typography variant="body2" color="text.secondary">{t('arena.nonlinearPriceHelp')}</Typography>
    <TableContainer tabIndex={0} role="region" aria-label={t('arena.nonlinearPrice')}><Table size="small" aria-label={t('arena.nonlinearPrice')} sx={{ maxWidth: 420 }}>
      <TableHead><TableRow><TableCell>{t('arena.function')}</TableCell><TableCell align="right">MUL</TableCell><TableCell align="right">ADD</TableCell></TableRow></TableHead>
      <TableBody>{rows.map(([name, value]) => <TableRow key={name}><TableCell>{name}</TableCell><TableCell align="right">{integer(count(value, 'mul'))}</TableCell><TableCell align="right">{integer(count(value, 'add'))}</TableCell></TableRow>)}</TableBody>
    </Table></TableContainer>
  </Stack></Paper>
}
