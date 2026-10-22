"""Which PRBs, symbols, modulation and level every user of a test model gets (TS 38.141-1 4.9.2.2 / 4.9.2.3.2).

Pure functions on integers; the RBG formulas use integer arithmetic only (nine (SCS, bandwidth, x) combinations give an
exactly integral x (N_RB - 3) / P, where a float comparison would be fragile).

Notation: N = N_RB - 3 is the size of the notional bandwidth part that starts at PRB 3; P is the nominal RBG size of
TS 38.214 table 5.1.2.2.1-1 configuration 1; RBG i of that part is indexed from 0 at its (partial) first RBG.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tables import MODEL_BY_ID, ModelDef, TddPattern, dmrs_symbols

CONTROL_PRBS = 3     # PRB 0-2 carry the PDCCH (and the n_RNTI = 2 user): "N_RB - 3" in tables 4.9.2.2-2 / 4.9.2.2.1-1


def rbg_size(n_bwp: int) -> int:
    """Nominal RBG size P, TS 38.214 table 5.1.2.2.1-1, configuration 1."""
    return 2 if n_bwp <= 36 else 4 if n_bwp <= 72 else 8 if n_bwp <= 144 else 16


def n_target_rbg(nrb: int, x_percent: int) -> int:
    """N_RBG = min(floor(x N / P), (A - A mod 2) / 2 + 1), A = floor((N + (3 mod P) - P) / P)  (tables 4.9.2.2.2-1, .7-1, .8-1)."""
    n = nrb - CONTROL_PRBS
    p = rbg_size(n)
    a = (n + CONTROL_PRBS % p - p) // p
    return min(x_percent * n // (100 * p), (a - a % 2) // 2 + 1)


def target_rbg_indices(nrb: int, x_percent: int) -> tuple[int, ...]:
    """0-based RBG indices {floor((N + (3 mod P)) / P) - 1} U {1, 3, ..., 2 (N_RBG - 2) + 1} (the odd set only if N_RBG > 1)."""
    n = nrb - CONTROL_PRBS
    p = rbg_size(n)
    count = n_target_rbg(nrb, x_percent)
    indices = {(n + CONTROL_PRBS % p) // p - 1}
    if count > 1:
        indices.update(range(1, 2 * (count - 2) + 2, 2))
    return tuple(sorted(indices))


def rbg_prbs(index: int, nrb: int) -> range:
    """PRBs of RBG `index`: the first RBG has P - (3 mod P) PRBs, the others P (the last one is clipped at N_RB)."""
    p = rbg_size(nrb - CONTROL_PRBS)
    first = p - CONTROL_PRBS % p
    if index == 0:
        return range(CONTROL_PRBS, CONTROL_PRBS + first)
    low = CONTROL_PRBS + first + (index - 1) * p
    return range(low, min(low + p, nrb))


def target_prbs(nrb: int, x_percent: int) -> np.ndarray:
    prbs = [prb for i in target_rbg_indices(nrb, x_percent) for prb in rbg_prbs(i, nrb)]
    return np.array(sorted(set(prbs)), dtype=int)


def balance_level_db(nrb: int, target_count: int, target_db: float) -> float:
    """Level of the complementary PRBs so that the PRB powers still sum to N_RB - 3:
    10 log10((N - 10^(L/10) T) / (N - T)), T = number of target PRBs (the spec's P N_RBG)."""
    n = nrb - CONTROL_PRBS
    return 10 * math.log10((n - 10 ** (target_db / 10) * target_count) / (n - target_count))


def tm2_prb(nrb: int, slot_in_frame: int) -> int:
    """Table 4.9.2.2.3-1: slot 3n -> PRB 0, 3n+1 -> floor(N_RB / 2), 3n+2 -> N_RB - 1 (slot numbered within the frame)."""
    return (0, nrb // 2, nrb - 1)[slot_in_frame % 3]


@dataclass(frozen=True)
class Allocation:
    """One user's PDSCH in one slot. sym_end is exclusive and equals the number of downlink symbols of the slot."""
    rnti: int
    qm: int
    prbs: np.ndarray
    sym_start: int
    sym_end: int
    power_db: float
    slot_class: int = 0      # TM2 family: slot index within the frame modulo 3 (the position of the single PRB)


def model_levels(model: ModelDef, nrb: int) -> dict[int, float]:
    """Level (dB) of every n_RNTI of the model relative to the unboosted resource element."""
    levels = {u.rnti: 0.0 for u in model.users}
    if model.structure == "rbg":
        count = len(target_prbs(nrb, model.x_percent))
        other = 0 if model.target_rnti == 1 else 1
        levels[model.target_rnti] = model.target_level_db
        levels[other] = balance_level_db(nrb, count, model.target_level_db)
    return levels


def slot_allocations(model: ModelDef, nrb: int, slot_in_frame: int, dl_symbols: int) -> list[Allocation]:
    """PDSCH allocations of a slot with `dl_symbols` leading downlink symbols (14 = full slot, 0 = nothing is sent)."""
    out: list[Allocation] = []

    def add(rnti, prbs, power_db=0.0, slot_class=0):
        user = model.user(rnti)
        if dl_symbols > user.start_symbol:
            out.append(Allocation(rnti, user.qm, np.asarray(prbs, dtype=int), user.start_symbol, dl_symbols, power_db, slot_class))

    levels = model_levels(model, nrb)
    if model.structure == "uniform":
        add(0, np.arange(CONTROL_PRBS, nrb))
        add(2, np.arange(0, CONTROL_PRBS))
    elif model.structure == "rbg":
        target = target_prbs(nrb, model.x_percent)
        taken = set(target.tolist())
        rest = np.array([prb for prb in range(CONTROL_PRBS, nrb) if prb not in taken], dtype=int)
        add(0, target if model.target_rnti == 0 else rest, levels[0])
        add(1, target if model.target_rnti == 1 else rest, levels[1])
        add(2, np.arange(0, CONTROL_PRBS))
    elif model.structure == "single_prb":
        add(2, [tm2_prb(nrb, slot_in_frame)], 0.0, slot_in_frame % 3)
    else:
        raise ValueError(model.structure)
    return out


def pdsch_data_res(allocation: Allocation) -> int:
    """PDSCH data resource elements per layer: allocated REs minus the DM-RS comb of the DM-RS symbols (port 1000, type 1)."""
    n_sc = 12 * len(allocation.prbs)
    dm = set(dmrs_symbols(allocation.sym_end))
    return sum(n_sc // 2 if l in dm else n_sc for l in range(allocation.sym_start, allocation.sym_end))


def dl_symbols_in_slot(duplex: str, pattern: TddPattern | None, slot: int) -> int:
    """Leading downlink symbols of absolute slot `slot` (FDD: 14; TDD: from the 5 ms pattern that repeats every period)."""
    if duplex == "fdd":
        return 14
    return pattern.dl_symbols_of_slot(slot % pattern.slots)


def prb_ranges(prbs: np.ndarray) -> list[list[int]]:
    """Inclusive [first, last] runs of consecutive PRBs."""
    if not len(prbs):
        return []
    p = np.sort(np.asarray(prbs, dtype=int))
    cuts = np.flatnonzero(np.diff(p) > 1)
    starts = np.r_[p[0], p[cuts + 1]]
    stops = np.r_[p[cuts], p[-1]]
    return [[int(a), int(b)] for a, b in zip(starts, stops)]


def reference_slot_allocations(model_id: str, nrb: int) -> list[list[Allocation]]:
    """Allocations of the reference full-DL slots: one entry for most models, the three position classes for TM2*."""
    model = MODEL_BY_ID[model_id]
    if model.structure == "single_prb":
        return [slot_allocations(model, nrb, c, 14) for c in range(3)]
    return [slot_allocations(model, nrb, 0, 14)]
