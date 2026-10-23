"""Resource grid of one test-model carrier: PDCCH, PDSCH of every user and DM-RS, slot by slot.

``build_carrier_grid`` returns a K x L matrix (K = 12 N_RB subcarriers from the lowest, L = 14 x slots OFDM symbols) at
unit resource-element amplitude, plus the mask of PDSCH data resource elements used as the EVM reference.

Two readings of the PN23 rules of TS 38.141-1 4.9.2.3 are implemented (``data_convention``):

* ``"standard"``: the text. One PN23 stream per user (n_RNTI) and one for the PDCCH, every stream restarted from the
  all-ones state in the first allocated slot of each radio frame and continuous over the slot boundaries. For TDD the
  stream is drawn only for downlink symbols (``pn_scope="dl_symbols"``) or as if every slot were a full downlink slot,
  the uplink and special-slot remainder being discarded (``pn_scope="all_symbols"``).
* ``"matlab"``: what MATLAB 5G Toolbox does, verified bit-exact against R2026a: one stream per PDSCH allocation object
  (user x slot class x full/special slot), one for the PDCCH, started once and never restarted. ``pn_scope`` is ignored.

In a single 10 ms FDD frame of TM1.1 / TM1.2 / TM3.x the two conventions give identical grids.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .sequences import PnStreams, dmrs_pdcch, dmrs_pdsch, gold, qam_map, scrambling_cinit_pdcch, scrambling_cinit_pdsch
from .spec import Allocation, dl_symbols_in_slot, pdsch_data_res, slot_allocations
from .tables import MU, TDD_PATTERNS, ModelDef, dmrs_symbols

# Hook for in-band overlays (NB-IoT in NR-N-TM). Called once per slot, after the NR content of the slot is written, as
# ``overlay(slot_grid, slot_in_frame)`` where ``slot_grid`` is a writable K x 14 copy of the exported port's slot. It returns
# either the modified K x 14 grid or ``(grid, excluded)`` with ``excluded`` a K x 14 boolean mask of the resource elements
# that no longer carry NR PDSCH data (they are removed from the EVM reference).
Overlay = Callable[[np.ndarray, int], "np.ndarray | tuple[np.ndarray, np.ndarray]"]

_PDCCH_DMRS_SC = np.array([k for k in range(36) if k % 12 in (1, 5, 9)])
_PDCCH_DATA_SC = np.array([k for k in range(36) if k % 12 not in (1, 5, 9)])


@dataclass
class CarrierGrid:
    values: np.ndarray       # [K, L] complex128, exported port
    data_mask: np.ndarray    # [K, L] bool, PDSCH data resource elements (EVM reference)
    n_slots: int


def _pdsch_bits_per_slot(alloc: Allocation, layers: int) -> int:
    return layers * alloc.qm * pdsch_data_res(alloc)


def build_carrier_grid(model: ModelDef, nrb: int, scs_khz: int, duplex: str, n_cell_id: int, n_subframes: int,
                       data_convention: str = "standard", pn_scope: str = "dl_symbols", layers: int = 1,
                       output_port: int = 0, overlay: Overlay | None = None) -> CarrierGrid:
    mu = MU[scs_khz]
    spf = 10 * 2 ** mu                       # slots per radio frame
    n_slots = n_subframes * 2 ** mu
    k_total = 12 * nrb
    pattern = TDD_PATTERNS[scs_khz] if duplex == "tdd" else None
    standard = data_convention == "standard"
    all_symbols = standard and pn_scope == "all_symbols"
    grids = [np.zeros((k_total, 14 * n_slots), dtype=np.complex128) for _ in range(layers)]
    mask = np.zeros((k_total, 14 * n_slots), dtype=bool)
    pn = PnStreams()
    scrambling: dict[int, np.ndarray] = {}
    dmrs_cache: dict[tuple[int, int], np.ndarray] = {}

    def scrambling_bits(rnti: int, n: int) -> np.ndarray:
        have = scrambling.get(rnti)
        if have is None or len(have) < n:
            have = scrambling[rnti] = gold(scrambling_cinit_pdsch(rnti, n_cell_id), max(n, 2 * (0 if have is None else len(have))))
        return have[:n]

    def dmrs(slot_in_frame: int, symbol: int) -> np.ndarray:
        key = (slot_in_frame, symbol)
        if key not in dmrs_cache:
            dmrs_cache[key] = dmrs_pdsch(slot_in_frame, symbol, n_cell_id, 6 * nrb)   # m counted from CRB 0 (carrier edge)
        return dmrs_cache[key]

    pdcch_scrambling = gold(scrambling_cinit_pdcch(0, n_cell_id), 108)

    for s in range(n_slots):
        sf = s % spf
        if standard and sf == 0 and s > 0:
            pn.restart()
        n_dl = dl_symbols_in_slot(duplex, pattern, s)
        col0 = 14 * s
        if n_dl >= 2:
            bits = pn.take("pdcch", 108)
            symbols = qam_map(bits ^ pdcch_scrambling, 2)
            for li in range(2):                        # TS 38.211 7.3.2.5: k first (27 data REs per symbol), then l
                grids[0][_PDCCH_DATA_SC, col0 + li] = symbols[li * 27:(li + 1) * 27]
                grids[0][_PDCCH_DMRS_SC, col0 + li] = dmrs_pdcch(sf, li, n_cell_id, 9)   # r(3n + k'), n = PRB 0..2
        allocations = slot_allocations(model, nrb, sf, n_dl)
        drawn: dict[int, np.ndarray] = {}
        if all_symbols:
            for full in slot_allocations(model, nrb, sf, 14):
                drawn[full.rnti] = pn.take(full.rnti, _pdsch_bits_per_slot(full, layers))
        for alloc in allocations:
            n_bits = _pdsch_bits_per_slot(alloc, layers)
            if all_symbols:
                bits = drawn[alloc.rnti][:n_bits]
            elif standard:
                bits = pn.take(alloc.rnti, n_bits)
            else:
                bits = pn.take((alloc.rnti, alloc.slot_class, alloc.sym_end), n_bits)
            _write_pdsch(grids, mask, alloc, bits ^ scrambling_bits(alloc.rnti, n_bits), s, sf, nrb, layers, dmrs)
        if overlay is not None and n_dl:
            sl = slice(col0, col0 + 14)
            result = overlay(grids[output_port][:, sl].copy(), sf)
            new, excluded = result if isinstance(result, tuple) else (result, None)
            grids[output_port][:, sl] = new
            if excluded is not None:
                mask[:, sl] &= ~excluded
    return CarrierGrid(grids[output_port], mask, n_slots)


def _write_pdsch(grids, mask, alloc: Allocation, scrambled: np.ndarray, slot: int, slot_in_frame: int, nrb: int,
                 layers: int, dmrs) -> None:
    sc = (alloc.prbs[:, None] * 12 + np.arange(12)[None, :]).reshape(-1)           # allocated subcarriers, ascending
    symbols = np.arange(alloc.sym_start, alloc.sym_end)
    dm_symbols = set(dmrs_symbols(alloc.sym_end))
    is_dmrs_symbol = np.fromiter((l in dm_symbols for l in symbols), dtype=bool, count=len(symbols))
    comb = sc % 2 == 0                                                                # port 1000, type 1: even subcarriers
    data = ~(is_dmrs_symbol[:, None] & comb[None, :])                                 # [symbol, subcarrier], data REs
    amp = 10 ** (alloc.power_db / 20)
    modulated = qam_map(scrambled, alloc.qm) * amp
    columns = 14 * slot + symbols
    comb_sc = sc[comb]
    for layer in range(layers):
        block = np.zeros((len(symbols), len(sc)), dtype=np.complex128)
        block[data] = modulated[layer::layers]                                        # TS 38.211 7.3.1.3: x(i)(j) = d(layers j + i)
        for row in np.flatnonzero(is_dmrs_symbol):
            r = dmrs(slot_in_frame, int(symbols[row]))[comb_sc // 2] * amp            # DM-RS follows the PDSCH level
            if layer == 1:
                r = r * np.where((comb_sc // 2) % 2 == 0, 1.0, -1.0)                  # port 1001: w_f(k') = (+1, -1), TS 38.211 table 7.4.1.1.2-1
            block[row, comb] = r
        grids[layer][sc[:, None], columns[None, :]] = block.T
    mask[sc[:, None], columns[None, :]] = data.T
