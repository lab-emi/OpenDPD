# Instrument adapters and the fail-closed interlock (S16)

`opendpd/instruments/` is the automatic control path of plan S16: play an
exported signal through a generator, capture the PA output with an analyser,
and hand the files to the same import that scores manual captures. Only the
**mock** adapter ships. A real adapter is written for one laboratory chain at a
time, by a person who can see that chain, and is verified there (§4); nothing
in this package lifts a limit on its own.

## 1. Contract (`base.py`)

An `Instrument` is one generator + analyser pair:

| Method | Meaning | Rule |
|---|---|---|
| `connect()` / `disconnect()` | open/close the sessions | `disconnect` turns the output off |
| `heartbeat()` | prove the link is alive | raises `LinkLost` otherwise |
| `rf_off()` | output off | idempotent; **never raises**; safe at any moment |
| `play(signal, fs)` | loop a complex baseband signal at digital full scale 1.0 | called only by the interlock |
| `capture(n, fs)` | `n` complex samples of the analyser output | may raise `InstrumentError` |

`AdapterInfo` declares `kind` (`mock` or `real`), `rf_output_capable` and the
adapter's `default_limits`. The mock declares `rf_output_capable = False`.

## 2. Interlock (`safety.py`)

```
disarmed ──arm(operator)──▶ armed ──play/capture──▶ active ──done──▶ armed ──disarm──▶ disarmed
    ▲                                                   │
    └─────────── (never from tripped) ◀── tripped ◀─────┘  limit, timeout, lost link, exception, abort, block exit
```

- **RF output is off by default.** Nothing plays until a named operator arms
  the session; an empty name is refused.
- **Real output needs the environment gate**: an adapter with
  `rf_output_capable = True` arms only when `OPENDPD_ALLOW_RF_OUTPUT=1` is set
  in the process environment. Continuous integration never sets it, and
  AGENTS.md forbids agents from setting it. The mock follows the same arming
  rule without the gate, so the whole procedure is exercised in CI.
- **Limits are checked before anything is sent**: peak |signal| against
  `max_peak_abs`, the requested output power against `max_output_power_dbm`,
  empty or non-finite signals.
- **Every guarded operation runs under a watchdog**: longer than `timeout_s`
  → `rf_off`, `tripped`; a `heartbeat` failure (`link_timeout_s`) → `rf_off`,
  `tripped`; any exception → `rf_off`, `tripped`; `abort()` → same.
- **A tripped interlock cannot be re-armed**; a new session is needed. Leaving
  the `with Interlock(...)` block always calls `rf_off` and `disconnect`,
  including on exceptions.
- The interlock keeps a timestamped log (armed, play, capture, rf_off,
  tripped with reason) that the session record stores.

Unit tests: `tests/unit/test_instruments.py` (timeout, lost link, adapter
failure, abort, limits, the environment gate, the block exit).

## 3. Session record (`session.py`)

`run_capture_session` arms, plays, captures, turns the output off and writes
`<out>.npy` (`(n, 2)` float32) plus `<out>.session.json`: adapter info, limits,
operator, timestamps, the played and captured hashes, the interlock log and
`mock: true|false`. `opendpd instruments dry-run` runs it twice (with `u`,
then with `x`) and writes a `conditions.json` template for the import.

## 4. Adding a real adapter (human-gated)

1. Implement `Instrument` for the chain (SCPI over VISA/socket is the usual
   route) with `kind="real"`, `rf_output_capable=True` and conservative
   `default_limits` (the generator's level ceiling, a short timeout).
2. Register it in `session.ADAPTERS`. CI must keep running only the mock:
   the environment gate guarantees that a real adapter cannot arm on a runner.
3. Verify on the bench with the operating procedure in
   `docs/protocols/measured-dpd.md` §6 and record the trial there. Until such
   a record exists, the S16 acceptance item "one instrument chain completed a
   supervised export → play → capture → import → evaluate trial" stays pending.
4. One successful chain is evidence for that chain only; the support matrix
   lists adapters individually.
