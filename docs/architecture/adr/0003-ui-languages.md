# ADR-0003: Seven UI languages with a per-workspace setting

- **Status:** accepted (maintainer decision, 2026-09-11)
- **Date:** 2026-09-11
- **Plan step:** post-S20 (Studio internationalisation)
- **Deciders:** OpenDPD maintainers

## Context

ADR-0001 kept every user-facing string in `frontend/src/i18n/en.json` so a
later locale would not touch components. The maintainer asked for English,
French, German, Spanish, Chinese, Japanese and Korean, with a selector that
users cannot miss, marked with the flag and the native name.

## Decision

| Concern | Decision | Notes |
|---|---|---|
| Languages | `en fr de es zh ja ko`; tags `en fr de es zh-CN ja ko` | native names, never translated |
| Catalogues | one flat JSON per language under `frontend/src/i18n`, exactly the keys of `en.json`, typed and tested for completeness and placeholders; non-English catalogues are lazy chunks | `t()` keeps its signature |
| Selector | `LanguageMenu` in the top bar of every page and in Settings; flag (bundled SVG, decorative) plus native name | emoji flags rejected: Windows renders letters |
| Persistence | `<workspace>/settings.json` through `GET/PUT /api/v1/settings` (`WorkspaceSettings`) | not `localStorage`: the native window is private, and the shell reads the same file |
| First run | stored value → browser languages → English; nothing written until the user chooses | |
| Formatting | counts, sizes and dates follow the language; metric values keep the decimal point of CSV and reports | |
| Server text | error messages, hints, Dataset Doctor findings, reports, logs and CLI output stay English | one wording across GUI, CLI and packages |
| Native shell | dialog and menu strings per language in `opendpd/studio/strings.py`, read from the same setting | |

## Alternatives considered

| Option | Why not |
|---|---|
| Translating server messages | duplicates scientific wording per language and breaks the GUI/CLI/package parity the plan requires |
| `localStorage` per browser | lost in the private native window, different per browser, invisible to the shell |
| A pluralisation library | 500 strings with a handful of counts; parenthesised plurals keep the catalogues flat |

## Consequences

- Every new string needs seven entries; the catalogue test fails otherwise.
- Long German and French strings must fit the 1366×768 layout (checked by the layout journey).
- Translations are maintained in the repository; corrections are ordinary PRs.

## Verification

`frontend/src/i18n/catalogues.test.ts`, `index.test.ts`, `LanguageMenu.test.tsx`, `LanguageGate.test.tsx`; `frontend/e2e/journey.spec.ts` (selector, reload, layout in `de` / `fr`); `tests/integration/test_studio_api.py` (settings routes); `tests/unit/test_window.py` (native strings).
