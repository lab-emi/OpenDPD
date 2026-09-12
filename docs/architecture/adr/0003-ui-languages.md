# ADR-0003: Nine UI languages with a per-workspace setting

- **Status:** accepted (maintainer decision, amended 2026-09-12)
- **Date:** 2026-09-11; expanded 2026-09-12
- **Plan step:** post-S20 (Studio internationalisation)
- **Deciders:** OpenDPD maintainers

## Context

ADR-0001 kept every user-facing string in `frontend/src/i18n/en.json` so a
later locale would not touch components. The maintainer asked for English,
French, German, Spanish, Chinese, Japanese and Korean, with a selector that
users cannot miss, marked with the flag and the native name. On September 12,
the maintainer added Dutch and Italian, prioritized English/Dutch/Chinese,
and requested localization of generated explanations as well as the interface.

## Decision

| Concern | Decision | Notes |
|---|---|---|
| Languages | `en nl zh fr de it ja ko es`; Chinese tag `zh-CN` | English, Dutch, Chinese first; the rest alphabetized by English language name; native names never translated |
| Catalogues | one flat JSON per language under `frontend/src/i18n`, exactly the keys of `en.json`, typed and tested for completeness and placeholders; non-English catalogues are lazy chunks | `t()` keeps its signature |
| Selector | `LanguageMenu` in the top bar of every page and in Settings; flag (bundled SVG, decorative) plus native name | emoji flags rejected: Windows renders letters |
| Persistence | `<workspace>/settings.json` through `GET/PUT /api/v1/settings` (`WorkspaceSettings`) | latest selection wins overlapping catalogue loads; settings mutations are serialized; startup reads the preference once |
| First run | stored value → English; nothing written until the user chooses (updated 2026-09-13) | |
| Formatting | counts, sizes and dates follow the language; metric values keep the decimal point of CSV and reports | |
| Generated explanations | `opendpd/studio/locales/*.json` is shared by the UI and report renderer; exact application phrases and named templates are translated only when displayed | covers diagnostics, live phases, registry help, plots and report prose; captured numbers and identifiers stay exact |
| Downloads | report and package endpoints accept an optional allowlisted `language`; otherwise use the workspace preference, then English | localized HTML/Markdown and plot legends; CLI/Python report calls default to English |
| Source records | configuration JSON, commands, user names, logs and GitHub commit messages retain their original text | technical records remain reproducible; original log/commit text is identified in the UI; unknown third-party text is never guessed |
| Native shell | dialog and menu strings per language in `opendpd/studio/strings.py`, read from the same setting | |

## Alternatives considered

| Option | Why not |
|---|---|
| Rewriting stored records in the selected language | would change provenance and scientific records; translate their presentation instead, using shared catalogues and the existing compute services |
| `localStorage` per browser | lost in the private native window, different per browser, invisible to the shell |
| A pluralisation library | 500 strings with a handful of counts; parenthesised plurals keep the catalogues flat |

## Consequences

- Every new interface or registered generated string needs nine entries; catalogue and placeholder tests fail otherwise.
- Long German and French strings must fit the 1366×768 layout (checked by the layout journey).
- Translations are maintained in the repository; corrections are ordinary PRs.

## Verification

`frontend/src/i18n/catalogues.test.ts`, `index.test.ts`, `LanguageMenu.test.tsx`, `LanguageGate.test.tsx`; `frontend/e2e/journey.spec.ts` (selector, reload, layout in `de` / `fr`); `tests/integration/test_studio_api.py` (settings routes); `tests/unit/test_window.py` (native strings).

The nine-language extension adds `frontend/src/i18n/messages.test.ts`,
`tests/unit/test_localization.py`, and real PA/DPD report/API/export checks in
`tests/integration/test_packages.py`. Browser evidence is recorded in
`docs/baseline/studio-nine-languages-2026-09-12.md`.
