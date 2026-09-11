# Seven UI Languages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** The Studio workbench runs in English, French, German, Spanish, Chinese, Japanese and Korean, with a flag-and-name selector in the top bar and the choice stored per workspace.

**Architecture:** The server keeps a per-workspace `settings.json` behind `GET/PUT /api/v1/settings`. The frontend's `t()` stays synchronous over an active catalogue held in a tiny external store; catalogues other than English are lazy chunks; the language gate resolves stored → browser → English before the shell renders, and `App` re-renders the route tree with an MUI theme merged with the matching locale pack. The desktop shell reads the same setting for its native strings.

**Tech Stack:** Pydantic v2, FastAPI, React 19, TanStack Query 5, MUI 9 (`@mui/material/locale`), Vite dynamic imports, Vitest + Testing Library, Playwright, pytest.

Spec: `docs/superpowers/specs/2026-09-11-ui-languages-design.md`. Depends on the native window plan (`2026-09-11-native-window-shell.md`) for Task 4 only.

## Global Constraints

- Code comments, docstrings, commit messages and documentation are in English; catalogues hold the translated UI text only.
- Language codes: `en`, `fr`, `de`, `es`, `zh`, `ja`, `ko`; BCP-47 tags `en`, `fr`, `de`, `es`, `zh-CN`, `ja`, `ko`; native names `English`, `Français`, `Deutsch`, `Español`, `中文`, `日本語`, `한국어`; flags `gb fr de es cn jp kr` (flag-icons 7.5.0, MIT; `es` hand-drawn three stripes).
- Server-generated text stays English. Identifiers, metric abbreviations, `MOCK`, `OpenDPD Studio`, `Dataset Doctor`, placeholders `{name}` and inline code in backticks are kept verbatim in every catalogue.
- Metric values keep `toFixed` with a decimal point; counts, sizes and dates use the UI language.
- `t(key, vars)` keeps its signature; no component is rewritten to a hook for translation.
- Every commit ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- After any schema or route change: `.venv/bin/python scripts/export_openapi.py` then `cd frontend && npm run types:generate`; after any example change: `.venv/bin/python -m opendpd.schemas export-mocks --out frontend/mocks`.

---

## File structure

| File | Responsibility |
|---|---|
| `opendpd/schemas/settings.py` (new), `opendpd/schemas/__init__.py`, `opendpd/schemas/examples.py` | `WorkspaceSettings` contract and its example |
| `opendpd/services/workspace.py` | `settings()` / `save_settings()` over `<workspace>/settings.json` |
| `opendpd/server/routes.py` | `GET/PUT /settings` |
| `frontend/src/i18n/index.ts` (rewrite), `frontend/src/i18n/{fr,de,es,zh,ja,ko}.json` (new), `frontend/src/i18n/LanguageGate.tsx` (new), `frontend/src/assets/flags/*` (new) | language store, catalogues, boot-time resolution, flags |
| `frontend/src/components/LanguageMenu.tsx` (new) | the selector (top bar and Settings) |
| `frontend/src/App.tsx`, `frontend/src/theme/index.ts`, `frontend/src/layout/AppShell.tsx`, `frontend/src/pages/SettingsPage.tsx`, `frontend/src/api/{client,hooks,types}.ts` | wiring: theme per language, gate, menu placement, settings hooks, `api.put` |
| the 13 files with `toLocaleString()` / `toLocaleTimeString()` | `formatNumber` / `formatDateTime` / `formatTime` |
| `frontend/e2e/mock-api.ts`, `frontend/e2e/journey.spec.ts` | fake settings endpoint, language journey, layout in `de` and `fr` |
| `opendpd/studio/strings.py`, `opendpd/studio/launcher.py` | native strings in seven languages, `preferred_language(workspace)` |
| `docs/architecture/adr/0003-ui-languages.md` (new), `docs/architecture/ux-spec.md`, `docs/tutorials/gui-quickstart.md`, `README.md`, `docs/releases/release-notes-2.2.0.md`, `OpenDPD_Studio_Development_Plan.md` | decision record and user documentation |

---

### Task 1: Workspace settings contract, storage and API

**Files:**
- Create: `opendpd/schemas/settings.py`
- Modify: `opendpd/schemas/__init__.py`, `opendpd/schemas/examples.py` (`all_examples`), `opendpd/services/workspace.py`, `opendpd/server/routes.py`
- Regenerate: `frontend/mocks/workspace_settings_default.json`, `docs/contracts/openapi.json`, `frontend/src/api/schema.ts`
- Test: `tests/unit/test_workspace.py`, `tests/integration/test_studio_api.py`

**Interfaces:**
- Produces: `UILanguage = Literal["en","fr","de","es","zh","ja","ko"]`, `UI_LANGUAGES: Tuple[str, ...]`, `WorkspaceSettings(language: Optional[UILanguage] = None)` (extra fields forbidden); `Workspace.settings_path: Path`, `Workspace.settings() -> WorkspaceSettings`, `Workspace.save_settings(settings) -> WorkspaceSettings`; routes `GET /api/v1/settings` (session) and `PUT /api/v1/settings` (CSRF) returning `WorkspaceSettings`; example name `workspace_settings_default`.

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_workspace.py`:

```python
from opendpd.schemas import WorkspaceSettings


def test_settings_default_roundtrip_and_corruption(tmp_path):
    ws = Workspace.create(tmp_path / "ws")
    assert ws.settings().language is None and not ws.settings_path.exists()
    ws.save_settings(WorkspaceSettings(language="ja"))
    assert Workspace.open(ws.root).settings().language == "ja"
    assert json.loads(ws.settings_path.read_text())["language"] == "ja"
    ws.settings_path.write_text("{not json")
    with pytest.raises(WorkspaceError, match="settings.json"):
        ws.settings()
    ws.settings_path.write_text('{"language": "xx"}')
    with pytest.raises(WorkspaceError, match="settings.json"):
        ws.settings()
```

Append to `tests/integration/test_studio_api.py` (after the session tests):

```python
def test_settings_default_roundtrip_validation_and_csrf(client, session):
    from pathlib import Path
    assert client.get("/api/v1/settings").json() == {"language": None}
    r = client.put("/api/v1/settings", json={"language": "de"})
    assert r.status_code == 200 and r.json() == {"language": "de"}
    stored = json.loads((Path(client.app.state.ws.root) / "settings.json").read_text())
    assert stored == {"language": "de"}
    assert client.get("/api/v1/settings").json()["language"] == "de"
    r = client.put("/api/v1/settings", json={"language": "xx"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    r = client.put("/api/v1/settings", json={"language": "fr", "theme": "dark"})
    assert r.status_code == 422, "unknown settings are refused, not ignored"
    bare = TestClient(client.app, base_url="http://127.0.0.1:8765")
    bare.cookies = client.cookies
    assert bare.put("/api/v1/settings", json={"language": "fr"}).status_code == 403
    assert client.put("/api/v1/settings", json={"language": None}).json() == {"language": None}
```

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_workspace.py -q -k settings` and `.venv/bin/python -m pytest tests/integration/test_studio_api.py -q -k settings`
Expected: FAIL (`ImportError: cannot import name 'WorkspaceSettings'`; 404 on `/api/v1/settings`).

- [x] **Step 3: Schema, example, service, routes**

`opendpd/schemas/settings.py`:

```python
"""Per-workspace preferences of the workbench (never scientific): what the GUI remembers between launches."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from .common import StrictModel

UILanguage = Literal["en", "fr", "de", "es", "zh", "ja", "ko"]
UI_LANGUAGES: Tuple[str, ...] = ("en", "fr", "de", "es", "zh", "ja", "ko")


class WorkspaceSettings(StrictModel):
    """Stored as ``<workspace>/settings.json``; ``language`` None means "follow the browser or system language"."""

    language: Optional[UILanguage] = None
```

`opendpd/schemas/__init__.py`: add `from .settings import UI_LANGUAGES, UILanguage, WorkspaceSettings` next to the other imports and the three names to `__all__`.

`opendpd/schemas/examples.py`: import `WorkspaceSettings`, add

```python
def workspace_settings_default() -> WorkspaceSettings:
    """No language stored yet: the client follows the browser or system language."""
    return WorkspaceSettings()
```

and the entry `"workspace_settings_default": workspace_settings_default(),` at the end of `all_examples()`.

`opendpd/services/workspace.py` (import `WorkspaceSettings` from `opendpd.schemas`; add after `preflight`):

```python
    # -- workbench preferences ----------------------------------------------
    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"

    def settings(self) -> WorkspaceSettings:
        """A missing file means defaults; a broken or unknown one is an error that names the file."""
        if not self.settings_path.exists():
            return WorkspaceSettings()
        try:
            return WorkspaceSettings.model_validate(read_json(self.settings_path))
        except (OSError, ValueError) as err:     # pydantic's ValidationError is a ValueError
            raise WorkspaceError(f"cannot read {self.settings_path}: {err}") from err

    def save_settings(self, settings: WorkspaceSettings) -> WorkspaceSettings:
        write_json_atomic(self.settings_path, settings)
        return settings
```

`opendpd/server/routes.py` (import `WorkspaceSettings`; add before the `# --- system` block):

```python
# --- workbench settings -----------------------------------------------------------

@router.get("/settings", response_model=WorkspaceSettings, tags=["settings"], dependencies=[Depends(require_session)])
def settings_get(request: Request):
    return _ws(request).settings()


@router.put("/settings", response_model=WorkspaceSettings, tags=["settings"], dependencies=[Depends(require_csrf)])
def settings_put(body: WorkspaceSettings, request: Request):
    return _ws(request).save_settings(body)
```

Then regenerate: `.venv/bin/python -m opendpd.schemas export-mocks --out frontend/mocks`, `.venv/bin/python scripts/export_openapi.py`, `cd frontend && npm run types:generate`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_workspace.py tests/unit/test_frontend_mocks_in_sync.py tests/unit/test_schemas.py -q` and `.venv/bin/python -m pytest tests/integration/test_studio_api.py -q -k "settings or session"` and `.venv/bin/python scripts/export_openapi.py --check` and `cd frontend && npm run types:check`
Expected: all PASS / up to date.

- [x] **Step 5: Commit**

```bash
git add opendpd/schemas/settings.py opendpd/schemas/__init__.py opendpd/schemas/examples.py opendpd/services/workspace.py opendpd/server/routes.py frontend/mocks/workspace_settings_default.json docs/contracts/openapi.json frontend/src/api/schema.ts tests/unit/test_workspace.py tests/integration/test_studio_api.py
git commit -m "feat(studio): per-workspace settings with a UI language, GET/PUT /api/v1/settings

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Language store, flags and the six catalogues

**Files:**
- Create: `frontend/src/assets/flags/{gb,fr,de,es,cn,jp,kr}.svg`, `frontend/src/assets/flags/LICENSE-flag-icons.txt`
- Rewrite: `frontend/src/i18n/index.ts`
- Modify: `frontend/src/i18n/en.json` (four new keys)
- Create: `frontend/src/i18n/{fr,de,es,zh,ja,ko}.json`
- Test: `frontend/src/i18n/catalogues.test.ts`, `frontend/src/i18n/index.test.ts`

**Interfaces:**
- Produces (from `@/i18n`): `MessageKey`, `Catalogue = Record<MessageKey, string>`, `LanguageCode`, `Language { code, tag, name, flag }`, `LANGUAGES: readonly Language[]`, `DEFAULT_LANGUAGE`, `t(key, vars?)`, `getLanguage()`, `useLanguage()`, `languageInfo(code?)`, `isLanguageCode(x)`, `resolveLanguage(stored, navigatorLanguages?)`, `loadCatalogue(code)`, `setLanguage(code): Promise<void>`, `resetLanguage()` (tests), `formatNumber(value, options?)`, `formatDateTime(value)`, `formatTime(value)`.
- New English keys: `language.label` = "Language", `language.saveFailed` = "The language could not be saved to the workspace: {error}. It stays selected for this session.", `settings.language` = "Language", `settings.language.note` = "Applies at once and is remembered in this workspace. Server messages, worker logs and reports stay in English."

- [x] **Step 1: Write the failing tests**

`frontend/src/i18n/catalogues.test.ts`:

```ts
import { describe, expect, test } from 'vitest'
import type { Catalogue } from './index'
import de from './de.json'
import en from './en.json'
import es from './es.json'
import fr from './fr.json'
import ja from './ja.json'
import ko from './ko.json'
import zh from './zh.json'

// Compile-time completeness: a catalogue missing a key does not typecheck.
const typed: Catalogue[] = [fr, de, es, zh, ja, ko]
const all: Record<string, Record<string, string>> = { fr, de, es, zh, ja, ko }
const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()
const code = (s: string) => [...s.matchAll(/`[^`]+`/g)].map((m) => m[0]).sort()

describe.each(Object.entries(all))('catalogue %s', (_code, catalogue) => {
  test('has exactly the English keys', () => {
    expect(typed.length).toBe(6)
    expect(Object.keys(catalogue).sort()).toEqual(Object.keys(en).sort())
  })
  test('keeps every placeholder and inline code, and has no empty string', () => {
    for (const [key, value] of Object.entries(en)) {
      const translated = catalogue[key] ?? ''
      expect(placeholders(translated), key).toEqual(placeholders(value))
      expect(code(translated), key).toEqual(code(value))
      expect(translated.trim().length, key).toBeGreaterThan(0)
    }
  })
  test('keeps the markers that must never be translated', () => {
    expect(catalogue['evidence.mock']).toBe('MOCK')
    expect(catalogue['app.title']).toBe('OpenDPD Studio')
  })
})
```

`frontend/src/i18n/index.test.ts`:

```ts
import { afterEach, expect, test } from 'vitest'
import { LANGUAGES, formatDateTime, formatNumber, getLanguage, resetLanguage, resolveLanguage, setLanguage, t } from './index'

afterEach(() => resetLanguage())

test('seven languages with tags, native names and flags', () => {
  expect(LANGUAGES.map((l) => l.code)).toEqual(['en', 'fr', 'de', 'es', 'zh', 'ja', 'ko'])
  expect(LANGUAGES.map((l) => l.name)).toEqual(['English', 'Français', 'Deutsch', 'Español', '中文', '日本語', '한국어'])
  expect(LANGUAGES.find((l) => l.code === 'zh')?.tag).toBe('zh-CN')
  for (const l of LANGUAGES) expect(l.flag, l.code).toMatch(/\.svg$|^data:image\/svg\+xml/)
})

test('resolution: stored value, then the browser languages, then English', () => {
  expect(resolveLanguage('ja', ['fr-CA'])).toBe('ja')
  expect(resolveLanguage(null, ['fr-CA', 'en-US'])).toBe('fr')
  expect(resolveLanguage(null, ['pt-BR', 'zh-TW'])).toBe('zh')
  expect(resolveLanguage(null, ['pt-BR'])).toBe('en')
  expect(resolveLanguage('xx', [])).toBe('en')
})

test('switching the language translates t() and updates <html lang>', async () => {
  expect(t('nav.datasets')).toBe('Datasets')
  await setLanguage('de')
  expect(getLanguage()).toBe('de')
  expect(t('nav.datasets')).toBe('Datensätze')
  expect(document.documentElement.lang).toBe('de')
  expect(t('topbar.nowRunning', { count: 2 })).toContain('2')
})

test('numbers and dates follow the UI language; English is the default', async () => {
  expect(formatNumber(1234567)).toBe('1,234,567')
  await setLanguage('de')
  expect(formatNumber(1234567)).toBe('1.234.567')
  expect(formatDateTime('2026-09-11T10:30:00Z')).toMatch(/2026/)
})
```

- [x] **Step 2: Run them to verify they fail**

Run: `cd frontend && npx vitest run src/i18n`
Expected: FAIL (missing catalogue modules and exports).

- [x] **Step 3: Flags**

Copy `gb.svg`, `fr.svg`, `de.svg`, `cn.svg`, `jp.svg`, `kr.svg` from the flag-icons 7.5.0 package (`flags/4x3/`) into `frontend/src/assets/flags/`. Write `es.svg` by hand:

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 480"><path fill="#AA151B" d="M0 0h640v480H0z"/><path fill="#F1BF00" d="M0 120h640v240H0z"/></svg>
```

Write `LICENSE-flag-icons.txt` with the MIT text of flag-icons (copyright Panayiotis Lipiridis) and the line "Files gb.svg, fr.svg, de.svg, cn.svg, jp.svg, kr.svg come from https://github.com/lipis/flag-icons (7.5.0); es.svg is drawn here (civil flag, no coat of arms)."

- [x] **Step 4: Rewrite `frontend/src/i18n/index.ts`**

```ts
/**
 * User-facing strings and the active UI language.
 *
 * `t()` stays synchronous: it reads the catalogue of the active language and
 * falls back to English per key. Non-English catalogues are lazy chunks. The
 * active language is a tiny external store so `App` can re-render the route
 * tree (and pick the MUI locale pack) when it changes.
 */
import { useSyncExternalStore } from 'react'
import cn from '@/assets/flags/cn.svg'
import de from '@/assets/flags/de.svg'
import es from '@/assets/flags/es.svg'
import fr from '@/assets/flags/fr.svg'
import gb from '@/assets/flags/gb.svg'
import jp from '@/assets/flags/jp.svg'
import kr from '@/assets/flags/kr.svg'
import en from './en.json'

export type MessageKey = keyof typeof en
export type Catalogue = Record<MessageKey, string>
export type LanguageCode = 'en' | 'fr' | 'de' | 'es' | 'zh' | 'ja' | 'ko'

export interface Language {
  code: LanguageCode
  /** BCP-47 tag for `<html lang>` and Intl formatting. */
  tag: string
  /** Native name; never translated. */
  name: string
  /** Flag image URL (bundled SVG). */
  flag: string
}

export const LANGUAGES: readonly Language[] = [
  { code: 'en', tag: 'en', name: 'English', flag: gb },
  { code: 'fr', tag: 'fr', name: 'Français', flag: fr },
  { code: 'de', tag: 'de', name: 'Deutsch', flag: de },
  { code: 'es', tag: 'es', name: 'Español', flag: es },
  { code: 'zh', tag: 'zh-CN', name: '中文', flag: cn },
  { code: 'ja', tag: 'ja', name: '日本語', flag: jp },
  { code: 'ko', tag: 'ko', name: '한국어', flag: kr },
]
export const DEFAULT_LANGUAGE: LanguageCode = 'en'

// A catalogue missing a key of en.json fails to typecheck here.
const loaders: Record<Exclude<LanguageCode, 'en'>, () => Promise<{ default: Catalogue }>> = {
  fr: () => import('./fr.json'),
  de: () => import('./de.json'),
  es: () => import('./es.json'),
  zh: () => import('./zh.json'),
  ja: () => import('./ja.json'),
  ko: () => import('./ko.json'),
}

const catalogues: Partial<Record<LanguageCode, Catalogue>> = { en }
let current: LanguageCode = DEFAULT_LANGUAGE
const listeners = new Set<() => void>()

export function isLanguageCode(value: unknown): value is LanguageCode {
  return LANGUAGES.some((l) => l.code === value)
}

export function getLanguage(): LanguageCode {
  return current
}

export function languageInfo(code: LanguageCode = current): Language {
  return LANGUAGES.find((l) => l.code === code) ?? LANGUAGES[0]!
}

/** Stored choice → first supported browser language → English. */
export function resolveLanguage(stored: string | null | undefined, navigatorLanguages: readonly string[] = navigator.languages ?? [navigator.language]): LanguageCode {
  if (isLanguageCode(stored)) return stored
  for (const tag of navigatorLanguages) {
    const base = tag.toLowerCase().split('-')[0]
    if (isLanguageCode(base)) return base
  }
  return DEFAULT_LANGUAGE
}

export async function loadCatalogue(code: LanguageCode): Promise<Catalogue> {
  const cached = catalogues[code]
  if (cached) return cached
  const loaded = (await loaders[code as Exclude<LanguageCode, 'en'>]()).default
  catalogues[code] = loaded
  return loaded
}

function notify(): void {
  document.documentElement.lang = languageInfo(current).tag
  for (const listener of listeners) listener()
}

/** Loads the catalogue if needed, then switches every `t()` call and re-renders subscribers. */
export async function setLanguage(code: LanguageCode): Promise<void> {
  await loadCatalogue(code)
  if (code === current) return
  current = code
  notify()
}

/** Back to English synchronously (English is always loaded); for tests. */
export function resetLanguage(): void {
  if (current === DEFAULT_LANGUAGE) return
  current = DEFAULT_LANGUAGE
  notify()
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useLanguage(): LanguageCode {
  return useSyncExternalStore(subscribe, getLanguage, getLanguage)
}

export function t(key: MessageKey, vars: Record<string, string | number> = {}): string {
  const template: string = catalogues[current]?.[key] ?? en[key]
  return template.replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? `{${name}}`))
}

export function formatNumber(value: number, options?: Intl.NumberFormatOptions): string {
  return value.toLocaleString(languageInfo().tag, options)
}

export function formatDateTime(value: string | number | Date): string {
  return new Date(value).toLocaleString(languageInfo().tag)
}

export function formatTime(value: string | number | Date): string {
  return new Date(value).toLocaleTimeString(languageInfo().tag)
}
```

Add the four English keys to `en.json` (after `"settings.contract"` for the settings ones; `language.*` after `"topbar.idle"`).

- [x] **Step 5: Write the six catalogues**

For each of `fr`, `de`, `es`, `zh`, `ja`, `ko`: a flat JSON object with every key of `en.json`, in the same order, translated according to the spec's terminology table and constraints. Style samples that every catalogue follows:

| key | fr | de | zh |
|---|---|---|---|
| `nav.datasets` | Jeux de données | Datensätze | 数据集 |
| `topbar.nowRunning` | {count} en cours | {count} laufen | {count} 个运行中 |
| `state.error.retry` | Réessayer | Erneut versuchen | 重试 |
| `status.cancel_requested` | Arrêt en cours… | Wird gestoppt… | 正在停止… |
| `evidence.dpd_surrogate` | DPD · substitut | DPD · Surrogat | DPD · 替代模型 |
| `session.token.invalid` | Le jeton n'est pas valide pour le serveur en cours. Redémarrez `opendpd gui` et utilisez l'URL qu'il affiche. | Das Token gilt nicht für den laufenden Server. Starten Sie `opendpd gui` neu und verwenden Sie die ausgegebene URL. | 该令牌对当前运行的服务器无效。请重新启动 `opendpd gui` 并使用其打印的 URL。 |

Japanese uses polite form (です・ます), Korean uses the formal polite ending (-습니다 / -하세요), Chinese is Simplified. Sentences keep the caveats of the English original; nothing is shortened into a friendlier claim.

- [x] **Step 6: Run the tests to verify they pass**

Run: `cd frontend && npx vitest run src/i18n && npm run typecheck && npm run lint`
Expected: PASS, no type errors, no lint errors.

- [x] **Step 7: Commit**

```bash
git add frontend/src/assets/flags frontend/src/i18n
git commit -m "feat(frontend): language store with lazy catalogues in seven languages

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Selector, gate, theme locale, formatting, journeys

**Files:**
- Modify: `frontend/src/api/client.ts` (PUT), `frontend/src/api/hooks.ts` (`keys.settings`, `useSettings`, `useUpdateSettings`), `frontend/src/api/types.ts` (`WorkspaceSettings`)
- Create: `frontend/src/i18n/LanguageGate.tsx`, `frontend/src/components/LanguageMenu.tsx`
- Modify: `frontend/src/theme/index.ts` (`themeFor`), `frontend/src/App.tsx`, `frontend/src/layout/AppShell.tsx`, `frontend/src/pages/SettingsPage.tsx`
- Modify (formatting): `ImportDatasetDialog.tsx`, `RunTimeline.tsx`, `PreprocessDialog.tsx`, `StateBlock.tsx`, `RobustnessPage.tsx`, `DatasetsPage.tsx`, `ResultDetailPage.tsx`, `ExperimentsPage.tsx`, `DatasetDetailPage.tsx`, `ResultsPage.tsx`
- Modify: `frontend/e2e/mock-api.ts`, `frontend/e2e/journey.spec.ts`
- Test: `frontend/src/components/LanguageMenu.test.tsx`, `frontend/src/i18n/LanguageGate.test.tsx`

**Interfaces:**
- Consumes: Task 2 exports; `WorkspaceSettings` from the generated schema (Task 1).
- Produces: `api.put<T>(path, body)`; `keys.settings = ['settings']`; `useSettings()`; `useUpdateSettings()` (mutation over `WorkspaceSettings`); `themeFor(code: LanguageCode): Theme`; `<LanguageGate>`; `<LanguageMenu variant?: 'toolbar' | 'settings' />` with `data-testid="language-menu"`.

- [x] **Step 1: Write the failing tests**

`frontend/src/components/LanguageMenu.test.tsx`:

```tsx
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, test } from 'vitest'
import { resetLanguage, t } from '@/i18n'
import { mockApi, renderWithProviders } from '@/test/utils'
import { LanguageMenu } from './LanguageMenu'

afterEach(() => resetLanguage())

test('lists seven languages with flags and names, switches at once and stores the choice', async () => {
  const { calls } = mockApi({ 'PUT /api/v1/settings': (_url, init) => JSON.parse(String(init.body)) })
  renderWithProviders(<LanguageMenu />)
  await userEvent.click(screen.getByRole('button', { name: 'Language' }))
  const items = screen.getAllByRole('menuitem')
  expect(items.map((i) => i.textContent)).toEqual(['English', 'Français', 'Deutsch', 'Español', '中文', '日本語', '한국어'])
  expect(items[2]?.querySelector('img')).toHaveAttribute('alt', '')
  await userEvent.click(screen.getByRole('menuitem', { name: 'Deutsch' }))
  await waitFor(() => expect(t('nav.datasets')).toBe('Datensätze'))
  expect(screen.getByRole('button', { name: 'Sprache' })).toHaveTextContent('Deutsch')
  await waitFor(() => expect(calls.filter((c) => c.method === 'PUT')).toEqual([{ method: 'PUT', path: '/api/v1/settings', body: { language: 'de' } }]))
})

test('a failed save keeps the language on screen and says so', async () => {
  mockApi({ 'PUT /api/v1/settings': () => ({ status: 500, body: { error: { code: 'workspace_error', message: 'disk full', details: [], hint: null } } }) })
  renderWithProviders(<LanguageMenu />)
  await userEvent.click(screen.getByRole('button', { name: 'Language' }))
  await userEvent.click(screen.getByRole('menuitem', { name: '日本語' }))
  await waitFor(() => expect(t('nav.datasets')).toBe('データセット'))
  expect(await screen.findByRole('alert')).toHaveTextContent('disk full')
})
```

`frontend/src/i18n/LanguageGate.test.tsx`:

```tsx
import { screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { getLanguage, resetLanguage } from '@/i18n'
import { LanguageGate } from '@/i18n/LanguageGate'
import { mockApi, renderWithProviders } from '@/test/utils'

afterEach(() => resetLanguage())

test('applies the stored language before rendering the children', async () => {
  mockApi({ 'GET /api/v1/settings': () => ({ language: 'ja' }) })
  renderWithProviders(<LanguageGate><p>ready</p></LanguageGate>)
  expect(await screen.findByText('ready')).toBeInTheDocument()
  expect(getLanguage()).toBe('ja')
  expect(document.documentElement.lang).toBe('ja')
})

test('without a stored language the browser language decides', async () => {
  vi.spyOn(navigator, 'languages', 'get').mockReturnValue(['fr-CA', 'en'])
  mockApi({ 'GET /api/v1/settings': () => ({ language: null }) })
  renderWithProviders(<LanguageGate><p>ready</p></LanguageGate>)
  expect(await screen.findByText('ready')).toBeInTheDocument()
  expect(getLanguage()).toBe('fr')
})

test('a settings failure shows the error state with retry, not a blank page', async () => {
  mockApi({ 'GET /api/v1/settings': () => ({ status: 500, body: { error: { code: 'workspace_error', message: 'cannot read settings.json', details: [], hint: null } } }) })
  renderWithProviders(<LanguageGate><p>ready</p></LanguageGate>)
  expect(await screen.findByText(/cannot read settings.json/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
})
```

- [x] **Step 2: Run them to verify they fail**

Run: `cd frontend && npx vitest run src/components/LanguageMenu.test.tsx src/i18n/LanguageGate.test.tsx`
Expected: FAIL (modules missing).

- [x] **Step 3: API client and hooks**

`client.ts`: change the union to `'GET' | 'POST' | 'PUT'` and add `put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),` to `api`.

`types.ts`: add `export type WorkspaceSettings = Schemas['WorkspaceSettings']`.

`hooks.ts`: add `settings: ['settings'] as const,` to `keys`, import `WorkspaceSettings`, and

```ts
export const useSettings = () => useQuery({ queryKey: keys.settings, queryFn: () => api.get<WorkspaceSettings>('/settings'), staleTime: Infinity })

/** Full replacement of the workbench settings; the response is the stored state. */
export function useUpdateSettings() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (settings: WorkspaceSettings) => api.put<WorkspaceSettings>('/settings', settings),
    onSuccess: (saved) => qc.setQueryData(keys.settings, saved),
  })
}
```

- [x] **Step 4: Gate, theme and App**

`frontend/src/i18n/LanguageGate.tsx`:

```tsx
import { useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useSettings } from '@/api/hooks'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { resolveLanguage, setLanguage } from '@/i18n'

/**
 * Applies the workspace's language (or the browser's) before the workbench
 * renders. Keyed by the stored value: a refetch returning the same value never
 * re-applies it, so a choice whose save failed is not silently reverted.
 */
export function LanguageGate({ children }: { children: ReactNode }) {
  const settings = useSettings()
  const stored = settings.data?.language ?? null
  const applied = useQuery({
    queryKey: ['language', stored],
    queryFn: async () => {
      const code = resolveLanguage(stored)
      await setLanguage(code)
      return code
    },
    enabled: settings.isSuccess,
    staleTime: Infinity,
    gcTime: Infinity,
  })
  if (settings.isError) return <ErrorState error={settings.error} onRetry={() => void settings.refetch()} />
  if (applied.isError) return <ErrorState error={applied.error} onRetry={() => void applied.refetch()} />
  if (!applied.isSuccess) return <LoadingState />
  return <>{children}</>
}
```

`theme/index.ts`: add

```ts
import { deDE, enUS, esES, frFR, jaJP, koKR, zhCN, type Localization } from '@mui/material/locale'
import type { LanguageCode } from '@/i18n'

const MUI_LOCALES: Record<LanguageCode, Localization> = { en: enUS, fr: frFR, de: deDE, es: esES, zh: zhCN, ja: jaJP, ko: koKR }

/** The design tokens theme merged with MUI's own strings (pagination, …) for a language. */
export function themeFor(code: LanguageCode) {
  return createTheme(theme, MUI_LOCALES[code])
}
```

`App.tsx`: import `useMemo`, `useLanguage`, `themeFor`, `LanguageGate`; in `App`:

```tsx
export default function App({ queryClient = createQueryClient() }: { queryClient?: QueryClient }) {
  const language = useLanguage()
  const muiTheme = useMemo(() => themeFor(language), [language])
  return (
    <ThemeProvider theme={muiTheme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <SessionGate>
            <LanguageGate>
              <AppRoutes />
            </LanguageGate>
          </SessionGate>
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
```

(`App` re-renders on a language change, which re-creates every route element, so every page re-renders with its state kept.)

- [x] **Step 5: The selector and its two placements**

`frontend/src/components/LanguageMenu.tsx`:

```tsx
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import Menu from '@mui/material/Menu'
import MenuItem from '@mui/material/MenuItem'
import Snackbar from '@mui/material/Snackbar'
import { useState, type MouseEvent } from 'react'
import { useUpdateSettings } from '@/api/hooks'
import { LANGUAGES, languageInfo, setLanguage, t, useLanguage, type LanguageCode } from '@/i18n'

function Flag({ src }: { src: string }) {
  return <img src={src} alt="" width={20} height={15} style={{ display: 'block', borderRadius: 2, boxShadow: '0 0 0 1px rgba(0,0,0,0.15)' }} />
}

/** Flag-and-name language switcher: applies the choice at once, then stores it in the workspace. */
export function LanguageMenu({ variant = 'toolbar' }: { variant?: 'toolbar' | 'settings' }) {
  const code = useLanguage()
  const current = languageInfo(code)
  const update = useUpdateSettings()
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const [failure, setFailure] = useState<string | null>(null)

  const choose = async (next: LanguageCode) => {
    setAnchor(null)
    if (next === code) return
    await setLanguage(next)
    try {
      await update.mutateAsync({ language: next })
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <>
      <Button
        onClick={(e: MouseEvent<HTMLElement>) => setAnchor(e.currentTarget)}
        color="inherit"
        variant={variant === 'settings' ? 'outlined' : 'text'}
        size="small"
        aria-label={t('language.label')}
        aria-haspopup="menu"
        aria-expanded={anchor ? 'true' : undefined}
        startIcon={<Flag src={current.flag} />}
        endIcon={<ArrowDropDownIcon />}
        data-testid="language-menu"
        sx={{ textTransform: 'none', flexShrink: 0 }}
      >
        <span lang={current.tag}>{current.name}</span>
      </Button>
      <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)} slotProps={{ list: { 'aria-label': t('language.label') } }}>
        {LANGUAGES.map((l) => (
          <MenuItem key={l.code} selected={l.code === code} onClick={() => void choose(l.code)} lang={l.tag}>
            <ListItemIcon>
              <Flag src={l.flag} />
            </ListItemIcon>
            <ListItemText>{l.name}</ListItemText>
          </MenuItem>
        ))}
      </Menu>
      <Snackbar open={failure !== null} autoHideDuration={8000} onClose={() => setFailure(null)}>
        <Alert severity="error" onClose={() => setFailure(null)}>
          {t('language.saveFailed', { error: failure ?? '' })}
        </Alert>
      </Snackbar>
    </>
  )
}
```

`AppShell.tsx`: import `LanguageMenu` and render `<LanguageMenu />` in the `Toolbar` between `<Box sx={{ flex: 1 }} />` and the running `Chip`.

`SettingsPage.tsx`: import `LanguageMenu`; add as the first `Paper` after the title:

```tsx
      <Paper sx={{ p: 2 }} component="section" aria-labelledby="settings-language">
        <Typography variant="h2" id="settings-language" gutterBottom>
          {t('settings.language')}
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {t('settings.language.note')}
        </Typography>
        <LanguageMenu variant="settings" />
      </Paper>
```

- [x] **Step 6: Locale-aware formatting**

Replace every `x.toLocaleString()` on a number with `formatNumber(x)`, `x.toLocaleString(undefined, opts)` with `formatNumber(x, opts)`, `new Date(v).toLocaleString()` with `formatDateTime(v)` and `d.toLocaleTimeString()` with `formatTime(d)` in the ten files listed above (import from `@/i18n`). Metric values formatted with `toFixed` are left alone.

- [x] **Step 7: Playwright**

`e2e/mock-api.ts`: add `language: string | null` to `FakeState` (initial `null`, or the value passed as `installFakeApi(page, { language })`), and in the router before the `/session` line:

```ts
    if (path === '/settings' && method === 'GET') return json(route, { language: state.language })
    if (path === '/settings' && method === 'PUT') {
      state.language = (req.postDataJSON() as { language: string | null }).language
      return json(route, { language: state.language })
    }
```

`e2e/journey.spec.ts`, in the J1 describe block:

```ts
  test('the language selector is in the top bar; a choice applies at once and survives a reload', async ({ page }) => {
    await installFakeApi(page)
    await page.goto('/')
    await page.getByRole('button', { name: 'Language' }).click()
    const items = page.getByRole('menuitem')
    await expect(items).toHaveText(['English', 'Français', 'Deutsch', 'Español', '中文', '日本語', '한국어'])
    await items.filter({ hasText: '中文' }).click()
    await expect(page.getByRole('link', { name: '数据集' })).toBeVisible()
    await expect(page.getByRole('heading', { level: 1, name: '首页' })).toBeVisible()
    await page.reload()
    await expect(page.getByRole('link', { name: '数据集' })).toBeVisible()
    await expect(page.locator('html')).toHaveAttribute('lang', 'zh-CN')
  })
```

and extend the layout test to run its loop for `[null, 'de', 'fr']` (`installFakeApi(page, { language })` before the loop, so the longest strings are checked at 1366×768).

- [x] **Step 8: Run everything**

Run: `cd frontend && npm run typecheck && npm run lint && npm test && npm run build && npx playwright test --project=chromium-1366 --update-snapshots` (the gallery baselines change only if the top bar is inside the captured regions; they are element screenshots of the StatusChip / MetricCard regions, so expect no change) then `npx playwright test --project=chromium-1366`.
Expected: all green.

- [x] **Step 9: Commit**

```bash
git add frontend/src frontend/e2e
git commit -m "feat(frontend): language selector with flags in the top bar and Settings, locale-aware formatting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Native strings in seven languages and the launcher wiring

**Files:**
- Modify: `opendpd/studio/strings.py`, `opendpd/studio/launcher.py` (`_default_window_runner`, new `preferred_language`)
- Test: `tests/unit/test_window.py`, `tests/unit/test_launcher.py`

**Interfaces:**
- Consumes: `Workspace.settings()` (Task 1), `UI_LANGUAGES`.
- Produces: `STRINGS: Dict[str, ShellStrings]` for the seven codes; `shell_strings(language) -> ShellStrings` (unknown → English); `launcher.preferred_language(workspace: Path) -> str`.

- [x] **Step 1: Write the failing tests**

Append to `tests/unit/test_window.py`:

```python
from opendpd.schemas import UI_LANGUAGES
from opendpd.studio.strings import ENGLISH, STRINGS


def test_every_language_has_every_native_string():
    assert set(STRINGS) == set(UI_LANGUAGES)
    for code, s in STRINGS.items():
        assert "{count}" in s.quit_body, code
        assert set(s.localization) == set(ENGLISH.localization), code
        assert all(v.strip() for v in s.localization.values()), code
    assert shell_strings("xx") is ENGLISH
    assert shell_strings("ja").quit_title != ENGLISH.quit_title
```

Append to `tests/unit/test_launcher.py`:

```python
def test_preferred_language_reads_the_workspace_then_the_os_locale(tmp_path, monkeypatch):
    from opendpd.schemas import WorkspaceSettings
    from opendpd.services.workspace import Workspace
    ws = Workspace.create(tmp_path / "ws")
    monkeypatch.setattr(launcher.locale, "getlocale", lambda: ("fr_FR", "UTF-8"))
    assert launcher.preferred_language(ws.root) == "fr"
    ws.save_settings(WorkspaceSettings(language="ja"))
    assert launcher.preferred_language(ws.root) == "ja"
    monkeypatch.setattr(launcher.locale, "getlocale", lambda: ("pt_BR", "UTF-8"))
    assert launcher.preferred_language(tmp_path / "missing") == "en"
    ws.settings_path.write_text("{broken")
    assert launcher.preferred_language(ws.root) == "en", "a broken file never blocks the window"
```

- [x] **Step 2: Run them to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_window.py tests/unit/test_launcher.py -q -k "native_string or preferred_language"`
Expected: FAIL (`STRINGS` and `preferred_language` missing).

- [x] **Step 3: Implement**

`opendpd/studio/strings.py`: keep `ShellStrings` and `ENGLISH`; extend `ENGLISH.localization` with the macOS menu keys (`cocoa.menu.about` "About", `cocoa.menu.services` "Services", `cocoa.menu.view` "View", `cocoa.menu.edit` "Edit", `cocoa.menu.hide` "Hide", `cocoa.menu.hideOthers` "Hide Others", `cocoa.menu.showAll` "Show All", `cocoa.menu.quit` "Quit", `cocoa.menu.fullscreen` "Enter Fullscreen", `cocoa.menu.cut` "Cut", `cocoa.menu.copy` "Copy", `cocoa.menu.paste` "Paste", `cocoa.menu.selectAll` "Select All", `windows.fileFilter.allFiles` "All files", `windows.fileFilter.otherFiles` "Other file types", `linux.openFile` "Open file", `linux.openFiles` "Open files", `linux.openFolder` "Open folder"); add the six other `ShellStrings` with the same keys, for example French:

```python
FRENCH = ShellStrings(
    quit_title="Quitter OpenDPD Studio ?",
    quit_body="{count} expérience(s) en cours. Quitter OpenDPD Studio et les arrêter ?",
    localization={"global.quit": "Quitter", "global.cancel": "Annuler", "global.ok": "OK", "global.saveFile": "Enregistrer le fichier",
                  "global.quitConfirmation": "Voulez-vous vraiment quitter ?", "cocoa.menu.about": "À propos", "cocoa.menu.services": "Services",
                  "cocoa.menu.view": "Présentation", "cocoa.menu.edit": "Édition", "cocoa.menu.hide": "Masquer", "cocoa.menu.hideOthers": "Masquer les autres",
                  "cocoa.menu.showAll": "Tout afficher", "cocoa.menu.quit": "Quitter", "cocoa.menu.fullscreen": "Activer le mode plein écran",
                  "cocoa.menu.cut": "Couper", "cocoa.menu.copy": "Copier", "cocoa.menu.paste": "Coller", "cocoa.menu.selectAll": "Tout sélectionner",
                  "windows.fileFilter.allFiles": "Tous les fichiers", "windows.fileFilter.otherFiles": "Autres types de fichiers",
                  "linux.openFile": "Ouvrir un fichier", "linux.openFiles": "Ouvrir des fichiers", "linux.openFolder": "Ouvrir un dossier"},
)
```

then `STRINGS = {"en": ENGLISH, "fr": FRENCH, "de": GERMAN, "es": SPANISH, "zh": CHINESE, "ja": JAPANESE, "ko": KOREAN}` and

```python
def shell_strings(language: Optional[str] = None) -> ShellStrings:
    return STRINGS.get(language or "", ENGLISH)
```

`opendpd/studio/launcher.py`: `import locale` at the top and

```python
def preferred_language(workspace: Path) -> str:
    """The workspace's stored UI language, else the OS locale when supported, else English."""
    from opendpd.schemas import UI_LANGUAGES
    try:
        from opendpd.services.workspace import Workspace
        stored = Workspace.open(workspace).settings().language
        if stored:
            return stored
    except Exception:  # noqa: BLE001 - no workspace yet or a broken file: never blocks the window
        pass
    try:
        code = (locale.getlocale()[0] or "").split("_")[0].lower()
    except ValueError:
        code = ""
    return code if code in UI_LANGUAGES else "en"


def _default_window_runner(url: str, active_runs: Callable[[], int], workspace: Path) -> None:
    from opendpd.studio import window as window_shell
    from opendpd.studio.strings import shell_strings
    window_shell.run_window(url, active_runs=active_runs, strings=lambda: shell_strings(preferred_language(workspace)),
                            icon=window_shell.icon_path())
```

and in `launch()` bind the workspace: `window_runner = window_runner or (lambda url, active: _default_window_runner(url, active, workspace))`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_window.py tests/unit/test_launcher.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add opendpd/studio/strings.py opendpd/studio/launcher.py tests/unit/test_window.py tests/unit/test_launcher.py
git commit -m "feat(studio): native window strings in seven languages, following the workspace setting

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Documentation, ADR-0003 and the full verification run

**Files:**
- Create: `docs/architecture/adr/0003-ui-languages.md`
- Modify: `docs/architecture/ux-spec.md`, `docs/tutorials/gui-quickstart.md`, `README.md`, `docs/releases/release-notes-2.2.0.md`, `OpenDPD_Studio_Development_Plan.md`

- [x] **Step 1: ADR-0003**

```markdown
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
| A pluralisation library | 496 strings with a handful of counts; parenthesised plurals keep the catalogues flat |

## Consequences

- Every new string needs seven entries; the catalogue test fails otherwise.
- Long German and French strings must fit the 1366×768 layout (checked by the layout journey).
- Translations are maintained in the repository; corrections are ordinary PRs.

## Verification

`frontend/src/i18n/catalogues.test.ts`, `index.test.ts`, `LanguageMenu.test.tsx`, `LanguageGate.test.tsx`; `frontend/e2e/journey.spec.ts` (selector, reload, layout in `de`/`fr`); `tests/integration/test_studio_api.py` (settings routes); `tests/unit/test_window.py` (native strings).
```

- [x] **Step 2: User documentation**

`docs/architecture/ux-spec.md`: add `| \`LanguageMenu\` | flag-and-name language switcher (top bar and Settings); applies at once, stored per workspace |` to the component catalogue and, in §2, the line "The top bar carries the language selector on every page."

`docs/tutorials/gui-quickstart.md`: add

```markdown
## Language

The top bar shows the current language with its flag; the menu lists
English, Français, Deutsch, Español, 中文, 日本語 and 한국어. A choice applies
at once and is stored in `<workspace>/settings.json`, so it is the same in
the native window and in any browser. The first launch follows your browser
or system language. Server messages, worker logs, reports and the CLI stay in
English so that a GUI screen, a package and a terminal say the same thing.
```

`README.md`: append to the Studio paragraph "The interface is available in English, French, German, Spanish, Chinese, Japanese and Korean."

`docs/releases/release-notes-2.2.0.md`: bullet `- **Seven UI languages**: English, French, German, Spanish, Chinese, Japanese and Korean, chosen from the flag menu in the top bar and stored per workspace; server text stays English.`

`OpenDPD_Studio_Development_Plan.md` §4.1, after the table: `> **修订（2026-09-11，维护者决定）**：界面提供英、法、德、西、中、日、韩七种语言，顶部栏提供带国旗和语言名称的切换菜单，选择按工作区保存（`settings.json`）；服务端文本、日志、报告与 CLI 保持英文。详见 ADR-0003。`

- [x] **Step 3: Full verification**

Run: `.venv/bin/python -m pytest tests/unit tests/integration/test_studio_api.py tests/integration/test_launcher_ownership.py -q`, `.venv/bin/python scripts/export_openapi.py --check`, `cd frontend && npm run types:check && npm run typecheck && npm run lint && npm test && npm run build && npx playwright test --project=chromium-1366`.
Expected: all green; the built bundle in `opendpd/studio/static` serves the selector (check with `opendpd gui` on a scratch workspace: the menu is visible and 中文 renders).

- [x] **Step 4: Commit**

```bash
git add docs/architecture/adr/0003-ui-languages.md docs/architecture/ux-spec.md docs/tutorials/gui-quickstart.md README.md docs/releases/release-notes-2.2.0.md OpenDPD_Studio_Development_Plan.md
git commit -m "docs(studio): ADR-0003 and documentation for the seven UI languages

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
