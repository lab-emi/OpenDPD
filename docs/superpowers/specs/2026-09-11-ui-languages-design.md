# Seven UI languages for OpenDPD Studio — design

- **Date:** 2026-09-11
- **Status:** approved by the maintainer (English, French, German, Spanish,
  Chinese, Japanese, Korean; a visible selector with flag and name)
- **Records:** ADR-0003 (written with the implementation)
- **Companion spec:** `2026-09-11-native-window-shell-design.md`

## 1. Goal

The workbench can be used in seven languages. A language selector showing the
flag and the native language name is visible in the top bar on every page; a
choice applies immediately, without reload, and is remembered per workspace;
the first launch follows the browser or system language and falls back to
English.

### Non-goals

Translating server-generated text: API error messages and hints, Dataset
Doctor findings, reports, worker logs and CLI output stay English so that the
GUI, the CLI and exported packages say the same thing. No plural-rules engine,
no right-to-left layout, no translation of scientific identifiers (metric
names, profile ids, recipe ids, model keys, status values in exports), no
change to the format of metric values, no visual redesign.

## 2. Languages

| Code | BCP-47 tag (`<html lang>`, `Intl`) | Native name | Flag file |
|---|---|---|---|
| `en` | `en` | English | `gb.svg` |
| `fr` | `fr` | Français | `fr.svg` |
| `de` | `de` | Deutsch | `de.svg` |
| `es` | `es` | Español | `es.svg` |
| `zh` | `zh-CN` | 中文 | `cn.svg` |
| `ja` | `ja` | 日本語 | `jp.svg` |
| `ko` | `ko` | 한국어 | `kr.svg` |

Flags are SVG files from flag-icons 7.5.0 (MIT), copied into
`frontend/src/assets/flags/` together with the licence text; Spain uses the
plain civil three-stripe flag drawn by hand because the package version embeds
an 80 KB coat of arms. Emoji flags are not used: Windows renders them as
letter pairs. Flags are decorative (`alt=""`); the native name next to them is
the accessible label, and each entry carries its `lang` attribute.

## 3. Where the choice lives

On the server, per workspace: `<workspace>/settings.json` holds
`{"language": "fr"}`. Schema `WorkspaceSettings` in
`opendpd/schemas/settings.py` (`language: Optional[UILanguage] = None`, with
`UILanguage` the seven codes) and an example registered for the mock export.
Workspace service: `Workspace.settings()` (missing file → defaults;
unreadable or invalid → `WorkspaceError` naming the file) and
`Workspace.save_settings()` (atomic write). API: `GET /api/v1/settings`
(session) and `PUT /api/v1/settings` (session + CSRF, full replacement; an
unknown code is a 422 `invalid_request`). Headless equivalent: edit the file;
no new CLI command.

Why the server and not `localStorage`: the native window runs in private
mode (no persistent WebKit storage), the preference must survive restarts and
be the same in the window and in any browser, and the desktop shell needs it
for its own native dialogs.

Resolution order in the client: the stored value → the first entry of
`navigator.languages` whose base language is supported → `en`. Nothing is
written until the user picks a language.

## 4. Frontend architecture

- `src/i18n/index.ts` keeps `t(key, vars)` so the 496 call sites are
  untouched, and adds a small external store: the `LANGUAGES` table,
  `getLanguage()`, `useLanguage()` (through `useSyncExternalStore`),
  `loadCatalogue(code)` (dynamic `import()` per language; English stays a
  static import), `setLanguage(code)` (only after its catalogue is loaded) and
  `formatNumber` / `formatDateTime` helpers bound to the BCP-47 tag. `t()`
  reads the active catalogue and falls back to English per key; the tests keep
  every catalogue complete, the fallback is defensive.
- Re-rendering without remount: `AppRoutes` subscribes with `useLanguage()`
  and re-creates the route elements on a change, so every page re-renders
  with its state preserved. `useMemo` / `useCallback` sites whose outputs
  contain `t()` text add the language to their dependencies (ConfigDiff, chart
  layouts). `<html lang>` follows the tag; the document title stays
  "OpenDPD Studio".
- MUI's own component strings (for example pagination labels) follow the
  language through the `@mui/material/locale` packs merged into the theme per
  language.
- Boot: `SessionGate`, once authenticated, loads the settings and the
  catalogue before rendering the shell (the loading state stays English; a
  settings failure shows the error state with retry).
- `LanguageMenu` (`src/components/LanguageMenu.tsx`): in the top bar, right
  side before the "running" chip, a button showing the current flag and native
  name with an accessible label ("Language" in the current language); a MUI
  `Menu` with the seven entries (flag, native name, `selected` on the current
  one). Choosing one loads the catalogue, switches the UI, then PUTs the
  setting. A failed PUT keeps the chosen language on screen and shows a MUI
  `Snackbar` with the error; nothing fails silently. The Settings page gets a
  "Language" section that renders the same component as a labelled control.
- Catalogues `src/i18n/{fr,de,es,zh,ja,ko}.json`: flat, exactly the keys of
  `en.json`, typed `Record<MessageKey, string>`. Placeholders (`{count}`, …),
  inline code in backticks, product names, metric abbreviations, profile /
  recipe / model identifiers and the `MOCK` marker are kept verbatim. Phrases
  that carry a scientific caveat ("simulated", "not verified", "pending",
  "never ranked") keep their explicit meaning.
- Formatting: counts, sizes and dates go through `formatNumber` /
  `formatDateTime` (the existing `toLocaleString()` sites); metric values keep
  `toFixed` with a decimal point because they mirror CSV, report and CLI
  output.
- The Playwright mock API answers `GET /settings` with `{language: null}` and
  stores what `PUT` sends.

## 5. Desktop shell strings

`opendpd/studio/strings.py` provides, per language, the window's native
strings: the confirm-quit title and body (`{count}`), and pywebview's
`localization` keys (`global.quit`, `global.cancel`, `global.ok`,
`global.saveFile`, `global.quitConfirmation`, the macOS `cocoa.menu.*`
items). The launcher reads `Workspace.settings().language` when the window
opens (menus) and again at each close request (dialog), falling back to the
OS locale's base language when supported, then English.

## 6. Testing

- Vitest: every catalogue has exactly the key set of `en.json`, the same
  placeholder set per key and no empty string; `LanguageMenu` lists seven
  entries with flag images and native names, choosing "Deutsch" turns the
  Datasets navigation label into "Datensätze" and PUTs `{language: "de"}`, a
  failed PUT shows the snackbar and keeps the language; resolution (stored
  `ja` beats the navigator; `fr-CA` → `fr`; unsupported → `en`);
  `formatNumber` / `formatDateTime` use the tag.
- Playwright (Chromium): the top-bar selector is visible on `/`, switching to
  中文 changes the navigation labels and the choice survives a reload (the mock
  stores it); the layout test at 1366×768 also runs in `de` and `fr` (the
  longest strings) to catch overflow; the accessibility audit covers the menu
  through the top bar it already visits on every page.
- Python: the settings routes (default `null`; a valid PUT persists to
  `settings.json`; an unknown code → 422; a PUT without CSRF → 403);
  `Workspace.settings()` on a corrupt file raises `WorkspaceError`; the
  strings table is complete for every language; the mock export is in sync;
  the OpenAPI contract and the generated TypeScript types are regenerated (CI
  checks both).

## 7. Documentation

ADR-0003 extends ADR-0001's i18n row (seven languages, per-workspace
setting, lazily loaded catalogues, translations maintained under
`frontend/src/i18n`, server text stays English); `docs/architecture/ux-spec.md`
(component row `LanguageMenu`, navigation note); `docs/tutorials/gui-quickstart.md`
("Language" section); README sentence; release notes; a short amendment in
`OpenDPD_Studio_Development_Plan.md` recording the decision.

## 8. Terminology

Translators keep these renderings consistent across the catalogues.

| English | Français | Deutsch | Español | 中文 | 日本語 | 한국어 |
|---|---|---|---|---|---|---|
| workspace | espace de travail | Arbeitsbereich | espacio de trabajo | 工作区 | ワークスペース | 워크스페이스 |
| dataset | jeu de données | Datensatz | conjunto de datos | 数据集 | データセット | 데이터셋 |
| run | exécution | Lauf | ejecución | 运行 | 実行 | 실행 |
| experiment | expérience | Experiment | experimento | 实验 | 実験 | 실험 |
| recipe | recette | Rezept | receta | 配方 | レシピ | 레시피 |
| smoke recipe | recette smoke | Smoke-Rezept | receta smoke | 冒烟配方 | スモークレシピ | 스모크 레시피 |
| result | résultat | Ergebnis | resultado | 结果 | 結果 | 결과 |
| evidence type | type de preuve | Evidenztyp | tipo de evidencia | 证据类型 | エビデンス種別 | 증거 유형 |
| PA model | modèle de PA | PA-Modell | modelo del PA | PA 模型 | PA モデル | PA 모델 |
| surrogate | substitut | Surrogat | sustituto | 替代模型 | サロゲート | 서로게이트 |
| measured | mesuré | gemessen | medido | 实测 | 実測 | 실측 |
| checkpoint | point de contrôle | Checkpoint | punto de control | 检查点 | チェックポイント | 체크포인트 |
| metric profile | profil de métriques | Metrikprofil | perfil de métricas | 指标配置 | メトリクスプロファイル | 메트릭 프로필 |
| split | découpage | Aufteilung | partición | 划分 | 分割 | 분할 |
| preprocessing version | version de prétraitement | Vorverarbeitungsversion | versión de preprocesamiento | 预处理版本 | 前処理バージョン | 전처리 버전 |
| import / export | importer / exporter | importieren / exportieren | importar / exportar | 导入 / 导出 | インポート / エクスポート | 가져오기 / 내보내기 |
| package | paquet | Paket | paquete | 包 | パッケージ | 패키지 |
| report | rapport | Bericht | informe | 报告 | レポート | 보고서 |
| compare | comparer | vergleichen | comparar | 比较 | 比較 | 비교 |
| robustness | robustesse | Robustheit | robustez | 鲁棒性 | ロバスト性 | 강건성 |
| streaming | streaming | Streaming | streaming | 流式 | ストリーミング | 스트리밍 |
| deployment | déploiement | Bereitstellung | despliegue | 部署 | デプロイ | 배포 |
| session | session | Sitzung | sesión | 会话 | セッション | 세션 |
| epoch | époque | Epoche | época | 轮次 | エポック | 에포크 |
| artifact | artefact | Artefakt | artefacto | 产物 | 成果物 | 아티팩트 |
| log | journal | Protokoll | registro | 日志 | ログ | 로그 |
| queued / running / succeeded / failed / cancelled / interrupted | en attente / en cours / réussi / échoué / annulé / interrompu | wartend / läuft / erfolgreich / fehlgeschlagen / abgebrochen / unterbrochen | en cola / en ejecución / completado / fallido / cancelado / interrumpido | 排队中 / 运行中 / 成功 / 失败 / 已取消 / 已中断 | 待機中 / 実行中 / 成功 / 失敗 / キャンセル済み / 中断 | 대기 중 / 실행 중 / 성공 / 실패 / 취소됨 / 중단됨 |

Kept verbatim in every language: OpenDPD Studio, Dataset Doctor, MOCK, NMSE,
ACLR, EVM, ACPR, PSD, PA, DPD, GRU, CSV, JSON, GUI, CLI, and every identifier
such as `legacy-opendpd-v1` or `pa-gru-smoke-v1`.
