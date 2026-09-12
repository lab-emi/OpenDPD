# README and documentation review — 2026-09-12

Scope: README navigation, current Studio onboarding, canonical user guides,
MkDocs navigation and shared asset links. No training code, scientific metric,
split, golden reference, benchmark result or acceptance threshold changed.

## Environment

macOS arm64, Python 3.13, Node.js 26.8.1, npm 11.19.0, Playwright WebKit.
The real Studio server used a separate clean local workspace and the existing
production frontend from the same application source tree. No mocked API or
synthetic UI state was used for the homepage screenshot.

## Validation

| Check | Result |
| --- | --- |
| `python -m mkdocs build --strict` | Passed; no missing includes, internal links or anchors. |
| `python -m pytest tests/integration/test_docs_commands.py -q` | 2 passed in 61.94 seconds; documented CLI flags and real CPU command-family workflows. |
| Relative Markdown file-link check | 131 links in the changed entry pages/guides resolved in the repository. |
| WebKit site inspection | Home, Installation, Training, About, Visualization and FAQ loaded all article images; at 390 px viewport all six pages had 390 px document width. Desktop homepage also inspected at 1440 px. |
| Real Studio onboarding | Home → Get Started → built-in DPA_200MHz → inspect → configure → Quick trial → start. PA run succeeded; final metrics, I/Q geometry and time/spectrum comparisons appeared. |
| Screenshot | `pics/studio-home.png`, captured directly from the clean Home page at 1440 × 850. |
| `git diff --check` | Passed. |

PyPI metadata was checked on this date: the released package was 2.1.0, with
only the `dev` extra. Studio installation instructions therefore use the
2.2 development source tree and explicitly build its frontend. The minimum
Node.js version follows the current lockfile's strictest runtime dependency
(`react-router`: >=22.22.0).

The historical paper table and BibTeX were moved into linked references;
numerical values were preserved. Scientific explanations link to the existing
protocols. GPU execution, physical RF measurements and additional native
platforms were not verified by this documentation change.

Browser diagnostics, test workspaces and authentication state stayed local.
The homepage image is the only browser capture shipped with the user docs.
