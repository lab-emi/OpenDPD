# Maintaining documentation

The GitHub README is the short entry point; the documentation site is the place to complete a task and look up details. Both draw from the same repository. Keep one maintained home for each substantial explanation and link to it from the other entry points.

## Where to edit

| Content | Source of truth | Reuse |
| --- | --- | --- |
| Product introduction, current highlights, homepage screenshot | Root `README.md` sections `intro`, `studio-features`, `hero` | Included by the site home page. |
| Minimal source-install command block | Root `README.md`, `source-install` section | Included by `docs/install.md`; edit once. |
| Setup options and troubleshooting | `docs/install.md` | Linked from README and tutorials. |
| Task walkthroughs and explanations | `docs/tutorials/`, `training.md`, `visualization.md`, `advanced.md`, `faq.md` | Read directly on GitHub or through the site. |
| Dataset format and Python examples | `datasets/README.md`, `examples/README.md` | Included by `docs/datasets.md` and `docs/examples.md`. |
| Benchmark report | `benchmark/benchmark_report.md` | Included by `docs/benchmark/index.md`; scientific review rules apply. |
| Citation metadata and paper entries | `CITATION.cff`; `docs/community/citation.md` | GitHub's citation button; links from README and About. |
| Test commands and contributor workflow | `docs/testing.md`; root `CONTRIBUTING.md` | Linked from README and the Project section. |
| Scientific definitions and acceptance evidence | `docs/protocols/`, `docs/releases/` | Link from guides; do not restate thresholds in the README. |
| Site navigation | `mkdocs.yml` | Organize by user task without moving existing page URLs. |
| Shared images | `pics/` | Published by `docs/hooks/assets.py`; keep one copy. |
| Studio and EMI logos | `frontend/src/assets/`; root `README.md`, `brand` section | The site includes the same brand section and publishes the original SVGs through the asset hook. |

`docs/index.md`, `docs/datasets.md`, `docs/examples.md` and `docs/benchmark/index.md` are site entry pages. They compose canonical content with `pymdownx.snippets`; they are not a second place to edit that content. The installation page reuses only the short command block. When reading its Markdown on GitHub, that block is available in the root README.

## Writing and linking

- Keep README focused on the screenshot, current capability, first run, workflow and next steps. Move long option lists, experiment settings and paper tables to a guide.
- Use normal relative Markdown links in guides. Link to the canonical source, not to a site wrapper when writing a GitHub entry point.
- Site-only links belong in `docs/index.md` or `mkdocs.yml`. Shared README snippets contain context-neutral text, external links, or repository-root image paths published by the asset hook so they work when included on the site home page.
- For images in a standalone guide, use the repository-relative path, such as `../pics/platform.png` from `docs/about.md`. The asset hook adjusts that path for the site build.
- Keep existing page filenames when reorganizing navigation. Update links and anchors if a heading changes.
- Mark development previews separately from released packages. Check the actual package extras, CLI flags and UI before documenting an installation or feature.
- Keep scientific meaning in the shared core and versioned protocols. A prose cleanup must not change a split rule, threshold, expected value or evidence type.

## Preview and check

In the source checkout, with a Python environment active:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

Open the local address to inspect desktop and narrow layouts, images, navigation and copied commands. Before opening a PR:

```bash
python -m mkdocs build --strict
```

The build checks internal links, anchors and snippet paths. It does not validate every external URL or execute code blocks. If tutorial commands change, also run the [documentation integration test](testing.md#documentation-checks) and verify the described UI path against a real local server.

The existing **Docs** workflow builds PRs and publishes the site from `main`. No separate copy or manual edit of generated `site/` files is needed.

## Refresh the Studio screenshot

Use the current app with a separate clean workspace, English selected, the light theme active and no private paths, captures or user run names visible. Capture the real Home page after it has loaded, with **Get Started** visible, and replace `pics/studio-home.png`. Keep browser diagnostics and session files out of Git. The README and site will both pick up the same image.

## Update the brand logos

Reuse the transparent SVGs from `frontend/src/assets/`, including their light and dark variants. Keep the EMI inverse variant's geometry identical to `emi-logo.svg`; only the ink colors differ. The site header uses the Studio emblem; its favicon reuses `frontend/public/favicon.svg`. Do not rasterize the logos or copy them into `pics/`.

The README's `brand` snippet uses GitHub-compatible `<picture class="brand-logo">` elements, each with a dark `source` and a light fallback `img`. Keep these child tags self-closing. During the site build, `docs/hooks/assets.py` converts them to Material's `#only-light` / `#only-dark` images, so the site's own theme toggle works independently of the system theme. Check both themes and a narrow viewport when changing this section.
