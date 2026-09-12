# Testing and continuous integration

--8<-- "README.md:testing"

## Documentation site

This site is built with [MkDocs](https://www.mkdocs.org/) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from the `docs/` directory and deployed to
GitHub Pages by the `Docs` workflow on every push to `main`. The pages are single-sourced: they include sections of
`README.md`, `datasets/README.md`, `examples/README.md` and `benchmark/benchmark_report.md` between `--8<--`
markers, so editing those files updates the site. To preview it locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

`mkdocs build --strict` is what CI runs: a broken link or a missing include fails the build.
