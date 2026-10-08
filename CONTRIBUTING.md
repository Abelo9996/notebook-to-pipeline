# Contributing

```
uv sync
uv run pytest -q
uv run ruff check src tests
```

The tests are offline and build small notebooks with nbformat in temp directories. Kernel tests run
on the interpreter that runs pytest, so `uv sync` installs pandas, numpy and scikit-learn as dev
dependencies.

Ground rules:

- The tool stays deterministic and never calls an LLM. The agent using it does the thinking.
- Every claim the tool prints must come from something it ran or measured, and its JSON output
  must contain the evidence behind it.
- Files in `src/notebook_to_pipeline/runtime/` run inside the user's interpreter. They may import
  only the standard library at module level and must stay Python 3.8 compatible.
- Never run `nb2p setup` against your real home directory while developing. The tests use a
  temporary `HOME`.

See AGENTS.md for the layout.
