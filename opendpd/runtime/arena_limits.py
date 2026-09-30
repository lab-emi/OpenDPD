"""Operational Arena guard, independent of the frozen scientific training budget."""

# A full-window DPA_160MHz sweep is over a million updates per seed. The
# operational ceiling must not truncate its fixed scientific budget.
ARENA_MAX_RUNTIME_SECONDS = 7 * 24 * 60 * 60
