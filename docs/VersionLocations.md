# Where the Neon version number lives

Checklist of every place to update when bumping the Neon version (e.g. `0.5.2a5` → `0.5.2a6`).
Keep this file current if a new location is added.

## Must update on every release

| File | What to change | Notes |
|------|----------------|-------|
| `pyproject.toml` | `version = "..."` under `[project]` | Source of truth for the `neon_gpu` wheel name and metadata. |
| `release.sh` | `VERSION="..."` | Sets the git tag (`v<VERSION>`) and selects `dist-multi/neon_gpu-<VERSION>-*.whl` for upload. Review `RELEASE_NOTES` in the same file too. |
| `CMakeLists.txt` | `set(NEON_VERSION_FULL "...")` | Full version, including the pre-release suffix, printed in the CMake configure summary. |
| `CMakeLists.txt` | `project(Neon VERSION X.Y.Z ...)` | Only when the numeric `X.Y.Z` part changes; CMake does not accept pre-release suffixes here. |

## Derived automatically (no edit needed)

| File | How |
|------|-----|
| `py/neon/__init__.py` | `neon.__version__` is read from the installed `neon_gpu` distribution via `importlib.metadata`, so it follows `pyproject.toml`. From an uninstalled source tree it reports `0.0.0+local`. |

## Optional / documentation

| File | What to change |
|------|----------------|
| `docker/README.md` | Example wheel filename `neon_gpu-<VERSION>-cp311-...whl` in the multi-Python section. |
| `template-release.sh` | Example `VERSION="..."`; it is a template, so updating it is cosmetic. |

## Downstream projects

These live outside this repository but pin a specific Neon release and must be bumped once the new wheels are published:

| Project | File | What to change |
|---------|------|----------------|
| XLB | `setup.py` | `_NEON_VERSION = "..."`, used to build the GitHub release URL for the `[neon]` extra. |

## Quick check

From the repository root, list remaining occurrences of the old version (replace `0.5.2a5`):

```bash
rg -n "0\.5\.2a5" -g '!extern/**' -g '!build/**' -g '!dist*/**'
```
