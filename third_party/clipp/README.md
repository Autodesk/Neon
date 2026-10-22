# clipp (vendored)

Single-header C++ command-line argument parser by André Müller.

- Upstream: https://github.com/muellan/clipp
- Version:  1.2.3
- License:  MIT (see the header at the top of `clipp.h`)

## Why vendor it

Carbon's reference, [Neon](../../Neon/), uses clipp for the CLI of its examples and
benchmark drivers (e.g. `Neon/libNeonSolver/tests/perf/solverPt_Poisson/`). To keep
carbon's example CLIs idiomatically aligned with Neon's, we vendor the same
single-header library here. The file is unmodified from upstream / Neon's bundled
copy at `Neon/libNeonCore/include/Neon/core/tools/clipp.h`.

clipp is not part of carbon's public API; it is exposed only as the
`carbon::third_party::clipp` CMake `INTERFACE` target and consumed by binaries
under `examples/`. carbon library headers and tests do not depend on clipp.

## Updating

If a newer upstream version is needed, replace `clipp.h` and update the version /
URL in this README. The MIT license notice at the top of the header must remain.
