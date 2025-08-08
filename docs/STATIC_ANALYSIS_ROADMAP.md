# Static Analysis Roadmap

This document outlines a staged plan to bring the `src` tree into
compliance with basic static-analysis tooling.

1. **Syntax Stabilization**
   - Fix malformed constructs and missing delimiters that prevent parsing.
   - Ensure all modules pass `python -m py_compile` and basic `ruff` checks.
2. **Import and Typing Hygiene**
   - Remove dead imports and swap deprecated `typing` collections for
     their built‑in counterparts.
   - Prefer PEP 604 union syntax and PEP 585 generics.
3. **Style and Safety Rules**
   - Address higher level lint rules (e.g. B008, B904, SIM102) and
     standardize logging and error handling.
4. **Type Checking and Tests**
   - Gradually enable `mypy` once syntax issues are resolved.
   - Add unit tests to verify behaviour and prevent regressions.

Initial remediation focused on
`acceleration/gpu_cognitive_accelerator.py` and
`api/routers/cognitive_architecture_router.py`, establishing a template
for broader cleanup across the repository. Subsequent passes have
stabilized additional routers such as
`api/routers/unified_thermodynamic_router.py`, bringing them under the
same lint and compile guarantees.

A repository-wide `ruff` sweep still reports over **19,000** issues—
including more than 15,000 invalid-syntax errors—highlighting the scale
of work remaining across the `src` tree. Continued cleanup will focus on
modern typing practices and safer exception handling to gradually reduce
these counts.
