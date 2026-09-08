# Contributing to ArchytasZX

ArchytasZX is a research prototype under active development. The API is unstable and the
internals move quickly, so the most useful contributions right now are bug reports,
counterexamples, and questions/concerns about the mathematics.

**Please send bug reports rather than code.** Unsolicited patches are hard to take at this
stage. A precise report of something the engine gets wrong is worth far more than a pull
request, and will be acted on faster.

## Reporting a soundness bug

A rewrite that does not preserve the denoted linear map — scalar included — is the most
serious kind of defect this project can have, and the most valuable thing to report. The
best report is a short script that builds two diagrams and shows the engine claiming they
are equal when the oracle at a small concrete dimension says otherwise:

```python
from archytaszx.semantics.check import compare

compare(before, after, {"d": 3}).matched   # expected True, got False
```

Please include the dimension and multiplicity values you used, and the certificate or
`RewriteStep` provenance if a rewrite was involved.

## Conventions

- **Layers depend strictly downward**: `algebra` ← `diagram` ← `rewrite` ← `semantics`. The
  REPL depends on the engine; the engine never depends on the REPL.
- **Every module raises only its own exception hierarchy**, with a base `<Area>Error` split
  into `<Area>GrammarError` and `<Area>DomainError`. No foreign exception crosses a module
  boundary.
- **Scalars are never silently discarded.** A rule that introduces a global factor records it
  exactly.
- **New behaviour arrives with an oracle test.** A rewrite is checked before-and-after by
  numeric contraction at small concrete instantiations, or by symbolic contraction with `d`
  formal, or both.
- **Determinism.** Match ordering, set iteration, and printed output must not depend on hash
  order or the wall clock.
- Docstrings are brief and describe either *what* a function is or *how* it works.
- Every source file carries the Apache 2.0 header.

## License

By contributing you agree that your contributions are licensed under the Apache License 2.0,
the same terms that cover the project.

<!--
Copyright 2026 Arkhip A. Dmitriev
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
