<!--
Part of the Trident project, under the MIT License.
SPDX-License-Identifier: MIT
-->

# Core Lit tests

The directory follows the LLVM and torch-mlir convention of organizing
regression tests by the implementation layer under test:

- `Dialect/<Dialect>/` covers parsing, printing, verification, and dialect
  transforms. Invalid IR uses `-verify-diagnostics`.
- `Conversion/<Pass>/` covers one conversion pass and checks its pass-local
  rewrite. The directory name matches the corresponding `core/lib` pass.
- `Conversion/Pipeline/` covers representative end-to-end lowering through
  `--trident-lowering-pipeline`.

Keep each file focused on one operation family or lowering concern. Use
descriptive filenames, `CHECK-LABEL` for each function, and named FileCheck
captures when a value is reused. Bind function arguments and reuse those
captures when checking data flow instead of relying on printed names such as
`%arg0`. Spell FileCheck directives with one space after the colon and use
`[a-zA-Z0-9_]+` for ordinary SSA captures. Do not use anonymous wildcard
matches in place of a capture. Shared auxiliary inputs belong in an `Inputs/`
directory and are excluded by the root Lit configuration.
