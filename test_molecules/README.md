# One-electron molecule inputs

Coordinates are in **bohr**; charge and multiplicity come from the
`* xyz charge multiplicity` header.

The 35 names in
[`suite.json`](../benchmarks/one_electron_engine/suite.json) are the archived
JCC D3 production suite. Their bytes also match the archived JCC inputs.
`tamapin.inp` is a later additional fixture and is excluded from `--molecules
all`; select it explicitly to include it.
