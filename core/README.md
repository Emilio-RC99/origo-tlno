# Core modules

The public namespaces follow the module ownership:

| Module | Namespace |
| --- | --- |
| `basis/` | `tlno::core::basis` |
| `molecule/` | `tlno::core::molecule` |

Basis construction accepts the molecule module's shared type:

```cpp
#include "molecule/molecule_reader.hpp"
#include "basis/basis_general.hpp"

const auto molecule = tlno::core::molecule::read_inp_xyz_block(path);
const auto result = tlno::core::basis::build_basis_general(
    molecule, tlno::core::basis::BasisKind::Sto3g);
```

Compile consumers with `core/` on the include path. External callers using the
previous namespaces must update their qualifications and rebuild; no aliases
to the previous namespaces are introduced. The namespace rename changes no
basis parameters, mathematical operations, or reader behavior.

From the repository root, validate basis and molecule with:

```sh
python3 -B from_TLNO/core/tests/check_core.py --reference-root ../TLNOv1
```

The script builds core and the TLNOv1 reference separately. It adapts the
generated core regression driver to the current namespaces and public names,
while the reference keeps its original API. Strict reference comparison also
checks diagnostic wording, which already differs from the original engine.

Math and SIMD files under `from_TLNO/core/` remain verbatim source snapshots
with their original namespaces. They are separate from the implementations
being developed here.
