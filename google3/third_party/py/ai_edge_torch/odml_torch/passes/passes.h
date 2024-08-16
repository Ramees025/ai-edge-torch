// Copyright 2024 The AI Edge Torch Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
#ifndef THIRD_PARTY_PY_AI_EDGE_TORCH_ODML_TORCH_PASSES_PASSES_H_
#define THIRD_PARTY_PY_AI_EDGE_TORCH_ODML_TORCH_PASSES_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func

#define GEN_PASS_DECL
#include "third_party/py/ai_edge_torch/odml_torch/passes/passes.h.inc"  // IWYU pragma: keep

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "third_party/py/ai_edge_torch/odml_torch/passes/passes.h.inc"

}  // namespace mlir

#endif  // THIRD_PARTY_PY_AI_EDGE_TORCH_ODML_TORCH_PASSES_PASSES_H_
