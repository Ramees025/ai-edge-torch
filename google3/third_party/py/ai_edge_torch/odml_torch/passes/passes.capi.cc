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
#include "third_party/py/ai_edge_torch/odml_torch/passes/passes.h"

#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "third_party/py/ai_edge_torch/odml_torch/passes/passes.capi.h.inc"

using namespace mlir;

#ifdef __cplusplus
extern "C" {
#endif

#include "third_party/py/ai_edge_torch/odml_torch/passes/passes.capi.cc.inc"

#ifdef __cplusplus
}
#endif
