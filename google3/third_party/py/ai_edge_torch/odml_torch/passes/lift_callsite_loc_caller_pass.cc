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
#include <iostream>

#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Location.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/IR/Operation.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Support/TypeID.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Transforms/DialectConversion.h"
#include "third_party/py/ai_edge_torch/odml_torch/passes/passes.h"

namespace mlir {
#define GEN_PASS_DEF_LIFTCALLSITELOCCALLERPASS
#include "third_party/py/ai_edge_torch/odml_torch/passes/passes.h.inc"

namespace {

class LiftCallSiteLocCallerPass
    : public impl::LiftCallSiteLocCallerPassBase<LiftCallSiteLocCallerPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LiftCallSiteLocCallerPass);

  void runOnOperation() override {
    getOperation()->walk([](func::FuncOp func_op) {
      for (Operation& op : func_op.getOps()) {
        if (!op.getLoc().isa<CallSiteLoc>()) {
          continue;
        }

        auto loc = op.getLoc().dyn_cast<CallSiteLoc>();
        op.setLoc(loc.getCaller());
      }
    });
  }
};

}  // namespace
}  // namespace mlir
