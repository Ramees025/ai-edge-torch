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
include "third_party/llvm/llvm-project/mlir/include/mlir/Pass/PassBase.td"

def LiftCallSiteLocCallerPass : Pass<"lift-callsite-loc-caller", "ModuleOp"> {
  let summary = "";
  let description = [{
    Empty
  }];
  let dependentDialects = ["func::FuncDialect"];
}

def BuildStableHLOCompositePass : Pass<"build-stablehlo-composite", "ModuleOp"> {
  let summary = "Build stablehlo.composite from inlined stablehlo.custom_call mark_tensor ops.";
  let description = [{
    Empty
  }];
  let dependentDialects = ["func::FuncDialect", "stablehlo::StablehloDialect"];
}