# Copyright 2024 The AI Edge Torch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests Python bindings for ODML Torch MLIR passes."""
from ai_edge_torch.odml_torch.passes import passes
from jax._src.lib.mlir import ir

from absl.testing import absltest as googletest


class PassesModuleTest(googletest.TestCase):

  def test_lift_callsite_loc_caller_binding(self):
    with ir.Context(), ir.Location.unknown():
      module = ir.Module.create()
      passes.lift_callsite_loc_caller(module)
    pass

  def test_build_stablehlo_composite_binding(self):
    with ir.Context(), ir.Location.unknown():
      module = ir.Module.create()
      passes.build_stablehlo_composite(module)
    pass


if __name__ == "__main__":
  googletest.main()
