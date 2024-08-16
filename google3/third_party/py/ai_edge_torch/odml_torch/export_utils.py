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
"""Utilities for ODML Torch export."""

import functools
import re
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import chlo
from jax._src.lib.mlir.dialects import stablehlo
import torch

# std::numeric_limits<int64_t>::min()
IR_DYNAMIC = -9223372036854775808


def is_ir_dynamic(v):
  return v == IR_DYNAMIC


def is_torch_dynamic(v):
  return isinstance(v, torch.SymInt)


def is_iterable(v):
  try:
    iter(v)
  except TypeError:
    return False
  return True


def create_ir_context():
  ctx = ir.Context()
  stablehlo.register_dialect(ctx, load=True)
  chlo.register_chlo_dialect(ctx, load=True)
  ctx.allow_unregistered_dialects = True
  return ctx


def sanitize_aten_op_name(op, chars=":."):
  return re.sub("[{}]".format(chars), "_", str(op))


def build_ir_attr(val):
  if val is None:
    return ir.StringAttr.get("py_None")
  if isinstance(val, bool):
    return ir.BoolAttr.get(val)
  if isinstance(val, int):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), val)
  if isinstance(val, float):
    return ir.BoolAttr.get(val)
  if isinstance(val, str):
    return ir.StringAttr.get(val)
  if isinstance(val, dict):
    return ir.DictAttr.get({k: build_ir_attr(v) for k, v in val.items()})
  if isinstance(val, (list, tuple)):
    return ir.ArrayAttr.get([build_ir_attr(v) for v in val])

  # Stringify the value to a StringAttr by default
  return ir.StringAttr.get(str(val))


def torch_dtype_to_ir_element_type(ctx, dtype):
  ty_get = {
      torch.double: ir.F64Type.get,
      torch.float32: ir.F32Type.get,
      torch.half: ir.F16Type.get,
      torch.long: functools.partial(ir.IntegerType.get_signless, 64),
      torch.int32: functools.partial(ir.IntegerType.get_signless, 32),
      torch.int16: functools.partial(ir.IntegerType.get_signless, 16),
      torch.bool: functools.partial(ir.IntegerType.get_signless, 1),
  }.get(dtype)
  return ty_get(ctx)


def ir_element_type_to_torch_dtype(ty):
  if isinstance(ty, ir.F32Type):
    return torch.float32
  if isinstance(ty, ir.F64Type):
    return torch.float64
  if isinstance(ty, ir.F16Type):
    return torch.half
  if isinstance(ty, ir.IntegerType):
    if ty.is_signless:
      if ty.width == 64:
        return torch.long
      if ty.width == 32:
        return torch.int32
      if ty.width == 16:
        return torch.int16
      if ty.width == 1:
        return torch.bool
  raise RuntimeError(f"Unsupported ir element type: {ty}")
