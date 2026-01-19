# PYTHON_IS_DUMB.py
# Because Python can't handle simple constant sharing without circular import tantrums

# Copyright 2025 Diego Cifrian Bueno
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


COMPILE_DISABLED = {"state":False}

COMPILE_OPTIONS = {"triton.cudagraphs": True,
                   "epilogue_fusion": True,
                   "shape_padding": True,
                   'coordinate_descent_tuning': True,
                   "max_autotune": True,
                   "b2b_gemm_pass": True,
                   "aggressive_fusion": True,
                   "max_autotune_gemm": True,
                   "memory_planning": False,  # Does reduce VRAM a bit, but also reduces speed a bit
                   "triton.cooperative_reductions": False,  # Slower
                   "triton.use_block_ptr": False,  # Slower, a few MB more of VRAM
                   "triton.max_tiles": 1,  # Slower with 4
                   "use_fast_math": False,  # No difference
                   "cuda.use_fast_math": True,
                   "max_autotune_pointwise": True,
                   "group_fusion": True,
                   "permute_fusion": False, # Issues with quantization
                   "force_pointwise_cat": True,
                   "max_fusion_size": 1024,
                   "max_pointwise_cat_inputs": 8,  # Maybe a tiny bit slower?
                   "triton.dense_indexing": True,
                   "size_asserts": False,  # No difference, might be a footgun.
                   "scalar_asserts": False,  # No difference, might be a footgun.
                   "triton.spill_threshold": 32,
                   "triton.min_split_scan_rblock": 16,
                   #"combo_kernels": True,
                   #"combo_kernels_autotune": 40,
                   "force_fuse_int_mm_with_mul": True
                   }