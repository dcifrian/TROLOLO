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

import math
from typing import Literal, Type

import torch
import torch.nn as nn

from TROLOLO.PYTHON_IS_DUMB import COMPILE_OPTIONS,COMPILE_DISABLED


def generate_orthogonal_matrix(rows: int, cols: int) -> torch.Tensor:
    """
    Generate a semi-orthogonal matrix of size (rows, cols) using QR decomposition.
    """
    # For QR decomposition, we need the first dimension >= second dimension
    # So we generate a matrix of shape (max(rows,cols), min(rows,cols))
    # and then transpose if necessary
    if rows >= cols:
        a = torch.randn(rows, cols)
        q, r = torch.qr(a)
        # q is already (rows, cols)
        signs = torch.sign(torch.diag(r))
        return q * signs.unsqueeze(0)
    else:
        # For rows < cols case, we transpose the problem and then transpose back
        a = torch.randn(cols, rows)
        q, r = torch.qr(a)
        signs = torch.sign(torch.diag(r))
        # Need to take first 'rows' columns of q and transpose
        return (q[:, :rows] * signs.unsqueeze(0)).t()


class SVDLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int,
            bias,
            init_method: Literal['orthogonal', 'normal', 'uniform'] = 'orthogonal',
            init_scale: float = 1.0,
            bits=None,
            v_LoRAs=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bits=bits
        self.quantize_training=self.bits is not None
        self.vLoRAs = v_LoRAs
        # Initialize matrices based on specified method
        if init_method == 'orthogonal':
            # Generate semi-orthogonal matrices and scale them
            u_init = generate_orthogonal_matrix(rank, out_features) * (init_scale / (rank) ** 0.25)
            v_init = generate_orthogonal_matrix(rank, in_features) * (init_scale / (rank) ** 0.25)

        elif init_method == 'normal':
            # Initialize with normal distribution
            u_init = torch.randn(rank, out_features) * (init_scale / (rank * out_features) ** 0.25)
            v_init = torch.randn(rank, in_features) * (init_scale / (rank * in_features) ** 0.25)

        else:  # uniform
            # Initialize with uniform distribution
            bound = init_scale * (3 / (rank * max(in_features, out_features))) ** 0.5
            u_init = torch.rand(rank, out_features).uniform_(-bound, bound)
            v_init = torch.rand(rank, in_features).uniform_(-bound, bound)
        v_init = v_init.T.contiguous()
        self.U = nn.Parameter(u_init)
        self.V = nn.Parameter(v_init)
        self.bias = bias
        if not isinstance(bias, nn.Parameter) and bias == True:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(size=(1, in_features, out_features), device="cpu", dtype=torch.int8))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Quantization parameters if enabled
        if self.quantize_training:
            self.forward=self.forward_quant
            self.forward_no_bias=self.forward_no_bias_quant
            # Initialize scale and zero_point for U and V
            self.u_scale = nn.Parameter(torch.ones((rank,1),device="cuda"))
            self.u_zero_point = nn.Parameter(torch.zeros((rank,1),device="cuda"))
            self.v_scale = nn.Parameter(torch.ones((rank,1),device="cuda"))
            self.v_zero_point = nn.Parameter(torch.zeros((rank,1),device="cuda"))

            # Initialize scales based on initial weight distribution per rank
            with torch.no_grad():
                quant_max = (2 ** self.bits) - 1
                # Initialize U quantization parameters per rank
                for i in range(rank):
                    u_min, u_max = self.U[i].min(), self.U[i].max()
                    self.u_scale[i][0] = (u_max - u_min) / quant_max
                    self.u_zero_point[i][0] = -u_min / self.u_scale[i][0]
                # Initialize V quantization parameters per rank (for transposed V)
                for i in range(rank):
                    v_min, v_max = self.V.T[i].min(), self.V.T[i].max()  # Transpose for quantization
                    self.v_scale[i][0] = (v_max - v_min) / quant_max
                    self.v_zero_point[i][0] = -v_min / self.v_scale[i][0]
        if self.vLoRAs:
            self.base_U = self.U
            self.base_V = self.V
            if self.quantize_training:
                self.base_v_scale=self.v_scale
                self.base_u_scale = self.u_scale
                self.base_v_zero_point = self.v_zero_point
                self.base_u_zero_point = self.u_zero_point

        if bias is not None:
            self._forward_train = torch.compile(self.forward, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
            self._forward_inference = torch.compile(self.forward, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
        else:
            self._forward_train = torch.compile(self.forward_no_bias, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
            self._forward_inference = torch.compile(self.forward_no_bias, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
        self.forward = self._forward_train



    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.forward = self._forward_train
        else:
            self.forward = self._forward_inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, self.V), self.U) + self.bias

    def forward_no_bias(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(x, self.V), self.U)

    def forward_quant(self, x: torch.Tensor) -> torch.Tensor:
        U = fake_quantize_tensor(self.U, self.u_scale.abs(), self.u_zero_point, self.bits)
        V = fake_quantize_tensor(self.V.T, self.v_scale.abs(), self.v_zero_point, self.bits).T
        return torch.matmul(torch.matmul(x, V), U) + self.bias

    def forward_no_bias_quant(self, x: torch.Tensor) -> torch.Tensor:
        U = fake_quantize_tensor(self.U, self.u_scale.abs(), self.u_zero_point, self.bits)
        V = fake_quantize_tensor(self.V.T, self.v_scale.abs(), self.v_zero_point, self.bits).T
        return torch.matmul(torch.matmul(x, V), U)

    def activate_vLoRAs(self,ids):
        for i,idL in enumerate(ids):
            vLora = self.vLoRAs[idL]
            if i == 0:
                self.U = self.base_U[vLora[0]:vLora[1],:]
                self.V = self.base_V[:,vLora[0]:vLora[1]]
                if self.quantize_training:
                    self.v_scale = self.base_v_scale[vLora[0]:vLora[1]]
                    self.v_zero_point = self.base_v_zero_point[vLora[0]:vLora[1],:]
                    self.u_scale = self.base_u_scale[vLora[0]:vLora[1]]
                    self.u_zero_point = self.base_u_zero_point[vLora[0]:vLora[1],:]
            else:
                self.U = torch.cat([self.U,self.base_U[vLora[0]:vLora[1],:]])
                self.V = torch.cat([self.V, self.base_V[:,vLora[0]:vLora[1]]])
                if self.quantize_training:
                    self.v_scale = torch.cat([self.v_scale,self.base_v_scale[vLora[0]:vLora[1],:]])
                    self.v_zero_point = torch.cat([self.v_zero_point,self.base_v_zero_point[vLora[0]:vLora[1],:]])
                    self.u_scale = torch.cat([self.u_scale,self.base_u_scale[vLora[0]:vLora[1],:]])
                    self.u_zero_point = torch.cat([self.u_zero_point,self.base_u_zero_point[vLora[0]:vLora[1],:]])

    # @property
    # def weight(self) -> torch.Tensor:
    #    """
    #    Returns the equivalent full weight matrix (for compatibility with nn.Linear)
    #    """
    #    print("Called")
    #    return torch.einsum('ki,kj->ij', self.U, self.V)


def analyze_matrices(svdlinear: SVDLinear):
    U = svdlinear.U
    V = svdlinear.V
    # Reconstruct full matrix
    W = torch.einsum('ik,ij->kj', U, V)

    # Look at how values distribute
    bins = 10
    u_dist = torch.histc(U.flatten(), bins=bins)
    v_dist = torch.histc(V.flatten(), bins=bins)
    w_dist = torch.histc(W.flatten(), bins=bins)
    # Check for structure/patterns
    u_correlation = torch.histc(torch.abs(torch.corrcoef(U)).flatten(), bins=bins).int()
    v_correlation = torch.histc(torch.abs(torch.corrcoef(V)).flatten(), bins=bins).int()
    # How orthogonal are the components?
    u_orthogonality = torch.histc(torch.abs(torch.mm(U.T, U)).flatten(), bins=bins).int()
    v_orthogonality = torch.histc(torch.abs(torch.mm(V.T, V)).flatten(), bins=bins).int()

    # Check zero patterns
    zero_thresh = 1e-5  # or some small threshold
    true_zeros = (W.abs() < zero_thresh).sum().item()
    zerosU = (U.abs() < zero_thresh).sum().item()
    zerosV = (V.abs() < zero_thresh).sum().item()

    # Look at row/column usage patterns
    row_norms = torch.norm(W, dim=1)  # How much each output uses inputs
    col_norms = torch.norm(W, dim=0)  # How much each input contributes

    # Check for correlation between rows/columns
    # High correlation might indicate redundant features
    row_corr = torch.corrcoef(W)

    return {
        'shape': W.shape,
        'size': W.numel(),
        'zeros': (true_zeros, zerosU, zerosV),
        'active_rows': (row_norms > zero_thresh).sum().item(),
        'active_cols': (col_norms > zero_thresh).sum().item(),
        'U_dist': u_dist.int().cpu().numpy(),
        'V_dist': v_dist.int().cpu().numpy(),
        'W_dist': w_dist.int().cpu().numpy(),
        'U_ortogonality_hist': u_orthogonality.cpu().numpy(),
        'V_ortogonality_hist': v_orthogonality.cpu().numpy(),
        'U_correlation_hist': u_correlation.cpu().numpy(),
        'V_correlation_hist': v_correlation.cpu().numpy(),
        # 'row_norms': row_norms.cpu().numpy(),
        # 'col_norms': col_norms.cpu().numpy(),
        # 'row_corr': row_corr.cpu().numpy()
    }


def pseudoSVs(svdlinear: SVDLinear):
    U = svdlinear.U
    V = svdlinear.V
    W = torch.einsum('ik,ij->kj', U, V)
    U_svd, S_svd, V_svd = torch.svd(W)
    u_norms = torch.norm(U, dim=1)
    v_norms = torch.norm(V, dim=1)
    SVs = u_norms * v_norms

    # Current utilization
    relative_last = SVs[-1] / SVs.mean()

    # Distribution sharpness
    SVs_spread = SVs.std() / SVs.mean()

    # Combine factors
    expansion_score = relative_last * (1 - SVs_spread)

    return {
        'pseudoSVs': SVs.cpu().numpy(),
        'trueSVD_SVs': S_svd[:SVs.numel()].cpu().numpy(),
        'Mean SV': SVs.mean().cpu().item(),
        # 'expansion_score':expansion_score.cpu().item(),
        # 'relative_last':relative_last.cpu().item(),
        'SVs_spread': SVs_spread.cpu().item()
    }


def replace_2Dlayers_with_svd(
        model,
        rank: float = 0.5,
        init_method: Literal['orthogonal', 'normal', 'uniform'] = 'orthogonal',
        init_scale: float = 1.0,
        linear_class: Type[nn.Linear] = nn.Linear):
    """
    Replaces all Linear layers within MLP blocks of a transformer with SVD structure.

    Args:
        model: The transformer model
        rank: Optional rank for SVD decomposition. If None, uses min(in_features, out_features)
        init_method: Initialization method for U and V matrices
            - 'orthogonal': Semi-orthogonal initialization using QR decomposition
            - 'normal': Normal distribution initialization
            - 'uniform': Uniform distribution initialization
        init_scale: Scaling factor for initialization
        linear_class: The class type to identify linear layers (default nn.Linear)

    Returns:
        Modified model with SVD structure in place of MLP linear layers
    """

    def is_replaceable(name: str) -> bool:
        mlp_patterns = ['mlp', 'ffn', 'feed_forward', 'self_attention']
        return any(pattern in name.lower() for pattern in mlp_patterns)

    for name, module in model.named_modules():
        if isinstance(module, linear_class) and is_replaceable(name):
            r = int(rank * min(module.in_features, module.out_features))
            print(f"Replacing {name} {module.in_features} {module.out_features} with SVD r:{r}")
            svd_layer = SVDLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=r,
                bias=module.bias,
                init_method=init_method,
                init_scale=init_scale
            )
            svd_layer.cuda()
            # Replace the linear layer with SVD layer
            parent = model
            for part in name.split('.')[:-1]:
                parent = getattr(parent, part)

            last_part = name.split('.')[-1]
            setattr(parent, last_part, svd_layer)

    return model

class QuantizationFunction(torch.autograd.Function):
    """Straight-through estimator for quantization with per-row quantization"""

    @staticmethod
    def forward(ctx, x, scale, zero_point, bits=8):
        # Store for backward pass
        ctx.save_for_backward(x, scale, zero_point)
        ctx.bits = bits

        #print(x.shape,scale.shape,zero_point.shape)
        # Quantize
        quant_min = 0
        quant_max = (2 ** bits) - 1

        # x is (rank, features), scale and zero_point are (rank,)
        # Scale and shift per row
        x_scaled = x / scale + zero_point

        # Clamp and round
        x_quant = torch.clamp(torch.round(x_scaled), quant_min, quant_max)

        # Dequantize
        x_dequant = (x_quant - zero_point) * scale

        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, zero_point = ctx.saved_tensors

        # Gradient w.r.t. input: straight-through
        grad_x = grad_output.clone()

        # Gradient w.r.t. scale and zero_point (per-row)
        quant_min = 0
        quant_max = (2 ** ctx.bits) - 1
        x_scaled = x / scale + zero_point
        x_quant = torch.clamp(torch.round(x_scaled), quant_min, quant_max)

        # Gradient for scale (sum over features dimension)
        grad_scale = torch.sum(grad_output * (x_quant - zero_point), dim=1, keepdim=True)
        #print(grad_scale.shape,grad_output.shape, x_quant.shape, zero_point.shape)

        # Gradient for zero_point (sum over features dimension)
        grad_zero_point = torch.sum(grad_output * (-scale), dim=1, keepdim=True)

        return grad_x, grad_scale, grad_zero_point, None


def fake_quantize_tensor(x, scale, zero_point, bits=8):
    """Apply fake quantization with learnable scale and zero_point"""
    return QuantizationFunction.apply(x, scale, zero_point, bits)