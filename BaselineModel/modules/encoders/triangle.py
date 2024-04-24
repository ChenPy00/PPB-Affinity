# +
import math
from typing import Callable, Optional, Tuple

import einops
from einops.layers.torch import Rearrange

import torch
from torch import nn
from torch import nn, einsum
import torch.nn.functional as F


# +
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


# -

def variance_scaling_init_(
    weight: torch.Tensor,
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "truncated_normal",
) -> None:
    fan_out, fan_in = weight.shape
    if mode == "fan_in":
        scale = scale / max(1.0, fan_in)
    elif mode == "fan_out":
        scale = scale / max(1.0, fan_out)
    elif mode == "fan_avg":
        scale = scale / max(1.0, (fan_in + fan_out) / 2.0)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if distribution == "truncated_normal":
        std = math.sqrt(scale) / 0.87962566103423978
        nn.init.trunc_normal_(weight, 0.0, std)
    elif distribution == "normal":
        std = math.sqrt(scale)
        nn.init.normal_(weight, 0.0, std)
    elif distribution == "uniform":
        limit = math.sqrt(3.0 * scale)
        nn.init.uniform_(weight, -limit, limit)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                variance_scaling_init_(self.weight, 1.0, "fan_in", "truncated_normal")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "relu":
                variance_scaling_init_(self.weight, 2.0, "fan_in", "truncated_normal")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "glorot":
                variance_scaling_init_(self.weight, 1.0, "fan_avg", "uniform")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "normal":
                variance_scaling_init_(self.weight, 1.0, "fan_in", "normal")
                if bias:
                    nn.init.zeros_(self.bias)
            elif init == "gating":
                nn.init.zeros_(self.weight)
                if bias:
                    nn.init.ones_(self.bias)
            elif init == "final":
                nn.init.zeros_(self.weight)
                if bias:
                    nn.init.zeros_(self.bias)
            else:
                raise ValueError(f"Invalid init: {init}")


class Attention(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.inf = 2.0 ** 15
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.q_proj = Linear(embed_dim, num_heads * head_dim, bias=False, init="glorot")
        self.k_proj = Linear(embed_dim, num_heads * head_dim, bias=False, init="glorot")
        self.v_proj = Linear(embed_dim, num_heads * head_dim, bias=False, init="glorot")
        self.gate_proj = Linear(embed_dim, num_heads * head_dim, init="gating")
        self.out_proj = Linear(num_heads * head_dim, embed_dim, init="final")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        import pdb
        pdb.set_trace()
        x = self.norm(x)
        query = einops.rearrange(
            self.q_proj(x),
            "... i (h c) -> ... h i c",
            h=self.num_heads,
            c=self.head_dim,
        )
        key = einops.rearrange(
            self.k_proj(x),
            "... j (h c) -> ... h j c",
            h=self.num_heads,
            c=self.head_dim,
        )
        value = einops.rearrange(
            self.v_proj(x),
            "... j (h c) -> ... h j c",
            h=self.num_heads,
            c=self.head_dim,
        )
        gate = einops.rearrange(
            torch.sigmoid(self.gate_proj(x)),
            "... i (h c) -> ... h i c",
            h=self.num_heads,
            c=self.head_dim,
        )
        import pdb
        pdb.set_trace()
        logits = torch.einsum("...ic,...jc->...ij", self.scale * query, key)
        if attn_bias is not None:
            logits += attn_bias
        attn_mask = einops.rearrange(mask, "... j -> ... 1 1 j")
        logits = logits.masked_fill(attn_mask < 0.5, -self.inf)
        attn = torch.softmax(logits, dim=-1)
        out = gate * torch.einsum("...ij,...jc -> ...ic", attn, value)
        out = einops.rearrange(out, "... h i c -> ... i (h c)")
        out = self.out_proj(out)
        return out


try:
    from transformers.models.esm.modeling_esmfold import EsmFoldTriangleAttention as TriangleAttention
except:
    class TriangleAttention(nn.Module):
        def __init__(
            self,
            *,
            dim,
            hidden_dim = None,
            num_heads = None,
            mode = 'ingoing'
        ):
            super().__init__()
            if mode not in ("starting", "ending"):
                raise ValueError(f"Invalid mode: {mode}")

            hidden_dim = default(hidden_dim, dim)
            num_heads = default(num_heads, 8)
            self.attn = Attention(dim, hidden_dim, num_heads)
            self.mode = mode

        def forward(self, pair: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
            if self.mode == "ending":
                pair = einops.rearrange(pair, "... i j d -> ... j i d")
                mask_2d = einops.rearrange(mask_2d, "... i j -> ... j i")
            out = self.attn(pair, mask_2d)
            if self.mode == "ending":
                out = einops.rearrange(out, "... j i d -> ... i j d")
            return out


# +
class TriangleAttention(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None,
        num_heads = None,
        mode = 'starting'
    ):
        super().__init__()
        if mode not in ("starting", "ending"):
            raise ValueError(f"Invalid mode: {mode}")

        hidden_dim = default(hidden_dim, dim)
        num_heads = default(num_heads, 8)
        self.attn = Attention(dim, hidden_dim, num_heads)
        self.mode = mode

    def forward(self, pair: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        if self.mode == "ending":
            pair = einops.rearrange(pair, "... i j d -> ... j i d")
            mask_2d = einops.rearrange(mask_2d, "... i j -> ... j i")
        out = self.attn(pair, mask_2d)
        if self.mode == "ending":
            out = einops.rearrange(out, "... j i d -> ... i j d")
        return out
    
# TriangleAttention(128,128,2, 'ending')(torch.ones(8,25,17,128), torch.ones(8,25,17)).shape


# +
class TriangleMultiplicativeModule(nn.Module):
        def __init__(
            self,
            *,
            dim,
            hidden_dim = None,
            mode = 'ingoing'
        ):
            super().__init__()
            assert mode in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

            hidden_dim = default(hidden_dim, dim)
            self.norm = nn.LayerNorm(dim)

            self.left_proj = nn.Linear(dim, hidden_dim)
            self.right_proj = nn.Linear(dim, hidden_dim)

            self.left_gate = nn.Linear(dim, hidden_dim)
            self.right_gate = nn.Linear(dim, hidden_dim)
            self.out_gate = nn.Linear(dim, hidden_dim)

            # initialize all gating to be identity

            for gate in (self.left_gate, self.right_gate, self.out_gate):
                nn.init.constant_(gate.weight, 0.)
                nn.init.constant_(gate.bias, 1.)

            if mode == 'outgoing':
                self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
            elif mode == 'ingoing':
                self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

            self.to_out_norm = nn.LayerNorm(hidden_dim)
            self.to_out = nn.Linear(hidden_dim, dim)

        def forward(self, x, mask = None):
#             assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
            if exists(mask):
                mask = rearrange(mask, 'b i j -> b i j ()')

            x = self.norm(x)

            left = self.left_proj(x)
            right = self.right_proj(x)

            if exists(mask):
                left = left * mask
                right = right * mask
            import pdb
            pdb.set_trace()
            left_gate = self.left_gate(x).sigmoid()
            right_gate = self.right_gate(x).sigmoid()
            out_gate = self.out_gate(x).sigmoid()

            left = left * left_gate
            right = right * right_gate

            out = einsum(self.mix_einsum_eq, left, right)

            out = self.to_out_norm(out)
            out = out * out_gate
            return self.to_out(out)

TriangleMultiplicativeModule(dim=128)(torch.randn(8,25,17,128))
# -

ligand_hidden = torch.ones(8,12,1)
receptor_hidden = torch.ones(8,25,1)
out = einsum('bic, bjc->bic', ligand_hidden, receptor_hidden)
print(out.shape, out)

ligand_hidden = torch.ones(8,12,1)
receptor_hidden = torch.ones(8,25,1)
c = torch.randn(8,25,1)
out = einsum('bic, bjc, bkc->bic', ligand_hidden, receptor_hidden, c)
print(out.shape, out)



try:
    from transformers.models.esm.modeling_esmfold import EsmFoldTriangleMultiplicativeUpdate as TriangleMultiplicativeModule
except:
    class TriangleMultiplicativeModule(nn.Module):
        def __init__(
            self,
            *,
            dim,
            hidden_dim = None,
            mode = 'ingoing'
        ):
            super().__init__()
            assert mode in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

            hidden_dim = default(hidden_dim, dim)
            self.norm = nn.LayerNorm(dim)

            self.left_proj = nn.Linear(dim, hidden_dim)
            self.right_proj = nn.Linear(dim, hidden_dim)

            self.left_gate = nn.Linear(dim, hidden_dim)
            self.right_gate = nn.Linear(dim, hidden_dim)
            self.out_gate = nn.Linear(dim, hidden_dim)

            # initialize all gating to be identity

            for gate in (self.left_gate, self.right_gate, self.out_gate):
                nn.init.constant_(gate.weight, 0.)
                nn.init.constant_(gate.bias, 1.)

            if mode == 'outgoing':
                self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
            elif mode == 'ingoing':
                self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

            self.to_out_norm = nn.LayerNorm(hidden_dim)
            self.to_out = nn.Linear(hidden_dim, dim)

        def forward(self, x, mask = None):
            assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
            if exists(mask):
                mask = rearrange(mask, 'b i j -> b i j ()')

            x = self.norm(x)

            left = self.left_proj(x)
            right = self.right_proj(x)

            if exists(mask):
                left = left * mask
                right = right * mask

            left_gate = self.left_gate(x).sigmoid()
            right_gate = self.right_gate(x).sigmoid()
            out_gate = self.out_gate(x).sigmoid()

            left = left * left_gate
            right = right * right_gate

            out = einsum(self.mix_einsum_eq, left, right)

            out = self.to_out_norm(out)
            out = out * out_gate
            return self.to_out(out)





# +
# Copyright 2022 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import math

import torch
import torch.distributed as dist
from torch import Tensor

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(numerator, denominator)

def divide(numerator, denominator):
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[dim], torch.distributed.get_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    output = tensor_list[torch.distributed.get_rank()]

    return output

def _gather(tensor: Tensor, dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    if dim == 0:
        output_shape = list(tensor.shape)
        output_shape[dim] *= torch.distributed.get_world_size()
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = output.chunk(torch.distributed.get_world_size(), dim=dim)
        dist.all_gather(list(tensor_list),
                        tensor,
                        async_op=False)
    else:
        tensor_list = [
            torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        dist.all_gather(tensor_list,
                        tensor,
                        async_op=False)
        output = torch.cat(tensor_list, dim=dim)

    return output

def scatter(input: Tensor, dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input

    input = Scatter.apply(input, dim)
    return input

def gather(input: Tensor, dim: int = -1, dtype=torch.float) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input
    
    input_dtype = input.dtype
    input = input.to(dtype)
    input = Gather.apply(input, dim)
    input = input.to(input_dtype)
    return input

def _all_to_all(tensor: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return tensor

    split_size = divide(tensor.shape[in_dim], torch.distributed.get_world_size())
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)

    input_tensor_list = [tensor_.contiguous() for tensor_ in input_tensor_list]
    output_tensor_list = [torch.ones_like(tensor_) for tensor_ in input_tensor_list]
    dist.all_to_all(output_tensor_list,
                    input_tensor_list,
                    async_op=False)
    output = torch.cat(output_tensor_list, dim=out_dim)

    return output

def col_to_row(input_: Tensor, dtype=torch.float) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input_
    input_dtype = input_.dtype
    input_ = input_.to(dtype)
    input_ = AlltoAll.apply(input_, 0, 1)
    input_ = input_.to(input_dtype)
    return input_

def row_to_col(input_: Tensor, dtype=torch.float) -> Tensor:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return input_
    input_dtype = input_.dtype
    input_ = input_.to(dtype)
    input_ = AlltoAll.apply(input_, 1, 0)
    input_ = input_.to(input_dtype)
    return input_

class Scatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Scatter", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: "Scatter", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim=int(dim)), None

class Gather(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Gather", input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: "Gather", grad_output: Tensor) -> Tuple[Tensor]:
        dim, = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None

class AlltoAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "AlltoAll", input_: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([in_dim, out_dim]))
        return _all_to_all(input_, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(ctx: "AlltoAll", grad_output: Tensor) -> Tuple[Tensor]:
        saved_tensors = ctx.saved_tensors[0]
        return _all_to_all(grad_output, in_dim=int(saved_tensors[1]),
                           out_dim=int(saved_tensors[0])), None, None


# +
from typing import Optional, Callable, List, Tuple, Union
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import truncnorm
from functools import partialmethod, partial

def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = [i for i in range(len(tensor.shape[:zero_index]))]
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class LayerNorm(nn.Module):

    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True, comm_dtype=torch.float, low_mem=False):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.comm_dtype = comm_dtype
        self.low_mem = low_mem

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor
            mask:
                [*, N_res, N_res] input mask
        Returns:
            [*, N_res, N_res, C_z] output tensor
        """
        if mask is None:
            mask = torch.ones(z.shape[:-1]).to(device=z.device, dtype=z.dtype)

        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)

        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        a = a * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        b = b * mask

        if self._outgoing:
            b = gather(b, dim=0, dtype=self.comm_dtype)
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = gather(a, dim=1, dtype=self.comm_dtype)
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))
        
        #x = torch.bmm(a, b)
        x = einsum('bcij,bcji->bijc',a,b)

#         x = permute_final_dims(x, (1, 2, 0))
        x = self.layer_norm_out(x)

        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        z = x * g

        return z


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)


# -

from torch import einsum
TriangleMultiplicationOutgoing(128,128)(torch.randn(8,25,17,128))
