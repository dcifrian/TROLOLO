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

import torch
import torch.nn as nn
import math
from collections import OrderedDict
from functools import partial
from typing import Callable
from TROLOLO.SVD_layer import SVDLinear

torch._dynamo.config.recompile_limit=640

from TROLOLO.PYTHON_IS_DUMB import COMPILE_OPTIONS,COMPILE_DISABLED

class Conv2dAsymPad(nn.Conv2d):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size ,
        stride  = 1,
        padding: tuple[int,int,int,int] = (0,0,0,0),
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,):
        super(Conv2dAsymPad,self).__init__(in_channels,out_channels,kernel_size,stride,0,dilation,groups,bias,"zeros",device,dtype)
        self.asym_pad=nn.ZeroPad2d(padding)
        self.forward = torch.compile(self.forward, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])

    def forward(self,x):
        return  super(Conv2dAsymPad,self).forward(self.asym_pad(x))

class PosEmbedding2D(nn.Module):

    def __init__(self, seq_length, n_class_tokens, x_size, y_size, **kwargs):
        super().__init__()
        self.reserved_dims=2
        n_class_tokens=int(n_class_tokens)
        x_size=int(x_size)
        y_size=int(y_size)
        seq_length=int(seq_length)
        pos_embedding_x = torch.zeros(seq_length)
        pos_embedding_y = torch.zeros(seq_length)
        pos_embedding_x[n_class_tokens:] = (torch.arange(start=1,end=x_size+1).repeat(y_size, 1).flatten() / (x_size + 1))[:seq_length-n_class_tokens] # 0 to 1
        pos_embedding_y[n_class_tokens:] = (torch.arange(start=1,end=y_size+1).repeat(x_size, 1).T.flatten() / (x_size + 1))[:seq_length-n_class_tokens]  # 0 to 1
        self.embedding_scale = torch.nn.Parameter(torch.ones(1))
        pos_embedding = torch.stack([pos_embedding_x, pos_embedding_y], dim=1)  # [256, 2]
        #self.pos_embedding = self.pos_embedding[n_class_tokens:,:]
        class_embedding = -torch.arange(start=1,end=n_class_tokens+1) / n_class_tokens
        class_embedding = torch.stack([class_embedding,-torch.arange(start=n_class_tokens+1,end=1,step=-1) / (n_class_tokens + 1)],dim=1)
        pos_embedding[:n_class_tokens,:]=class_embedding
        self.register_buffer("pos_embedding",pos_embedding, persistent=False)
        # Now each token is [190 features + x_coord + y_coord] = 192 dim
        self.forward = torch.compile(self.forward, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])

    def forward(self, input):
        # Create position embeddings with scale and bias
        pos_embed_adjusted = (self.embedding_scale + 0.01) * self.pos_embedding
        pos_embed_batch = pos_embed_adjusted.expand(input.shape[0], -1, -1)
        return torch.cat([input, pos_embed_batch], dim=2)


class PosEmbedding1D(nn.Module):

    def __init__(self, seq_length, n_class_tokens, **kwargs):
        super().__init__()
        self.reserved_dims = 1
        pos_embedding = torch.zeros(seq_length)
        pos_embedding[n_class_tokens:] = (torch.arange(start=1,end=seq_length+1-n_class_tokens) / (seq_length+1-n_class_tokens))[:seq_length-n_class_tokens] # 0 to 1
        self.embedding_scale = torch.nn.Parameter(torch.ones(1))
        class_embedding = -torch.arange(start=1,end=n_class_tokens+1) / n_class_tokens
        pos_embedding[:n_class_tokens]=class_embedding
        pos_embedding=pos_embedding.unsqueeze(dim=1)
        self.register_buffer("pos_embedding",pos_embedding, persistent=False)
        # Now each token is [191 features + pos] = 192 dim
        self.forward = torch.compile(self.forward, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])

    def forward(self, input):
        # Create position embeddings with scale and bias
        pos_embed_adjusted = (self.embedding_scale + 0.01) * self.pos_embedding
        pos_embed_batch = pos_embed_adjusted.expand(input.shape[0], -1, -1)
        return torch.cat([input, pos_embed_batch], dim=2)

class MLPBlockSVD(torch.nn.Sequential):
    """Transformer MLP block with SVD layers."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float, rank: float, out_dim:int=None,quantize_bits: int=None, activation: Callable=nn.GELU):
        params = {}
        layers = []
        hidden_channels=[mlp_dim, in_dim]
        activation_layer=activation
        for embed_dim in hidden_channels[:-1]:
            r=max(1,int(rank*min(in_dim,embed_dim)))
            if rank == 1 or rank == 1.0:
                layers.append(nn.Linear(in_dim, embed_dim, bias=True))
            else:
                bias = nn.Parameter(torch.empty(embed_dim))
                layers.append(SVDLinear(in_dim, embed_dim,r,bias=bias,bits=quantize_bits))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = embed_dim
        r = max(1,int(rank * min(in_dim, hidden_channels[-1])))
        if out_dim is None:
            out_dim=hidden_channels[-1]
        if rank == 1  or rank == 1.0:
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
        else:
            layers.append(SVDLinear(in_dim, out_dim , r,bias=None,bits=quantize_bits) )
        layers.append(torch.nn.Dropout(dropout, **params))
        super().__init__(*layers)
        for m in self.modules():
            if isinstance(m, SVDLinear):
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

class EncoderBlockSVD(nn.Module):
    """Transformer encoder block with SVD layers."""

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        mlp_dim: int,
        mlp_rank: float,
        qkv_rank: float,
        attnproj_rank: float,
        dropout: float,
        attention_dropout: float,
        n_class_tokens: int,
        out_dim: int = None,
        attn_dim = None,
        attn_rank: int=32,
        learnable_skips: int= 1,
        activation: Callable=nn.GELU,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        quantize_bits: int = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim=out_dim
        self.embed_dim=embed_dim
        self.n_class_tokens = n_class_tokens
        # Attention block
        self.ln_1 = norm_layer(embed_dim)
        """if attn_rank is None:
            self.self_attention = timm_Attention(dim=embed_dim,num_heads=num_heads, attn_drop=attention_dropout,norm_layer=None,qkv_bias=False)
            if qkv_rank != 1:
                self.self_attention.qkv = SVDLinear(embed_dim, embed_dim*3, rank=int(embed_dim*qkv_rank), bias=self.self_attention.qkv.bias,bits=quantize_bits)
            if attnproj_rank != 1:
                self.self_attention.proj = SVDLinear(embed_dim, embed_dim, rank=int(embed_dim*attnproj_rank), bias=self.self_attention.proj.bias,bits=quantize_bits)
        else:
            proj_rank=attnproj_rank
            self.self_attention = LowRankAttention(embed_dim,num_heads,qkv_rank,proj_rank,attn_rank,attention_dropout,quantize_bits=quantize_bits)
        """
        self.self_attention = LowRankAttention(embed_dim=embed_dim,
                                               num_heads=num_heads,
                                               qkv_rank=qkv_rank,
                                               proj_rank=attnproj_rank,
                                               n_bottleneck=attn_rank,
                                               attn_drop=attention_dropout,
                                               attn_dim=attn_dim,
                                               proj_bias= True,
                                               quantize_bits=quantize_bits)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation()
        # MLP block
        self.ln_2 = norm_layer(embed_dim)
        self.mlp = MLPBlockSVD(in_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout, rank=mlp_rank, out_dim=out_dim, activation=activation, quantize_bits=quantize_bits)
        if out_dim is not None:
            self.forward=self.forward_outdim
            self.interp_stream = torch.cuda.Stream(priority=1)
            self.default_stream = torch.cuda.default_stream() # Workaround because inductor can't figure out why something in torch.* doesn't return a tensor
            self.interp_event = torch.cuda.Event()
            side = int(math.sqrt(embed_dim))
            height_frac = (embed_dim / side) - (embed_dim // side)
            direction = 1 - 2 * round(height_frac)
            self.side_w = int(side + direction * (height_frac > 1e-10))
            self.side_h = embed_dim // self.side_w
            target_side = int(math.sqrt(self.out_dim))
            height_frac = (self.out_dim / target_side) - (self.out_dim // target_side)
            direction = 1 - 2 * round(height_frac)
            self.target_w = int(target_side + direction * (height_frac > 1e-10))
            self.target_h = self.out_dim // self.target_w
        self.skip_scale =torch.ones(max(1,learnable_skips))
        self.skip_scale_attn = torch.ones(1)
        if learnable_skips > 0:
            self.skip_scale =  nn.Parameter(self.skip_scale, requires_grad=True)
            self.skip_scale_attn = nn.Parameter(self.skip_scale_attn, requires_grad=True)
        self._forward_train = torch.compile(self.forward,fullgraph=True,dynamic=True,options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
        inf_options = COMPILE_OPTIONS.copy()["triton.cudagraphs"] = False
        self._forward_inference = torch.compile(self.forward,fullgraph=True,dynamic=True,options=inf_options, disable=COMPILE_DISABLED["state"])

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.forward = self._forward_train
        else:
            self.forward = self._forward_inference

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, embed_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = x + self.skip_scale * input
        y = self.ln_2(x)
        y = self.mlp(y)
        return y + self.skip_scale_attn * x

    def forward_outdim(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, embed_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = x + self.skip_scale * input
        y = self.ln_2(x)
        y = self.mlp(y)
        # Pool x to match y's dimensionality for skip connection

        with torch.cuda.stream(self.interp_stream):
            self.interp_stream.wait_stream(self.default_stream)
            class_token = x[:, :self.n_class_tokens, :]
            class_token_y = y[:, :self.n_class_tokens, :]
            class_token_y = nn.functional.interpolate(class_token_y, size=self.embed_dim, mode="nearest-exact")
            class_token = class_token + class_token_y
            self.interp_event.record()
        spatial_tokens = x[:, self.n_class_tokens:, :]
        x_pooled = nn.functional.adaptive_avg_pool1d(spatial_tokens, self.out_dim)
        y = y[:, self.n_class_tokens:, :]
        x = y + x_pooled * self.skip_scale_attn
        self.interp_event.wait()
        return x , class_token


    def forward_outdim2(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, embed_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        batch_size = x.shape[0]
        # Pool x to match y's dimensionality for skip connection
        # Assuming embed_dim is a perfect square and we want 2x2 pooling
        class_token = x[:, :self.n_class_tokens, :]
        spatial_tokens = x[:, self.n_class_tokens:, :]
        # print(x.shape, spatial_tokens.shape, embed_dim)
        x_pooled = nn.functional.adaptive_avg_pool1d(spatial_tokens, output_size=self.out_dim)
        #class_token_y = y[:, :self.n_class_tokens, :].reshape(batch_size, class_token.shape[1], self.target_w, self.target_h)
        # print(class_token_y.shape)
        #class_token_y = nn.functional.interpolate(class_token_y, size=(self.side_w, self.side_h), mode="nearest-exact")
        #class_token_y = class_token_y.view(batch_size, class_token.shape[1], class_token.shape[2])

        class_token_y = y[:, :self.n_class_tokens, :]
        # print(class_token_y.shape)
        class_token_y = nn.functional.interpolate(class_token_y, size=self.embed_dim, mode="nearest-exact")
        class_token = class_token + class_token_y
        x = x_pooled
        y = y[:, self.n_class_tokens:, :]
        return x + y, class_token

class EncoderSVD(nn.Module):
    """Transformer Model Encoder with SVD layers."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        mlp_dim: int,
        n_class_tokens: int,
        mlp_rank: float,
        qkv_rank: float,
        attnproj_rank: float,
        dropout: float,
        attention_dropout: float,
        patch_grid_size: int,
        attn_dim = None,
        sequence_pyramid=[(2, 4)],
        attn_rank_pyramid = [(0, 32), (1, 32)],
        rank_pyramid_begin = 2,
        rank_pyramid_factor=1.0,
        activation: Callable=nn.GELU,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embedding: Callable[..., torch.nn.Module] = partial(PosEmbedding2D),
        quantize_bits: int = None
    ):
        super().__init__()
        img_size=int(math.sqrt(seq_length-n_class_tokens))
        self.n_class_tokens= n_class_tokens
        self.pos_embedding = pos_embedding(seq_length=seq_length, n_class_tokens=n_class_tokens, x_size=img_size, y_size=img_size)
        self.dropout = nn.Dropout(dropout)
        self.rank_pyramid_begin = rank_pyramid_begin
        self.rank_pyramid_factor = rank_pyramid_factor
        self.sequence_pyramid = sequence_pyramid
        self.attn_rank_pyramid = attn_rank_pyramid
        if rank_pyramid_begin is None:
            self.rank_pyramid_begin = 10*num_layers # A higher than num_layers number makes it never apply
        if rank_pyramid_factor is None:
            self.rank_pyramid_factor = 1.0
        if sequence_pyramid is None:
            self.sequence_pyramid = []
        if attn_rank_pyramid is None:
            self.attn_rank_pyramid = []
        seq_length_layer = seq_length - n_class_tokens
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            isSequencePyramid=False
            seqRed = 1
            for p in self.sequence_pyramid:
                if p[0] == i:
                    isSequencePyramid = True
                    seqRed = p[1]
                    break
            attn_rank= None
            for ar in self.attn_rank_pyramid:
                if ar[0]==i:
                    attn_rank = ar[1]
                    break
            layers[f"encoder_layer_{i}"] = EncoderBlockSVD(
                num_heads=num_heads,
                embed_dim=embed_dim,
                mlp_dim=mlp_dim,
                mlp_rank=mlp_rank if i < self.rank_pyramid_begin else mlp_rank/(i*self.rank_pyramid_factor),
                qkv_rank=qkv_rank if i < self.rank_pyramid_begin else qkv_rank/(i*self.rank_pyramid_factor),
                attnproj_rank=attnproj_rank if i < self.rank_pyramid_begin else attnproj_rank/(i*self.rank_pyramid_factor),
                dropout=dropout,
                attention_dropout=attention_dropout,
                norm_layer=norm_layer,
                n_class_tokens=n_class_tokens,
                out_dim= embed_dim//seqRed if isSequencePyramid else None,
                attn_dim= attn_dim,
                attn_rank= attn_rank,
                learnable_skips= 1,
                activation= activation,
                quantize_bits = quantize_bits
            )
            if isSequencePyramid:
                seq_length_layer = seq_length_layer // seqRed
                reduced_dim=embed_dim // seqRed
                layers[f"encoder_pooling_{i}"] = sequence_stitch(patch_grid_shape=(patch_grid_size, patch_grid_size),
                                                                    n_class_tokens=n_class_tokens,
                                                                    reduced_dim=reduced_dim,
                                                                    full_dim=embed_dim,
                                                                    seq_len=seq_length_layer,
                                                                    pos_embedding=pos_embedding,concat_shape=(2,2))
                patch_grid_size=int(patch_grid_size // math.sqrt(seqRed))
            print(patch_grid_size)
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(embed_dim)
        coptions=COMPILE_OPTIONS.copy()
        coptions["triton.dense_indexing"]=False
        self.forward=torch.compile(self.forward,fullgraph=True,options=coptions, disable=COMPILE_DISABLED["state"])

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, embed_dim) got {input.shape}")
        input = self.pos_embedding(input)
        return self.ln(self.layers(self.dropout(input)))

class sequence_stitch(nn.Module):

    def __init__(self, patch_grid_shape,n_class_tokens,reduced_dim,full_dim,seq_len,pos_embedding: Callable[..., torch.nn.Module],concat_shape: tuple=(2,2)):
        super().__init__()
        self.concat_shape=concat_shape
        ndim=len(concat_shape)
        assert ndim == len(patch_grid_shape), "The concat_shape should have the same number of dimensions as the patch_grid_shape."
        print(patch_grid_shape,concat_shape)
        concat_n = sum(concat_shape)
        self.concat_n = concat_n
        self.ndim = ndim
        self.patch_grid_shape = patch_grid_shape
        self.n_class_tokens=n_class_tokens
        self.pos_embed = pos_embedding(seq_length=seq_len, n_class_tokens=0, x_size=math.sqrt(seq_len), y_size=math.sqrt(seq_len))

        # New grid shape after stitching
        self.new_grid_shape = tuple(g // s for g, s in zip(patch_grid_shape, concat_shape))
        self.new_seq_len = math.prod(self.new_grid_shape)

        self.view_shape_template = []
        for grid_size, stitch_factor in zip(self.patch_grid_shape, self.concat_shape):
            self.view_shape_template.extend([grid_size // stitch_factor, stitch_factor])
        self.view_shape_template.append(reduced_dim)

        # Pre-compute full permutation
        self.perm = [0]  # batch
        for i in range(ndim):
            self.perm.append(1 + i * 2)  # new_grid dimensions
        for i in range(ndim):
            self.perm.append(2 + i * 2)  # stitch dimensions
        self.perm.append(len(self.view_shape_template)) # Last dimension (reduced_dim


        self.mismatch = reduced_dim * concat_n + self.pos_embed.reserved_dims - full_dim
        if self.mismatch != 0:
            print(f"The sequence reduction causes a mismatch:{self.mismatch}")
            print(f"You can avoid mismatches by setting the model dim such that model_dim//seq_red + concat_embedding_dim == 0 ")
            if self.mismatch >0:
                print(f"A positive mismatch means that after embedding we have a larger dimension than the model, will solve it by summing the first components.")
                self.forward = torch.compile(self.forward_folded, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
            elif self.mismatch < 0:
                print("A negative mismatch means that after stitching the patches we are still lacking dimensions so we will pad them.")
                self.forward = torch.compile(self.forward_padded, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
        else:
            self.forward=torch.compile(self.forward,fullgraph=True,dynamic=True,options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])

    def forward(self,x):
        """
        x: [batch, patch_grid_size^2, dim/4]
        Stitch neighboring patches together to get a shorter sequence length with full dim
        """
        data_tokens = x[0]
        class_token = x[1]
        stitched = self.stitch(data_tokens)
        return torch.cat([class_token, stitched], dim=1)

    def forward_padded(self,x):
        data_tokens = x[0]
        class_token = x[1]
        stitched = self.stitch(data_tokens)
        stitched = torch.nn.functional.pad(input=stitched,pad=self.mismatch,value=0.01) # Better than 0, if it is 0 the weights are dead, if it isn't they can act as input biases.
        return torch.cat([class_token, stitched], dim=1)

    def forward_folded(self,x):
        data_tokens = x[0]
        class_token = x[1]
        stitched = self.stitch(data_tokens)
        batch_size, seq_len, _ = stitched.shape
        mismatch2 = 2*self.mismatch
        folded_start = stitched[...,:mismatch2].reshape(batch_size,seq_len,self.mismatch,2).sum(dim=-1)
        stitched = torch.cat([folded_start, stitched[...,mismatch2:]],dim=-1)
        return torch.cat([class_token, stitched], dim=1)

    def stitch(self, x):
        batch_size, _, reduced_dim = x.shape
        # Reshape to N-dimensional grid
        spatial_grid = x.view(batch_size, *self.patch_grid_shape, reduced_dim)
        # Apply pre-computed view (just insert batch_size)
        view_shape = [batch_size] + self.view_shape_template
        spatial_grid = spatial_grid.view(*view_shape)
        # Apply pre-computed permutation
        spatial_grid = spatial_grid.permute(*self.perm).contiguous()
        # Final reshape
        stitched = spatial_grid.view(batch_size, self.new_seq_len, self.concat_n * reduced_dim)
        return self.pos_embed(stitched)

class REncoderSVD(nn.Module):
    """Transformer Model Encoder with SVD layers for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        mlp_dim: int,
        n_class_tokens: int,
        mlp_rank: float,
        qkv_rank: float,
        attnproj_rank: float,
        dropout: float,
        attention_dropout: float,
        patch_grid_size: int,
        sequence_pyramid=[(2, 4)],
        attn_rank_pyramid = [(0, 32), (1, 32)],
        rank_pyramid_begin = 2,
        rank_pyramid_factor=1.0,
        activation: Callable = nn.GELU,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embedding: Callable[..., torch.nn.Module] = partial(PosEmbedding2D),
        quantize_bits: int = None
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        img_size=int(math.sqrt(seq_length-n_class_tokens))
        self.n_class_tokens= n_class_tokens
        self.num_layers=num_layers
        self.layerid_scale = torch.nn.Parameter(torch.ones(1) * 0.2)
        self.pos_embedding = pos_embedding(seq_length=seq_length, n_class_tokens=n_class_tokens, x_size=img_size, y_size=img_size)
        self.dropout = nn.Dropout(dropout)
        self.rank_pyramid_begin = rank_pyramid_begin
        self.rank_pyramid_factor = rank_pyramid_factor
        self.sequence_pyramid = sequence_pyramid
        self.attn_rank_pyramid = attn_rank_pyramid
        if rank_pyramid_begin is None:
            self.rank_pyramid_begin = 10*num_layers # A higher than num_layers number makes it never apply
        if rank_pyramid_factor is None:
            self.rank_pyramid_factor = 1.0
        if sequence_pyramid is None:
            self.sequence_pyramid = []
        if attn_rank_pyramid is None:
            self.attn_rank_pyramid = []
        attn_rank = None
        for ar in self.attn_rank_pyramid:
            attn_rank = ar[1]
            break
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        enc= EncoderBlockSVD(
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            mlp_rank=mlp_rank,
            qkv_rank=qkv_rank,
            attnproj_rank=attnproj_rank,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            n_class_tokens=n_class_tokens,
            out_dim= None,
            attn_rank= attn_rank,
            activation=activation,
            quantize_bits = quantize_bits
        )
        layers[f"encoder_layer"]=enc
        self.base_qkv = enc.self_attention.qkv
        self.base_mlp = enc.mlp
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(embed_dim)
        coptions=COMPILE_OPTIONS.copy()
        coptions["triton.dense_indexing"]=False
        self.forward=torch.compile(self.forward,fullgraph=True,options=coptions, disable=COMPILE_DISABLED["state"])

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, embed_dim) got {input.shape}")
        input = self.pos_embedding(input)
        #input = self.dropout(input)
        for i in range(self.num_layers):
            input = input + (self.layerid_scale+0.001) * i/self.num_layers
            input = self.layers(input)
        return self.ln(self.layers(self.dropout(input)))


class LowRankAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_rank, proj_rank, n_bottleneck, attn_drop=0.0, attn_dim=None, qkv_bias: bool = False, proj_bias: bool = False, quantize_bits: int=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if isinstance(num_heads, int):
            self.num_heads = (num_heads,)
        num_heads = 0
        for nh in self.num_heads:
            num_heads += nh
        self.n_head_sizes = len(self.num_heads)
        self.n_bottleneck = n_bottleneck
        self.attn_drop = nn.Dropout(attn_drop)
        self.head_dim = []
        self.scale = []
        self.attn_dims = []
        if attn_dim is None:
            attn_dim = embed_dim
        if attn_dim == "ceilheads" or attn_dim == "floorheads":
            mode = attn_dim
            attn_dim = 0
            for nh in self.num_heads:
                attn_dim2 = 0
                if mode == "ceilheads":
                    attn_dim2 = math.ceil(embed_dim / (nh * self.n_head_sizes)) * nh
                elif mode == "floorheads":
                    attn_dim2 = math.floor(embed_dim / (nh * self.n_head_sizes)) * nh
                self.attn_dims.append(3*attn_dim2)
                attn_dim += attn_dim2
                head_dim = attn_dim2 // nh
                self.head_dim.append(head_dim)
                self.scale.append(1.0 / math.sqrt(head_dim))
        elif self.n_head_sizes == 1:
            self.attn_dims.append(3*attn_dim)
            head_dim = embed_dim//self.num_heads[1]
            self.head_dim.append(head_dim)
            self.scale.append(1.0 / math.sqrt(head_dim))
        self.attn_dim = attn_dim
        effectivedim=0
        for i in range(self.n_head_sizes):
            effectivedim+=self.num_heads[i]*self.head_dim[i]
        assert attn_dim ==effectivedim, f"The attention dim {attn_dim} == {effectivedim} should be divisible by num_heads {num_heads}. Setting attn_dim to ceilheads or floorheads will adjust it automatically."
        if qkv_rank == 1 or qkv_rank == 1.0:
            self.qkv = nn.Linear(embed_dim, attn_dim*3, bias=qkv_bias)
        else:
            self.qkv = SVDLinear(embed_dim, attn_dim*3, rank=max(1,int(embed_dim * qkv_rank)), bias=qkv_bias,bits=quantize_bits)
        if proj_rank == 1 or proj_rank == 1.0:
            self.proj = nn.Linear(attn_dim, embed_dim, bias=proj_bias)
        else:
            self.proj = SVDLinear(attn_dim, embed_dim, rank=max(1,int(embed_dim * proj_rank)), bias=proj_bias,bits=quantize_bits)
        # Bottleneck tokens
        self.streams =[torch.cuda.default_stream(),torch.cuda.Stream()]
        #self.streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for i in range(self.n_head_sizes-1):
            self.streams.append(torch.cuda.Stream())
            self.streams.append(torch.cuda.Stream())
        self.bottleneck_tokens = nn.ParameterList()
        # TODO: Experiment with dynamic bottlenecks figuring out a way to project down the sequence, maybe by convolution.
        # It may also be worth having separate ones for q and k even if static
        if n_bottleneck is not None and n_bottleneck >0:
            for i in range(self.n_head_sizes):
                self.bottleneck_tokens.append(nn.Parameter(torch.randn(self.num_heads[i], n_bottleneck, self.head_dim[i]) * self.scale[i]))
            self.forward = torch.compile(self.forward, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])
        else:
            self.forward = torch.compile(self.forward_quadratic_attention, fullgraph=True, dynamic=True, options=COMPILE_OPTIONS, disable=COMPILE_DISABLED["state"])

    def reshapeHeads(self,x,B,N,num_heads,head_dim):
        qkv = x.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        return qkv

    def forward(self, x):
        B, N, _ = x.shape
        #C = self.proj.in_features
        bottlenecks_list = []
        for i in range(self.n_head_sizes):
            with torch.cuda.stream(self.streams[i+self.n_head_sizes]):
                bottlenecks_list.append(self.bottleneck_tokens[i].unsqueeze(0).expand(B, -1, -1, -1))  # Get bottlenecks [n_heads, n_bottleneck, d_k] -> [B, n_heads, n_bottleneck, d_k]
        qkv0 = self.qkv(x) # Single projection for Q, K, V
        qkv0 = torch.split(tensor= qkv0, split_size_or_sections=self.attn_dims, dim=-1)
        x_list = []
        for i in range(self.n_head_sizes):
            stream = self.streams[i]
            with torch.cuda.stream(stream):
                qkv = self.reshapeHeads(qkv0[i], B, N, num_heads=self.num_heads[i], head_dim=self.head_dim[i])
                q, k, v = qkv.unbind(0)
                stream.wait_stream(self.streams[i + self.n_head_sizes]) # Waiting for the bottleneck streams
                bottlenecks=bottlenecks_list[i]
                global_V = nn.functional.scaled_dot_product_attention(
                    bottlenecks,  # [B, heads, n_bottleneck, d_k]
                    k,  # [B, heads, N, d_k]
                    v,  # [B, heads, N, d_k]
                    dropout_p=self.attn_drop.p if self.training else 0.
                )  # Output: [B, heads, n_bottleneck, d_k]
                out2 = nn.functional.scaled_dot_product_attention( # Step 2: Original positions attend to bottlenecks
                    q,  # [B, heads, N, d_k]
                    bottlenecks,  # [B, heads, n_bottleneck, d_k]
                    global_V,  # [B, heads, n_bottleneck, d_k]
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )  # Output: [B, heads, N, d_k]
                out2 = out2.transpose(1, 2) # Transpose to [B, N, num_heads[i], head_dim[i]]
                out2 = out2.reshape(B, N, -1) # Flatten heads: [B, N, num_heads[i] * head_dim[i]]
                x_list.append(out2)
        for stream in self.streams:
            stream.synchronize()
        out = torch.cat(x_list, dim=-1)  # [B, N, attention_dim]
        #out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def forward_quadratic_attention(self, x):
        B, N, _ = x.shape
        #C = self.proj.in_features
        # Single projection for Q, K, V
        qkv0 = self.qkv(x) # Single projection for Q, K, V
        #print(qkv0.shape, self.n_head_sizes,self.num_heads,self.attn_dims)
        qkv0 = torch.split(tensor= qkv0, split_size_or_sections=self.attn_dims, dim=-1)
        x_list = []
        #print(qkv0,self.n_head_sizes)
        for i in range(self.n_head_sizes):
            with torch.cuda.stream(self.streams[i]):
                qkv = self.reshapeHeads(qkv0[i],B,N,num_heads=self.num_heads[i],head_dim=self.head_dim[i])
                q, k, v = qkv.unbind(0)
                x2 = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
                x2 = x2.transpose(1, 2) # Transpose to [B, N, num_heads[i], head_dim[i]]
                x2 = x2.reshape(B, N, -1) # Flatten heads: [B, N, num_heads[i] * head_dim[i]]
                x_list.append(x2)
        #print(x_list[0].shape,x_list[1].shape)
        for stream in self.streams:
            stream.synchronize()
        x = torch.cat(x_list, dim=-1)  # [B, N, attention_dim]
        return self.proj(x)