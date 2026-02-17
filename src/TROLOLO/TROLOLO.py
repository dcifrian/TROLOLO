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
import pathlib
import time

import torch
import torch.nn as nn
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
from TROLOLO.TROLOLO_Layers import Conv2dAsymPad, EncoderSVD, PosEmbedding2D, PosEmbedding1D, REncoderSVD

torch._dynamo.config.recompile_limit=640

from TROLOLO.PYTHON_IS_DUMB import COMPILE_OPTIONS,COMPILE_DISABLED

def disable_compilation(disable: bool=True):
    from TROLOLO import PYTHON_IS_DUMB
    PYTHON_IS_DUMB.COMPILE_DISABLED["state"]=disable


class TROLOLO(nn.Module):

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        mlp_dim: int,
        attention_dim: int = None,
        n_class_tokens: int = 2,
        sequence_length: int = None,
        image_size: int = None,
        img_channels: int  = None,
        patch_size: int  = None,
        kernel_size: int  = None,
        group_conv: bool = False,
        mlp_rank: float = 0.1,
        qkv_rank: float = 0.2,
        attnproj_rank: float = 0.1,
        sequence_pyramid=[(2, 4)],
        attn_rank_pyramid=[(0, 32), (1, 32)],
        rank_pyramid_begin=2,
        rank_pyramid_factor=1.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 10,
        head_constriction="ALL_CLASS_TOKENS",
        representation_size: Optional[int] = None,
        activation: Callable = nn.GELU,
        pos_embedding: Callable[..., torch.nn.Module] = partial(PosEmbedding2D),
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        quantize_bits: int=None
    ):
        super().__init__()
        torch._assert(image_size is None or image_size % patch_size == 0, "Input shape indivisible by patch size!")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True
        self.loss_fn = nn.CrossEntropyLoss()
        self.n_class_tokens = n_class_tokens
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.num_layers=num_layers

        # If the kernel size is larger than the stride, the number of patches will be smaller unless we pad the convolutions.
        if (kernel_size is not None and kernel_size>0):
            image_height = self.image_size[0] if isinstance(image_size, list) else self.image_size
            image_width = self.image_size[1] if isinstance(image_size, list) else self.image_size
            target_output_height = image_height // patch_size  # Calculate required output size (ideal sequence length)
            target_output_width = image_width // patch_size
            padding_height = ((target_output_height - 1) * patch_size + kernel_size - image_height) / 2
            padding_width = ((target_output_width - 1) * patch_size + kernel_size - image_width) / 2
            asym_h_pad = 0
            asym_w_pad = 0
            if (padding_height - int(padding_height)) > 0.01:
                asym_h_pad = 1
            if (padding_width - int(padding_width)) > 0.01:
                asym_w_pad = 1
            padding_height = max(0, int(padding_height))
            padding_width = max(0, int(padding_width))
            padding = [padding_height, padding_width]
            if kernel_size == patch_size:
                padding = 0
            # Moved the instantiation of the convolutional stem after the encoder to account for possible reserved dimensions
            seq_length = (image_size // patch_size) ** 2
        if sequence_length is not None:
            seq_length = sequence_length

        seq_length += n_class_tokens
        self.encoder = EncoderSVD(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            n_class_tokens=n_class_tokens,
            mlp_rank=mlp_rank,
            qkv_rank=qkv_rank,
            attnproj_rank=attnproj_rank,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            sequence_pyramid=sequence_pyramid,
            attn_rank_pyramid=attn_rank_pyramid,
            rank_pyramid_begin=rank_pyramid_begin,
            rank_pyramid_factor=rank_pyramid_factor,
            activation=activation,
            attn_dim=attention_dim,
            patch_grid_size=int(math.sqrt(seq_length-n_class_tokens)),
            pos_embedding = pos_embedding,
            quantize_bits=quantize_bits
        )
        self.seq_length = seq_length
        self.reserved_dims = 0
        if self.encoder.pos_embedding is not None:
            self.reserved_dims = self.encoder.pos_embedding.reserved_dims

        # Add a class token
        self.cls_token_stream = torch.cuda.Stream(priority=1)
        self.class_token = nn.Parameter(torch.zeros(1, n_class_tokens, self.embed_dim-self.reserved_dims))

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.head_input_size = embed_dim
        if head_constriction=="ALL_CLASS_TOKENS":
            self.head_input_size = embed_dim * self.n_class_tokens
        elif head_constriction == "ALL_TOKENS":
            self.head_input_size = embed_dim * self.seq_length
        elif head_constriction == "ONE_CLASS_TOKEN":
            self.head_input_size = embed_dim
        if representation_size is None:
            heads_layers["head"] = nn.Linear(self.head_input_size, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.head_input_size, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)
        self.head_pool = nn.AdaptiveAvgPool1d(self.head_input_size//self.embed_dim)
        self.heads = nn.Sequential(heads_layers)

        if (kernel_size is not None and kernel_size>0):
            outchannels = embed_dim - self.reserved_dims
            groups = 1
            if group_conv:
                assert outchannels % img_channels == 0, "Group convolutions require the input channels to be a multiple of the output channels.\nThe output channels are the model dimension minus any components reserved for positional embeddings. Another option is disabling it with group_conv=False."
                groups = img_channels
            self.conv_proj = nn.Conv2d(in_channels=img_channels, out_channels=outchannels, kernel_size=kernel_size, stride=patch_size, padding=padding, groups=groups)
            if asym_h_pad > 0 or asym_w_pad > 0:
                padding = (padding_width, padding_width + asym_w_pad, padding_height, padding_height + asym_h_pad)
                self.conv_proj = Conv2dAsymPad(in_channels=img_channels, out_channels=outchannels, kernel_size=kernel_size, stride=patch_size, padding=padding, groups=groups)
            if isinstance(self.conv_proj, nn.Conv2d):
                # Init the patchify stem
                fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
                nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
                if self.conv_proj.bias is not None:
                    nn.init.zeros_(self.conv_proj.bias)
            elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
                # Init the last 1x1 conv of the conv stem
                nn.init.normal_(
                    self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
                )
                if self.conv_proj.conv_last.bias is not None:
                    nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print("Parameters per layer:\n")
        for name, param in self.named_parameters(): # Workaround to print the input layer first, due to the deferred initialization would appear last if I don't do this.
            if "conv_proj" in name and param.requires_grad:
                print(name, param.numel(), param.shape)
        for name, param in self.named_parameters():
            if not "conv_proj" in name and param.requires_grad:
                print(name, param.numel(), param.shape)
        print(f"TROLOLO with {trainable_params} parameters")
        coptions=COMPILE_OPTIONS.copy()
        coptions["triton.dense_indexing"]=False
        self._process_input=self._skip_process_input
        if kernel_size is not None:
            self._process_input=self._process_input_cnn
        self.forward=torch.compile(self.forward,fullgraph=True,options=coptions, disable=COMPILE_DISABLED["state"])

    def init_heads(self,scale=1.0):
        for hl in self.heads:
            if hasattr(hl, "weight"):
                torch.nn.init.kaiming_uniform_(hl.weight,mode="fan_out", a=math.sqrt(5))
                hl.weight=nn.Parameter(hl.weight * scale)

    def _process_input_cnn(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p
        # (n, c, h, w) -> (n, embed_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, embed_dim, n_h, n_w) -> (n, embed_dim, (n_h * n_w))
        x = x.reshape(n, self.conv_proj.out_channels, n_h * n_w)
        # (n, embed_dim, (n_h * n_w)) -> (n, (n_h * n_w), embed_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x

    def _skip_process_input(self,x:torch.Tensor):
        return x.permute(0, 2, 1) # Maybe we could not even permute

    def forward(self, x: torch.Tensor):
        cls_token_event = torch.cuda.Event()
        with torch.cuda.stream(self.cls_token_stream):
            # Expand the class token to the full batch
            n = x.shape[0]
            batch_class_token = self.class_token.expand(n, -1, -1)
            cls_token_event.record()
        x = self._process_input(x)  # Patchify the input tensor and run it through the cnn if there is one, else do nothing
        cls_token_event.wait()
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x[:,:self.n_class_tokens,:]
        x = x.transpose(1, 2)
        x = self.head_pool(x)  # (batch, embed_dim, output_size)
        x = x.reshape(x.shape[0], -1)  # (batch, embed_dim * output_size)
        x = self.heads(x)
        return x

    def preallocate_inputs(self,batch_size:int,dtype:torch.dtype = torch.bfloat16):
        """
        Preallocates the model inputs in the gpu so they can be reused avoiding the allocation overhead and the memory fragmentation caused by repeated deallocations.
        """
        if self.image_size is not None and self.image_size>0:
            x_gpu = torch.zeros([batch_size, self.conv_proj.in_channels, self.image_size, self.image_size], dtype=dtype, device="cuda")
        else:
            x_gpu = torch.zeros([batch_size, self.embed_dim - self.reserved_dims, self.seq_length - self.n_class_tokens], dtype=dtype, device="cuda")
        return x_gpu

    def preallocate_targets(self, batch_size: int,dtype:torch.dtype = torch.bfloat16):
        """
        Preallocates the model targets in the gpu so they can be reused avoiding the allocation overhead and the memory fragmentation caused by repeated deallocations.
        """
        return  torch.zeros([batch_size,self.num_classes], dtype=dtype, device="cuda")

    def preallocate_class_indices(self, batch_size: int,dtype:torch.dtype = torch.long):
        """
        Preallocates the target class indices in the gpu so they can be reused avoiding the allocation overhead and the memory fragmentation caused by repeated deallocations.
        """
        return torch.zeros([batch_size], dtype=dtype, device="cuda")

    def copy_to_preallocated_inputs(self, x: torch.Tensor,x_gpu: torch.Tensor,batch_size: int):
        """
        Copies the model inputs from cpu to gpu.
        Meant to be used with a preallocated gpu tensor.
        Returns a view of the gpu tensor with the same shape as x
        """
        return x_gpu[:batch_size, ...].copy_(x, non_blocking=True)[:batch_size, ...]

    def copy_to_preallocated_class_indices(self, y: torch.Tensor,y_gpu: torch.Tensor,batch_size: int):
        """
        Copies the target class indices from cpu to gpu.
        Meant to be used with a preallocated gpu tensor.
        Returns a view of the gpu tensor with the same shape as y
        """
        return y_gpu[:batch_size].copy_(y, non_blocking=False)[:batch_size]

    def convert_to_onehot_preallocated(self,y: torch.Tensor,y_gpu_onehot: torch.Tensor,batch_size: int):
        """
        Converts the targets from a class index to onehot.
        Meant to be used with a preallocated gpu tensor.
        Returns a view of the gpu tensor cropped to the batch size
        """
        return y_gpu_onehot[:batch_size, :].copy_(torch.nn.functional.one_hot(y, self.num_classes).float(), non_blocking=True)[:batch_size, :]

    def classify(self,inputs):
        import torchvision
        if not isinstance(inputs,torch.Tensor):
            tensors = []
            if not isinstance(inputs,list):
                inputs = [inputs]
            for input in inputs:
                if isinstance(input,str) or isinstance(input,pathlib.Path):
                    tensors.append(torchvision.io.decode_image(input))
                elif isinstance(input,torch.Tensor):
                    tensors.append(input)
            inputs=torch.stack(tensors,dim=0)
        self.eval()
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with (torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=dtype)):
            y=self(inputs)
            y=y.detach()
            cls=y.argmax(1)
            y=y.cpu()
            cls=cls.cpu()
        return cls,y

class RETROLOLO(nn.Module):
    def __init__(
        self,
        image_size: int,
        img_channels: int,
        patch_size: int,
        kernel_size: int,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        mlp_dim: int,
        n_class_tokens: int,
        mlp_rank: float = 0.1,
        qkv_rank: float = 0.2,
        attnproj_rank: float = 0.1,
        sequence_pyramid=[(2, 4)],
        attn_rank_pyramid=[(0, 32), (1, 32)],
        rank_pyramid_begin=2,
        rank_pyramid_factor=1.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 10,
        head_constriction="ALL_CLASS_TOKENS",
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        quantize_bits: int=None
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.n_class_tokens = n_class_tokens
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.num_layers=num_layers
        self.acc_stream = torch.cuda.Stream(priority=1)
        self.acc_event = torch.cuda.Event()
        self.onehot_stream = torch.cuda.Stream(priority=1)
        self.onehot_event = torch.cuda.Event()
        self.data_stream = torch.cuda.Stream()
        self.data_ready_event = torch.cuda.Event()
        self.data_used_event = torch.cuda.Event()

        # If the kernel size is larger than the stride, the number of patches will be smaller unless we pad the convolutions.
        image_height = self.image_size[0] if isinstance(image_size, list) else self.image_size
        image_width = self.image_size[1] if isinstance(image_size, list) else self.image_size
        target_output_height = image_height // patch_size  # Calculate required output size (ideal sequence length)
        target_output_width = image_width // patch_size
        padding_height = ((target_output_height - 1) * patch_size + kernel_size - image_height) / 2
        padding_width = ((target_output_width - 1) * patch_size + kernel_size - image_width) / 2
        asym_h_pad = 0
        asym_w_pad = 0
        if (padding_height - int(padding_height)) > 0.01:
            asym_h_pad = 1
        if (padding_width - int(padding_width)) > 0.01:
            asym_w_pad = 1
        padding_height = max(0, int(padding_height))
        padding_width = max(0, int(padding_width))

        # h_out =1+ (image_height + 2*padding_height -kernel_size) / patch_size
        # w_out =1+ (image_width + 2*padding_width -kernel_size) / patch_size

        padding = [padding_height, padding_width]
        if kernel_size == patch_size:
            padding = 0
        self.conv_proj = nn.Conv2d(in_channels=img_channels, out_channels=embed_dim - 2, kernel_size=kernel_size, stride=patch_size, padding=padding)
        if asym_h_pad > 0 or asym_w_pad > 0:
            padding = (padding_width, padding_width + asym_w_pad, padding_height, padding_height + asym_h_pad)
            self.conv_proj = Conv2dAsymPad(in_channels=img_channels, out_channels=embed_dim - 2, kernel_size=kernel_size, stride=patch_size, padding=padding)
        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.cls_token_stream = torch.cuda.Stream(priority=1)
        self.class_token = nn.Parameter(torch.zeros(1, n_class_tokens, self.conv_proj.out_channels))
        seq_length += n_class_tokens

        self.encoder = REncoderSVD(
            seq_length=seq_length,
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            n_class_tokens=n_class_tokens,
            mlp_rank=mlp_rank,
            qkv_rank=qkv_rank,
            attnproj_rank=attnproj_rank,
            dropout=dropout,
            attention_dropout=attention_dropout,
            norm_layer=norm_layer,
            sequence_pyramid=sequence_pyramid,
            attn_rank_pyramid=attn_rank_pyramid,
            rank_pyramid_begin=rank_pyramid_begin,
            rank_pyramid_factor=rank_pyramid_factor,
            patch_grid_size=int(math.sqrt(seq_length-n_class_tokens)),
            quantize_bits=quantize_bits
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.head_input_size = embed_dim
        if head_constriction=="ALL_CLASS_TOKENS":
            self.head_input_size = embed_dim * self.n_class_tokens
        elif head_constriction == "ALL_TOKENS":
            self.head_input_size = embed_dim * self.seq_length
        elif head_constriction == "ONE_CLASS_TOKEN":
            self.head_input_size = embed_dim
        if representation_size is None:
            heads_layers["head"] = nn.Linear(self.head_input_size, num_classes)
            #heads_layers["head"] = nn.Linear(embed_dim*seq_length, num_classes)
            #self.heads = nn.Linear(self.head_input_size, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.head_input_size, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)
        #self.head_pool = nn.AdaptiveAvgPool1d(self.head_input_size) if self.head_input_size > self.embed_dim else nn.Identity()
        self.head_pool = nn.AdaptiveAvgPool1d(self.head_input_size//self.embed_dim)
        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print("Parameters per layer:\n")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.numel(), param.shape)
        print(f"TROLOLO with {trainable_params} parameters")
        coptions=COMPILE_OPTIONS.copy()
        coptions["triton.dense_indexing"]=False
        self.forward=torch.compile(self.forward,fullgraph=True,options=coptions)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p
        # (n, c, h, w) -> (n, embed_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, embed_dim, n_h, n_w) -> (n, embed_dim, (n_h * n_w))
        x = x.reshape(n, self.conv_proj.out_channels, n_h * n_w)
        # (n, embed_dim, (n_h * n_w)) -> (n, (n_h * n_w), embed_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        cls_token_event = torch.cuda.Event()
        with torch.cuda.stream(self.cls_token_stream):
            # Expand the class token to the full batch
            n = x.shape[0]
            batch_class_token = self.class_token.expand(n, -1, -1)
            cls_token_event.record()
        x = self._process_input(x)  # Patchify the input tensor and run it through the cnn
        cls_token_event.wait() # Yoloing it, the class token expansion is way faster than the CNN, should always be done at this point
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        x = x[:,:self.n_class_tokens,:]
        x = x.transpose(1, 2)
        x = self.head_pool(x)  # (batch, embed_dim, output_size)
        x = x.reshape(x.shape[0], -1)  # (batch, embed_dim * output_size)
        x = self.heads(x)
        return x



