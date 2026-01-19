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
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional
from TROLOLO.TROLOLOLR_Scheduler import TROLOLOLR_Scheduler
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
        self.default_stream = torch.cuda.default_stream()  # Workaround because inductor can't figure out why something in torch.* doesn't return a tensor
        self.data_ready_event = torch.cuda.Event()
        self.data_used_event = torch.cuda.Event()


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

    def preallocate_inputs(self,batch_size:int):
        """
        Preallocates the model inputs in the gpu so they can be reused avoiding the allocation overhead and the memory fragmentation caused by repeated deallocations.
        """
        if self.image_size is not None and self.image_size>0:
            x_gpu = torch.zeros([batch_size, self.conv_proj.in_channels, self.image_size, self.image_size], dtype=torch.bfloat16, device="cuda")
        else:
            x_gpu = torch.zeros([batch_size, self.embed_dim - self.reserved_dims, self.seq_length - self.n_class_tokens], dtype=torch.bfloat16, device="cuda")
        return x_gpu

    def preallocate_targets(self, batch_size: int):
        """
        Preallocates the model targets in the gpu so they can be reused avoiding the allocation overhead and the memory fragmentation caused by repeated deallocations.
        """
        return  torch.zeros([batch_size,self.num_classes], dtype=torch.bfloat16, device="cuda")

    def preallocate_class_indices(self, batch_size: int):
        """
        Preallocates the target class indices in the gpu so they can be reused avoiding the allocation overhead and the memory fragmentation caused by repeated deallocations.
        """
        return torch.zeros([batch_size], dtype=torch.long, device="cuda")

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

    def pretraining_loop(self,train_data,lr,lr_mid,lr_min,n_epochs,batch_size,cut_size=10):
        from torchvision.transforms import v2, InterpolationMode
        self.cuda()
        #Performing head surgery
        oldHead=self.heads
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(self.head_input_size,self.conv_proj.in_channels * cut_size ** 2, device="cuda")
        self.heads = nn.Sequential(heads_layers)
        if isinstance(train_data,torch.utils.data.dataset.Dataset):
            train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        else:
            train_dataloader = train_data
        datalength = len(train_dataloader.dataset)
        n_batches =  datalength // batch_size
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        lr_sched = TROLOLOLR_Scheduler.semiauto(optimizer=optimizer,
                                                lr_peak=lr,lr_mid=lr_mid,lr_min=lr_min,
                                                n_epochs=n_epochs,n_batches=n_batches,batch_size=batch_size,
                                                num_classes=self.num_classes
                                                )
        print("constant epochs: ",lr_sched.constantLr_epochs)
        print("transition samples: ",lr_sched.transition_steps*batch_size)
        loss_fn = nn.MSELoss()
        x_gpu = torch.zeros([batch_size,self.conv_proj.in_channels,self.image_size,self.image_size], dtype=torch.float16, device="cuda")
        for epoch in range(n_epochs):
            self.train()
            progressBar = tqdm(total=datalength, desc=f"Pre Training epoch {epoch}/{n_epochs}: ", unit="images", colour="green", position=0, leave=True)
            loss_sum=0
            i=0
            for data in train_dataloader:
                X_batch = data[0]
                curr_bs = X_batch.shape[0]
                re = torchvision.transforms.RandomErasing.get_params(X_batch, scale=((cut_size/ self.image_size) ** 2, (cut_size/self.image_size) ** 2), ratio=(1.0, 1.0), value=[0])
                y_Batch = v2.functional.crop(X_batch,re[0],re[1],re[2],re[3]).cuda().reshape(curr_bs, -1)
                X_batch = v2.functional.erase(X_batch,re[0],re[1],re[2],re[3],re[4])
                with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
                    x=x_gpu[:curr_bs,:,:,:].copy_(X_batch, non_blocking=True)[:curr_bs,:,:,:]
                    y_pred = self(x)
                    loss = loss_fn(y_pred, y_Batch)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss=loss.item()
                loss_sum = loss_sum+loss
                i+=1
                progressBar.set_postfix(loss=loss,loss_avg=loss_sum/i, lr=lr_sched._last_lr[0])
                progressBar.update(curr_bs)
                lr_sched.step()
            progressBar.close()
        self.eval()
        with torch.no_grad():
            self.heads = oldHead
            self.init_heads()
        optimizer.zero_grad()


    def validation_loop(self, dataloader, epoch, batch_size):
        self.eval()
        acc_stream = torch.cuda.Stream(priority=1)
        acc_event = torch.cuda.Event()
        onehot_stream = torch.cuda.Stream(priority=1)
        onehot_event = torch.cuda.Event()
        datalength=None
        try:
            datalength = len(dataloader.dataset)
        except:
            datalength = dataloader.len
        loss_fn = self.loss_fn
        accuracies = []
        accuracies5 = []
        with (torch.inference_mode(),torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16)):
            x_gpu = self.preallocate_inputs(batch_size)
            y_gpu = self.preallocate_class_indices(batch_size)
            y_gpu_onehot = self.preallocate_targets(batch_size)
            progressBar = tqdm(total=datalength, desc=f"Val epoch {epoch}: ", unit="images", colour="green", position=0, leave=True)
            for data in dataloader:
                X_batch=data[0]
                y_batch=data[1]
                x, y, y_onehot = self.copy_batch_to_preallocated(y_batch=y_batch, y_gpu=y_gpu, onehot_stream=onehot_stream, y_gpu_onehot=y_gpu_onehot, x_gpu=x_gpu, X_batch=X_batch)
                loss,accuracy,acc5 = self.validation_batch(x, y, y_onehot , acc_stream, loss_fn)
                accuracies.append(accuracy)
                accuracies5.append(acc5)
                loss=loss.item()
                accuracy=accuracy.item()
                progressBar.set_postfix(loss=loss, acc=accuracy ,best=self.best_acc)
                progressBar.update(len(y_batch))
            accuracy = torch.mean(torch.stack(accuracies))
            accuracy5 = torch.mean(torch.stack(accuracies5))
            if accuracy > self.best_acc:
                self.best_acc = accuracy.item()
                torch.save(self.state_dict(),"trololo.weight")
            if self.num_classes > 50:
                force_pbar_setpostfix(progressBar,loss=loss, acc=f"{accuracy.item():.5f}", acc5=f"{accuracy5.item():.5f}", best=f"{self.best_acc:.5f}")
            else:
                force_pbar_setpostfix(progressBar,loss=loss, acc=f"{accuracy.item():.5f}", best=f"{self.best_acc:.5f}")
            progressBar.close()
        return accuracy,accuracy5

    def validation_batch(self, x, y, y_onehot , acc_stream, loss_fn):
        with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            y_pred = self(x)
            # torch.cuda.synchronize()  # Force everything to finish
            with torch.cuda.stream(acc_stream):
                acc_stream.wait_stream(self.default_stream)
                accuracy = (y_pred.detach().argmax(1) == y).float().mean()
                _, top5_preds = torch.topk(y_pred.detach(), k=5, dim=1)
                acc5 = (top5_preds == y.unsqueeze(1)).any(dim=1).float().mean()
            #   acc_event.record()
            # torch.cuda.synchronize()  # Force everything to finish
            # onehot_event.wait()
            loss = loss_fn(y_pred, y_onehot)
        return loss,accuracy,acc5

    def transfer_loop(self,lr,epochs,train_dataloader,batch_size):
        self.train()
        acc_stream = torch.cuda.Stream(priority=1)
        onehot_stream = torch.cuda.Stream(priority=1)
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        loss_fn_nosmooth = nn.CrossEntropyLoss(reduction="none") # not really needed in this loop but required to reuse the batch training method
        x_gpu = self.preallocate_inputs(batch_size)
        y_gpu = self.preallocate_class_indices(batch_size)
        y_gpu_onehot = self.preallocate_targets(batch_size)
        loss_sum = 0
        i = 0
        stuck = 1000000
        best_loss = 1000000
        nograd = 0
        for param in self.parameters():
            param.requires_grad = False
            nograd += 1
        transferring = True
        self.heads.requires_grad_()
        for epoch in range(epochs):
            if not transferring:
                break
            progressBar = tqdm(total=len(train_dataloader.dataset), desc=f"Transfer epoch {epoch}/{epochs}: ", unit="images", colour="green", position=0, leave=True)
            for data in train_dataloader:
                if stuck > 16384:
                    transferring = False
                    for param in reversed(list(self.parameters())):
                        if not param.requires_grad:
                            param.requires_grad = True
                            transferring = True
                            nograd = nograd - 1
                            break
                    stuck = 0
                    if not transferring:
                        break
                X_batch = data[0]
                y_batch = data[1]
                with torch.compiler.set_stance("force_eager"):
                    x, y, y_onehot = self.copy_batch_to_preallocated(y_batch=y_batch, y_gpu=y_gpu, onehot_stream=onehot_stream, y_gpu_onehot=y_gpu_onehot, x_gpu=x_gpu, X_batch=X_batch)
                    loss,_,_,accuracy,acc5 = self.train_batch(
                        x=x, y=y, y_onehot=y_onehot, acc_stream=acc_stream, loss_fn=loss_fn, loss_fn_nosmooth=loss_fn_nosmooth, optimizer=optimizer, scaler=scaler)
                loss = loss.item()
                accuracy = accuracy.item()
                i += 1
                loss_sum = loss + loss_sum
                loss_avg = loss_sum / i
                if (best_loss - loss_avg) / (best_loss + 0.0000001) > 0.003*math.sqrt(epoch/epochs)  * len(list(self.parameters())) / (nograd + 1):
                    best_loss = loss_avg
                    stuck = 0
                else:
                    stuck += batch_size
                if self.num_classes > 50:
                    progressBar.set_postfix(loss=loss, loss_avg=loss_avg, stuck=stuck, acc=accuracy, acc5=acc5, nograd=nograd)
                else:
                    progressBar.set_postfix(loss=loss, loss_avg=loss_avg, stuck=stuck, acc=accuracy, nograd=nograd)
                progressBar.update(len(y_batch))
            progressBar.close()
        for param in self.parameters():
            param.requires_grad = True

    def training_loop(self,train_data,val_data,lr,lr_mid,lr_min,n_epochs,batch_size,transfer=0):
        self.cuda()
        coptions=COMPILE_OPTIONS.copy()
        coptions["triton.cudagraphs"] = False  # The cudagraphs don't work with cpu tensors and the next function has both cpu and gpu tensors.
        #self.copy_batch_to_preallocated = torch.compile(self.copy_batch_to_preallocated, fullgraph=True, options=coptions, disable=COMPILE_DISABLED["state"])
        # Validation is either slower or crashes when compiled.
        #self.validation_batch = torch.compile(self.validation_batch, fullgraph=True, options=coptions, disable=COMPILE_DISABLED["state"])
        # The scaler is not compatible with fullgraph, inductor crashes with cudagraphs, if no compile options are provided it works, but it is slower
        #self.train_batch = torch.compile(self.train_batch, fullgraph=False, dynamic=True, options=None, disable=COMPILE_DISABLED["state"])
        acc_stream = torch.cuda.Stream(priority=1)
        acc_event = torch.cuda.Event()
        onehot_stream = torch.cuda.Stream(priority=1)
        onehot_event = torch.cuda.Event()
        loss_fn_nosmooth=nn.CrossEntropyLoss(reduction="none")
        labelsmoothing=0.1
        start_smoothing_th = 0.3 + 0.05* math.log(0.1*self.num_classes)
        notSmoothing=True
        expected_random_loss=math.log(self.num_classes)
        val_batch_size=1*batch_size
        if isinstance(train_data,torch.utils.data.dataset.Dataset):
            train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
            datalength = len(train_data)
        else:
            datalength = train_data.len
            #datalength = train_data.dataset.num_samples
            train_dataloader = train_data
            #sampleWeights= train_data.dataset.weights.copy()
        #datalength = len(train_dataloader.dataset)
        #datalength = train_dataloader.len
        if isinstance(val_data,torch.utils.data.dataset.Dataset):
            val_dataloader = DataLoader(dataset=val_data, batch_size= val_batch_size, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        else:
            val_dataloader=val_data
        n_batches =  datalength // batch_size
        if transfer:
            self.transfer_loop(lr=lr_mid/20.0,epochs=transfer,train_dataloader=train_dataloader,batch_size=batch_size)
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,weight_decay=1e-4)
        lr_sched = TROLOLOLR_Scheduler.semiauto(optimizer=optimizer,
                                                lr_peak=lr,lr_mid=lr_mid,lr_min=lr_min,
                                                n_epochs=n_epochs,n_batches=n_batches,batch_size=batch_size,
                                                num_classes=self.num_classes
                                                )
        print("constant epochs: ",lr_sched.constantLr_epochs)
        print("transition samples: ",lr_sched.transition_steps*batch_size)
        loss_fn = nn.CrossEntropyLoss()
        x_gpu = self.preallocate_inputs(batch_size)
        y_gpu = self.preallocate_class_indices(batch_size)
        y_gpu_onehot = self.preallocate_targets(batch_size)
        for epoch in range(n_epochs):
            loss_sum=0
            acc_sum=0
            acc5_sum=0
            i=0
            self.train()
            progressBar = tqdm(total=datalength, desc=f"Training epoch {epoch}/{n_epochs}: ", unit="images", colour="green", position=0, leave=True)
            for data in train_dataloader:
                X_batch = data[0]
                y_batch = data[1]
                indexes = data[2] if len(data)==3 else None
                x, y, y_onehot = self.copy_batch_to_preallocated(y_batch=y_batch,y_gpu=y_gpu,onehot_stream=onehot_stream,y_gpu_onehot=y_gpu_onehot,x_gpu=x_gpu,X_batch=X_batch)
                loss, unsmooth_loss, unsmooth_loss_batch, accuracy, acc5 = self.train_batch(
                    x=x, y=y, y_onehot=y_onehot, acc_stream=acc_stream,loss_fn=loss_fn,loss_fn_nosmooth=loss_fn_nosmooth, optimizer=optimizer,scaler=scaler)
                loss = loss.item()
                unsmooth_loss = unsmooth_loss.item()
                accuracy = accuracy.item()
                acc5 = acc5.item()
                loss_sum = loss_sum+loss
                acc_sum = acc_sum+accuracy
                acc5_sum = acc5_sum+acc5
                i+=1
                if indexes is not None:
                    train_dataloader.dataset.sample_weights[indexes.cuda()] = unsmooth_loss_batch
                del unsmooth_loss_batch
                if notSmoothing and (loss / expected_random_loss < start_smoothing_th):
                    print("Begin label smoothing")
                    loss_fn = nn.CrossEntropyLoss(label_smoothing=labelsmoothing)
                    notSmoothing = False
                #acc_event.wait() # Yolo! the accuracy calculations should take no time compared to a backwards pass
                hard = False
                if indexes is not None:
                    hard = train_dataloader.hard
                if self.num_classes > 50:
                    progressBar.set_postfix(loss=loss, acc=accuracy, acc5=acc5, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
                else:
                    progressBar.set_postfix(loss=loss, acc=accuracy, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
                progressBar.update(len(y_batch))
                lr_sched.step()
            progressBar.last_print_t = progressBar.last_print_t - progressBar.mininterval
            progressBar.last_print_n = progressBar.last_print_n - progressBar.miniters
            if self.num_classes > 50:
                progressBar.set_postfix(loss=loss_sum/i, acc=acc_sum/i, acc5=acc5_sum/i, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
            else:
                progressBar.set_postfix(loss=loss_sum/i, acc=acc_sum/i, lr=lr_sched._last_lr[0], lsratio=loss / unsmooth_loss, hard=hard)
            progressBar.close()
            #if indexes is not None:
            #    hard=epoch % 10 == 0 and epoch > (lr_sched.constantLr_epochs + 5)
            #    train_dataloader.set_hard(hard)
            self.validation_loop(val_dataloader,epoch,val_batch_size)

    def copy_batch_to_preallocated(self,y_batch,y_gpu,onehot_stream,y_gpu_onehot,x_gpu,X_batch):
        with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            curr_bs = y_batch.shape[0]
            y = self.copy_to_preallocated_class_indices(y_batch, y_gpu, curr_bs)
            with torch.cuda.stream(onehot_stream):
                onehot_stream.wait_stream(self.default_stream)
                y_onehot = self.convert_to_onehot_preallocated(y, y_gpu_onehot, curr_bs)
            #    onehot_event.record()
            x = self.copy_to_preallocated_inputs(X_batch, x_gpu, curr_bs)
        return x,y,y_onehot

    def train_batch(self, x, y, y_onehot, acc_stream, loss_fn, loss_fn_nosmooth, optimizer, scaler):
        acc5=0
        with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
            y_pred = self(x)
            with torch.inference_mode():
                with torch.cuda.stream(acc_stream):
                    acc_stream.wait_stream(self.default_stream)
                    accuracy = (y_pred.detach().argmax(1) == y).float().mean()
                    _, top5_preds = torch.topk(y_pred.detach(), k=5, dim=1)
                    acc5 = (top5_preds == y.unsqueeze(1)).any(dim=1).float().mean()
                #    acc_event.record()
                # onehot_event.wait() # Yolo! the one hot calculations should take no time compared to a forward pass
                unsmooth_loss_batch = loss_fn_nosmooth(y_pred, y_onehot).detach()
                unsmooth_loss = unsmooth_loss_batch.mean()
            loss = loss_fn(y_pred, y_onehot)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss,unsmooth_loss,unsmooth_loss_batch,accuracy,acc5


    def validation_loop_nocnn(self,dataloader,epoch,batch_size):
        self.eval()
        acc_stream = torch.cuda.Stream(priority=1)
        acc_event = torch.cuda.Event()
        onehot_stream = torch.cuda.Stream(priority=1)
        onehot_event = torch.cuda.Event()
        #if isinstance(dataloader, torch.utils.data.dataloader.DataLoader):
        datalength=None
        try:
            datalength = len(dataloader.dataset)
        except:
            datalength = dataloader.len
        #else:
        #    datalength = dataloader.len
        loss_fn = nn.MSELoss()
        x_gpu = self.preallocate_inputs(batch_size)
        y_gpu = self.preallocate_targets(batch_size)
        losses=[]
        with (torch.inference_mode(),torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16)):
            progressBar = tqdm(total=datalength, desc=f"Val epoch {epoch}: ", unit="seq", colour="green", position=0, leave=True)
            for dat in dataloader:
                data=dat["features"]
                X_batch = data
                y_batch = dat["target"]
                curr_bs = y_batch.shape[0]
                y = y_gpu[:curr_bs].copy_(y_batch, non_blocking=False)
                x_gpu[:curr_bs, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
                x = x_gpu[:curr_bs, :, :]
                y_pred = self(x)
                loss = loss_fn(y_pred, y)
                losses.append(loss.item())

                progressBar.set_postfix(loss=loss.item(), best=self.best_acc)
                progressBar.update(curr_bs)
            #loss_avg = torch.mean(torch.stack(losses)).item()
            loss_avg = sum(losses) / len(losses)

            if loss_avg < self.best_acc:
                self.best_acc = loss_avg
                torch.save(self.state_dict(),"trololo.weight")
            force_pbar_setpostfix(progressBar,loss=loss_avg, best=self.best_acc)
            progressBar.close()
        return loss_avg

    def training_loop_nocnn(self,train_data,val_data,lr,lr_mid,lr_min,n_epochs,batch_size,transfer=0):
        self.cuda()
        val_batch_size=1*batch_size

        #train_dataloader = DataLoader(dataset=train_data,worker_init_fn=train_data._worker_init_fn, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, pin_memory=True, prefetch_factor=10, num_workers=8, persistent_workers=True)
        datalength = len(train_dataloader.dataset)
        val_dataloader = DataLoader(dataset=val_data, batch_size=val_batch_size, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=8, persistent_workers=True)
        #val_dataloader = DataLoader(dataset=val_data,worker_init_fn=val_data._worker_init_fn, batch_size= val_batch_size, shuffle=False, pin_memory=True, prefetch_factor=10, num_workers=15, persistent_workers=True)
        n_batches =  datalength // batch_size
        scaler = torch.amp.GradScaler()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        lr_sched = TROLOLOLR_Scheduler.semiauto(optimizer=optimizer,
                                                lr_peak=lr,lr_mid=lr_mid,lr_min=lr_min,
                                                n_epochs=n_epochs,n_batches=n_batches,batch_size=batch_size,
                                                num_classes=self.num_classes
                                                )
        print("constant epochs: ",lr_sched.constantLr_epochs)
        print("transition samples: ",lr_sched.transition_steps*batch_size)
        loss_fn = nn.MSELoss()
        x_gpu = self.preallocate_inputs(batch_size)
        y_gpu = self.preallocate_targets(batch_size)
        if transfer:
            self.transfer_loop(lr=lr_mid/20,epochs=transfer,train_dataloader=train_dataloader,batch_size=batch_size)
        self.best_acc=1000000.0
        for epoch in range(n_epochs):
            self.train()
            progressBar = tqdm(total=datalength, desc=f"Training epoch {epoch}/{n_epochs}: ", unit="seq", colour="green", position=0, leave=True)
            for dat in train_dataloader:
                data=dat["features"]
                X_batch = data
                y_batch = dat["target"]
                with torch.autocast(device_type='cuda', enabled=True, cache_enabled=True, dtype=torch.bfloat16):
                    curr_bs = y_batch.shape[0]
                    y = y_gpu[:curr_bs].copy_(y_batch, non_blocking=False)
                    x_gpu[:curr_bs, :X_batch.shape[2], :].copy_(X_batch.permute(0, 2, 1), non_blocking=True)
                    x = x_gpu[:curr_bs, :, :]
                    y_pred = self(x)
                    loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss=loss.item()
                progressBar.set_postfix(loss=loss, lr=lr_sched._last_lr[0])
                progressBar.update(len(y_batch))
                lr_sched.step()
            progressBar.close()
            val_loss = self.validation_loop_nocnn(val_dataloader,epoch,val_batch_size)

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

def force_pbar_setpostfix(progressBar,**kwargs):
    progressBar.last_print_t = progressBar.last_print_t - progressBar.mininterval
    progressBar.last_print_n = progressBar.last_print_n - progressBar.miniters
    progressBar.set_postfix(**kwargs)

