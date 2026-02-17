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

from TROLOLO.TROLOLO import *
from TROLOLO.TROLOLO_Trainer import TROLOLO_Trainer
from torchvision.transforms import v2, InterpolationMode
import torchvision
from TROLOLO.TROLOLOLR_Scheduler import *

def PCAM(quantize=True):
    disable_compilation(False)
    # PCAM has 96x96 images but for some reason they decided that the label depends only on the central 32x32 region.
    # Using the full image leads to adversarial examples.
    trololo = TROLOLO(image_size=32,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=8,
                      group_conv=False,
                      num_layers=8,
                      num_heads=16,
                      embed_dim=256,
                      attention_dim="ceilheads",
                      mlp_dim=512,
                      n_class_tokens=1,
                      num_classes=2,
                      mlp_rank=0.10,
                      qkv_rank=0.2,
                      attnproj_rank=0.10,
                      sequence_pyramid=[(4, 4)],
                      attn_rank_pyramid=[(0, 16),(1, 16), (2, 16),(3, 8), (4, 8),(5, 8),(6, 8), (7, 8)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=0.7,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
    trainer = TROLOLO_Trainer(trololo,experiment_name="PCAM")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        v2.ToDtype(trainer.input_dtype),
        v2.CenterCrop(size=(32, 32)),
    ])
    val_data = torchvision.datasets.PCAM(root="data/PCAM", download=True, split="val", transform=transform)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        v2.ToDtype(trainer.input_dtype),
        v2.CenterCrop(size=(32, 32)),
        v2.RandomHorizontalFlip(),
        v2.RandomChoice([
            v2.RandomAdjustSharpness(sharpness_factor=0.8, p=0.15),
            v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.15)
        ]),
        v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
        v2.AugMix(severity=1),
    ])
    transform_gpu = torchvision.transforms.Compose([
         v2.Lambda(lambda x: nn.functional.pad(x, (31, 31, 31, 31), mode='circular')),
         v2.RandomRotation(degrees=180, expand=False),
         v2.RandomApply(torch.nn.ModuleList([
             v2.RandomAffine(degrees=0, scale=(0.95, 1.05), translate=(0.25, 0.25), interpolation=InterpolationMode.NEAREST),
         ]), p=0.5),
         v2.RandomApply(torch.nn.ModuleList([
             v2.ElasticTransform(alpha=50, sigma=5)
         ]), p=0.25),
         v2.CenterCrop(size=(32,32)),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.05), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.05), value='random'),
         #torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    train_data = torchvision.datasets.PCAM(root="data/PCAM", download=True, split="train", transform=transform)
    batch_size=256
    lr_scaling = TROLOLOLR_Scheduler.lr_scale(batch_size=(batch_size, 64), dims=[(trololo.embed_dim, 192), (trololo.mlp_dim, 512)], num_layers=(trololo.num_layers, 6))
    #trainer.pretraining_loop(train_data=pretrain_data, lr=lr_scaling*1e-3, lr_mid=lr_scaling*2.0e-4, lr_min=lr_scaling*1e-6, n_epochs=100, batch_size=batch_size)
    trainer.training_loop(train_data=train_data,val_data=val_data,lr=lr_scaling*2.0e-3,lr_mid=lr_scaling*2.5e-4,lr_min=lr_scaling*1e-6,n_epochs=250,batch_size=batch_size,transfer=0,transforms=transform_gpu)


if __name__ == "__main__":
    PCAM(quantize=False)