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
import torchvision
from torch.utils.data import random_split

from TROLOLO.TROLOLO import *
from TROLOLO.TROLOLOLR_Scheduler import TROLOLOLR_Scheduler
from TROLOLO.TROLOLO_Trainer import TROLOLO_Trainer
from torchvision.transforms import v2, InterpolationMode

def Eurosat(quantize=False):
    disable_compilation(False)
    trololo = TROLOLO(image_size=64,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=8,
                      group_conv=False,
                      num_layers=8,
                      #num_heads=(25,9),
                      num_heads=96,
                      embed_dim=384,
                      attention_dim="ceilheads",
                      mlp_dim=1024,
                      n_class_tokens=1,
                      num_classes=10,
                      mlp_rank=0.04,
                      qkv_rank=0.05,
                      attnproj_rank=0.05,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[(0, 16),(1, 8), (2, 8),(3, 8), (4, 8),(5, 8),(6, 8), (7, 8)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1.0,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
    trainer = TROLOLO_Trainer(trololo=trololo,experiment_name="Eurosat_420K")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),v2.ToDtype(trainer.input_dtype)])
    dataset = torchvision.datasets.ImageFolder("data/eurosat", transform=transform)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    _, val_data = random_split(dataset=dataset, lengths = [0.9, 0.1], generator=generator)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        v2.ToDtype(trainer.input_dtype),
        v2.RandomHorizontalFlip(),
        v2.RandomChoice([
            v2.RandomAdjustSharpness(sharpness_factor=0.9, p=0.05),
            v2.RandomAdjustSharpness(sharpness_factor=1.15, p=0.05)
        ]),
        v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
        v2.AugMix(severity=3),
    ])
    transform_gpu = torchvision.transforms.Compose([
        v2.Lambda(lambda x: nn.functional.pad(x, (63, 63, 63, 63), mode='circular')),
        v2.RandomAffine(degrees=180, interpolation=InterpolationMode.BILINEAR),
        v2.RandomChoice([
            v2.RandomApply(torch.nn.ModuleList([
             v2.RandomAffine(degrees=0, translate=(0.25, 0.25), interpolation=InterpolationMode.NEAREST),
            ]), p=0.75),
            v2.RandomApply(torch.nn.ModuleList([
             v2.RandomAffine(degrees=0, scale=(0.95, 1.05), interpolation=InterpolationMode.BILINEAR),
            ]), p=0.25),
            v2.RandomPerspective(distortion_scale=0.04, p=0.75),
            v2.ElasticTransform(alpha=50, sigma=5),
        ]),
        v2.CenterCrop(size=(64, 64)),
        v2.RandomErasing(p=0.8, scale=(0.0, 0.05), value='random'),
        v2.RandomErasing(p=0.5, scale=(0.0, 0.05), value='random'),
        torchvision.transforms.v2.GaussianNoise(sigma=0.002),
    ])
    dataset = torchvision.datasets.ImageFolder("data/eurosat", transform=transform)
    train_data, _ = random_split(dataset=dataset, lengths=[0.9, 0.1], generator=generator)
    #show_batch(train_data)
    batch_size=100
    lr_scaling = TROLOLOLR_Scheduler.lr_scale(batch_size=(batch_size, 64), dims=[(trololo.embed_dim, 192), (trololo.mlp_dim, 512)], num_layers=(trololo.num_layers, 6))
    #infinitesat = torchvision.datasets.ImageFolder("data/infinitesat/images", transform=transform)
    #trainer.pretraining_loop(train_data=infinitesat, lr=2e-3, lr_mid=4.0e-4, lr_min=1e-5, n_epochs=2, batch_size=batch_size)
    #trainer.pretraining_loop(train_data=train_data, lr=2e-3, lr_mid=4.0e-4, lr_min=1e-5, n_epochs=300, batch_size=batch_size)
    trainer.training_loop(train_data=train_data,val_data=val_data,lr=2e-3*lr_scaling,lr_mid=1.0e-4*lr_scaling,lr_min=5e-6*lr_scaling,n_epochs=2000,batch_size=batch_size,transfer=0,transforms=transform_gpu)

if __name__ == "__main__":
    Eurosat()