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
from TROLOLO.TROLOLO_Trainer import TROLOLO_Trainer, show_batch
from torchvision.transforms import v2, InterpolationMode
import torchvision
from TROLOLO.TROLOLOLR_Scheduler import *

def CIFAR10(quantize=True):
    disable_compilation(False)
    trololo = TROLOLO(image_size=32,
                      img_channels=3,
                      patch_size=2,
                      kernel_size=8,
                      group_conv=False,
                      num_layers=8,
                      #num_heads=(25,9),
                      num_heads=16,
                      embed_dim=384,
                      attention_dim="ceilheads",
                      mlp_dim=1024,
                      n_class_tokens=1,
                      num_classes=10,
                      mlp_rank=0.05,
                      qkv_rank=0.07,
                      attnproj_rank=0.05,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[(0, 32),(1, 16), (2, 16),(3, 8), (4, 8),(5, 8),(6, 8), (7, 8)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=0.5,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.07,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )

    trainer = TROLOLO_Trainer(trololo, experiment_name="CIFAR10")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    val_data = torchvision.datasets.CIFAR10(root="data/CIFAR10", download=True, train=False, transform=transform)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(torch.uint8, scale=True),
         v2.RandomVerticalFlip(),
         v2.RandomHorizontalFlip(),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.8, p=0.15),  # Not sure it helps, experiment more, not sure if sharpness_factor varies or is fixed
             v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.15)
         ]),
         v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.5),
         v2.RandomResizedCrop(size=(32,32), scale=(0.5,1.0),antialias=False),
         v2.ElasticTransform(alpha=50.0, fill=127),
         v2.ToDtype(torch.float16, scale=True),
         ],
    )
    pretrain_data = torchvision.datasets.CIFAR10(root="data/CIFAR10", download=True, train=True, transform=transform)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         #v2.ToDtype(torch.uint8, scale=True),
         v2.RandomHorizontalFlip(),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.8, p=0.15),
             v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.15)
         ]),
         v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
         v2.AugMix(severity=2),
         v2.Lambda(lambda x: nn.functional.pad(x, (31, 31, 31, 31), mode='circular')),
         v2.RandomAffine(degrees=30, interpolation=InterpolationMode.BILINEAR),
         v2.RandomChoice([
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=0, translate=(0.25, 0.15), interpolation=InterpolationMode.BILINEAR),
             ]), p=0.75),
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=0, scale=(0.8, 1.2), interpolation=InterpolationMode.BILINEAR),
             ]), p=0.25),
             v2.RandomPerspective(distortion_scale=0.2, p=1.0),
             v2.ElasticTransform(alpha=50, sigma=5),
         ]),
         v2.CenterCrop(size=(32, 32)),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.04), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.04), value='random'),
         #v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    transform2 = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(torch.uint8, scale=True),
         v2.RandomVerticalFlip(),
         v2.RandomHorizontalFlip(),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.8, p=0.15),
             v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.15)
         ]),
         v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
         v2.RandomChoice([
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=0, translate=(0.15, 0.15), interpolation=InterpolationMode.NEAREST),
             ]), p=0.75),
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=10, scale=(1.0, 1.10), interpolation=InterpolationMode.BILINEAR),
             ]), p=0.25),
             v2.RandomPerspective(distortion_scale=0.3, p=1.0),
             v2.ElasticTransform(alpha=150, sigma=7, fill=127),
         ]),
         v2.AugMix(severity=3),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.04), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.04), value='random'),
         v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )

    train_data = torchvision.datasets.CIFAR10(root="data/CIFAR10", download=True, train=True, transform=transform)
    batch_size=100
    lr_scaling = TROLOLOLR_Scheduler.lr_scale(batch_size=(batch_size, 64), dims=[(trololo.embed_dim, 192), (trololo.mlp_dim, 512)], num_layers=(trololo.num_layers, 6))
    #trainer.pretraining_loop(train_data=pretrain_data, lr=lr_scaling*1e-3, lr_mid=lr_scaling*2.0e-4, lr_min=lr_scaling*1e-6, n_epochs=100, batch_size=batch_size)
    show_batch(train_data)
    trainer.training_loop(train_data=train_data,val_data=val_data,lr=lr_scaling*4e-3,lr_mid=lr_scaling*2.0e-4,lr_min=lr_scaling*1e-6,n_epochs=2000,batch_size=batch_size,transfer=0)


if __name__ == "__main__":
    CIFAR10(quantize=False)