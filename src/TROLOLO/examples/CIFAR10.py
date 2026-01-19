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

def CIFAR10(quantize=True):
    from torchvision.transforms import v2, InterpolationMode
    disable_compilation(False)
    trololo = TROLOLO(image_size=32,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=6,
                      group_conv=True,
                      num_layers=8,
                      #num_heads=(25,9),
                      num_heads=16,
                      embed_dim=194,
                      attention_dim="ceilheads",
                      mlp_dim=512,
                      n_class_tokens=2,
                      num_classes=10,
                      mlp_rank=0.1,
                      qkv_rank=0.15,
                      attnproj_rank=0.1,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=0.5,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
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
         #v2.RandomPerspective(distortion_scale=0.3, p=0.1),
         #v2.RandomChoice([
         #    v2.RandomApply(torch.nn.ModuleList([
         #        v2.RandomAffine(degrees=10, scale=(1.05, 1.10), interpolation=InterpolationMode.BILINEAR),
         #    ]), p=0.25),
         #    v2.ElasticTransform(alpha=50.0)
         #]),
         v2.ElasticTransform(alpha=50.0, fill=127),
         v2.ToDtype(torch.float16, scale=True),
         ],
    )
    pretrain_data = torchvision.datasets.CIFAR10(root="data/CIFAR10", download=True, train=True, transform=transform)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(torch.uint8, scale=True),
         v2.RandomVerticalFlip(),
         v2.RandomHorizontalFlip(),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.8, p=0.15),  # Not sure it helps, experiment more, not sure if sharpness_factor varies or is fixed
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
             v2.ElasticTransform(alpha=250, sigma=7, fill=127),
         ]),
         v2.AugMix(severity=3),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.04), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.04), value='random'),
         v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    train_data = torchvision.datasets.CIFAR10(root="data/CIFAR10", download=True, train=True, transform=transform)
    batch_size=64
    trololo.pretraining_loop(train_data=pretrain_data, lr=1e-3, lr_mid=2.0e-4, lr_min=1e-6, n_epochs=300, batch_size=batch_size)
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=1e-3,lr_mid=2.0e-4,lr_min=1e-6,n_epochs=1500,batch_size=batch_size,transfer=200)
    print("best acc: ",trololo.best_acc)

if __name__ == "__main__":
    CIFAR10(quantize=False)