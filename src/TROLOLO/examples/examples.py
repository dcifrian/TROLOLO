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
import torchvision
from torch.utils.data import DataLoader, random_split
from TROLOLO.TROLOLOLR_Scheduler import TROLOLOLR_Scheduler
from TROLOLO.TROLOLO import *


def TROLOLO_Hahahahaha(quantize=True):
    from torchvision.transforms import v2, InterpolationMode
    disable_compilation(False)
    trololo = TROLOLO(image_size=64,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=8,
                      group_conv=True,
                      num_layers=6,
                      num_heads=49,
                      embed_dim=194,
                      attention_dim="ceilheads",
                      mlp_dim=194,
                      n_class_tokens=2,
                      num_classes=10,
                      mlp_rank=0.05,
                      qkv_rank=0.06,
                      attnproj_rank=0.05,
                      sequence_pyramid=[(2, 4)],
                      attn_rank_pyramid=[(0, 32),(1, 32), (2, 32)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=0.81,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/eurosat", transform=transform)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    _, val_data = random_split(dataset=dataset, lengths = [0.9, 0.1], generator=generator)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(torch.uint8, scale=True),
         v2.RandomVerticalFlip(),
         v2.RandomHorizontalFlip(),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.9, p=0.05),  # Not sure it helps, experiment more, not sure if sharpness_factor varies or is fixed
             v2.RandomAdjustSharpness(sharpness_factor=1.15, p=0.05)
         ]),
         v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
         v2.RandomChoice([
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=0, translate=(0.02, 0.02), interpolation=InterpolationMode.NEAREST),
             ]), p=0.75),
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=5, scale=(1.0, 1.05), interpolation=InterpolationMode.BILINEAR),
             ]), p=0.25),
         ]),
         v2.AugMix(severity=2),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.05), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.05), value='random'),
         v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    dataset = torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/eurosat", transform=transform)
    train_data, _ = random_split(dataset=dataset, lengths=[0.9, 0.1], generator=generator)
    batch_size=64
    infinitesat = torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/infinitesat/images", transform=transform)
    trololo.pretraining_loop(train_data=infinitesat, lr=2e-3, lr_mid=4.0e-4, lr_min=1e-5, n_epochs=2, batch_size=batch_size)
    trololo.pretraining_loop(train_data=train_data, lr=2e-3, lr_mid=4.0e-4, lr_min=1e-5, n_epochs=300, batch_size=batch_size)
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=2e-3,lr_mid=2.0e-4,lr_min=5e-6,n_epochs=2000,batch_size=batch_size,transfer=500) # transfer 500
    print("best acc: ",trololo.best_acc)


def MNIST(quantize=True):
    from torchvision.transforms import v2, InterpolationMode
    disable_compilation(False)
    trololo = TROLOLO(image_size=28,
                      img_channels=1,
                      patch_size=2,
                      kernel_size=8,
                      group_conv=False,
                      num_layers=6,
                      num_heads=(24,8),
                      embed_dim=192,
                      attention_dim="ceilheads",
                      mlp_dim=192,
                      n_class_tokens=2,
                      num_classes=10,
                      mlp_rank=0.05,
                      qkv_rank=0.06,
                      attnproj_rank=0.05,
                      sequence_pyramid=[(2, 4)],
                      attn_rank_pyramid=[(0, 32),(1, 32), (2, 32)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=0.81,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    val_data = torchvision.datasets.MNIST(root="data/MNIST", download=True, train=False, transform=transform)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(torch.uint8, scale=True),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.9, p=0.05),  # Not sure it helps, experiment more, not sure if sharpness_factor varies or is fixed
             v2.RandomAdjustSharpness(sharpness_factor=1.15, p=0.05)
         ]),
         v2.ColorJitter(brightness=0.12, contrast=0.18),
         v2.RandomChoice([
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=0, translate=(0.02, 0.02), interpolation=InterpolationMode.NEAREST),
             ]), p=0.75),
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=5, scale=(1.0, 1.05), interpolation=InterpolationMode.BILINEAR),
             ]), p=0.25),
             v2.ElasticTransform(alpha=150, sigma=6, fill=127),
         ]),
         v2.AugMix(severity=3),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.03), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.03), value='random'),
         v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    train_data = torchvision.datasets.MNIST(root="data/MNIST", download=True, train=True, transform=transform)
    batch_size=64
    #trololo.pretraining_loop(train_data=train_data, lr=2e-3, lr_mid=4.0e-4, lr_min=1e-5, n_epochs=200, batch_size=batch_size)
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=2e-3,lr_mid=2.0e-4,lr_min=5e-6,n_epochs=300,batch_size=batch_size,transfer=0)
    print("best acc: ",trololo.best_acc)


def CIFAR10(quantize=True):
    from torchvision.transforms import v2, InterpolationMode
    disable_compilation(False)
    trololo = TROLOLO(image_size=32,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=6,
                      group_conv=True,
                      num_layers=8,
                      num_heads=(25,9),
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
             v2.ElasticTransform(alpha=50.0)
         ]),
         v2.AugMix(severity=4),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.08), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.08), value='random'),
         v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.005),
         ],
    )
    train_data = torchvision.datasets.CIFAR10(root="data/CIFAR10", download=True, train=True, transform=transform)
    batch_size=64
    trololo.pretraining_loop(train_data=pretrain_data, lr=1e-3, lr_mid=2.0e-4, lr_min=1e-6, n_epochs=300, batch_size=batch_size)
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=1e-3,lr_mid=2.0e-4,lr_min=1e-6,n_epochs=1500,batch_size=batch_size,transfer=200)
    print("best acc: ",trololo.best_acc)

def TROLOLO_Hahahahaha2(quantize=True):
    # from dataloaders import get_dali_train_loader,get_dali_val_loader
    from torchvision.transforms import v2, InterpolationMode
    from TROLOLO.dali_hard_mining import DALIHardMiningWrapper, extract_files_from_torch_dataset
    from TROLOLO.torch_to_dali_converter import validate_conversion
    trololo = TROLOLO(image_size=64,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=8,
                      group_conv=True,
                      num_layers=6,
                      num_heads=49,
                      embed_dim=194,
                      attention_dim="ceilheads",
                      mlp_dim=194,
                      n_class_tokens=2,
                      num_classes=10,
                      mlp_rank=0.05,
                      qkv_rank=0.06,
                      attnproj_rank=0.05,
                      sequence_pyramid=[(2, 4)],
                      attn_rank_pyramid=[(0, 32),(1, 32), (2, 32)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=0.81,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
    batch_size = 64
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/eurosat", transform=transform)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    _, val_data = random_split(dataset=dataset, lengths = [0.9, 0.1], generator=generator)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    transform_config = validate_conversion(transform)
    val_data = DALIHardMiningWrapper(
        data_path=val_data,
        image_size=64,
        batch_size=batch_size,
        workers=16,
        num_classes=10,
        one_hot=False,
        transform_config=transform_config
    )
    val_data.len=2700
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(torch.uint8, scale=True),
         v2.RandomVerticalFlip(),
         v2.RandomHorizontalFlip(),
         #v2.RandomChoice([
         #    v2.RandomAdjustSharpness(sharpness_factor=0.9, p=0.05),  # Not sure it helps, experiment more, not sure if sharpness_factor varies or is fixed
         #    v2.RandomAdjustSharpness(sharpness_factor=1.15, p=0.05)
         #]),
         #v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
         #v2.RandomChoice([
         #    v2.RandomApply(torch.nn.ModuleList([
         #        v2.RandomAffine(degrees=0, translate=(0.02, 0.02), interpolation=InterpolationMode.NEAREST),
         #    ]), p=0.75),
         #    v2.RandomApply(torch.nn.ModuleList([
         #        v2.RandomAffine(degrees=5, scale=(1.0, 1.05), interpolation=InterpolationMode.BILINEAR),
         #    ]), p=0.25),
         #]),
         v2.ToDtype(torch.float16, scale=True),
         #torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    dataset = torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/eurosat", transform=transform)
    train_data, _ = random_split(dataset=dataset, lengths=[0.9, 0.1], generator=generator)
    transform_config=validate_conversion(transform)
    train_data = DALIHardMiningWrapper(
        data_path=train_data,
        image_size=64,
        batch_size=batch_size,
        workers=16,
        num_classes=10,
        one_hot=False,
        transform_config=transform_config
    )
    train_data.len=24400
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=2e-3,lr_mid=4.0e-4,lr_min=3e-5,n_epochs=1000,batch_size=batch_size)
    print("best acc: ",trololo.best_acc)

def Eurosat(quantize=False):
    # from dataloaders import get_dali_train_loader,get_dali_val_loader
    from torchvision.transforms import v2, InterpolationMode
    # from dali_hard_mining import DALIHardMiningWrapper, extract_files_from_torch_dataset
    # from torch_to_dali_converter import validate_conversion
    trololo = TROLOLO(image_size=64,
                      img_channels=3,
                      patch_size=4,
                      kernel_size=6,
                      num_layers=6,
                      num_heads=48,
                      embed_dim=192,
                      mlp_dim=512,
                      n_class_tokens=2,
                      num_classes=10,
                      mlp_rank=0.1,
                      qkv_rank=0.2,
                      attnproj_rank=0.1,
                      sequence_pyramid=[(2, 4)],
                      attn_rank_pyramid=[(0, 32), (1, 32)],
                      rank_pyramid_begin=None,
                      rank_pyramid_factor=None,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.2,
                      attention_dropout=0.01,
                      quantize_bits=None if not quantize else 8
                      )
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),v2.ToDtype(torch.uint8, scale=True),v2.ToDtype(torch.float16, scale=True)])
    dataset = torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/eurosat", transform=transform)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    _, val_data = random_split(dataset=dataset, lengths = [0.9, 0.1], generator=generator)
    generator = torch.Generator().manual_seed(42)  # To always produce the same split.
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(torch.uint8, scale=True),
         v2.RandomVerticalFlip(),
         v2.RandomHorizontalFlip(),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.9, p=0.05), # Not sure it helps, experiment more, not sure if sharpness_factor varies or is fixed
             v2.RandomAdjustSharpness(sharpness_factor=1.15, p=0.05)
         ]),
         v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
         v2.RandomChoice([
             v2.RandomApply(torch.nn.ModuleList([
                v2.RandomAffine(degrees=0, translate=(0.02,0.02), interpolation=InterpolationMode.NEAREST),
                ]), p=0.75),
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=5, scale=(1.0,1.05), interpolation=InterpolationMode.BILINEAR),
             ]), p=0.25),
         ]),
         v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    dataset = torchvision.datasets.ImageFolder("/home/pickman/Escritorio/TROLOLO/data/eurosat", transform=transform)
    train_data, _ = random_split(dataset=dataset, lengths=[0.9, 0.1], generator=generator)
    batch_size=64
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=1e-3,lr_mid=2.0e-4,lr_min=3e-5,n_epochs=1000,batch_size=batch_size)
    print("best acc: ",trololo.best_acc)

def Imagenet():
    from TROLOLO.dataloaders import get_dali_train_loader,get_dali_val_loader
    # from torchvision.transforms import v2, InterpolationMode
    from TROLOLO.dali_hard_mining import DALIHardMiningWrapper, extract_files_from_torch_dataset
    # from torch_to_dali_converter import validate_conversion
    trololo = TROLOLO(image_size=256,
                      img_channels=3,
                      patch_size=16,
                      kernel_size=18,
                      num_layers=12,
                      num_heads=24,
                      embed_dim=974,
                      attention_dim="floorheads",
                      mlp_dim=2048,
                      n_class_tokens=4,
                      num_classes=1000,
                      mlp_rank=0.1,
                      qkv_rank=0.1,
                      attnproj_rank=0.1,
                      #sequence_pyramid=[(3, 4),(6,4)],
                      sequence_pyramid=[(8, 4)],
                      attn_rank_pyramid=[(0, 64), (1, 64), (2, 64), (3, 64), (4, 32), (5, 32), (6, 32), (7, 32), (8, 32)],
                      rank_pyramid_begin=3,
                      rank_pyramid_factor=0.8,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01
                      )
    batch_size=128

    train_loader, train_loader_len = get_dali_train_loader()(
        data_path="/home/pickman/Escritorio/TROLOLO/data/Imagenet/",
        image_size=256,
        batch_size=batch_size,
        num_classes=1000,
        one_hot=False,
        interpolation="bilinear",
        augmentation="disabled",
        start_epoch=0,
        workers=16,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format
    )
    """
    train_loader = DALIHardMiningWrapper(
        data_path="/home/pickman/Escritorio/TROLOLO/data/Imagenet/train",
        image_size=256,
        batch_size=batch_size,
        workers=16,
        num_classes=1000,
        one_hot=False
    )
    train_loader_len=1280000//batch_size
    """
    val_loader, val_loader_len = get_dali_val_loader()(
        data_path="/home/pickman/Escritorio/TROLOLO/data/Imagenet/",
        image_size=256,
        batch_size=batch_size,
        num_classes=1000,
        one_hot=False,
        interpolation="bilinear",
        crop_padding=32,
        workers=16,
        _worker_init_fn=None,
        memory_format=torch.contiguous_format
    )
    train_loader.len=train_loader_len*batch_size
    val_loader.len=val_loader_len*batch_size
    train_data=train_loader
    val_data=val_loader
    #transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),v2.Resize([256]),v2.CenterCrop(224)])
    #train_data = torchvision.datasets.ImageNet(root="/home/pickman/Escritorio/TROLOLO/data/Imagenet",split="train",transform=transform)
    #val_data = torchvision.datasets.ImageNet(root="/home/pickman/Escritorio/TROLOLO/data/Imagenet",split="val",transform=transform)
    lr_scaling=TROLOLOLR_Scheduler.lr_scale(batch_size=(batch_size,64),dims=[(trololo.embed_dim,192),(trololo.mlp_dim,512)],num_layers=(trololo.num_layers,6))
    pre_epochs=0
    if pre_epochs > 0:
        trololo.pretraining_loop(train_data=train_data, lr=lr_scaling * 8e-4, lr_mid=lr_scaling * 3.0e-4, lr_min=lr_scaling * 1e-5, n_epochs=pre_epochs, batch_size=batch_size)
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=lr_scaling*2e-3,lr_mid=lr_scaling*1.0e-4,lr_min=lr_scaling*1e-6,n_epochs=300,batch_size=batch_size,transfer=pre_epochs)
    print("best acc: ",trololo.best_acc)


if __name__ == "__main__":
    #calcTROLOPS()
    #Imagenet()
    #Eurosat()
    #TROLOLO_Hahahahaha2(quantize=True)
    #TROLOLO_Hahahahaha(quantize=True)
    MNIST(quantize=True)