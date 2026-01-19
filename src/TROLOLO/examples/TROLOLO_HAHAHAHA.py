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
    dataset = torchvision.datasets.ImageFolder("data/eurosat", transform=transform)
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
             v2.ElasticTransform(alpha=150, sigma=6, fill=127),
         ]),
         v2.AugMix(severity=2),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.05), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.05), value='random'),
         v2.ToDtype(torch.float16, scale=True),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    transform.__call__ = torch.compile(transform.__call__)
    dataset = torchvision.datasets.ImageFolder("data/eurosat", transform=transform)
    train_data, _ = random_split(dataset=dataset, lengths=[0.9, 0.1], generator=generator)
    batch_size=64
    infinitesat = torchvision.datasets.ImageFolder("data/infinitesat/images", transform=transform)
    #trololo.pretraining_loop(train_data=infinitesat, lr=2e-3, lr_mid=4.0e-4, lr_min=1e-5, n_epochs=2, batch_size=batch_size)
    #trololo.pretraining_loop(train_data=train_data, lr=2e-3, lr_mid=4.0e-4, lr_min=1e-5, n_epochs=300, batch_size=batch_size)
    trololo.training_loop(train_data=train_data,val_data=val_data,lr=2e-3,lr_mid=2.0e-4,lr_min=5e-6,n_epochs=2000,batch_size=batch_size,transfer=0) # transfer 500



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
    dataset = torchvision.datasets.ImageFolder("data/eurosat", transform=transform)
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
    dataset = torchvision.datasets.ImageFolder("data/eurosat", transform=transform)
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

if __name__ == "__main__":
    #TROLOLO_Hahahahaha2(quantize=True)
    TROLOLO_Hahahahaha(quantize=True)

