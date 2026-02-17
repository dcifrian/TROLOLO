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
from torchvision.transforms import v2, InterpolationMode
from TROLOLO.TROLOLO import *
from TROLOLO.TROLOLO_Trainer import TROLOLO_Trainer, show_batch


def MNIST(quantize=True):
    disable_compilation(False)
    trololo = TROLOLO(image_size=28,
                      img_channels=1,
                      patch_size=2,
                      kernel_size=8,
                      group_conv=False,
                      num_layers=8,
                      num_heads=16,
                      embed_dim=192,
                      attention_dim="ceilheads",
                      mlp_dim=192,
                      n_class_tokens=2,
                      num_classes=10,
                      mlp_rank=0.05,
                      qkv_rank=0.08,
                      attnproj_rank=0.05,
                      sequence_pyramid=[(2, 4)],
                      attn_rank_pyramid=[(0, 32),(1, 16), (2, 16)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1.0,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.05,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
    trainer = TROLOLO_Trainer(trololo=trololo,experiment_name="MNIST",)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),v2.ToDtype(trainer.input_dtype)])
    val_data = torchvision.datasets.MNIST(root="data/MNIST", download=True, train=False, transform=transform)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         v2.ToDtype(trainer.input_dtype),
         v2.RandomChoice([
             v2.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
             v2.RandomAdjustSharpness(sharpness_factor=1.15, p=0.1)
         ]),
         v2.ColorJitter(brightness=0.12, contrast=0.18),
         v2.RandomChoice([
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=0, translate=(0.02, 0.02), interpolation=InterpolationMode.NEAREST),
             ]), p=0.75),
             v2.RandomApply(torch.nn.ModuleList([
                 v2.RandomAffine(degrees=5, scale=(0.95, 1.05), interpolation=InterpolationMode.BILINEAR),
             ]), p=0.25),
             v2.ElasticTransform(alpha=150, sigma=6, fill=127),
             v2.RandomPerspective(distortion_scale=0.1, p=0.75),
         ]),
         v2.AugMix(severity=3),
         v2.RandomErasing(p=0.8, scale=(0.0, 0.03), value='random'),
         v2.RandomErasing(p=0.5, scale=(0.0, 0.03), value='random'),
         torchvision.transforms.v2.GaussianNoise(sigma=0.002),
         ],
    )
    train_data = torchvision.datasets.MNIST(root="data/MNIST", download=True, train=True, transform=transform)
    batch_size=64
    show_batch(train_data)
    trainer.training_loop(train_data=train_data,val_data=val_data,lr=2e-3,lr_mid=2.0e-4,lr_min=5e-6,n_epochs=300,batch_size=batch_size,transfer=0)

if __name__ == "__main__":
    MNIST(quantize=True)