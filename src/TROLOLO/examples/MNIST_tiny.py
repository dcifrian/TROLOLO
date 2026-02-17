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

from TROLOLO.TROLOLO import *
from TROLOLO.TROLOLOLR_Scheduler import TROLOLOLR_Scheduler
from TROLOLO.TROLOLO_Trainer import TROLOLO_Trainer

def MNIST(quantize=True):
    disable_compilation(False)
    trololo = TROLOLO(image_size=28,
                      img_channels=1,
                      patch_size=2,
                      kernel_size=8,
                      group_conv=False,
                      num_layers=4,
                      num_heads=12,
                      embed_dim=72,
                      attention_dim="ceilheads",
                      mlp_dim=72,
                      n_class_tokens=1,
                      num_classes=10,
                      mlp_rank=0.08,
                      qkv_rank=0.14,
                      attnproj_rank=0.07,
                      sequence_pyramid=[],
                      attn_rank_pyramid=[(0, 16),(1, 12), (2,8 ),(3,8)],
                      rank_pyramid_begin=2,
                      rank_pyramid_factor=1,
                      head_constriction="ONE_CLASS_TOKEN",
                      dropout=0.09,
                      attention_dropout=0.01,
                      quantize_bits= None if not quantize else 8,
                      activation=nn.Hardswish
                      )
    trainer = TROLOLO_Trainer(trololo=trololo,experiment_name="MNIST")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.v2.ToDtype(trainer.input_dtype)])
    val_data = torchvision.datasets.MNIST(root="data/MNIST", download=True, train=False, transform=transform)
    train_data = torchvision.datasets.MNIST(root="data/MNIST", download=True, train=True, transform=transform)
    batch_size=64
    lr_scaling = TROLOLOLR_Scheduler.lr_scale(batch_size=(batch_size, 64), dims=[(trololo.embed_dim, 192), (trololo.mlp_dim, 512)], num_layers=(trololo.num_layers, 6))
    trainer.training_loop(train_data=train_data,val_data=val_data,lr=2e-3*lr_scaling,lr_mid=1.0e-4*lr_scaling,lr_min=1e-6*lr_scaling,n_epochs=300,batch_size=batch_size,transfer=0,weight_decay=1e-3)

if __name__ == "__main__":
    MNIST(quantize=True)