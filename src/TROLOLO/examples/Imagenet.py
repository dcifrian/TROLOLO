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

from TROLOLO.TROLOLOLR_Scheduler import *
from TROLOLO.TROLOLO import *
from TROLOLO.TROLOLO_Trainer import TROLOLO_Trainer

def Imagenet():
    #from TROLOLO.dataloaders import get_dali_train_loader,get_dali_val_loader
    from torchvision.transforms import v2, InterpolationMode
    from TROLOLO.dali_hard_mining import DALIHardMiningWrapper, extract_files_from_torch_dataset
    from TROLOLO.torch_to_dali_converter import validate_conversion
    disable_compilation(False)
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
    trainer = TROLOLO_Trainer(trololo=trololo)
    batch_size=128
    """
    train_loader, train_loader_len = get_dali_train_loader()(
        data_path="data/Imagenet/",
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
    )"""
    train_loader = DALIHardMiningWrapper(
        data_path="data/Imagenet/train",
        image_size=256,
        batch_size=batch_size,
        workers=16,
        num_classes=1000,
        one_hot=False
    )
    train_loader_len=1280000//batch_size
    """val_loader, val_loader_len = get_dali_val_loader()(
        data_path="data/Imagenet/",
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
    val_data=val_loader
    """
    train_data=train_loader
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),v2.Resize([256]),v2.CenterCrop(224)])
    #train_data = torchvision.datasets.ImageNet(root="data/Imagenet",split="train",transform=transform)
    val_data = torchvision.datasets.ImageNet(root="data/Imagenet",split="val",transform=transform)
    lr_scaling=TROLOLOLR_Scheduler.lr_scale(batch_size=(batch_size,64),dims=[(trololo.embed_dim,192),(trololo.mlp_dim,512)],num_layers=(trololo.num_layers,6))
    pre_epochs=0
    if pre_epochs > 0:
        trainer.pretraining_loop(train_data=train_data, lr=lr_scaling * 8e-4, lr_mid=lr_scaling * 3.0e-4, lr_min=lr_scaling * 1e-5, n_epochs=pre_epochs, batch_size=batch_size)
    trainer.training_loop(train_data=train_data,val_data=val_data,lr=lr_scaling*2e-3,lr_mid=lr_scaling*1.0e-4,lr_min=lr_scaling*1e-6,n_epochs=300,batch_size=batch_size,transfer=pre_epochs)

if __name__ == "__main__":
    Imagenet()