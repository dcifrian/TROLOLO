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

import math
import os
from importlib.metadata import files

import torch
import numpy as np
from collections import defaultdict
from nvidia.dali import pipeline_def
from nvidia.dali import fn
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator
import nvidia.dali.types as types
from sympy.core.random import shuffle

from torch_to_dali_converter import apply_transforms


class DALIHardMiningWrapper:
    """Wrapper that provides (input, target, index) tuples"""

    # def __init__(self, loader, num_classes, one_hot, memory_format):
    def __init__(self, data_path, image_size, batch_size, num_classes, one_hot, transform_config=None,interpolation="bilinear", augmentation="disabled", start_epoch=0, workers=5, memory_format=torch.contiguous_format):
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.transform_config = transform_config
        self.interpolation = interpolation
        self.augmentation = augmentation
        self.start_epoch = start_epoch
        self.workers = workers
        self.memory_format = memory_format
        self.num_classes = num_classes
        self.one_hot = one_hot
        self.memory_format = memory_format
        self.hard = False
        self.dataset = HardSampleMiningDataset(data_path)
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        self.interpolation = {
            "bicubic": types.INTERP_CUBIC,
            "bilinear": types.INTERP_LINEAR,
            "triangular": types.INTERP_TRIANGULAR,
        }[interpolation]
        self.output_layout = "HWC" if self.memory_format == torch.channels_last else "CHW"
        # Calculate epoch size
        self.steps_per_epoch = math.ceil(len(self.dataset) / (world_size * self.batch_size))
        self.total_samples = len(self.dataset)
        # Pre-sample files and labels for this epoch
        files, labels, indices = self.resample()
        self.files = files
        self.labels = labels
        self.indices = indices
        self.pipeline_kwargs = {
            "batch_size": self.batch_size,
            "num_threads": self.workers,
            "device_id": rank % torch.cuda.device_count(),
            "seed": 12 + rank % torch.cuda.device_count(),
        }
        self.loader = self.regenerate_loader()

    def __iter__(self):
        for data in self.loader:
            if self.memory_format == torch.channels_last:
                shape = data[0]["data"].shape
                stride = data[0]["data"].stride()
                def nhwc_to_nchw(t):
                    return t[0], t[3], t[1], t[2]
                input = torch.as_strided(
                    data[0]["data"],
                    size=nhwc_to_nchw(shape),
                    stride=nhwc_to_nchw(stride),
                )
            else:
                input = data[0]["data"].contiguous(memory_format=self.memory_format)
            target = torch.reshape(data[0]["label"], [-1]).cuda().long()
            if self.one_hot:
                target = self._expand(self.num_classes, torch.float, target)
            yield input, target, data[0]["indices"]  # Only return input and target for now
        # self.regenerate_loader() # Regenerating it here is problematic, it will happen right after the last batch and we may have not updated the sampling or mode.
        # self.loader.reset()

    def __len__(self):
        return math.ceil(len(self.dataset)/self.batch_size)

    def set_hard(self, hard=True):
        if hard:
            self.hard = hard
            self.resample()
            self.regenerate_loader()
        elif self.hard: # If we are changing back to easy from hard
            self.hard = hard
        self.resample()
        self.regenerate_loader()
        # If we change from easy to easy we don't need to resample or regenerate the loader
        self.hard = hard


    def resample(self):
        if self.hard:
            weights = self.dataset.sample_weights
            weights = weights / weights.add(1e-6).mean()
            # This undersamples easy samples but doesn't exclude any
            # It also oversamples hard ones not fully capping the value but compresses the range so no single sample can dominate.
            self.dataset.sample_weights = 0.1 + (1 + 3 * weights.asinh().pow(2)).log()
        else:
            self.dataset.sample_weights =  torch.ones(self.dataset.num_samples).cuda()
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        files, labels, indices = self.dataset.sample_epoch_files_and_labels(
            total_samples=self.total_samples,
            rank=rank,
            world_size=world_size
        )
        self.files=files
        self.labels=labels
        print (self.batch_size*math.ceil(len(files) / self.batch_size) - len(files))
        indices = np.pad(np.array(indices),(0,self.batch_size*math.ceil(len(files)/self.batch_size) - len(files)),mode="constant",constant_values=-1)
        self.indices = indices.reshape(-1, self.batch_size)
        return self.files, self.labels, self.indices

    def regenerate_loader(self):
        if self.hard:
            self.resample()
        pipe = training_pipe_file_list(
            files=self.files,
            labels=self.labels,
            interpolation=self.interpolation,
            image_size=self.image_size,
            output_layout2=self.output_layout,
            automatic_augmentation=self.augmentation,
            transform_config=self.transform_config,
            dali_device="gpu",
            indices=self.indices,
            #shuffle=not self.hard,
            shuffle=False,
            **self.pipeline_kwargs,
        )
        train_loader = DALIGenericIterator(
            pipe,
            ["data", "label", "indices"],
            reader_name="Reader",
            # size=1281024,
            auto_reset=False,
            # fill_last_batch=True,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            last_batch_padded=False
        )
        return train_loader

    def _expand(self, num_classes, dtype, tensor):
        e = torch.zeros(
            tensor.size(0), num_classes, dtype=dtype, device=torch.device("cuda")
        )
        e = e.scatter(1, tensor.unsqueeze(1), 1.0)
        return e


class HardSampleMiningDataset:
    """Dataset wrapper that provides weighted sampling and returns indices"""

    def __init__(self, data_source, initial_weights=None):
        """
        Args:
            data_source: Either a path string or a PyTorch dataset/subset
        """

        if isinstance(data_source, str):
            # Original path-based initialization
            self._init_from_path(data_source)
        else:
            # Initialize from PyTorch dataset
            self._init_from_torch_dataset(data_source)

        # Initialize weights
        if initial_weights is None:
            self.weights = torch.ones(self.num_samples).cuda()
        else:
            self.weights = initial_weights.copy()

    def __len__(self):
        return len(self.samples)

    def _init_from_path(self, data_path):
        """Original path-based initialization"""
        self.data_path = data_path
        self.samples = []
        self.labels = []

        for class_idx, class_name in enumerate(sorted(os.listdir(data_path))):
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png','.tif','.tiff','.bmp')):
                        self.samples.append(os.path.join(class_path, filename))
                        self.labels.append(class_idx)

        self.num_samples = len(self.samples)
        self.original_indices = list(range(self.num_samples))

    def _init_from_torch_dataset(self, torch_dataset):
        """Initialize from PyTorch dataset"""
        self.samples, self.labels, self.original_indices = extract_files_from_torch_dataset(torch_dataset)
        self.num_samples = len(self.samples)
        self.data_path = None  # Not path-based

    def update_weights(self, new_weights):
        """Update sample weights for hard mining"""
        self.weights = new_weights.copy()

    def sample_epoch_files_and_labels(self, total_samples, rank=0, world_size=1):
        """Pre-sample files and labels for one epoch using weighted sampling"""
        # Calculate samples per worker
        samples_per_worker = total_samples // world_size
        #print(samples_per_worker,world_size,rank)
        # Create weighted sampler for this worker's portion
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.weights,
            num_samples=samples_per_worker,
            replacement=True
        )

        # Sample files, labels, and indices
        sampled_files = []
        sampled_labels = []
        sampled_indices = []

        for i in sampler:
            sampled_files.append(self.samples[i])
            sampled_labels.append(self.labels[i])
            sampled_indices.append(i)

        return sampled_files, sampled_labels, sampled_indices


@pipeline_def(enable_conditionals=True)
def training_pipe_file_list(
        files,
        labels,
        interpolation,
        image_size,
        output_layout2,
        indices,
        automatic_augmentation="disabled",
        transform_config = None,
        dali_device="gpu",
        shuffle=False,
        rank=0,
        world_size=1,
):
    """Training pipeline using pre-sampled file list"""
    print(len(files),indices.shape[0]*indices.shape[1],indices.shape)
    # Read images from file list
    jpegs, file_labels = fn.readers.file(files=files, labels=labels, name="Reader",
                                         shard_id=rank,num_shards=world_size,random_shuffle=shuffle,prefetch_queue_depth=10,pad_last_batch=False,)
    # Process images (same as original efficientnet_processing_training)
    decoder_device = "mixed" if dali_device == "gpu" else "cpu"
    if transform_config is None:
        images = fn.decoders.image_random_crop(
            jpegs,
            device=decoder_device,
            output_type=types.RGB,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.5, 1.0],
        )
    else:
        images = fn.decoders.image(
            jpegs,
            device=decoder_device,
            output_type=types.RGB,
        )
    indices_data = fn.external_source(source=indices, dtype=types.DALIDataType.INT64, cycle=True)
    if transform_config is None:
        images = fn.resize(
            images,
            size=[image_size, image_size],
            interp_type=interpolation,
            antialias=False,
        )
    # Apply custom transforms if provided

    images = images.gpu()
    if transform_config is not None:
        output = apply_transforms(images, transform_config)
        output = fn.crop_mirror_normalize(
            output,
            dtype=types.FLOAT,
            output_layout=output_layout2,
            std=[255.0,255.0,255.0]
        )
    else:
        output = images  # No additional transforms
        rng = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, horizontal=rng)

        output = fn.crop_mirror_normalize(
            images,
            dtype=types.FLOAT,
            output_layout=output_layout2,
            crop=(image_size, image_size),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    return output, file_labels, indices_data


def extract_files_from_torch_dataset(torch_dataset):
    """
    Extract file paths and labels from PyTorch datasets

    Args:
        torch_dataset: ImageFolder, Subset, or other dataset with samples

    Returns:
        tuple: (file_paths, labels, indices_mapping)
    """

    if hasattr(torch_dataset, 'samples'):
        # Direct ImageFolder or similar
        samples = torch_dataset.samples
        files = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]
        indices = list(range(len(samples)))

    elif hasattr(torch_dataset, 'dataset') and hasattr(torch_dataset, 'indices'):
        # Subset from random_split
        base_dataset = torch_dataset.dataset
        subset_indices = torch_dataset.indices

        # Get the original samples
        if hasattr(base_dataset, 'samples'):
            base_samples = base_dataset.samples
            files = [base_samples[i][0] for i in subset_indices]
            labels = [base_samples[i][1] for i in subset_indices]
            indices = subset_indices
        else:
            raise ValueError("Base dataset doesn't have 'samples' attribute")

    else:
        raise ValueError("Unsupported dataset type")
    return files, labels, indices
