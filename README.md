# TROLOLO

**Transformer for Rapid Optimized Learning with Overlapping Lightweight Operations**

TROLOLO is a transformer architecture that achieves competitive accuracy with dramatically fewer parameters through principled low-rank factorization of all linear layers and linear-complexity attention. A 96KB model matches ResNet-50 on EuroSAT; a similarly compact model reaches 99.652% on MNIST.

This is not a distilled or pruned model—it trains from scratch. The low-rank structure doesn't limit the optimization landscape; it exploits the inherent low-rank nature of learned weight matrices while allowing the optimizer to explore a space equivalent to much larger networks.

## Key Innovations

### SVD-Factorized Linear Layers

Every linear layer (MLP projections, QKV projections, attention output) is factorized as:

```
y = xVU + b    where V ∈ ℝ^(in × rank), U ∈ ℝ^(rank × out)
```

Instead of storing and computing with a full `in × out` matrix, we use two smaller matrices through a bottleneck of dimension `rank`. With rank set to 5-20% of the smaller dimension, this provides substantial parameter reduction while—counterintuitively—often improving generalization. The regularization effect is similar to dropout: the network cannot memorize through brute force.
Layers closer to the output tolerate much stronger rank reduction, so we support pyramidal ranks.

### Bottleneck Attention

Standard self-attention has O(N²) complexity in sequence length. TROLOLO uses learned "bottleneck tokens"—a small set of latent vectors (typically 8-64) that mediate attention:

1. Bottleneck tokens attend to all keys/values → compressed global representation
2. Queries attend to bottleneck tokens → output

This achieves O(N × B) complexity where B is the number of bottlenecks. Beyond efficiency, this acts as a strong regularizer: the network cannot memorize token-specific patterns when information must pass through a bottleneck. The result is often better validation accuracy despite slightly higher training loss.

### Decoupled Attention Dimension

Unlike standard transformers where attention dimension equals embedding dimension, TROLOLO decouples these. This has several implications:

- **Embedding dimension is independent of head configuration.** You can choose embed_dim for capacity and num_heads for attention patterns separately.
- **Head dimension is freed from embedding constraints.** No need to ensure embed_dim is divisible by num_heads.
- **Multiple head sizes become natural.** You can specify e.g. `num_heads=(12, 6)` for two groups with different head dimensions. Combined with attention dimension control, this allows overlapping heads (larger attention_dim), split-input style attention (same attention_dim), or anything in between—though what actually happens depends on what the QKV projection learns.

Set `attention_dim="ceilheads"` or `"floorheads"` to auto-adjust, or specify manually.

### Post-Attention Activation

TROLOLO applies a nonlinearity between the attention output projection and the MLP block. In standard transformers, the linear attention output connects directly to the linear MLP input—which algebraically could collapse to a single wider layer. The activation breaks this, adding representational capacity at negligible cost (no additional parameters, minimal compute). We observe small but consistent improvements.

### Concatenated Positional Embeddings

Rather than adding positional information to token embeddings (which can be washed out over many layers), TROLOLO concatenates 2D coordinates as dedicated dimensions:

```
token = [features..., x_coord, y_coord]
```

Class tokens receive negative coordinates to distinguish them from spatial positions. A learnable scale parameter allows the network to modulate positional influence. This consistently outperforms RoPE and similar schemes in our experiments, matching only fully learnable embeddings (which require more parameters).

1D positional embeddings are also supported. 

### Overlapping Patch Embeddings

The convolutional stem uses `kernel_size > patch_size` (stride), so adjacent patches share input pixels. This seemingly minor change significantly speeds up early convergence. It also decouples sequence length from receptive field size, providing another tuning knob for compute vs. parameters tradeoffs.

### Sequence Pyramid

For longer sequences, TROLOLO can progressively reduce sequence length by spatially stitching neighboring patches:

```
Before: 256 tokens × 192 dims
After:   64 tokens × 192 dims (4 patches stitched, dims preserved via reduced MLP output)
```

This is nearly free in both parameters and compute—the previous layer's MLP simply outputs to a smaller dimension, and spatial neighbors are concatenated. Class tokens bypass this reduction through a dedicated skip path.

### Learnable skip connections

Skips are vital for deep neural network trainability by preventing the vanishing gradient of the layers that are far from the output.
But using skips forces the network to learn deltas and in some sense to have inputs and the different layers fight each other.
As the vanishing gradient is only a problem when using non-linear functions, adding a learnable scale parameter to the skip connection doesn't impede the gradient flow and allows the network to learn how much the skip should be contributing for each layer and at each moment during training.
It makes a small difference but costs very little parameters and compute 


## Installation

### From source (editable install, recommended for development)

```bash
git clone https://github.com/dcifrian/TROLOLO.git
cd TROLOLO
git checkout develop  # Main branch is currently empty
pip install -e .
```

### Dependencies only

If you prefer to use the modules directly without installing:

```bash
git clone https://github.com/dcifrian/TROLOLO.git
cd TROLOLO
git checkout develop
pip install torch torchvision tqdm tensorboard
```

### Optional dependencies

For running the example scripts:
```bash
pip install -e ".[examples]"
```

For faster data loading (experimental, may have stability issues):
```bash
pip install nvidia-dali-cuda120  # adjust for your CUDA version
```

### PyTorch Version Note

**torch.compile is currently broken in PyTorch 2.10.0 for this project and compilation is automatically disabled for it.** If maximum performance is desired, PyTorch 2.7 or 2.7.1  use the most aggresive compilation flags but the difference is minor when compared agains 2.9.1. To change the compilation behavior (see [Compilation](#compilation) section).

## Quick Start

TROLOLO can be trained like any PyTorch model, but for best results use the included `TROLOLO_Trainer` which handles learning rate scheduling, mixed precision, label smoothing, and other optimizations.

Here's a minimal MNIST example that reaches 99.2% without any augmentation:

```python
import torch.nn as nn
import torchvision
from TROLOLO.TROLOLO import TROLOLO, disable_compilation
from TROLOLO.TROLOLO_Trainer import TROLOLO_Trainer

# Optional: disable compilation for faster startup during experimentation
# disable_compilation(True)

# Define model - this tiny 24k parameters config works well without augmentation
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
                  quantize_bits=8,
                  activation=nn.Hardswish
                  )

# Create trainer
trainer = TROLOLO_Trainer(trololo, experiment_name="MNIST")

# Load data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.v2.ToDtype(trainer.input_dtype)]
)
train_data = torchvision.datasets.MNIST(root="data/MNIST", train=True, download=True, transform=transform)
val_data = torchvision.datasets.MNIST(root="data/MNIST", train=False, transform=transform)

# Train
trainer.training_loop(
    train_data=train_data,
    val_data=val_data,
    lr=2e-3,
    lr_mid=1e-4,
    lr_min=1e-6,
    n_epochs=300,
    batch_size=64,    
)
```

For the full setup with augmentation that reaches 99.652%, see `MNIST.py`.

For a more complete example with all optimizations on a harder dataset, see `TROLOLO_HAHAHAHA.py` (EuroSAT, 97% accuracy with 96K parameters).

### Beyond Image Classification

TROLOLO is not limited to image classification. The architecture works for any task where transformers apply—sequence modeling, time series, tabular data, etc. The main constraint is that variable sequence length is not yet supported.

For an example outside vision, see [Moisturizer](https://github.com/dcifrian/Moisturizer): soil moisture imputation using sequences of meteorological data from nearby weather stations. It uses the `training_loop_nocnn` variant for non-image inputs.

## Architecture Configuration

The potential of having models orders of magnitude smaller and faster makes feasible customizing the architecture to the dataset which opens the door for further efficiency and accuracy improvements.

See [CONFIGURATION.md](CONFIGURATION.md) for detailed guidance on choosing hyperparameters, including:

- Patch size and sequence length tradeoffs
- Embedding and attention dimensions
- Rank parameters and their effects
- Bottleneck attention configuration
- Sequence pyramid setup
- And more

## Results

### EuroSAT

| Model | Parameters | Top-1 Accuracy | Notes |
|-------|------------|----------------|-------|
| TROLOLO_HAHAHAHA | 99K | 97.05% | From scratch, heavy augmentation |
| ResNet-50 | 25M        | 96.36% | From scratch (original paper) |
| ResNet-50 | 25M        | 98.56% | ImageNet pretrained (original paper) |

The comparison uses published numbers from the EuroSAT paper. Our augmentation pipeline and LR scheduling differ and may contribute to the result.

### MNIST

| Model | Parameters | Error Rate | Training Time | Notes |
|-------|------------|------------|---------------|-------|
| TROLOLO | ~110K      | 0.348%     | ~2 hours | Full augmentation |

This matches or exceeds standard Vision Transformers with 85M+ parameters, and is competitive with well-tuned single-model CNNs. Lower error rates for single models exist (best ~0.17% with a branching CNN+capsule architecture), but typically require architectural innovations such as convolutional tokenization — the best pure transformer result we found is ~0.35% with a hyperparameter-optimized ViT.

![Training curves](https://github.com/user-attachments/assets/6857c5de-7897-4f44-a7e0-dd6ebd73b921)



### Other Datasets (Preliminary)

- **CIFAR-10:** 82% without pretraining. Transformers generally struggle here without pretraining; best augmentation-only transformer results are ~91%.
- **ImageNet-1K:** 60% with 7M parameters, but severely bottlenecked by data loading. Work in progress.
- **PCAM:** Overfits aggressively due to dataset issues (labels depend only on central 32×32 pixels; all images from only 400 slides).

## Training

### Learning Rate Scheduler

Learning Rate Scheduler
TROLOLO includes a custom scheduler (`TROLOLOLR_Scheduler`) that maintains the highest stable learning rate for as long as possible, allowing the optimizer to travel far from initialization before settling. It uses three reference learning rates:

- `lr_peak`: Highest LR that won't diverge immediately (but won't converge well alone)
- `lr_mid`: Highest LR that trains stably without scheduling
- `lr_min`: Very small LR for final refinement

The schedule:
1. **Brief constant** at lr_mid (stabilization)
2. **Ramp up** to lr_peak (push into fast-training regime)
3. **Constant at lr_peak** with optional ReduceLROnPlateau exit (main training phase)
4. **Ramp down** to lr_mid (stabilize gains)
5. **Cosine decay** to lr_min (fine-tune to final minimum)

Why this works
In high-dimensional loss landscapes, local minima are abundant — including near any random initialization. A conventional learning rate will settle into whatever minimum is nearby, but since the quality of minima varies significantly, "nearest" is unlikely to mean "best."
A higher learning rate increases the effective step size, letting the optimizer skip over nearby minima and traverse a much larger region of the landscape. The longer you sustain this, the further the model moves from its arbitrary starting point. When the learning rate eventually decays, the model settles into a minimum within this broader explored region — statistically likely to be better than whatever happened to be closest to initialization.
The challenge is that lr_peak alone would underperform (e.g. plateauing at 90% where lr_mid reaches 95%) because some components of the loss can't stabilize at that rate. The brief lr_mid phase at the start resolves this: it gives those components just enough time to converge, after which lr_peak trains far more effectively than it would from a cold start. The final cosine decay then refines the solution in directions that the earlier aggressive rates were too coarse to optimize, yielding results slightly better than lr_mid alone would achieve.

### Learning Rate Scaling

When changing batch size or model architecture, use `TROLOLOLR_Scheduler.lr_scale()` to adjust learning rates:

```python
from TROLOLO.TROLOLOLR_Scheduler import TROLOLOLR_Scheduler

# Scale LR when changing from reference configuration
lr_scaling = TROLOLOLR_Scheduler.lr_scale(
    batch_size=(new_batch_size, reference_batch_size),     # √ scaling
    dims=[(new_embed_dim, ref_embed_dim), (new_mlp_dim, ref_mlp_dim)],  # √ scaling
    num_layers=(new_layers, ref_layers)                     # linear scaling
)

# Apply to your learning rates
lr = base_lr * lr_scaling
lr_mid = base_lr_mid * lr_scaling
lr_min = base_lr_min * lr_scaling
```

**What this gives you:**
- **Batch size scaling:** Loss curves become nearly invariant to batch size changes (tested 1 to a few thousand).
- **Architecture scaling:** The optimal and maximum stable learning rates stay consistent across model size changes. Find them once, then scale.
- **LR invariant:** The rank, bottleneck attention, number of heads, patch size or kernel size don't change the maximum learning rate.

The example files demonstrate this—their learning rates were found once and scaled from a reference configuration.
Changing datasets may still require tuning the learning rate.

### Label Smoothing

The trainer automatically enables label smoothing once the loss drops below a threshold (computed from the number of classes). No configuration required. This is intentional: early training benefits from hard targets to establish correct predictions; later training benefits from smoothed targets to improve calibration and prevent overconfidence.

The progress bar shows `lsratio`—the ratio of smoothed to unsmoothed loss. This helps monitor the smoothing effect. TensorBoard graphs always show unsmoothed loss for consistent comparison across training phases.

### TensorBoard Integration

Pass `experiment_name` to the trainer to enable TensorBoard logging:

```python
trainer = TROLOLO_Trainer(model, experiment_name="my_experiment")
```

Logs are saved to `runs/{experiment_name}{timestamp}/`. View with:

```bash
tensorboard --logdir runs/
```

Logged metrics include train/val loss, accuracy (and top-5 for large class counts), and learning rate.

### Augmentation

Heavy augmentation is often essential, as with any transformer. The example files show effective pipelines using `torchvision.transforms.v2`. Key techniques:

- Random affine transforms (rotation, translation, scale)
- Color jittering
- AugMix
- Random erasing
- Elastic deformation

GPU-side batched augmentation is supported by passing `transforms` to the training loop—recommended for speed when data loading is the bottleneck. 

Although batched augmentation carries the downside of all images in a batch sharing the same random augmentation reducing the augmentation diversity when compared to CPU augmentation in the dataset transform.

Transforms can also be split between GPU and CPU.

## Compilation

TROLOLO uses `torch.compile` by default, providing roughly 2× speedup. The first run will take several minutes to compile.

### PyTorch Version Compatibility

**torch.compile is currently broken in PyTorch 2.10.0 for this project.** Use PyTorch 2.9.1 for best performance, or disable compilation.

### Disabling Compilation

```python
from TROLOLO.TROLOLO import disable_compilation
disable_compilation(True)  # Must be called before model creation
```

### Modifying Compilation Options

Compilation options are in `PYTHON_IS_DUMB.py` (Yes, it is a workaround for its complaints about circular imports). To modify before model creation:

```python
from TROLOLO.PYTHON_IS_DUMB import COMPILE_OPTIONS

# Example: disable cudagraphs
COMPILE_OPTIONS["triton.cudagraphs"] = False

# Then create your model
model = TROLOLO(...)
```

Changes must be made before instantiating TROLOLO to take effect.

### When to Disable Compilation

- Quick iteration during development (avoids multi-minute compilation delay)
- Debugging (clearer stack traces, better error messages)
- Dynamic batch sizes (triggers recompilation each size)
- Freezing/unfreezing layers during training (triggers recompilation)
- Using PyTorch 2.10.0 (currently broken)

## Experimental Features

These work but are not fully validated or complete:

### Quantization-Aware Training

Setting `quantize_bits=8` enables fake quantization during training with learnable scale and zero-point per rank. Models train successfully and sometimes achieve better accuracy than non-quantized (additional regularization effect). However, **quantized inference is not yet implemented**—you get the regularization benefit during training but not deployment speedup yet.

### DALI Data Loading

[DALI_HardMinig](https://github.com/dcifrian/DALI_HardMining):
It wraps Nvidia's DALI dataloader in a drop in replacement for pytorch's dataloader and dataset classes and adds hard sample mining and sample identification capabilities.

DALI improves dataloading performance by providing hardware image decoding and resizing in addition to GPU transforms. 

`TROLOLO_HAHAHAHA.py` Includes a working implementation of DALI_HardMining but it is not the default yet. Current status:

- Basic loading works
- May randomly crash on large datasets (observed on ImageNet, not on EuroSAT)
- DALI's own augmentations not fully integrated
- Hard sample mining implemented but not validated

### MAE Pretraining

`pretraining_MAE_loop` in the trainer implements masked autoencoder pretraining. Results have been inconsistent—sometimes minor improvement, sometimes none. Work in progress.

## File Overview

- `TROLOLO.py`: Main model definition
- `TROLOLO_Layers.py`: Encoder blocks, attention, positional embeddings
- `SVD_layer.py`: SVDLinear implementation and utilities
- `TROLOLO_Trainer.py`: Training loops and utilities
- `TROLOLOLR_Scheduler.py`: Learning rate scheduler
- `PYTHON_IS_DUMB.py`: Compilation flags and options
- `MNIST.py`, `MNIST_tiny.py`, `CIFAR10.py`, `PCAM.py`, `Imagenet.py` , `EuroSAT.py`: Dataset-specific examples
- `TROLOLO_HAHAHAHA.py`: Full-featured EuroSAT example (PyTorch and DALI variants) , EuroSAT.py is over 4x larger but only does 0.3% better. 

## Roadmap / Future Work

No timeline, but planned:
- [ ] Cpu inference
- [ ] Variable sequence length
- [ ] Causal masked attention
- [ ] Sequence to sequence models
- [ ] Optimized CUDA kernels (PyTorch overhead dominates for small models)
- [ ] Quantized inference implementation
- [ ] Improved quantization schemes like multi level block quantization
- [ ] Sharpness-aware minimization (SAM)
- [ ] More dataset results and controlled comparisons
- [ ] TROLOLOLO aka TROLOLO Looks Once or TROLOLO with a YOLO head trained on COCO
- [ ] Paper
- [ ] A TROLOLO model that can sing Trololo
- [ ] Pre training results
- [ ] Alternative methods of sequence length reduction
- [ ] More positional embedding schemes
- [ ] Bottleneck attention variants with bottleneck tokens derived from the input
- [ ] Growable architectures
- [ ] Virtual LoRAs
- [ ] Automatic optimal learning rate determination
- [ ] A tiny language model
- [ ] Better documentation including an augmentation guide.

## License

Apache 2.0. See LICENSE file.

## Citation

Paper forthcoming. For now, please cite this repository:

```bibtex
@misc{trololo2025,
  author = {Cifrian Bueno, Diego},
  title = {TROLOLO: Transformer for Rapid Optimized Learning with Overlapping Lightweight Operations},
  year = {2025},
  url = {https://github.com/dcifrian/TROLOLO}
}
```

## Acknowledgments

README drafted with assistance from Claude (Anthropic).
