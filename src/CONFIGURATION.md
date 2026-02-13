# TROLOLO Configuration Guide

This guide walks through configuring a TROLOLO model, roughly in the order decisions should be made. Later choices often depend on earlier ones.

## Decision Order Overview

1. Patch size → determines sequence length
2. Kernel size → controls overlap and receptive field
3. Group convolution → parameter savings vs. constraints
4. Embedding dimension → model capacity
5. Attention dimension → usually auto-configured
6. MLP dimension → feedforward capacity
7. Number of heads → attention pattern granularity
8. Number of layers → depth and task complexity
9. Rank parameters → compression vs. capacity tradeoff
10. Attention bottleneck pyramid → per-layer bottleneck counts
11. Sequence pyramid → where to reduce sequence length
12. Rank pyramid → per-layer rank reduction
13. Class tokens → usually 1, sometimes more
14. Other parameters → dropout, activation, etc.

---

## 1. Patch Size and Sequence Length

`patch_size` determines how many pixels per patch and thus your sequence length:

```
sequence_length = (image_size / patch_size)²
```

**Guidelines:**

| Sequence Length | Speed | Accuracy | Notes |
|-----------------|-------|----------|-------|
| < 64 | No benefit | — | SDPA pads to 64 internally |
| 64 | Fastest | Good | Good baseline |
| 256 | ~2× slower than 64 | Slightly better | Sweet spot for many tasks |
| 1024+ | Slower | Diminishing returns | Use sequence pyramid |

**Key insight:** This can be adjusted later. Decreasing patch_size (increasing sequence length) is almost always safe—you're giving the model more spatial resolution to work with.

Using bottleneck attention sequences much longer than 1024 are feasible, but with the current focus on computer vision are unnecessary.

---

## 2. Kernel Size

`kernel_size` controls the convolutional stem's receptive field (the patch_size acts as stride).

**Rules:**

- **Minimum:** `kernel_size ≥ patch_size` (or parts of the image aren't sampled)
- **Recommended:** `kernel_size ≥ patch_size + 2` (for meaningful overlap)
- **Larger:** Minor additional benefit, more stem parameters, can be worth it if not trying for minimal size.

**Why overlap matters:**

With `kernel_size > patch_size`, adjacent patches share input pixels. This seemingly minor change improves early convergence by a lot. The overlap also decouples sequence length from per-patch receptive field, giving you another knob for compute/parameter tradeoffs.

---

## 3. Group Convolution (Optional)

Setting `group_conv=True` makes the convolutional stem use grouped convolutions (groups = input channels), significantly reducing parameters.

**Tradeoff:**

| | group_conv=False | group_conv=True |
|---|---|---|
| Parameters | More | Fewer |
| Constraint | None | `(embed_dim - 2) % img_channels == 0` |
| Speed | Consistent | Varies by kernel_size (PyTorch quirk) |

**When to use:**
- Very small models where stem parameters matter
- When you can satisfy the divisibility constraint naturally

**Caution:** Some kernel sizes trigger inexplicably slow grouped convolution in PyTorch. Test speed if using unusual sizes.

**Note on mixing:** If `embed_dim` is significantly larger than `kernel_size² × img_channels`, the stem has room to learn channel-mixing filters. With group convolution, channels can't mix in the stem—this may hurt in high-capacity configurations.

---

## 4. Embedding Dimension

`embed_dim` is the token dimension throughout the network.

**Minimum considerations:**

| Threshold | Formula | Meaning                                      |
|-----------|---------|----------------------------------------------|
| Information preservation (patches) | `patch_size² × img_channels + 2` | All pixels per patch can be sampled          |
| Information preservation (overlap) | `kernel_size² × img_channels + 2` | All pixels in receptive field can be sampled |
| The +2 | Positional embedding dims | 2D concatenated coordinates                  |

Going below these thresholds forces information loss before the first transformer block. The model will still work but accuracy suffers.

**Going higher:** Allows the CNN stem to learn additional filters beyond information preservation. More capacity, more parameters.

**With group_conv:** Ensure `(embed_dim - 2) % img_channels == 0`.

**Practical note:** embed_dim interacts with attention_dim and num_heads. See those sections for how they relate.

---

## 5. Attention Dimension

`attention_dim` controls the total dimension of Q, K, V projections.

**Recommended:** Set to `"ceilheads"` or `"floorheads"` for automatic configuration:
- `"ceilheads"`: Rounds up to nearest multiple of num_heads × head_dim
- `"floorheads"`: Rounds down

**Manual setting:** Must be divisible by the number of heads (total across all head groups if using multiple sizes).

**Why it's decoupled:** Unlike standard transformers, attention_dim ≠ embed_dim. This means:
- embed_dim can be chosen for capacity without attention constraints
- Number of heads doesn't constrain embed_dim
- Head dimension is flexible

**With multiple head sizes:** If `num_heads=(h1, h2)`, the attention dimension is split between head groups. You can:
- Increase attention_dim for "overlapping" heads (more total attention capacity)
- Keep attention_dim same as single-head config for "split input" style
- Or anything in between

---

## 6. MLP Dimension

`mlp_dim` is the hidden dimension in feedforward blocks.

**Range:** 1× to 3× embed_dim

| Ratio | Effect |
|-------|--------|
| 1× | Minimal expansion, fewer parameters |
| 2× | Standard transformer ratio |
| 3× | More capacity, more overfitting risk |

**Recommendation:** Start with 1-2× and increase if clearly underfitting (training loss plateaus high, validation tracks training).

---

## 7. Number of Heads

`num_heads` determines attention granularity. Can be an integer or tuple.

**Head dimension:** With auto attention_dim, head_dim ≈ embed_dim / num_heads (adjusted for divisibility).

**Guidelines:**
- Head dimension shouldn't go much below 4 (attention becomes too coarse)
- Optimal head dimension varies by dataset: some prefer ~4, others ~12
- More heads = finer attention patterns, more parameters in attention

**Multiple head sizes:** Specify as tuple, e.g., `num_heads=(12, 6)`:
- First group: 12 heads (smaller head_dim)
- Second group: 6 heads (larger head_dim)

This sometimes improves over single head size—different heads can capture different scales of patterns.

---

## 8. Number of Layers

`num_layers` controls depth. Unlike dimensions, this relates to **task complexity**, not just capacity.

| Layers | Good for | Training behavior |
|--------|----------|-------------------|
| 4-6 | Simple tasks (MNIST, EuroSAT) | Fast convergence |
| 8 | Medium complexity | Notable improvement over 6 for harder tasks |
| 10-12+ | Complex tasks (ImageNet) | More epochs needed, higher overfitting risk |

**Key insight:** More layers always means more epochs to converge. The rank pyramid (below) makes additional layers cheap in parameters.

**Reference:** Look at popular ViT configurations for your target dataset as starting points.

---

## 9. Rank Parameters

`mlp_rank`, `qkv_rank`, `attnproj_rank` control the bottleneck ratio in SVD-factorized layers.

**Range:** 0.0 to 1.0 (or higher, but why?)

| Value | Effect |
|-------|--------|
| 1.0 | Disables low-rank, uses standard nn.Linear |
| 0.5 | Parameter savings begin (for square matrices) |
| 0.2 | Good balance, rarely need higher |
| 0.1 | Strong compression, works well for most tasks |
| 0.05 | Very strong compression, may limit hard tasks |
| 0.0 | Caps at absolute rank 1 |

**Key findings:**

1. **Lower rank often helps.** Moderate ranks (0.1-0.2) frequently train *faster* and to *better* validation accuracy than full-rank. The bottleneck acts as regularization.

2. **Going above 0.2 typically just increases overfitting** without improving validation.

3. **QKV often benefits from ~1.5× the rank** of other projections. The QKV projection is 3× wider, so the same relative rank creates a tighter bottleneck.

4. **Very low ranks work for simple tasks** but cause early plateaus on hard tasks (training and validation loss both cap, augmentation won't help).

**Recommended starting point:** `mlp_rank=0.1, qkv_rank=0.15, attnproj_rank=0.1`

**Tuning:** Once augmentation is set up, adjust ranks:
- Training loss much lower than validation? → Lower ranks
- Both plateauing too high? → Higher ranks (Can also be augmenting too hard or a too small model)

---

## 10. Attention Bottleneck Pyramid

`attn_rank_pyramid` specifies bottleneck token count per layer:

```python
attn_rank_pyramid = [(layer_idx, n_bottlenecks), ...]
```

**Example:** `[(0, 32), (1, 32), (2, 16), (3, 16), (4, 8), (5, 8)]`

**Guidelines:**

| Bottleneck count | Effect                                                    |
|------------------|-----------------------------------------------------------|
| Missing          | Uses full quadratic attention (with low rank projections) |
| 64               | Optimal speed benefit (SDPA pads to 64)                   |
| 32               | Good default                                              |
| 16               | Good for middle layers                                    |
| 8                | Good for later layers, very strong regularization         |
| < 8              | May hurt accuracy                                         |

**Key insights:**

1. **Later layers need fewer bottlenecks.** They operate on more abstract representations.

2. **Even without speed benefit, fewer bottlenecks improve generalization.** The regularization effect is valuable.

3. **Speed benefit requires sequence_length > 128.** If using sequence pyramid to reduce to 64 or less tokens, bottleneck attention still helps accuracy but can even hurt speed of those layers.

4. **This applies to image classifiers and moderate sequence lenghts, may behave differently in other domains or with large sequences.
---

## 11. Sequence Pyramid

`sequence_pyramid` specifies where to reduce sequence length:

```python
sequence_pyramid = [(layer_idx, reduction_factor), ...]
```

**Currently:** Only 4× reduction (2×2 spatial stitching) is fully functional.

**Example:** `[(2, 4)]` — reduces sequence 4× after layer 2.

**Guidelines:**

- **Don't reduce before layer 2-3.** Some full-resolution processing is valuable.
- **Reduce once or twice** in a typical model.
- **Only beneficial when starting sequence > 64.** Reducing from 64 to 16 doesn't help speed (SDPA padding).
- **Class tokens bypass reduction** via dedicated skip connection.

**How it works:** The layer before reduction outputs to `embed_dim / reduction_factor`. Spatial neighbors are concatenated to restore full embed_dim with fewer tokens.

---

## 12. Rank Pyramid

`rank_pyramid_begin` and `rank_pyramid_factor` progressively reduce ranks in later layers.

```python
rank_pyramid_begin = 2      # Start reducing at layer 2
rank_pyramid_factor = 0.7   # Multiply rank by this each layer
```

**Effect:** Layer `i` (for `i >= rank_pyramid_begin`) uses:
```
effective_rank = base_rank / (i * rank_pyramid_factor)
```

**Guidelines:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `rank_pyramid_begin` | 2-4 | When to start reducing |
| `rank_pyramid_factor` | 0.5-1.0 | How aggressive |

- **More layers → more aggressive reduction is safe.** Later layers in deep networks need less capacity.
- **This makes adding layers cheap.** Each new layer has lower rank than the previous.

---

## 13. Class Tokens

`n_class_tokens` is usually 1. Consider more when:

1. **Using sequence pyramid:** Class tokens get their spatial dimension reduced/interpolated. Multiple tokens provide redundancy. (There is a full-resolution skip connection, so this is less critical than it sounds.)

2. **Many classes ≈ embed_dim:** E.g., ImageNet-1K with embed_dim ~1000. More class tokens give the head more input features.

Can be freely adjusted without other changes.

---

## 14. Other Parameters

### Dropout

- `dropout`: Applied after attention and MLP blocks. 0.01-0.1 typical.
- `attention_dropout`: Applied to attention weights. 0.01 typical.

### Activation

`activation`: Default `nn.GELU`. `nn.Hardswish` is efficient and works well.

### Head Constriction

`head_constriction`: How classification head reads encoder output.

| Value | Head input | Use case |
|-------|------------|----------|
| `"ONE_CLASS_TOKEN"` | First class token × embed_dim | Standard classification |
| `"ALL_CLASS_TOKENS"` | All class tokens concatenated | More head capacity |
| `"ALL_TOKENS"` | Full sequence | High-dimensional output |

### Quantization

`quantize_bits`: Set to 8 for quantization-aware training. Adds regularization (and sometimes improves accuracy). Note: quantized inference not yet implemented.

### Representation Size

`representation_size`: Optional bottleneck before classification head. Usually not needed.

---

## Example Configurations

### Tiny (MNIST-scale, no augmentation needed)

```python
TROLOLO(
    image_size=28, img_channels=1,
    patch_size=2, kernel_size=8,
    num_layers=4, num_heads=12,
    embed_dim=72, attention_dim="ceilheads", mlp_dim=72,
    n_class_tokens=1, num_classes=10,
    mlp_rank=0.07, qkv_rank=0.14, attnproj_rank=0.07,
    sequence_pyramid=[],
    attn_rank_pyramid=[(0, 10), (1, 10), (2, 8), (3, 4)],
    rank_pyramid_begin=2, rank_pyramid_factor=1,
    dropout=0.07, attention_dropout=0.01,
    quantize_bits=8, activation=nn.Hardswish,
)
# Reaches 99.2% on MNIST without augmentation
```

### Small (EuroSAT-scale)

```python
TROLOLO(
    image_size=64, img_channels=3,
    patch_size=4, kernel_size=8,
    num_layers=6, num_heads=49,
    embed_dim=194, attention_dim="ceilheads", mlp_dim=194,
    n_class_tokens=2, num_classes=10,
    mlp_rank=0.05, qkv_rank=0.05, attnproj_rank=0.05,
    sequence_pyramid=[(2, 4)],
    attn_rank_pyramid=[(0, 32), (1, 16), (2, 16)],
    rank_pyramid_begin=2, rank_pyramid_factor=0.81,
    group_conv=True,
    activation=nn.Hardswish,
)
```

### Medium (ImageNet-scale, work in progress)

```python
TROLOLO(
    image_size=224, img_channels=3,
    patch_size=16, kernel_size=18,
    num_layers=8, num_heads=24,
    embed_dim=974, attention_dim="floorheads", mlp_dim=2048,
    n_class_tokens=4, num_classes=1000,
    mlp_rank=0.1, qkv_rank=0.1, attnproj_rank=0.1,
    sequence_pyramid=[(4, 4)],
    attn_rank_pyramid=[(i, 32 if i < 6 else 16) for i in range(8)],
    rank_pyramid_begin=3, rank_pyramid_factor=0.8,
)
# Note: ImageNet training is bottlenecked by data loading; this config is not fully validated
```

---

## Tuning Workflow

1. **Start with a reference configuration** from the examples or above or from popular ViT configurations with changes following this guide.

2. **Get training working** with minimal augmentation. Verify the model trains.

3. **Add augmentation** appropriate to your dataset. This is critical for transformers.

4. **Observe training vs. validation gap:**
   - Large gap (overfitting) → Lower ranks, more augmentation, fewer layers
   - No gap, both high loss → Higher ranks, more layers, larger dimensions

5. **Adjust sequence length** if speed is an issue (increase patch_size) or accuracy needs improvement (decrease patch_size). Can also use sequence pyramid and bottleneck attention.

6. **Fine-tune bottleneck counts** in attention. More bottlenecks for harder tasks in early layers.

7. **Scale learning rates** automatically using `TROLOLOLR_Scheduler.lr_scale()` to avoid the need of changing them while fine tuning the architecture.
