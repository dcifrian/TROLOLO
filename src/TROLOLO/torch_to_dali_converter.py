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

import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from typing import Dict, Any, List, Union
import warnings

from nvidia.dali import fn
import nvidia.dali.types as types

def torchTransforms_toDALI_config(torch_transforms: transforms.Compose) -> Dict[str, Any]:
    """
    Convert PyTorch transforms.Compose to DALI config dict
    
    Args:
        torch_transforms: torchvision.transforms.Compose object
        
    Returns:
        Dictionary of DALI transform configurations
        
    Example:
        >>> transform = transforms.Compose([
        ...     v2.RandomHorizontalFlip(p=0.5),
        ...     v2.ColorJitter(brightness=0.1, contrast=0.1)
        ... ])
        >>> dali_config = torchTransforms_toDALI_config(transform)
    """
    dali_config = {}
    unsupported_transforms = []
    
    for i, transform in enumerate(torch_transforms.transforms):
        transform_type = type(transform).__name__
        
        try:
            if transform_type in ["RandomVerticalFlip"]:
                dali_config[f"vertical_flip_{i}"] = {
                    "probability": getattr(transform, 'p', 0.5)
                }
                
            elif transform_type in ["RandomHorizontalFlip"]:
                dali_config[f"horizontal_flip_{i}"] = {
                    "probability": getattr(transform, 'p', 0.5)
                }
                
            elif transform_type in ["ColorJitter"]:
                # Extract ColorJitter parameters (they can be None, scalar, or tuple)
                def extract_param(param, default=0):
                    if param is None:
                        return default
                    elif isinstance(param, (list, tuple)) and len(param) == 2:
                        return max(abs(param[0]), abs(param[1]))  # Take max range
                    else:
                        return abs(param)
                
                dali_config[f"color_jitter_{i}"] = {
                    "brightness": extract_param(getattr(transform, 'brightness', None)),
                    "contrast": extract_param(getattr(transform, 'contrast', None)),
                    "saturation": extract_param(getattr(transform, 'saturation', None)),
                    "hue": extract_param(getattr(transform, 'hue', None))
                }
                
            elif transform_type in ["GaussianNoise"]:
                dali_config[f"gaussian_noise_{i}"] = {
                    "sigma": getattr(transform, 'sigma', 0.1)
                }
                
            elif transform_type in ["RandomChoice"]:
                # Handle RandomChoice by extracting probabilities and nested transforms
                choice_config = _parse_random_choice(transform, i)
                if choice_config:
                    dali_config.update(choice_config)
                else:
                    unsupported_transforms.append(f"RandomChoice_{i} (complex nested structure)")
                    
            elif transform_type in ["RandomApply"]:
                # Handle RandomApply wrapper
                apply_config = _parse_random_apply(transform, i)
                if apply_config:
                    dali_config.update(apply_config)
                else:
                    unsupported_transforms.append(f"RandomApply_{i} (unsupported nested transforms)")
                    
            elif transform_type in ["RandomAffine"]:
                dali_config[f"random_affine_{i}"] = _parse_random_affine(transform)
                
            elif transform_type in ["ToTensor", "ToDtype"]:
                # These are handled automatically by DALI pipeline
                pass
                
            else:
                unsupported_transforms.append(f"{transform_type}_{i}")
                
        except Exception as e:
            warnings.warn(f"Failed to parse {transform_type}_{i}: {e}")
            unsupported_transforms.append(f"{transform_type}_{i} (parsing error)")
    
    # Report unsupported transforms
    if unsupported_transforms:
        warnings.warn(
            f"The following transforms are not supported in DALI conversion: "
            f"{', '.join(unsupported_transforms)}. "
            f"They will be skipped in the DALI pipeline."
        )
    
    return dali_config

def _parse_random_choice(transform, idx: int) -> Dict[str, Any]:
    """Parse RandomChoice transform"""
    try:
        choices = transform.transforms
        if len(choices) == 2:
            # Handle binary choice (like your sharpness example)
            choice_1, choice_2 = choices
            
            # Check if it's RandomAdjustSharpness
            if (type(choice_1).__name__ == "RandomAdjustSharpness" and 
                type(choice_2).__name__ == "RandomAdjustSharpness"):
                
                return {
                    f"sharpness_choice_{idx}": {
                        "probabilities": [choice_1.p, choice_2.p],
                        "factors": [choice_1.sharpness_factor, choice_2.sharpness_factor]
                    }
                }
                
            # Check if it's RandomAffine choices
            elif (type(choice_1).__name__ == "RandomApply" and 
                  type(choice_2).__name__ == "RandomApply"):
                
                # Extract the nested RandomAffine transforms
                affine_1 = _extract_from_random_apply(choice_1)
                affine_2 = _extract_from_random_apply(choice_2)
                
                if (affine_1 and affine_2 and 
                    type(affine_1).__name__ == "RandomAffine" and
                    type(affine_2).__name__ == "RandomAffine"):
                    
                    return {
                        f"affine_choice_{idx}": {
                            "probabilities": [choice_1.p, choice_2.p],
                            "affine_1": _parse_random_affine(affine_1),
                            "affine_2": _parse_random_affine(affine_2)
                        }
                    }
        
        return None  # Unsupported RandomChoice structure
        
    except:
        return None

def _parse_random_apply(transform, idx: int) -> Dict[str, Any]:
    """Parse RandomApply transform"""
    try:
        nested_transforms = transform.transforms
        probability = getattr(transform, 'p', 0.5)
        
        if len(nested_transforms) == 1:
            nested = nested_transforms[0]
            nested_type = type(nested).__name__
            
            if nested_type == "RandomAffine":
                config = _parse_random_affine(nested)
                config["probability"] = probability
                return {f"random_affine_{idx}": config}
                
        return None
        
    except:
        return None

def _extract_from_random_apply(random_apply_transform):
    """Extract the first transform from a RandomApply wrapper"""
    try:
        if hasattr(random_apply_transform, 'transforms') and len(random_apply_transform.transforms) > 0:
            return random_apply_transform.transforms[0]
    except:
        pass
    return None

def _parse_random_affine(transform) -> Dict[str, Any]:
    """Parse RandomAffine transform parameters"""
    config = {}
    
    # Extract degrees
    degrees = getattr(transform, 'degrees', 0)
    if isinstance(degrees, (list, tuple)):
        config["degrees_range"] = degrees
    else:
        config["degrees_range"] = (-degrees, degrees) if degrees != 0 else (0, 0)
    
    # Extract translate
    translate = getattr(transform, 'translate', None)
    if translate:
        config["translate_range"] = translate
    else:
        config["translate_range"] = (0, 0)
    
    # Extract scale
    scale = getattr(transform, 'scale', None)
    if scale:
        config["scale_range"] = scale
    else:
        config["scale_range"] = (1.0, 1.0)
    
    # Extract interpolation mode
    interp = getattr(transform, 'interpolation', None)
    if interp:
        # Map PyTorch interpolation to DALI (simplified)
        interp_map = {
            'nearest': 'INTERP_NN',
            'bilinear': 'INTERP_LINEAR', 
            'bicubic': 'INTERP_CUBIC'
        }
        config["interpolation"] = interp_map.get(str(interp).lower(), 'INTERP_LINEAR')
    
    return config

# Extended apply_transforms function to handle converted configs
def apply_converted_transforms(images, converted_config):
    """Apply transforms using config from torchTransforms_toDALI_config"""
    
    for transform_name, params in converted_config.items():
        base_name = transform_name.split('_')[0] + '_' + transform_name.split('_')[1]
        
        if "vertical_flip" in transform_name:
            images = random_vertical_flip(images, **params)
        elif "horizontal_flip" in transform_name:
            images = random_horizontal_flip(images, **params)
        elif "color_jitter" in transform_name:
            images = color_jitter(images, **params)
        elif "gaussian_noise" in transform_name:
            images = gaussian_noise(images, **params)
        elif "sharpness_choice" in transform_name:
            images = random_sharpness_choice(images, **params)
        elif "affine_choice" in transform_name:
            images = random_affine_choice(images, **params)
        elif "random_affine" in transform_name:
            images = random_affine_single(images, **params)
    
    return images

def random_affine_single(images, degrees_range=(0, 0), translate_range=(0, 0), 
                        scale_range=(1.0, 1.0), probability=1.0, interpolation="INTERP_LINEAR"):
    """Single RandomAffine transform"""
    from nvidia.dali import fn
    import nvidia.dali.types as types
    
    should_apply = fn.random.coin_flip(probability=probability)
    
    # Generate random parameters
    angle = fn.random.uniform(range=degrees_range)
    translate_x = fn.random.uniform(range=(-translate_range[0], translate_range[0]))
    translate_y = fn.random.uniform(range=(-translate_range[1], translate_range[1]))
    scale = fn.random.uniform(range=scale_range)
    
    # Create transformation matrix
    matrix = fn.transforms.scale(scale=(scale, scale)) * \
             fn.transforms.rotation(angle=angle) * \
             fn.transforms.translation(offset=(translate_x, translate_y))
    
    # Apply conditionally
    interp_type = getattr(types, interpolation, types.INTERP_LINEAR)
    return fn.warp_affine(images, matrix=matrix * should_apply, interp_type=interp_type)


def random_vertical_flip(images, probability=0.5):
    """Random vertical flip"""
    should_flip = fn.random.coin_flip(probability=probability)
    return fn.flip(images, vertical=should_flip)


def random_horizontal_flip(images, probability=0.5):
    """Random horizontal flip"""
    should_flip = fn.random.coin_flip(probability=probability)
    return fn.flip(images, horizontal=should_flip)


def random_sharpness_choice(images, probabilities=[0.05, 0.05]):
    """Random choice between two sharpness adjustments"""
    # Generate random choice: 0=no change, 1=sharpen, 2=blur
    choice = fn.random.uniform(range=(0, 1))

    # Apply sharpness based on choice
    sharp_factor = fn.random.uniform(range=(0.9, 0.9))  # Fixed at 0.9
    blur_factor = fn.random.uniform(range=(1.15, 1.15))  # Fixed at 1.15

    # Use conditional to apply different sharpness
    should_sharp = choice < probabilities[0]
    should_blur = (choice >= probabilities[0]) & (choice < (probabilities[0] + probabilities[1]))

    factor = fn.cast(should_sharp, dtype=types.FLOAT) * sharp_factor + \
             fn.cast(should_blur, dtype=types.FLOAT) * blur_factor + \
             fn.cast(~(should_sharp | should_blur), dtype=types.FLOAT) * 1.0

    return fn.laplacian(images, kernel_size=3, scale=factor)  # Approximate sharpness


def color_jitter(images, brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02):
    """Color jitter with specified ranges"""
    # Generate random factors
    brightness_factor = fn.random.uniform(range=(1 - brightness, 1 + brightness))
    contrast_factor = fn.random.uniform(range=(1 - contrast, 1 + contrast))
    saturation_factor = fn.random.uniform(range=(1 - saturation, 1 + saturation))
    hue_shift = fn.random.uniform(range=(-hue, hue))

    # Apply transformations
    images = fn.brightness_contrast(images, brightness=brightness_factor, contrast=contrast_factor)
    images = fn.saturation(images, saturation=saturation_factor)
    images = fn.hue(images, hue=hue_shift)

    return images


def random_affine_choice(images, probabilities=[0.75, 0.25]):
    """Random choice between two affine transformations"""
    choice = fn.random.coin_flip(probability=probabilities[0])

    # First option: translation only
    translate_x = fn.random.uniform(range=(-0.02, 0.02))
    translate_y = fn.random.uniform(range=(-0.02, 0.02))

    # Second option: rotation and scale
    angle = fn.random.uniform(range=(-5, 5))
    scale = fn.random.uniform(range=(1.0, 1.05))

    # Apply based on choice
    # For translation (nearest neighbor approximation)
    matrix_translate = fn.transforms.translation(offset=(translate_x, translate_y))

    # For rotation+scale (bilinear)
    matrix_rotate_scale = fn.transforms.scale(scale=(scale, scale)) * \
                          fn.transforms.rotation(angle=angle)

    # Choose which transformation to apply
    # Note: This is simplified - DALI doesn't have exact equivalents to PyTorch's interpolation modes
    return fn.warp_affine(images,
                          matrix=fn.cast(choice, dtype=types.FLOAT) * matrix_translate + \
                                 fn.cast(~choice, dtype=types.FLOAT) * matrix_rotate_scale,
                          interp_type=types.INTERP_LINEAR)


def gaussian_noise(images, sigma=0.002):
    """Add Gaussian noise"""
    noise = fn.random.normal(device="gpu", shape=fn.shapes(images), stddev=sigma)
    return images + noise


# Transform configurations
EUROSAT_TRANSFORMS = {
    "vertical_flip": {"probability": 0.5},
    "horizontal_flip": {"probability": 0.5},
    "sharpness_choice": {"probabilities": [0.05, 0.05]},
    "color_jitter": {
        "brightness": 0.12,
        "contrast": 0.18,
        "saturation": 0.15,
        "hue": 0.02
    },
    "affine_choice": {"probabilities": [0.75, 0.25]},
    "gaussian_noise": {"sigma": 0.002}
}

IMAGENET_TRANSFORMS = {
    "horizontal_flip": {"probability": 0.5},
    # Add other ImageNet-specific transforms as needed
}


def apply_transforms(images, transform_config):
    """Apply a sequence of transforms based on configuration"""
    for transform_name, params in transform_config.items():
        if transform_name == "vertical_flip":
            images = random_vertical_flip(images, **params)
        elif transform_name == "horizontal_flip":
            images = random_horizontal_flip(images, **params)
        elif transform_name == "sharpness_choice":
            images = random_sharpness_choice(images, **params)
        elif transform_name == "color_jitter":
            images = color_jitter(images, **params)
        elif transform_name == "affine_choice":
            images = random_affine_choice(images, **params)
        elif transform_name == "gaussian_noise":
            images = gaussian_noise(images, **params)

    return images

# Example usage and validation
def validate_conversion(torch_transforms, test_on_sample=True):
    """
    Validate that conversion works and show what was converted
    
    Args:
        torch_transforms: torchvision.transforms.Compose object
        test_on_sample: Whether to test conversion on sample data
    """
    print("Converting PyTorch transforms to DALI config...")
    dali_config = torchTransforms_toDALI_config(torch_transforms)
    
    print("\nOriginal PyTorch transforms:")
    for i, transform in enumerate(torch_transforms.transforms):
        print(f"  {i}: {transform}")
    
    print(f"\nConverted DALI config:")
    for key, value in dali_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nConversion complete. {len(dali_config)} transforms converted.")
    
    return dali_config

# Example with your EuroSAT transforms
if __name__ == "__main__":
    # Recreate your EuroSAT transform
    transform = transforms.Compose([
        v2.RandomVerticalFlip(),
        v2.RandomHorizontalFlip(),
        v2.RandomChoice([
            v2.RandomAdjustSharpness(sharpness_factor=0.9, p=0.05),
            v2.RandomAdjustSharpness(sharpness_factor=1.15, p=0.05)
        ]),
        v2.ColorJitter(brightness=0.12, contrast=0.18, saturation=0.15, hue=0.02),
        v2.RandomChoice([
            v2.RandomApply([
                v2.RandomAffine(degrees=0, translate=(0.02, 0.02))
            ], p=0.75),
            v2.RandomApply([
                v2.RandomAffine(degrees=5, scale=(1.0, 1.05))
            ], p=0.25),
        ]),
        v2.GaussianNoise(sigma=0.002),
    ])
    
    dali_config = validate_conversion(transform)