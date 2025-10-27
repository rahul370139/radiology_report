"""
Shared LLaVA processor for consistent image and text processing.
This ensures the same logic is used in both training and inference.
"""

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class LLaVAProcessor:
    """
    Custom processor that handles image and text processing consistently
    for LLaVA models. This ensures the same 5-patch processing logic
    is used in both training and inference.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Image processing pipeline - same as in advanced_trainer.py
        self.image_proc = transforms.Compose([
            transforms.Resize((336, 336), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(336),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, text, images, return_tensors="pt", padding=True, truncation=True, max_length=512, **kwargs):
        """
        Process text and images for LLaVA model.
        
        Args:
            text: Input text string
            images: PIL Image or list of PIL Images
            return_tensors: Return format ("pt" for PyTorch tensors)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            **kwargs: Additional tokenizer arguments
            
        Returns:
            Dictionary with input_ids, attention_mask, pixel_values, and image_sizes
        """
        # Handle text-only processing
        if images is None:
            return self.tokenizer(
                text, 
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs
            )
        
        # Process images
        if isinstance(images, list):
            # Multiple images - stack them
            processed_images = [self.image_proc(img) for img in images]
            pixel_values = torch.stack(processed_images)
            # Get image sizes for all images
            image_sizes = [img.size for img in images]
        else:
            # Single image - process and add batch dimension
            pixel_values = self.image_proc(images).unsqueeze(0)
            # Get image size
            image_sizes = [images.size]
        
        # Process text
        text_encoding = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            **kwargs
        )
        
        # Combine results with required metadata for LLaVA-Next
        return {
            'input_ids': text_encoding['input_ids'],
            'attention_mask': text_encoding['attention_mask'],
            'pixel_values': pixel_values,
            'image_sizes': image_sizes,  # Required for LLaVA-Next
        }
    
    def process_images_only(self, images):
        """
        Process only images (for cases where text is handled separately).
        
        Args:
            images: PIL Image or list of PIL Images
            
        Returns:
            pixel_values tensor
        """
        if isinstance(images, list):
            processed_images = [self.image_proc(img) for img in images]
            return torch.stack(processed_images)
        else:
            return self.image_proc(images).unsqueeze(0)
