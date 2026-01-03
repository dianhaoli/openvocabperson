# embeddings.py
"""
Image Embedding Extraction using Qwen2.5-VL Vision Encoder.

This module extracts visual embeddings from the already-loaded Qwen2.5-VL model,
avoiding the need to load a separate CLIP model. The embeddings are used for
vector similarity search to find similar images/entities.

Architecture:
    - Reuses the VLM's vision encoder (no additional model loading)
    - Extracts patch-level features and pools them to a single vector
    - Normalizes for cosine similarity search

Usage:
    # Initialize with existing pipeline components
    extractor = QwenEmbeddingExtractor(
        vlm_model=pipeline.vlm,
        processor=pipeline.processor,
        device=pipeline.device
    )
    
    # Generate embedding for an image
    embedding = extractor.generate_embedding(pil_image)  # Returns numpy array
    
    # Batch generation (more efficient)
    embeddings = extractor.generate_embeddings_batch([img1, img2, img3])
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration


class QwenEmbeddingExtractor:
    """
    Extract image embeddings from Qwen2.5-VL vision encoder.
    
    Reuses the already-loaded VLM to avoid loading separate embedding models.
    The vision encoder outputs are pooled to create a fixed-size embedding
    vector suitable for similarity search.
    
    Attributes:
        vlm: The Qwen2.5-VL model
        processor: The AutoProcessor for the model
        device: Device string ("cuda", "mps", or "cpu")
        embedding_dim: Dimension of output embeddings (determined by model)
    """
    
    def __init__(
        self,
        vlm_model: "Qwen2_5_VLForConditionalGeneration",
        processor: "AutoProcessor",
        device: str,
    ):
        """
        Initialize with existing Qwen model components.
        
        Args:
            vlm_model: Already-loaded Qwen2_5_VLForConditionalGeneration
            processor: Already-loaded AutoProcessor
            device: Device string ("cuda", "mps", or "cpu")
        """
        self.vlm = vlm_model
        self.processor = processor
        self.device = device
        self.embedding_dim: Optional[int] = None
        
        # Discover embedding dimension from model config
        self._discover_embedding_dim()
    
    def _discover_embedding_dim(self):
        """Determine embedding dimension from model configuration."""
        try:
            # Qwen2.5-VL stores vision config in model.config.vision_config
            if hasattr(self.vlm.config, 'vision_config'):
                vision_config = self.vlm.config.vision_config
                if hasattr(vision_config, 'hidden_size'):
                    self.embedding_dim = vision_config.hidden_size
                elif hasattr(vision_config, 'embed_dim'):
                    self.embedding_dim = vision_config.embed_dim
            
            # Fallback: check model directly
            if self.embedding_dim is None and hasattr(self.vlm, 'visual'):
                # Try to get from visual encoder
                if hasattr(self.vlm.visual, 'config'):
                    self.embedding_dim = getattr(
                        self.vlm.visual.config, 'hidden_size', 
                        getattr(self.vlm.visual.config, 'embed_dim', 1024)
                    )
            
            # Default fallback
            if self.embedding_dim is None:
                self.embedding_dim = 1024  # Common default for vision models
                
            print(f"   Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            print(f"   Warning: Could not determine embedding dim: {e}")
            self.embedding_dim = 1024
    
    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Extracts visual features from the image using the VLM's vision encoder,
        pools the patch-level features to a single vector, and normalizes it
        for cosine similarity search.
        
        Args:
            image: PIL Image to embed
            
        Returns:
            Normalized embedding vector as numpy array, shape (embedding_dim,)
        """
        embeddings = self.generate_embeddings_batch([image])
        return embeddings[0]
    
    def generate_embeddings_batch(
        self, 
        images: List[Image.Image],
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images efficiently.
        
        Processes images in batches to balance memory usage and speed.
        
        Args:
            images: List of PIL Images
            batch_size: Maximum images per batch (default 8)
            
        Returns:
            Array of normalized embeddings, shape (num_images, embedding_dim)
        """
        if not images:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_embeddings = self._process_batch(batch_images)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def _process_batch(self, images: List[Image.Image]) -> np.ndarray:
        """
        Process a batch of images through the vision encoder.
        
        Args:
            images: List of PIL Images (single batch)
            
        Returns:
            Array of embeddings for this batch
        """
        from qwen_vl_utils import process_vision_info
        
        # Build conversations for processing
        conversations = []
        for img in images:
            conversations.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "describe"}  # Minimal text
                ]
            }])
        
        # Extract image inputs using Qwen's utility
        image_inputs = []
        for conv in conversations:
            imgs, _ = process_vision_info(conv)
            image_inputs.extend(imgs)
        
        # Prepare text inputs (needed for processor)
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations
        ]
        
        # Process through the full processor
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to device
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)
        
        # Extract embeddings from vision encoder
        with torch.no_grad():
            # Forward through the model to get hidden states
            # We use the model's forward but stop before generation
            outputs = self.vlm(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Get the hidden states from the vision part
            # For Qwen2.5-VL, the vision features are embedded into the hidden states
            # We extract from the first layer's hidden states (after vision encoding)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                # Use the first hidden state (after embedding layer)
                # This contains the embedded vision tokens
                hidden = outputs.hidden_states[0]  # [batch, seq_len, hidden_dim]
                
                # Pool over sequence (mean of all tokens including vision)
                # This captures the overall image representation
                pooled = hidden.mean(dim=1)  # [batch, hidden_dim]
                
                # Normalize for cosine similarity
                norms = torch.norm(pooled, dim=1, keepdim=True)
                normalized = pooled / (norms + 1e-8)
                
                embeddings = normalized.cpu().numpy()
                
                # Update embedding dimension if needed
                if self.embedding_dim != embeddings.shape[1]:
                    self.embedding_dim = embeddings.shape[1]
                
                return embeddings
            else:
                # Fallback: create zero embeddings (should not happen)
                print("Warning: Could not extract hidden states, using fallback")
                return np.zeros((len(images), self.embedding_dim or 1024))
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim or 1024


# Convenience function for creating extractor from pipeline
def create_embedding_extractor(pipeline) -> QwenEmbeddingExtractor:
    """
    Create an embedding extractor from an existing HierarchicalPipeline.
    
    Args:
        pipeline: Initialized HierarchicalPipeline instance
        
    Returns:
        QwenEmbeddingExtractor ready to use
    """
    if not pipeline.initialized:
        raise RuntimeError("Pipeline must be initialized before creating extractor")
    
    return QwenEmbeddingExtractor(
        vlm_model=pipeline.vlm,
        processor=pipeline.processor,
        device=pipeline.device
    )

