# storage_utils.py
"""
File system utilities for storing and loading images.
Handles saving images to disk and loading them back.
"""

import uuid
from pathlib import Path
from PIL import Image
from typing import Optional

# Define storage directories
STORAGE_ROOT = Path("storage")
SESSIONS_DIR = STORAGE_ROOT / "sessions"  # Full images
CROPS_DIR = STORAGE_ROOT / "crops"        # Cropped images


def ensure_storage_dirs():
    """Create storage directories if they don't exist."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)


def save_session_image(image: Image.Image, session_id: str) -> Path:
    """
    Save full session image to disk.
    
    Args:
        image: PIL Image to save
        session_id: Session ID (used as filename)
    
    Returns:
        Path to saved image file
    """
    ensure_storage_dirs()
    
    filename = f"{session_id}.jpg"
    filepath = SESSIONS_DIR / filename
    
    # Save as JPEG with good quality
    image.save(filepath, "JPEG", quality=85)
    
    return filepath


def save_crop_image(image: Image.Image, object_id: str) -> Path:
    """
    Save cropped entity image to disk.
    
    Args:
        image: PIL Image (cropped) to save
        object_id: Object ID (used as filename)
    
    Returns:
        Path to saved image file
    """
    ensure_storage_dirs()
    
    filename = f"{object_id}.jpg"
    filepath = CROPS_DIR / filename
    
    # Save as JPEG with good quality
    image.save(filepath, "JPEG", quality=85)
    
    return filepath


def load_image(image_path: Path) -> Image.Image:
    """
    Load image from disk.
    
    Args:
        image_path: Path to image file
    
    Returns:
        PIL Image object
    """
    return Image.open(image_path).convert("RGB")

