"""On-disk storage for originals, thumbnails and crops (all writes off the event loop)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from ..config import get_settings
from ..runtime import get_runtime

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".avif"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp"}


def media_kind(name: str, content_type: Optional[str] = None) -> Optional[str]:
    ext = Path(name or "").suffix.lower()
    if ext in VIDEO_EXT or (content_type or "").startswith("video/"):
        return "video"
    if ext in IMAGE_EXT or (content_type or "").startswith("image/"):
        return "image"
    return None


def safe_suffix(name: str, kind: str) -> str:
    """The file suffix stored media and temporary uploads get: never the client's choice, so an
    upload can not become e.g. a ``.pt`` file. Images are normalised to JPEG; videos keep a known
    container suffix (decoders pick the demuxer by content) or get ``.mp4``."""
    if kind == "image":
        return ".jpg"
    ext = Path(name or "").suffix.lower()
    return ext if ext in VIDEO_EXT else ".mp4"


class FileStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or get_settings().storage_dir)
        self.media_dir = self.root / "media"
        self.thumb_dir = self.root / "thumbs"
        self.crop_dir = self.root / "crops"
        for d in (self.media_dir, self.thumb_dir, self.crop_dir):
            d.mkdir(parents=True, exist_ok=True)

    # paths
    def media_path(self, media_id: str, ext: str) -> Path:
        return self.media_dir / f"{media_id}{ext.lower()}"

    def thumb_path(self, media_id: str) -> Path:
        return self.thumb_dir / f"{media_id}.jpg"

    def crop_path(self, entity_id: str) -> Path:
        return self.crop_dir / f"{entity_id}.jpg"

    # sync helpers
    @staticmethod
    def _save_jpeg(img: Image.Image, path: Path, quality: int = 88, max_side: Optional[int] = None) -> Path:
        img = img.convert("RGB")
        if max_side and max(img.size) > max_side:
            img = img.copy()
            img.thumbnail((max_side, max_side), Image.BILINEAR)
        tmp = path.with_suffix(".tmp")
        img.save(tmp, "JPEG", quality=quality)
        os.replace(tmp, path)
        return path

    @staticmethod
    def load_image(path: Path | str) -> Image.Image:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # respect phone camera orientation
        return img.convert("RGB")

    # async API
    async def save_crop(self, entity_id: str, img: Image.Image) -> Path:
        return await get_runtime().run_cpu(self._save_jpeg, img, self.crop_path(entity_id), 90)

    async def save_crops(self, items: list[tuple[str, Image.Image]]) -> list[Path]:
        def _all() -> list[Path]:
            return [self._save_jpeg(img, self.crop_path(eid), 90) for eid, img in items]

        return await get_runtime().run_cpu(_all)

    async def save_thumb(self, media_id: str, img: Image.Image) -> Path:
        return await get_runtime().run_cpu(self._save_jpeg, img, self.thumb_path(media_id), 80, 480)

    async def save_image_original(self, media_id: str, img: Image.Image) -> Path:
        # normalise to JPEG (EXIF orientation already applied) so boxes match what the browser shows
        return await get_runtime().run_cpu(self._save_jpeg, img, self.media_path(media_id, ".jpg"), 92)

    async def import_file(self, media_id: str, src: Path, move: bool = False) -> Path:
        """Copy (or move) a video into storage under a whitelisted suffix (see ``safe_suffix``)."""
        dst = self.media_path(media_id, safe_suffix(src.name, "video"))

        def _copy() -> Path:
            if move:
                shutil.move(str(src), dst)
            else:
                shutil.copyfile(src, dst)
            return dst

        return await get_runtime().run_cpu(_copy)

    async def open_image(self, path: Path | str) -> Image.Image:
        return await get_runtime().run_cpu(self.load_image, path)

    async def delete(self, paths: list[Optional[str]]) -> None:
        def _rm() -> None:
            for p in paths:
                if p:
                    try:
                        Path(p).unlink(missing_ok=True)
                    except OSError:
                        pass

        await get_runtime().run_cpu(_rm)


_store: Optional[FileStore] = None


def get_store() -> FileStore:
    global _store
    if _store is None:
        _store = FileStore()
    return _store


def reset_store(store: Optional[FileStore] = None) -> FileStore:
    global _store
    _store = store or FileStore()
    return _store
