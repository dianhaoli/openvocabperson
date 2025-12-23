# -*- coding: utf-8 -*-
"""
Hierarchical Vision Analysis Pipeline

A multi-stage detection and analysis pipeline with:
- Hierarchical routing (YOLO → Class Filter → VLM)
- Async streaming results
- Batched VLM inference
- PyTorch optimizations (quantization, compile, etc.)

Author: ML Engineer Portfolio Project
"""

import asyncio
import math
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from ultralytics import YOLO

from qwen_vl_utils import process_vision_info


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class ProcessingStage(Enum):
    """Stages in the hierarchical pipeline."""
    YOLO_ONLY = "yolo_only"           # Skipped VLM (not a priority class)
    LOW_CONFIDENCE = "low_confidence"  # Low confidence, minimal processing
    VLM_FULL = "vlm_full"             # Full VLM analysis


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    
    # ── Model Settings ──
    yolo_model: str = "yolo11l.pt"
    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"  # Accuracy priority
    
    # ── Detection Settings ──
    detection_confidence: float = 0.1
    detection_iou: float = 0.5
    expand_ratio: float = 0.1  # Expand bounding boxes by 10%
    
    # ── Hierarchical Routing Thresholds ──
    high_confidence_threshold: float = 0.3  # Above this → full VLM analysis
    priority_classes: tuple = ("person",)    # Classes that warrant VLM analysis
    
    # ── VLM Generation Settings ──
    max_new_tokens: int = 35  # Concise outputs
    
    # ── Optimization Settings ──
    use_quantization: bool = True
    quantization_bits: int = 8  # INT8 for accuracy (vs INT4 for speed)
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    
    # ── Batching Settings ──
    batch_size: int = 4  # VLM batch size
    
    # ── Async Settings ──
    max_workers: int = 2  # Thread pool workers


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED PROMPT - Focused on Observable Facts Only
# ══════════════════════════════════════════════════════════════════════════════

ANALYSIS_PROMPT = """Answer only what is directly visible. Do not infer intent, identity, or emotion. If uncertain, say so.

A. Actions visible now?
B. Objects person is interacting with?
C. Clothing (non-sensitive)?
D. What's NOT visible/uncertain?
E. Visibility limitations?

Be concise."""


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """A single detection from YOLO."""
    index: int
    box: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    class_id: int


@dataclass
class CropInfo:
    """Information about a cropped region."""
    detection: Detection
    crop_image: Image.Image
    expanded_box: tuple  # (x1, y1, x2, y2) after expansion


@dataclass
class AnalysisResult:
    """Result of analyzing a single crop."""
    crop_info: CropInfo
    stage: ProcessingStage
    analysis_text: Optional[str] = None
    reason: Optional[str] = None  # Why this stage was chosen
    
    def to_dict(self) -> dict:
        return {
            "index": self.crop_info.detection.index,
            "box": self.crop_info.detection.box,
            "confidence": self.crop_info.detection.confidence,
            "class": self.crop_info.detection.class_name,
            "stage": self.stage.value,
            "analysis": self.analysis_text,
            "reason": self.reason,
        }


@dataclass
class StreamEvent:
    """Event yielded during async streaming."""
    event_type: str  # "detection_complete", "crop_analyzed", "batch_complete", "complete"
    data: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class HierarchicalPipeline:
    """
    Multi-stage detection and analysis pipeline.
    
    Architecture:
        Stage 1: YOLO Detection (fast)
        Stage 2: Class-based Routing (instant)
        Stage 3: VLM Analysis (expensive, only for priority detections)
    
    Features:
        - Hierarchical routing based on class and confidence
        - Batched VLM inference
        - Async streaming results
        - PyTorch optimizations (quantization, compile)
    
    Usage:
        pipeline = HierarchicalPipeline()
        await pipeline.initialize()
        
        # Streaming mode
        async for event in pipeline.analyze_streaming(image_path):
            print(event)
        
        # Batch mode
        results = pipeline.analyze(image_path)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.yolo = None
        self.vlm = None
        self.processor = None
        self.device = None
        self.initialized = False
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
    
    # ──────────────────────────────────────────────────────────────────────────
    # INITIALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    
    async def initialize(self):
        """Initialize models asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._load_models)
    
    def _load_models(self):
        """Load YOLO and VLM models with optimizations."""
        print("🚀 Initializing Hierarchical Pipeline...")
        print("─" * 60)
        
        # ── PyTorch Backend Optimizations ──
        self._configure_pytorch_backends()
        
        # ── Load YOLO ──
        print(f"   Loading YOLO: {self.config.yolo_model}")
        self.yolo = YOLO(self.config.yolo_model)
        print("   ✓ YOLO loaded")
        
        # ── Determine Device ──
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {self.device}")
        
        # ── Load VLM with Optimizations ──
        self._load_vlm_optimized()
        
        self.initialized = True
        print("─" * 60)
        print("✅ Pipeline initialized!\n")
    
    def _configure_pytorch_backends(self):
        """Configure PyTorch for optimal inference."""
        if torch.cuda.is_available():
            # cuDNN optimizations
            torch.backends.cudnn.benchmark = True
            
            # TF32 for Ampere+ GPUs (slight precision trade-off, big speed gain)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable optimized attention backends
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            print("   ✓ PyTorch backends optimized (cuDNN, TF32, SDPA)")
    
    def _load_vlm_optimized(self):
        """Load VLM with quantization and compilation."""
        print(f"   Loading VLM: {self.config.vlm_model_id}")
        
        # ── Quantization Config ──
        quantization_config = None
        if self.config.use_quantization and self.device == "cuda":
            if self.config.quantization_bits == 8:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                print("   ✓ INT8 quantization enabled")
            elif self.config.quantization_bits == 4:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                print("   ✓ INT4 quantization enabled")
        
        # ── Load Model ──
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }
        
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        
        # Try to use SDPA attention implementation
        try:
            load_kwargs["attn_implementation"] = "sdpa"
        except Exception:
            pass  # Fallback to default attention
        
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.vlm_model_id,
            **load_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.config.vlm_model_id)
        
        # Fix padding side for decoder-only models
        if hasattr(self.processor, 'tokenizer'):
            self.processor.tokenizer.padding_side = 'left'
        
        print("   ✓ VLM loaded")
        
        # ── torch.compile() ──
        if self.config.use_torch_compile and not self.config.use_quantization:
            # Note: torch.compile may have issues with quantized models
            try:
                self.vlm = torch.compile(
                    self.vlm,
                    mode=self.config.compile_mode
                )
                print(f"   ✓ torch.compile enabled (mode={self.config.compile_mode})")
            except Exception as e:
                print(f"   ⚠ torch.compile failed: {e}")
    
    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 1: YOLO DETECTION
    # ──────────────────────────────────────────────────────────────────────────
    
    def detect(self, image_source) -> tuple[Image.Image, np.ndarray, list[Detection]]:
        """
        Run YOLO detection on image.
        
        Returns:
            tuple: (PIL Image, OpenCV image, list of Detection objects)
        """
        # Load image
        if isinstance(image_source, str):
            image_pil = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            image_pil = image_source.convert("RGB")
        else:
            raise ValueError("image_source must be a file path or PIL Image")
        
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Run YOLO
        results = self.yolo(
            np.array(image_pil),
            conf=self.config.detection_confidence,
            iou=self.config.detection_iou,
            verbose=False,
        )
        
        # Parse detections
        detections = []
        for r in results:
            for idx, box in enumerate(r.boxes):
                det = Detection(
                    index=idx,
                    box=tuple(map(int, box.xyxy[0].tolist())),
                    confidence=float(box.conf[0]),
                    class_name=r.names[int(box.cls[0])],
                    class_id=int(box.cls[0]),
                )
                detections.append(det)
        
        return image_pil, image_cv, detections
    
    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 2: CROP EXTRACTION
    # ──────────────────────────────────────────────────────────────────────────
    
    def extract_crops(
        self, image_cv: np.ndarray, detections: list[Detection]
    ) -> list[CropInfo]:
        """Extract cropped regions from detections with expanded bounding boxes."""
        h_img, w_img = image_cv.shape[:2]
        crops = []
        
        for det in detections:
            x1, y1, x2, y2 = det.box
            w, h = x2 - x1, y2 - y1
            
            # Expand bounding box
            dx = int(w * self.config.expand_ratio / 2)
            dy = int(h * self.config.expand_ratio / 2)
            
            x1_exp = max(0, x1 - dx)
            y1_exp = max(0, y1 - dy)
            x2_exp = min(w_img, x2 + dx)
            y2_exp = min(h_img, y2 + dy)
            
            # Crop and convert
            crop_bgr = image_cv[y1_exp:y2_exp, x1_exp:x2_exp]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            crops.append(CropInfo(
                detection=det,
                crop_image=crop_pil,
                expanded_box=(x1_exp, y1_exp, x2_exp, y2_exp),
            ))
        
        return crops
    
    # ──────────────────────────────────────────────────────────────────────────
    # STAGE 3: HIERARCHICAL ROUTING
    # ──────────────────────────────────────────────────────────────────────────
    
    def route_crops(self, crops: list[CropInfo]) -> dict[ProcessingStage, list[CropInfo]]:
        """
        Route crops to appropriate processing stage based on class and confidence.
        
        Routing Logic:
            - Non-priority class (e.g., car, dog) → YOLO_ONLY (skip VLM)
            - Priority class, low confidence → LOW_CONFIDENCE (minimal)
            - Priority class, high confidence → VLM_FULL (full analysis)
        """
        routed = {
            ProcessingStage.YOLO_ONLY: [],
            ProcessingStage.LOW_CONFIDENCE: [],
            ProcessingStage.VLM_FULL: [],
        }
        
        for crop in crops:
            det = crop.detection
            
            # Check if priority class
            if det.class_name not in self.config.priority_classes:
                routed[ProcessingStage.YOLO_ONLY].append(crop)
            elif det.confidence < self.config.high_confidence_threshold:
                routed[ProcessingStage.LOW_CONFIDENCE].append(crop)
            else:
                routed[ProcessingStage.VLM_FULL].append(crop)
        
        return routed
    
    # ──────────────────────────────────────────────────────────────────────────
    # VLM ANALYSIS (BATCHED)
    # ──────────────────────────────────────────────────────────────────────────
    
    def analyze_crops_batched(
        self,
        crops: list[CropInfo],
        prompt: str = ANALYSIS_PROMPT,
    ) -> list[str]:
        """
        Analyze multiple crops in batches using VLM.
        
        Uses batched inference for efficiency while maintaining accuracy.
        """
        all_results = []
        
        for i in range(0, len(crops), self.config.batch_size):
            batch = crops[i:i + self.config.batch_size]
            batch_results = self._analyze_batch(batch, prompt)
            all_results.extend(batch_results)
        
        return all_results
    
    def _analyze_batch(self, batch: list[CropInfo], prompt: str) -> list[str]:
        """Analyze a batch of crops in a single forward pass."""
        if not batch:
            return []
        
        # Build batch of messages
        conversations = []
        for crop_info in batch:
            conversations.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_info.crop_image},
                    {"type": "text", "text": prompt}
                ]
            }])
        
        # Prepare inputs
        texts = [
            self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in conversations
        ]
        
        image_inputs = []
        for conv in conversations:
            imgs, _ = process_vision_info(conv)
            image_inputs.extend(imgs)
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device with memory pinning for speed
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                if self.device == "cuda":
                    inputs[key] = inputs[key].pin_memory().to(self.device, non_blocking=True)
                else:
                    inputs[key] = inputs[key].to(self.device)
        
        # Generate with optimizations
        with torch.inference_mode():
            generated_ids = self.vlm.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        
        # Decode responses - handle variable input lengths in batch
        # Each sequence may have different input length due to padding
        responses = []
        for i in range(len(batch)):
            # Get the actual input length for this sequence (excluding padding)
            input_ids_i = inputs.input_ids[i]
            # Find where actual tokens end (non-pad tokens)
            if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer.pad_token_id is not None:
                pad_id = self.processor.tokenizer.pad_token_id
                # For left-padded, find first non-pad token
                non_pad_mask = input_ids_i != pad_id
                if non_pad_mask.any():
                    input_len = non_pad_mask.sum().item()
                else:
                    input_len = len(input_ids_i)
            else:
                input_len = inputs.input_ids.shape[1]
            
            # Decode only the generated part
            generated_tokens = generated_ids[i, inputs.input_ids.shape[1]:]
            response = self.processor.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            responses.append(response)
        
        return responses
    
    def analyze_single_crop(self, crop_info: CropInfo, prompt: str = ANALYSIS_PROMPT) -> str:
        """Analyze a single crop (for streaming or individual analysis)."""
        results = self._analyze_batch([crop_info], prompt)
        return results[0] if results else ""
    
    # ──────────────────────────────────────────────────────────────────────────
    # MAIN ANALYSIS (SYNCHRONOUS)
    # ──────────────────────────────────────────────────────────────────────────
    
    def analyze(self, image_source, prompt: str = ANALYSIS_PROMPT) -> list[AnalysisResult]:
        """
        Run full hierarchical analysis pipeline (synchronous).
        
        Args:
            image_source: Path to image or PIL Image
            prompt: Custom prompt (uses ANALYSIS_PROMPT by default)
        
        Returns:
            list of AnalysisResult objects
        """
        if not self.initialized:
            self._load_models()
        
        # Stage 1: Detection
        image_pil, image_cv, detections = self.detect(image_source)
        print(f"📊 Detected {len(detections)} objects")
        
        # Stage 2: Crop extraction
        crops = self.extract_crops(image_cv, detections)
        
        # Stage 3: Route crops
        routed = self.route_crops(crops)
        print(f"   → YOLO only: {len(routed[ProcessingStage.YOLO_ONLY])}")
        print(f"   → Low confidence: {len(routed[ProcessingStage.LOW_CONFIDENCE])}")
        print(f"   → Full VLM: {len(routed[ProcessingStage.VLM_FULL])}")
        
        results = []
        
        # Process YOLO_ONLY (instant, no VLM)
        for crop_info in routed[ProcessingStage.YOLO_ONLY]:
            results.append(AnalysisResult(
                crop_info=crop_info,
                stage=ProcessingStage.YOLO_ONLY,
                analysis_text=None,
                reason=f"Skipped: class '{crop_info.detection.class_name}' not in priority list",
            ))
        
        # Process LOW_CONFIDENCE (could add minimal VLM check here)
        for crop_info in routed[ProcessingStage.LOW_CONFIDENCE]:
            results.append(AnalysisResult(
                crop_info=crop_info,
                stage=ProcessingStage.LOW_CONFIDENCE,
                analysis_text=None,
                reason=f"Low confidence ({crop_info.detection.confidence:.2f})",
            ))
        
        # Process VLM_FULL (batched)
        vlm_crops = routed[ProcessingStage.VLM_FULL]
        if vlm_crops:
            print(f"\n🔍 Analyzing {len(vlm_crops)} crops with VLM...")
            analyses = self.analyze_crops_batched(vlm_crops, prompt)
            
            for crop_info, analysis in zip(vlm_crops, analyses):
                results.append(AnalysisResult(
                    crop_info=crop_info,
                    stage=ProcessingStage.VLM_FULL,
                    analysis_text=analysis,
                    reason="High confidence, priority class",
                ))
        
        # Sort by original detection index
        results.sort(key=lambda r: r.crop_info.detection.index)
        
        return results
    
    # ──────────────────────────────────────────────────────────────────────────
    # ASYNC STREAMING ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────
    
    async def analyze_streaming(
        self, image_source, prompt: str = ANALYSIS_PROMPT
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Run analysis with streaming results.
        
        Yields StreamEvent objects as processing progresses:
            - detection_complete: YOLO detection finished
            - crop_routed: Crop routing decision made
            - crop_analyzed: Single crop VLM analysis complete
            - complete: All processing finished
        
        Usage:
            async for event in pipeline.analyze_streaming(image_path):
                if event.event_type == "crop_analyzed":
                    print(f"Result: {event.data['analysis']}")
        """
        if not self.initialized:
            await self.initialize()
        
        loop = asyncio.get_event_loop()
        
        # Stage 1: Detection (run in executor)
        image_pil, image_cv, detections = await loop.run_in_executor(
            self.executor, self.detect, image_source
        )
        
        yield StreamEvent(
            event_type="detection_complete",
            data={
                "num_detections": len(detections),
                "detections": [
                    {"index": d.index, "class": d.class_name, "confidence": d.confidence}
                    for d in detections
                ],
            },
        )
        
        # Stage 2: Extract and route crops
        crops = self.extract_crops(image_cv, detections)
        routed = self.route_crops(crops)
        
        yield StreamEvent(
            event_type="routing_complete",
            data={
                "yolo_only": len(routed[ProcessingStage.YOLO_ONLY]),
                "low_confidence": len(routed[ProcessingStage.LOW_CONFIDENCE]),
                "vlm_full": len(routed[ProcessingStage.VLM_FULL]),
            },
        )
        
        # Yield immediate results for non-VLM crops
        for crop_info in routed[ProcessingStage.YOLO_ONLY]:
            yield StreamEvent(
                event_type="crop_analyzed",
                data=AnalysisResult(
                    crop_info=crop_info,
                    stage=ProcessingStage.YOLO_ONLY,
                    reason=f"Skipped: not priority class",
                ).to_dict(),
            )
        
        for crop_info in routed[ProcessingStage.LOW_CONFIDENCE]:
            yield StreamEvent(
                event_type="crop_analyzed",
                data=AnalysisResult(
                    crop_info=crop_info,
                    stage=ProcessingStage.LOW_CONFIDENCE,
                    reason=f"Low confidence ({crop_info.detection.confidence:.2f})",
                ).to_dict(),
            )
        
        # Stage 3: VLM analysis (stream as completed)
        vlm_crops = routed[ProcessingStage.VLM_FULL]
        if vlm_crops:
            # Process in batches, yield after each batch
            for i in range(0, len(vlm_crops), self.config.batch_size):
                batch = vlm_crops[i:i + self.config.batch_size]
                
                # Run batch analysis in executor
                analyses = await loop.run_in_executor(
                    self.executor, self._analyze_batch, batch, prompt
                )
                
                # Yield each result in the batch
                for crop_info, analysis in zip(batch, analyses):
                    result = AnalysisResult(
                        crop_info=crop_info,
                        stage=ProcessingStage.VLM_FULL,
                        analysis_text=analysis,
                        reason="Full VLM analysis",
                    )
                    yield StreamEvent(
                        event_type="crop_analyzed",
                        data=result.to_dict(),
                    )
        
        yield StreamEvent(event_type="complete", data={})
    
    # ──────────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    
    def visualize_results(
        self, results: list[AnalysisResult], cols: int = 4, figsize_mult: float = 5
    ):
        """Display results in a grid with analysis text."""
        import matplotlib.pyplot as plt
        
        # Filter to only VLM results with analysis
        vlm_results = [r for r in results if r.analysis_text]
        
        if not vlm_results:
            print("No VLM analysis results to visualize.")
            return
        
        num = len(vlm_results)
        rows = math.ceil(num / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_mult, rows * (figsize_mult + 1)))
        
        # Flatten axes
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for i, result in enumerate(vlm_results):
            row, col = i // cols, i % cols
            ax = axes[row][col] if rows > 1 else axes[0][col]
            
            ax.imshow(result.crop_info.crop_image)
            ax.axis("off")
            
            wrapped = textwrap.fill(result.analysis_text or "", width=35)
            ax.set_title(
                f"Crop {result.crop_info.detection.index}\n{wrapped}",
                fontsize=8,
                pad=5,
            )
        
        # Hide empty subplots
        for i in range(num, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row][col] if rows > 1 else axes[0][col]
            ax.axis("off")
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self, results: list[AnalysisResult]):
        """Print formatted analysis results."""
        print("\n" + "═" * 80)
        print("📋 HIERARCHICAL ANALYSIS RESULTS")
        print("═" * 80)
        
        for result in results:
            det = result.crop_info.detection
            print(f"\n┌─ Detection {det.index} ({det.class_name}, conf={det.confidence:.2f}) ─")
            print(f"│  Stage: {result.stage.value}")
            print(f"│  Reason: {result.reason}")
            if result.analysis_text:
                wrapped = textwrap.fill(
                    result.analysis_text, width=74,
                    initial_indent="│  ", subsequent_indent="│  "
                )
                print(wrapped)
            print(f"└{'─' * 77}")
        
        print("\n" + "═" * 80)
        
        # Summary
        stages = {}
        for r in results:
            stages[r.stage.value] = stages.get(r.stage.value, 0) + 1
        print(f"Summary: {len(results)} detections")
        for stage, count in stages.items():
            print(f"   • {stage}: {count}")
        print("═" * 80)
    
    # ──────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────────────────
    
    def cleanup(self):
        """Free GPU memory."""
        if self.vlm is not None:
            del self.vlm
            self.vlm = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.initialized = False
        print("🧹 Pipeline cleaned up, GPU memory freed.")


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def create_pipeline(
    use_quantization: bool = True,
    use_compile: bool = True,
    model_size: str = "3B",  # "3B" or "2B"
) -> HierarchicalPipeline:
    """
    Factory function to create a configured pipeline.
    
    Args:
        use_quantization: Enable INT8 quantization
        use_compile: Enable torch.compile
        model_size: VLM model size ("3B" for accuracy, "2B" for speed)
    
    Returns:
        Configured HierarchicalPipeline instance
    """
    model_id = {
        "3B": "Qwen/Qwen2.5-VL-3B-Instruct",
        "2B": "Qwen/Qwen2.5-VL-2B-Instruct",
    }.get(model_size, "Qwen/Qwen2.5-VL-3B-Instruct")
    
    config = PipelineConfig(
        vlm_model_id=model_id,
        use_quantization=use_quantization,
        use_torch_compile=use_compile,
    )
    
    return HierarchicalPipeline(config)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN (for testing)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    # Default test image
    image_path = sys.argv[1] if len(sys.argv) > 1 else "YFQdcXrsu64BJMMhEl6k2WPjorzYG4.jpg"
    
    print("=" * 60)
    print("HIERARCHICAL VISION ANALYSIS PIPELINE - TEST RUN")
    print("=" * 60)
    
    # Create and initialize pipeline
    pipeline = create_pipeline(use_quantization=True, use_compile=False)
    pipeline._load_models()
    
    # Run analysis
    results = pipeline.analyze(image_path)
    
    # Display results
    pipeline.print_results(results)
    pipeline.visualize_results(results)

