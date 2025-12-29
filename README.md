# Hierarchical Vision Analysis Pipeline

A multi-stage object detection and analysis system that combines YOLO for fast detection with a Vision-Language Model (VLM) for detailed scene understanding.
## Demo
https://youtu.be/xvKNw0Qv5gY
## What It Does

Given an image, the pipeline:
1. Detects all objects using YOLO
2. Routes detections based on class and confidence
3. Analyzes priority detections (e.g., people) with a VLM to describe visible actions, clothing, and objects

The hierarchical approach avoids running expensive VLM inference on every detection—only the ones that matter get full analysis.

## Architecture

```
                      <img width="670" height="823" alt="image" src="https://github.com/user-attachments/assets/81d30065-eaf9-461d-bddb-ee7183e1b14b" />

```

## Project Structure

```
.
├── app.py              # FastAPI server with REST endpoints
├── pipeline.py         # Core pipeline logic (detection, routing, VLM)
├── demo.py             # CLI demo for testing without the server
├── model.py            # Original notebook code (for reference)
├── static/
│   └── index.html      # Web frontend
├── yolo11l.pt          # YOLO model weights
└── requirements.txt    # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for reasonable performance. CPU works but is slow.

## Running the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in your browser.

Models load on startup—expect 30-60 seconds before the API is ready.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web frontend |
| `/health` | GET | Pipeline status |
| `/analyze` | POST | Analyze image, return JSON |
| `/analyze/stream` | POST | SSE streaming results |
| `/analyze/full-stream` | POST | Streaming with crop images |

### Example Request

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/analyze
```

### Response Format

```json
{
  "success": true,
  "elapsed_seconds": 2.34,
  "num_detections": 5,
  "results": [
    {
      "index": 0,
      "class": "person",
      "confidence": 0.87,
      "stage": "vlm_full",
      "analysis": "Person standing, wearing dark jacket...",
      "crop_image": "data:image/jpeg;base64,..."
    }
  ]
}
```

## CLI Demo

For testing without the server:

```bash
python demo.py                    # uses default test image
python demo.py path/to/image.jpg  # custom image
```

## Configuration

Key settings in `pipeline.py`:

```python
PipelineConfig(
    yolo_model="yolo11l.pt",
    vlm_model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    detection_confidence=0.1,
    high_confidence_threshold=0.3,
    priority_classes=("person",),
    use_quantization=True,        # INT8 for lower memory
    batch_size=4,                 # VLM batch size
)
```

## Models Used

- **YOLO11-Large**: Object detection
- **Qwen2.5-VL-3B-Instruct**: Vision-language model for analysis

Both models are loaded once at startup and reused for all requests.

## Performance Notes

- INT8 quantization reduces VRAM usage significantly
- Batched VLM inference processes multiple crops per forward pass
- Non-priority classes skip VLM entirely (no cost)
- Streaming endpoints return results as they complete

## Requirements

- Python 3.10+
- CUDA GPU (tested on RTX 3090, works on 8GB+ VRAM with quantization)
- ~6GB VRAM with INT8 quantization enabled

