# AntiGravity Annotator

A GPU-accelerated image annotation tool with a FastAPI backend and browser-based frontend. Supports bounding box, polygon, polyline, and keypoint annotations with AI-assisted labeling via YOLO, YOLO-World, and SAM2 models — including an agentic pipeline that automatically sweeps a dataset and flags uncertain detections for human review.

---

## Features

- Create and manage multiple annotation projects
- Upload and browse large image sets (paginated, search, filter by annotation status)
- Draw **bounding boxes**, **polygons**, **polylines**, and **keypoints**
- Undo/redo, zoom/pan, lock annotations, dark/light theme
- Annotation propagation (copy annotations forward to N images)
- Import/export: **YOLO TXT**, **COCO JSON**, **CSV**, native JSON
- Command palette (`Ctrl+K`) and full keyboard shortcut set
- AI-assisted annotation:
  - **YOLO detection** (`.pt`, `.onnx`) — single image or batch
  - **YOLO-World** open-vocabulary detection (text prompt)
  - **SAM2** click-to-segment and box-prompted segmentation
  - **Agent pipeline** — auto-annotate a whole dataset, accept high-confidence detections, flag uncertain ones for review
- GPU auto-selection (picks the GPU with the most free VRAM)
- LAN access — any device on the same network can open the tool
- Interactive API docs at `/docs`

---

## Requirements

- Python 3.8+
- (Optional) NVIDIA GPU + CUDA for fast AI inference

### Install

```bash
# 1. Install torch first — choose the right build for your hardware:

#   CPU only:
pip install torch torchvision

#   GPU — CUDA 12.1 (Tesla T4, RTX series):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

#   GPU — CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install all other dependencies:
pip install -r requirements.txt
```

> **Note:** `torch` is not listed in `requirements.txt` because the correct version depends on your CUDA version. Install it manually (step 1) before running `pip install -r requirements.txt`.

---

## Quick Start

```bash
python run.py
```

The server starts on `http://localhost:8001` and opens the browser automatically.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address (`0.0.0.0` = all interfaces for LAN access) |
| `--port` | `8001` | Port number |
| `--gpu` | auto | GPU device index (e.g. `--gpu 1` for Tesla T4) |
| `--no-browser` | — | Skip auto-opening browser |
| `--reload` | — | Enable hot-reload for development |

**Examples:**

```bash
python run.py --port 9000
python run.py --gpu 1 --no-browser
python run.py --reload
```

---

## Directory Structure

```
antigravity/
├── server.py          # FastAPI app — all API routes and handlers
├── run.py             # Launcher — GPU selection, browser open, server start
├── requirements.txt   # Python dependencies
├── frontend/
│   ├── index.html     # Browser UI (React, single file)
│   └── vendor/        # Vendored JS libraries (React, Tailwind, etc.)
├── models/            # YOLO / SAM model files (.pt, .onnx, .engine)
└── projects/          # Created automatically; one folder per project
    └── {project_id}/
        ├── project.json       # Project metadata and class definitions
        ├── annotations.json   # All annotations keyed by image filename
        └── images/            # Uploaded image files
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `V` | Select / move tool |
| `B` | Bounding box tool |
| `P` | Polygon tool |
| `L` | Polyline tool |
| `H` | Pan tool |
| `K` | Keypoints tool |
| `A` | Open AI panel |
| `G` | Open propagation panel |
| `N` / `Shift+N` | Next / previous image |
| `Tab` / `Shift+Tab` | Cycle annotations |
| `Delete` | Delete selected annotation |
| `1`–`9` | Quick-select class by index |
| `=` / `-` | Zoom in / out |
| `0` | Fit view |
| `\` | Toggle label visibility |
| `` ` `` | Toggle dark/light theme |
| `Ctrl+S` | Save |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| `Ctrl+D` | Duplicate selected annotation |
| `Ctrl+E` | Export |
| `Ctrl+K` | Command palette |
| `Ctrl+→` | Propagate annotation to next image |
| `Enter` | Finish drawing polygon / polyline / keypoints |
| `Escape` | Cancel draw / close dialog |
| `?` | Keyboard shortcut help |

---

## API Reference

### Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server status, GPU info, project/model counts |
| POST | `/api/projects` | Create a project |
| GET | `/api/projects` | List all projects |
| GET | `/api/projects/{id}` | Get project details |
| PUT | `/api/projects/{id}` | Update project (name, classes, settings) |
| DELETE | `/api/projects/{id}` | Delete project and all its data |

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/projects/{id}/images` | Upload images (multipart) |
| GET | `/api/projects/{id}/images` | List images — query: `page`, `limit`, `search`, `status` |
| GET | `/api/projects/{id}/images/{file}` | Serve an image file |
| DELETE | `/api/projects/{id}/images/{file}` | Delete an image and its annotations |

### Annotations

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/projects/{id}/annotations` | Get all annotations for a project |
| GET | `/api/projects/{id}/annotations/{file}` | Get annotations for one image |
| PUT | `/api/projects/{id}/annotations/{file}` | Save annotations for one image |
| PUT | `/api/projects/{id}/annotations` | Bulk save all annotations |

### Models & Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List available models + currently loaded model |
| POST | `/api/models` | Upload a model file |
| GET | `/api/models/{filename}` | Download a model file |
| POST | `/api/models/load` | Load a model into GPU memory |
| POST | `/api/models/unload` | Unload model and free VRAM |
| POST | `/api/infer` | Run inference on a single image (detect / SAM) |
| POST | `/api/infer/batch` | Batch detection across multiple images |

### Agent Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/agent/run` | Auto-annotate a dataset with configurable confidence thresholds |
| GET | `/api/agent/flagged/{id}` | Get flagged / accepted / unprocessed image lists |
| POST | `/api/agent/accept/{id}` | Accept or reject flagged annotations for an image |

Full interactive docs: `http://localhost:8001/docs`

---

## Annotation Format

Annotations are stored in `annotations.json` as **pixel coordinates** relative to the original image dimensions, keyed by image filename:

```json
{
  "photo.jpg": [
    {
      "id": "a1b2c3d4",
      "tp": "bb",
      "x": 120.5, "y": 45.0, "w": 230.0, "h": 180.0,
      "cid": "35089c"
    },
    {
      "id": "e5f6g7h8",
      "tp": "pg",
      "pts": [[100, 50], [200, 50], [250, 150], [100, 150]],
      "cid": "193959"
    },
    {
      "id": "i9j0k1l2",
      "tp": "pl",
      "pts": [[10, 10], [50, 80], [120, 60]],
      "cid": "35089c"
    },
    {
      "id": "m3n4o5p6",
      "tp": "kp",
      "keypoints": [{"x": 100.0, "y": 200.0, "v": 2}],
      "cid": "193959"
    }
  ]
}
```

**Annotation types (`tp`):**

| `tp` | Type | Fields |
|------|------|--------|
| `bb` | Bounding box | `x`, `y`, `w`, `h` (top-left origin, pixel dimensions) |
| `pg` | Polygon | `pts: [[x,y], ...]` (closed, ≥3 points) |
| `pl` | Polyline | `pts: [[x,y], ...]` (open path) |
| `kp` | Keypoints | `keypoints: [{x, y, v}]` (v: 0=unlabeled, 1=labeled, 2=visible) |

`cid` references a class `id` from `project.json`. Optional fields: `lk` (boolean, locked annotation), `agent_status` (`"accepted"` or `"flagged"`), `agent_score` (model confidence).

---

## Models

Place model files in the `models/` directory. Supported formats: `.pt`, `.onnx`, `.engine`.

Model type is inferred from the filename:
- `sam` in name → SAM segmentation (`ultralytics.SAM`)
- `world` in name → YOLO-World open-vocabulary detection (`ultralytics.YOLOWorld`)
- anything else → standard YOLO detection (`ultralytics.YOLO`)

**Models included:**

| File | Size | Type |
|------|------|------|
| `sam2.1_b.pt` | 155 MB | SAM 2.1 Base segmentation |
| `sam2.1_l.pt` | 429 MB | SAM 2.1 Large segmentation |
| `yolo11n.onnx` | 11 MB | YOLOv11 Nano detection |
| `yolo11s.onnx` | 37 MB | YOLOv11 Small detection |
| `yolov8x-worldv2.pt` | 140 MB | YOLOv8x-World open-vocabulary detection |

Only one model is loaded in GPU memory at a time. Loading a new model automatically unloads the previous one.
