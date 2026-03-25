# 🚀 AntiGravity Annotator

### GPU-Accelerated Image Annotation Platform with AI-Powered Auto-Labeling

A self-hosted, production-grade image annotation tool built with **FastAPI + React + PyTorch**. Supports bounding boxes, polygons, polylines, and keypoints — with integrated **YOLO11**, **YOLO-World**, and **SAM2** models for AI-assisted labeling. Includes an **agentic auto-annotation pipeline** that sweeps entire datasets, auto-accepts high-confidence detections, and flags uncertain ones for human review.

> **No cloud. No accounts. No internet required.** All data stays on your machine. Runs on LAN — start the server and your whole team can annotate from any browser.

---

## 📸 Screenshots

### Annotation Workspace
Full-featured canvas with bounding box, polygon, polyline, and keypoint tools. Class management, annotation statistics, zoom/pan, and dark theme.

![Annotation Workspace](screenshots/workspace.png)

### Project Dashboard
Create and manage multiple annotation projects. Track image counts, annotation progress, and last-modified timestamps.

![Project Dashboard](screenshots/dashboard.png)

### AI Assist Panel
One-click AI inference with YOLO11, YOLO-World (text-prompt detection), and SAM2 (click-to-segment). Adjustable confidence and NMS thresholds. Batch detection across entire projects.

![AI Assist](screenshots/ai-assist.png)

### Agentic Auto-Annotation Pipeline
Configure confidence thresholds → Agent auto-annotates the full dataset → High-confidence results are accepted, uncertain detections are flagged for human review. Optional SAM refinement for polygon-level precision.

![Agent Pipeline](screenshots/agent-pipeline.png)

---

## ✨ Key Features

| Category | Features |
|----------|----------|
| **Annotation Tools** | Bounding boxes, polygons, polylines, keypoints with 8-handle resize, drag-to-move, undo/redo, lock, and annotation propagation |
| **AI Models** | YOLO11 detection, YOLO-World open-vocabulary detection (text prompt), SAM2 click-to-segment and box-prompted segmentation |
| **Agent Pipeline** | Agentic auto-annotation — runs multi-stage inference (detect → filter by confidence → optionally refine with SAM → flag for human review), reducing manual annotation effort by ~70% |
| **Import/Export** | YOLO TXT, COCO JSON, CSV, native JSON — ready for direct model training |
| **GPU Management** | Auto-selects GPU with most free VRAM, single-model memory management, CPU fallback |
| **Collaboration** | LAN access — any device on the same network can use the tool simultaneously |
| **UX** | Command palette (Ctrl+K), full keyboard shortcuts, dark/light theme, paginated image browser with search and filter |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, Uvicorn |
| **AI/ML** | PyTorch, CUDA, Ultralytics (YOLO11, YOLO-World, SAM2) |
| **Frontend** | React 18, Tailwind CSS, HTML5 Canvas |
| **Storage** | File-based JSON (no database required) |
| **Deployment** | Single-command launch, LAN-accessible, zero configuration |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- (Optional) NVIDIA GPU + CUDA for fast AI inference

### Install

```bash
# 1. Install PyTorch (choose your CUDA version):

# CPU only:
pip install torch torchvision

# GPU — CUDA 12.1 (Tesla T4, RTX series):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies:
pip install -r requirements.txt
```

### Run

```bash
python run.py
```

Opens at `http://localhost:8001`. Your LAN IP is printed in the terminal for team access.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address (`0.0.0.0` = LAN access) |
| `--port` | `8001` | Port number |
| `--gpu` | auto | GPU device index (e.g. `--gpu 1`) |
| `--no-browser` | — | Skip auto-opening browser |
| `--reload` | — | Enable hot-reload for development |

---

## 📂 Project Structure

```
antigravity/
├── server.py          # FastAPI backend — all API routes (20+ endpoints)
├── run.py             # Launcher — GPU selection, browser open, server start
├── requirements.txt   # Python dependencies
├── frontend/
│   ├── index.html     # Full React UI (~2400 lines, single file)
│   └── vendor/        # Vendored JS libs (React, Tailwind, Babel, JSZip)
├── models/            # YOLO / SAM model files (.pt, .onnx)
└── projects/          # Auto-created; one folder per project
    └── {project_id}/
        ├── project.json       # Project metadata and class definitions
        ├── annotations.json   # All annotations keyed by image filename
        └── images/            # Uploaded image files
```

---

## 🤖 AI Models

Place model files in the `models/` directory. Model type is auto-detected from filename:

| File | Size | Type | Use Case |
|------|------|------|----------|
| `yolo11n.onnx` | 11 MB | YOLO detection | Fast object detection |
| `yolo11s.onnx` | 37 MB | YOLO detection | Better accuracy |
| `yolov8x-worldv2.pt` | 140 MB | YOLO-World | Open-vocabulary detection (text prompts) |
| `sam2.1_b.pt` | 155 MB | SAM2 segmentation | Click-to-segment |
| `sam2.1_l.pt` | 429 MB | SAM2 segmentation | Higher quality segmentation |

Only one model is loaded in GPU memory at a time. Loading a new model automatically unloads the previous one to manage VRAM.

---

## ⌨️ Keyboard Shortcuts

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `B` | Bounding box | `Ctrl+Z` | Undo |
| `P` | Polygon | `Ctrl+Shift+Z` | Redo |
| `L` | Polyline | `Ctrl+S` | Save |
| `K` | Keypoints | `Ctrl+E` | Export |
| `V` | Select/move | `Ctrl+K` | Command palette |
| `A` | AI panel | `N` / `Shift+N` | Next/prev image |
| `G` | Propagation | `1`–`9` | Quick-select class |
| `?` | Show all shortcuts | `` ` `` | Toggle theme |

---

## 📡 API Reference

Full interactive docs available at `http://localhost:8001/docs` when the server is running.

### Core Endpoints

| Category | Endpoints |
|----------|-----------|
| **Projects** | `POST /api/projects` · `GET /api/projects` · `GET /api/projects/{id}` · `PUT /api/projects/{id}` · `DELETE /api/projects/{id}` |
| **Images** | `POST /api/projects/{id}/images` · `GET /api/projects/{id}/images` · `GET /api/projects/{id}/images/{file}` · `DELETE /api/projects/{id}/images/{file}` |
| **Annotations** | `GET /api/projects/{id}/annotations` · `GET /api/projects/{id}/annotations/{file}` · `PUT /api/projects/{id}/annotations/{file}` · `PUT /api/projects/{id}/annotations` |
| **Models** | `GET /api/models` · `POST /api/models` · `POST /api/models/load` · `POST /api/models/unload` |
| **Inference** | `POST /api/infer` · `POST /api/infer/batch` |
| **Agent** | `POST /api/agent/run` · `GET /api/agent/flagged/{id}` · `POST /api/agent/accept/{id}` |

---

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of the system design, data flow, storage format, security model, and feature status.

**High-level architecture:**

```
Browser (React UI)  ←→  FastAPI Server (server.py)  ←→  PyTorch/CUDA (GPU)
                              ↕
                    File System (JSON + Images)
```

- Single-file React frontend (~2400 lines) — no build step required
- REST API with 20+ endpoints for project management, image serving, annotations, and ML orchestration
- Single-model GPU memory management — auto-selects GPU with most free VRAM
- File-based JSON storage — no database dependency

---

## 📝 License

This project is for personal and educational use.

---

## 👩‍💻 Author

**Archana K** — Associate Consultant, AI Computer Vision  
[LinkedIn](https://www.linkedin.com/in/archanaksnow) · [Email](mailto:archanaforstudy298@gmail.com)
