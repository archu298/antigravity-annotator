# AntiGravity Annotator — Architecture Overview

This document explains how the application is built, how its pieces connect, and what each part is responsible for. It is intended for developers who want to understand, extend, or contribute to the codebase.

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [How It Works — Big Picture](#2-how-it-works--big-picture)
3. [Tech Stack](#3-tech-stack)
4. [File Structure](#4-file-structure)
5. [Backend API](#5-backend-api)
6. [Frontend UI](#6-frontend-ui)
7. [AI & Model Pipeline](#7-ai--model-pipeline)
8. [Data Storage](#8-data-storage)
9. [Feature Status](#9-feature-status)
10. [Dependencies](#10-dependencies)
11. [Security & Network](#11-security--network)

---

## 1. What It Does

AntiGravity Annotator is a **self-hosted image annotation tool** built for teams labeling large datasets (hundreds to thousands of images). It runs as a local web server — you start it with one command, open a browser, and everyone on your network can use it immediately.

**Core workflow:**
1. Create a project and define your label classes (e.g., "car", "pedestrian")
2. Upload images (drag-and-drop, any quantity)
3. Draw annotations — bounding boxes, polygons, polylines, or keypoints
4. Optionally run AI-assisted annotation using YOLO or SAM2 models on your GPU
5. Export annotations in standard formats (YOLO, COCO, CSV)

An **Agent pipeline** takes this further: you describe what objects to find in plain text, the agent runs detection across your entire dataset automatically, accepts high-confidence results, and flags uncertain ones for you to review with a single click.

No cloud services, no accounts, no internet connection required. All data stays on your machine.

---

## 2. How It Works — Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                         Your Browser                        │
│                                                             │
│   React UI (frontend/index.html)                            │
│   ┌──────────┐  ┌────────────┐  ┌──────────────────────┐   │
│   │ Project  │  │ Annotation │  │   AI Panel /         │   │
│   │   List   │  │  Canvas    │  │   Agent Pipeline     │   │
│   └──────────┘  └────────────┘  └──────────────────────┘   │
│                        │  REST API calls (fetch)            │
└────────────────────────┼────────────────────────────────────┘
                         │ HTTP  (port 8001)
┌────────────────────────┼────────────────────────────────────┐
│                    server.py (FastAPI)                       │
│                                                             │
│   /api/projects   /api/images   /api/annotations            │
│   /api/models     /api/infer    /api/agent                  │
│                        │                                    │
│          ┌─────────────┴──────────────┐                     │
│          │                            │                     │
│   projects/ (disk)            models/ (disk)                │
│   └─ {id}/project.json        └─ yolo11n.onnx               │
│   └─ {id}/annotations.json    └─ yolov8x-worldv2.pt         │
│   └─ {id}/images/             └─ sam2.1_b.pt  ...           │
│                                            │                │
│                               Ultralytics / PyTorch (GPU)   │
└─────────────────────────────────────────────────────────────┘
```

**Request flow in plain English:**
- The browser loads `index.html` once — this contains the entire UI
- Every action (open project, save annotation, run AI) is a REST API call from the browser to `server.py`
- `server.py` reads/writes JSON files and image files on disk
- For AI inference, `server.py` loads the model into GPU memory and runs it, then returns results to the browser as JSON
- The browser never talks to the GPU directly — inference is always server-side

---

## 3. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Web server** | FastAPI + Uvicorn (Python) | Async Python framework; auto-generates API docs at `/docs` |
| **AI inference** | Ultralytics | Unified interface for YOLO, YOLOWorld, and SAM2 models |
| **GPU runtime** | PyTorch + CUDA | Required for fast inference; CPU fallback built-in |
| **Image processing** | Pillow | Validates uploaded images, reads dimensions |
| **UI framework** | React 18 | Component-based UI; runs in browser via CDN |
| **UI styling** | Tailwind CSS | Utility-first CSS; standalone build, no npm needed |
| **JSX transpiler** | Babel Standalone | Converts JSX to JavaScript in the browser at load time |
| **ZIP / download** | JSZip + FileSaver.js | Used for annotation export |
| **Storage** | Local filesystem | Plain JSON + image files; no database |

**No build step required.** The frontend is a single HTML file that loads React, Babel, and Tailwind from a local `vendor/` folder. You can edit `index.html` directly and refresh your browser to see changes.

---

## 4. File Structure

```
antigravity/
│
├── run.py                  Entry point — sets up GPU, checks environment,
│                           opens browser, starts the server
│
├── server.py               All backend logic — every API route lives here
│
├── requirements.txt        Python package dependencies
│
├── frontend/
│   ├── index.html          The entire UI — one file, ~2400 lines of React/JSX
│   └── vendor/             JavaScript libraries served locally (no CDN calls)
│       ├── react.production.min.js
│       ├── react-dom.production.min.js
│       ├── babel.min.js
│       ├── tailwind.min.js
│       ├── jszip.min.js
│       ├── FileSaver.min.js
│       └── ort.min.js + wasm files   (ONNX Runtime — vendored, not yet used)
│
├── models/                 Place your model files here
│   ├── sam2.1_b.pt         SAM 2.1 Base  — 155 MB — segmentation
│   ├── sam2.1_l.pt         SAM 2.1 Large — 429 MB — segmentation
│   ├── yolo11n.onnx        YOLOv11 Nano  —  11 MB — object detection
│   ├── yolo11s.onnx        YOLOv11 Small —  37 MB — object detection
│   └── yolov8x-worldv2.pt  YOLOv8x-World — 140 MB — open-vocabulary detection
│
└── projects/               Auto-created; one subfolder per project
    └── {8-char-id}/
        ├── project.json        Name, classes, last-modified timestamp
        ├── annotations.json    All annotations for every image in this project
        └── images/             Uploaded image files
```

### Why a separate `run.py`?

`server.py` focuses purely on API logic. `run.py` handles the "startup" concerns:
- Detecting which GPU has the most free memory and setting `CUDA_VISIBLE_DEVICES` before PyTorch loads
- Creating `projects/` and `models/` directories if they don't exist
- Printing the local and network URLs
- Opening the browser automatically

This separation is a common pattern: keep the server code free of ops concerns.

---

## 5. Backend API

All routes are defined in `server.py`. The FastAPI app is available at `http://localhost:8001`. Interactive API documentation (try-it-in-browser) is at `http://localhost:8001/docs`.

### Health Check

| Method | Endpoint | What it returns |
|--------|----------|-----------------|
| `GET` | `/api/health` | Server version, GPU availability and name, count of projects and models |

### Projects

A **project** is a named container for images and annotations. It stores its label classes (e.g., "dog", "cat") and settings.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/projects` | Create a project — body: `{"name": "...", "classes": [...]}` |
| `GET` | `/api/projects` | List all projects, sorted by most recently modified |
| `GET` | `/api/projects/{id}` | Get a project's details and its image list |
| `PUT` | `/api/projects/{id}` | Update project name, classes, or settings |
| `DELETE` | `/api/projects/{id}` | Delete project and all its images and annotations (irreversible) |

### Images

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/projects/{id}/images` | Upload one or more image files (multipart form) |
| `GET` | `/api/projects/{id}/images` | List images — supports `?page=`, `?limit=`, `?search=`, `?status=annotated\|unannotated` |
| `GET` | `/api/projects/{id}/images/{filename}` | Serve an image file — used directly as `<img src="...">` in the UI |
| `DELETE` | `/api/projects/{id}/images/{filename}` | Delete an image and its annotations |

**Accepted image formats:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`

Images are served directly from disk with browser caching — no base64 encoding. This makes the tool practical for large datasets where loading everything into memory would be slow.

### Annotations

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/projects/{id}/annotations` | Get all annotations for the entire project at once |
| `GET` | `/api/projects/{id}/annotations/{filename}` | Get annotations for one specific image |
| `PUT` | `/api/projects/{id}/annotations/{filename}` | Save annotations for one image (called on auto-save) |
| `PUT` | `/api/projects/{id}/annotations` | Replace all annotations in bulk (used by agent pipeline and import) |

### Models

The server keeps one model loaded in GPU memory at a time. Loading a new model automatically unloads the previous one to free VRAM.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List model files and show which one is currently loaded |
| `POST` | `/api/models` | Upload a new model file (`.pt`, `.onnx`, or `.engine`) |
| `POST` | `/api/models/load` | Load a model — body: `{"filename": "yolov8x-worldv2.pt"}` |
| `POST` | `/api/models/unload` | Unload the current model and free GPU memory |

**How model type is determined from filename:**

| Filename contains | Loaded as |
|-------------------|-----------|
| `sam` | SAM segmentation model |
| `world` | YOLOWorld open-vocabulary model |
| anything else | Standard YOLO detection model |

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/infer` | Run the loaded model on a single image |
| `POST` | `/api/infer/batch` | Run detection across multiple images in sequence |

**Single inference request body:**
```json
{
  "project_id": "a61f84f7",
  "filename": "photo.jpg",
  "conf": 0.25,
  "iou": 0.45,
  "classes": ["forklift", "person"],
  "points": [[120, 340]],
  "labels": [1],
  "box": [100, 80, 400, 350]
}
```

- `classes` — for YOLOWorld only: which object types to detect
- `points` + `labels` — for SAM only: click coordinates (label 1 = foreground, 0 = background)
- `box` — for SAM only: bounding box prompt instead of point clicks

**Response for detection models:**
```json
{
  "detections": [
    {"box": [x1, y1, x2, y2], "score": 0.87, "class_id": 0, "class_name": "forklift"}
  ],
  "model_type": "detect"
}
```

**Response for SAM segmentation:**
```json
{
  "masks": [
    {"pts": [[x, y], [x, y], ...], "score": 1.0}
  ],
  "model_type": "segment"
}
```

All coordinates in inference responses are **pixel values** relative to the original image size.

**CPU fallback:** If a CUDA or ONNX runtime error occurs (e.g., ONNX model compiled for a different GPU architecture), the server automatically retries the same inference on CPU. Results are identical, just slower.

### Agent Pipeline

The agent pipeline lets you auto-annotate an entire dataset with a single request.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/agent/run` | Run a full auto-annotation sweep over a project |
| `GET` | `/api/agent/flagged/{id}` | Get the list of images with uncertain (flagged) annotations |
| `POST` | `/api/agent/accept/{id}` | Accept or reject flagged annotations on a specific image |

**How it works — `/api/agent/run` request:**
```json
{
  "project_id": "a61f84f7",
  "description": "forklift, person, pallet",
  "detection_model": "yolov8x-worldv2.pt",
  "sam_model": "sam2.1_b.pt",
  "accept_threshold": 0.70,
  "flag_threshold": 0.30,
  "refine_with_sam": false,
  "scope": "unannotated",
  "detection_conf": 0.10,
  "detection_iou": 0.45
}
```

**Scope options:** `"all"` (every image), `"unannotated"` (skip already-annotated images), `"selected"` (a specific list of filenames).

**Accept/flag threshold logic:**

```
Detection confidence score
        │
        ├─ ≥ accept_threshold (default 0.70)  → Auto-accepted ✅
        │
        ├─ ≥ flag_threshold (default 0.30)    → Flagged for review ⚠️
        │   (if refine_with_sam=true, SAM runs on these detections
        │    to produce polygon masks instead of boxes)
        │
        └─ < flag_threshold                   → Discarded ✗
```

**After the agent runs:**
- Auto-accepted annotations are written directly to `annotations.json`
- Flagged annotations are also saved but marked with `"agent_status": "flagged"` and `"agent_score": <confidence>`
- The UI shows a review screen where you can accept or reject flagged images one by one
- Accepting removes the `agent_status` flag (annotation becomes permanent); rejecting deletes it

**`/api/agent/accept` actions:** `"accept_all"`, `"reject_all"`, or `"selective"` (with a list of annotation IDs to keep).

### Frontend Serving

Any URL that doesn't match an `/api/` route serves the frontend. If the URL matches a file in `frontend/` (e.g., a vendor JS file), that file is served directly. Otherwise, `index.html` is returned — this is the standard "SPA fallback" pattern that lets the React app handle its own routing.

---

## 6. Frontend UI

The entire UI lives in `frontend/index.html` — a single file containing ~2,400 lines of React written in JSX. Babel translates the JSX to JavaScript in the browser at load time. No npm, no webpack, no build step.

### Screen Flow

```
Browser opens
      │
      ▼
┌─────────────┐     User selects       ┌──────────────────────────┐
│ Project List │ ─── or creates ──────▶ │ Annotator (main editor)  │
│             │     a project           │                          │
└─────────────┘                        └──────────────────────────┘
```

### Annotator Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Toolbar  [Project Name ▼]  [Tools]  [Save] [AI] [Export]... │
├──────────┬───────────────────────────────────┬───────────────┤
│          │                                   │               │
│  Image   │                                   │  Annotations  │
│ Sidebar  │       Annotation Canvas           │    Panel      │
│          │       (zoom, pan, draw)           │  (list, edit, │
│ Thumbnail│                                   │   class pick) │
│   list   │                                   │               │
│          │                                   │               │
└──────────┴───────────────────────────────────┴───────────────┘
```

### Components

| Component | What it does |
|-----------|-------------|
| **Root** | App entry point — shows either the Project List or the Annotator |
| **Project List** | Shows all projects as cards; handles create, rename, open, delete |
| **Toolbar** | Top bar with drawing tool buttons, project name, and action buttons |
| **Image Sidebar** | Scrollable thumbnail strip on the left; shows annotation count per image; supports multi-select |
| **Annotation Canvas** | The main drawing area — handles all mouse events for drawing and editing annotations, zoom/pan, and SAM click placement |
| **Properties Panel** | Right-side panel listing annotations for the current image; click an annotation to select it; change its class |
| **Class Editor** | Modal for adding, renaming, recoloring, and deleting label classes |
| **AI Panel** | Modal for running YOLO detection or SAM segmentation on the current image — shows a preview before you apply results |
| **Agent Panel** | Modal for configuring and running the batch agent pipeline; shows results and flagged image review after the run |
| **Export Modal** | Export annotations to YOLO TXT, COCO JSON, CSV, or the native JSON format |
| **Import Modal** | Import annotations from YOLO TXT, COCO JSON, or CSV |
| **Propagation Modal** | Copy annotations from the current image forward to the next N images |
| **Command Palette** | `Ctrl+K` overlay — fuzzy-search through all available actions |
| **Keyboard Help** | `?` overlay — shows all keyboard shortcuts |
| **Context Menu** | Right-click on an annotation — change class, duplicate, copy to all images, lock, bring to front/back, delete |
| **Toast Notifications** | Brief slide-in messages (top-right corner) for success, error, and warning feedback |

### How Annotations Are Saved

```
User draws a shape on the canvas
        │
        ▼
Shape is added to the in-memory annotation list for the current image
        │
        ▼
A 2-minute auto-save timer resets (or user presses Ctrl+S)
        │
        ▼
Browser sends:  PUT /api/projects/{id}/annotations/{filename}
                Body: { "annotations": [ ...all annotations for this image... ] }
        │
        ▼
server.py reads annotations.json from disk
Updates the entry for this filename
Writes annotations.json back to disk
Updates the project's "modified" timestamp in project.json
        │
        ▼
Annotations are persisted — safe even if browser closes
```

All edits are kept in memory and periodically flushed. The undo/redo history is in-memory only and is lost on page refresh.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `V` | Select / move tool |
| `B` | Bounding box |
| `P` | Polygon |
| `L` | Polyline |
| `H` | Pan (drag canvas) |
| `K` | Keypoints |
| `A` | Open AI panel |
| `G` | Open propagation panel |
| `N` / `Shift+N` | Next / previous image |
| `Tab` / `Shift+Tab` | Cycle through annotations |
| `Delete` | Delete selected annotation |
| `1` – `9` | Select class by number |
| `=` / `-` | Zoom in / out |
| `0` | Fit image to view |
| `\` | Toggle annotation labels |
| `` ` `` | Toggle dark / light theme |
| `F` | Fit view |
| `Ctrl+S` | Save |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| `Ctrl+D` | Duplicate selected annotation |
| `Ctrl+E` | Open export |
| `Ctrl+K` | Command palette |
| `Ctrl+→` | Propagate to next image and advance |
| `Enter` | Finish drawing polygon / polyline / keypoints |
| `Escape` | Cancel drawing / close panel |
| `?` | Keyboard shortcut help |

---

## 7. AI & Model Pipeline

### Models Available

| File | Size | Purpose |
|------|------|---------|
| `yolo11n.onnx` | 11 MB | Fast, lightweight object detection (80 COCO classes) |
| `yolo11s.onnx` | 37 MB | Slightly larger/more accurate detection |
| `yolov8x-worldv2.pt` | 140 MB | Open-vocabulary detection — detect any object you name in text |
| `sam2.1_b.pt` | 155 MB | Click-to-segment: draw a mask from a point or box |
| `sam2.1_l.pt` | 429 MB | SAM Large — higher accuracy segmentation |

### Detection (YOLO)

1. User opens the AI panel and selects a YOLO model
2. Server loads the model into GPU memory (unloading any previous model)
3. User clicks **Preview** — the server runs detection on the current image and returns bounding boxes + confidence scores
4. A preview canvas shows the detections overlaid on the image
5. User clicks **Apply** — the detections become annotations in the project

For **YOLOWorld**, the user types object names (e.g., "forklift, person") before running detection. The model detects only those named objects, so you don't need a custom-trained model for new categories.

For **batch detection**, the same model runs across all selected images in sequence. Results are added as annotations directly without a preview step.

### Segmentation (SAM2)

1. User loads a SAM model and switches to the SAM tab
2. User clicks on the image to place foreground points (or draws a rough bounding box)
3. Server runs SAM with those prompts and returns a polygon outline of the detected object
4. The polygon is overlaid on the image
5. User clicks **Add as Annotation** — the polygon becomes a polygon annotation

SAM is useful for precise object outlines where a bounding box is not specific enough.

### Agent Pipeline (Auto-annotation)

The agent pipeline is designed to annotate large datasets with minimal manual work:

```
┌─────────────────────────────────────────────────────────────┐
│  User configures:                                           │
│   • What objects to find (text description)                 │
│   • Which detection model to use                            │
│   • Accept threshold (e.g., 70% confidence → auto-accept)  │
│   • Flag threshold  (e.g., 30% confidence → flag for review)│
│   • Scope: all images / unannotated only / selected         │
└──────────────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
                    For each image in scope:
                    Run detection model at low confidence (0.10)
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
               score ≥ 0.70  0.30 ≤ score < 0.70  score < 0.30
               Auto-accept    Flag for review       Discard
                                   │
                    (optional) Run SAM on flagged detections
                               to convert boxes → polygon masks
                                   │
                                   ▼
                    Save all accepted + flagged annotations to disk
                    Return summary + list of flagged images
                                   │
                                   ▼
                    User reviews flagged images one by one:
                    Accept ✓  or  Reject ✗  per image
```

All existing annotations in the project are preserved — the agent only adds new ones.

---

## 8. Data Storage

There is no database. All data is stored as plain files you can read and edit directly.

### Project Structure

```
projects/
└── a61f84f7/                    ← project ID (random 8-char hex)
    ├── project.json             ← project name, classes, last-modified
    ├── annotations.json         ← all annotations for every image
    └── images/
        ├── photo1.jpg
        └── photo2.png
```

### project.json

Stores the project name, its label classes, and when it was last changed.

```json
{
  "id": "a61f84f7",
  "name": "forklift",
  "created": 1773213561.0,
  "modified": 1773214893.7,
  "classes": [
    { "id": "35089c", "name": "forklift", "color": "#7ec8c8" },
    { "id": "193959", "name": "person",   "color": "#e8a87c" }
  ],
  "settings": {
    "ci": 2
  }
}
```

- `id` — matches the directory name
- `classes[].id` — 6-character hex, used as a reference from annotation objects
- `created` / `modified` — Unix timestamps (seconds since 1970)
- `settings.ci` — the index of the last image you were viewing, so you resume where you left off

### annotations.json

A single file holding all annotations for the project, grouped by image filename.

```json
{
  "photo1.jpg": [
    {
      "id": "a1b2c3d4",
      "tp": "bb",
      "x": 120.0, "y": 45.0, "w": 230.0, "h": 180.0,
      "cid": "35089c"
    },
    {
      "id": "e5f6g7h8",
      "tp": "pg",
      "pts": [[100, 50], [200, 50], [250, 150], [100, 150]],
      "cid": "193959"
    }
  ],
  "photo2.jpg": []
}
```

**Annotation types:**

| `tp` value | Shape | Extra fields |
|------------|-------|-------------|
| `bb` | Bounding box | `x`, `y` (top-left corner), `w`, `h` (width, height) — all in pixels |
| `pg` | Polygon | `pts`: list of `[x, y]` pixel coordinates, minimum 3 points, closed shape |
| `pl` | Polyline | `pts`: list of `[x, y]` pixel coordinates, open path (not closed) |
| `kp` | Keypoints | `keypoints`: list of `{x, y, v}` where `v` is visibility (0=unlabeled, 1=labeled, 2=visible) |

All coordinates are **pixel values** relative to the original image dimensions.

**Optional fields on any annotation:**
- `lk: true` — annotation is locked; user cannot move or resize it
- `agent_status: "accepted"` or `"flagged"` — set by the agent pipeline; absent on manual annotations
- `agent_score: 0.85` — model confidence score from the agent run

### Image Serving

Images are never encoded or transformed. They are served directly from disk over HTTP:

```
File on disk:   projects/{id}/images/photo.jpg
Browser URL:    /api/projects/{id}/images/photo.jpg
Used as:        <img src="/api/projects/{id}/images/photo.jpg">
```

The browser handles caching automatically. This approach is efficient for large datasets because only visible images are loaded into memory.

---

## 9. Feature Status

| Feature | Status | Notes |
|---------|--------|-------|
| Project management | ✅ Working | Create, rename, delete |
| Image upload | ✅ Working | Multi-file, drag-and-drop, validates format |
| Bounding box annotation | ✅ Working | 8-handle resize, drag to move |
| Polygon annotation | ✅ Working | Click to add points, Enter to close |
| Polyline annotation | ✅ Working | Open path, Enter to finish |
| Keypoint annotation | ✅ Working | Visibility flags per keypoint |
| Undo / redo | ✅ Working | In-memory; resets on page refresh |
| Auto-save | ✅ Working | Every 2 minutes + manual Ctrl+S |
| Class management | ✅ Working | Add, rename, recolor, delete |
| Annotation propagation | ✅ Working | Copy to next N images |
| Import — YOLO TXT | ✅ Working | |
| Import — COCO JSON | ✅ Working | |
| Import — CSV | ✅ Working | |
| Export — YOLO TXT | ✅ Working | |
| Export — COCO JSON | ✅ Working | |
| Export — CSV | ✅ Working | |
| Export — Native JSON | ✅ Working | The raw annotations.json format |
| Keyboard shortcuts | ✅ Working | Full set — see section 6 |
| Command palette | ✅ Working | Ctrl+K, fuzzy search |
| Dark / light theme | ✅ Working | |
| LAN access | ✅ Working | Any device on the same network |
| YOLO detection (single image) | ✅ Working | Preview before applying |
| YOLO detection (batch) | ✅ Working | Runs across multiple images |
| YOLOWorld text-prompt detection | ✅ Working | Describe objects in plain text |
| SAM2 click-to-segment | ✅ Working | Point and box prompts |
| Agent auto-annotation pipeline | ✅ Working | Full sweep + flagged review |
| GPU auto-selection | ✅ Working | Picks GPU with most free memory |
| Model upload via UI | ✅ Working | Upload .pt / .onnx / .engine |
| ONNX model support | ⚠️ Partial | Falls back to CPU if GPU architecture mismatch |
| In-browser ONNX inference | ❌ Not built | ONNX Runtime is vendored but not connected |
| Full project backup export | ❌ Stub | Shows message: copy `./projects/` manually |
| Full project backup import | ❌ Stub | Not yet implemented |
| Real-time agent progress | ❌ Not built | Agent run blocks until complete (no progress bar) |
| Video annotation | ❌ Not built | Image files only |
| Authentication | ❌ Not built | Open access — LAN only |
| Thumbnail caching | ❌ Not built | Full images used in sidebar |

---

## 10. Dependencies

### Python

Install with `pip install -r requirements.txt`. Install PyTorch separately first (see README).

| Package | What it does |
|---------|-------------|
| `fastapi` | Web framework — defines all API routes |
| `uvicorn` | ASGI server — runs FastAPI and handles HTTP connections |
| `python-multipart` | Required for file uploads (multipart form parsing) |
| `Pillow` | Opens and validates uploaded images, reads image dimensions |
| `ultralytics` | Loads and runs YOLO / YOLOWorld / SAM2 models |
| `numpy` | Array operations on model output tensors |
| `torch` | PyTorch — GPU runtime for model inference (install separately) |

**`torch` is not in `requirements.txt`** because the right version depends on your CUDA version. See the README for the correct install command for your hardware.

### Frontend (all files in `vendor/`, no internet required)

| Library | Purpose |
|---------|---------|
| React 18 | UI component framework |
| React DOM | Renders React components to the browser DOM |
| Babel Standalone | Converts JSX syntax to plain JavaScript at load time |
| Tailwind CSS | Styling — utility classes applied directly in JSX |
| JSZip | Creates ZIP archives for annotation export |
| FileSaver.js | Triggers file downloads in the browser |
| ONNX Runtime Web | Vendored for future in-browser inference — not currently used |

### GPU Requirements

| Task | GPU needed? |
|------|------------|
| Running the server, managing projects/images/annotations | No |
| YOLO or SAM inference | No — CPU works; GPU makes it significantly faster |
| Agent pipeline on 1,000+ images | GPU strongly recommended (minutes vs. hours) |

---

## 11. Security & Network

### Intended Use

This tool is designed for **LAN use only** — a team on the same network, or a single user running it locally. It is not designed to be exposed to the internet.

### Network Binding

The server binds to `0.0.0.0` by default, meaning it accepts connections from any device on your local network. The startup banner prints both the localhost URL and the LAN IP address so teammates can connect easily.

To restrict to local access only:
```bash
python run.py --host 127.0.0.1
```

### File Access Protection

Two measures prevent malicious requests from reading files outside the project directory:

1. **Project ID validation** — project IDs must match `[a-zA-Z0-9_-]+`. A request with `project_id = "../../etc/passwd"` is rejected immediately with a 400 error.

2. **Path resolution check** — for image and model file requests, the server resolves the full filesystem path and confirms it falls inside the expected directory. A filename like `../../server.py` would resolve outside the images folder and return a 403 error.

### File Upload Validation

- Image uploads are validated by Pillow after saving — if a file is not a valid image, it is deleted and the upload reports an error
- Model uploads are restricted to `.pt`, `.onnx`, `.engine` extensions
- All uploaded filenames are sanitized (special characters replaced with underscores)

### What Is Not Protected

| Gap | Impact |
|-----|--------|
| No authentication | Any device on your LAN can read and modify any project |
| No HTTPS | Traffic is unencrypted — fine for a local network, not for the internet |
| No rate limiting | A client could spam the API or upload very large files |
| CORS is fully open | Any webpage can make API calls to this server (acceptable for a LAN tool) |

If you need to expose the tool beyond your LAN, put it behind a reverse proxy (Nginx or Caddy) with authentication and HTTPS.
