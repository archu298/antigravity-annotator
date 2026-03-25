"""
AntiGravity Annotator - Backend Server
========================================
FastAPI application handling: projects, images, annotations, models, and frontend serving.

ARCHITECTURE:
    Browser Request -> Uvicorn (ASGI server) -> FastAPI (our app) -> Route Handlers

ROUTE MAP:
    GET  /api/health                                -> Server status
    POST /api/projects                              -> Create project
    GET  /api/projects                              -> List projects
    GET  /api/projects/{id}                         -> Get project
    PUT  /api/projects/{id}                         -> Update project
    DELETE /api/projects/{id}                        -> Delete project
    POST /api/projects/{id}/images                  -> Upload images
    GET  /api/projects/{id}/images                  -> List images (paginated)
    GET  /api/projects/{id}/images/{file}           -> Serve image
    DELETE /api/projects/{id}/images/{file}          -> Delete image
    GET  /api/projects/{id}/annotations             -> Get all annotations
    GET  /api/projects/{id}/annotations/{file}      -> Get image annotations
    PUT  /api/projects/{id}/annotations/{file}      -> Save image annotations
    PUT  /api/projects/{id}/annotations             -> Bulk save annotations
    GET  /api/models                                -> List YOLO models
    GET  /{path}                                    -> Serve frontend (catch-all)
"""

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List
import json
import time
import os
import shutil
import uuid
import re
from PIL import Image


# -- Path Configuration -------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
PROJECTS_DIR = BASE_DIR / "projects"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"


# -- GPU Model State (singleton — one model loaded at a time) --
loaded_model = None        # The ultralytics model object
loaded_model_name = None   # Filename of currently loaded model
loaded_model_type = None   # "detect", "segment", "world"
loaded_model_classes = None  # List of class names (if available)


# -- Create FastAPI App --------------------------------------------------------
app = FastAPI(
    title="AntiGravity Annotator API",
    version="1.0.0",
    description="Image annotation tool with GPU-accelerated AI inference"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# HELPERS
# ==============================================================================

def make_project_id():
    """Generate a short unique project ID (8 hex chars)."""
    return uuid.uuid4().hex[:8]


def read_json(path: Path, default=None):
    """Read JSON file safely. Returns default if file missing or corrupted.

    WHY: We read JSON files constantly. This avoids repeating try/except
    everywhere. The default parameter lets us return {} or [] gracefully.
    """
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError):
        pass
    return default if default is not None else {}


def write_json(path: Path, data):
    """Write data to JSON file with pretty formatting.

    indent=2 makes it human-readable (you can open project.json in a text editor).
    ensure_ascii=False allows non-English class names.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_project_dir(project_id: str) -> Path:
    """Get project directory path, with path-traversal prevention.

    SECURITY: If someone sends project_id='../../etc/passwd', they could
    read system files. We only allow alphanumeric + hyphen + underscore.
    """
    if not re.match(r'^[a-zA-Z0-9_-]+$', project_id):
        raise HTTPException(status_code=400, detail="Invalid project ID")
    return PROJECTS_DIR / project_id


def load_project_meta(project_id: str) -> dict:
    """Load project metadata, raise 404 if not found."""
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return read_json(project_dir / "project.json", {})


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

DEFAULT_COLORS = [
    "#e8a87c", "#7ec8c8", "#c38cd4", "#8bc98b", "#d4a5a5",
    "#6cb4d9", "#c9b95f", "#d47fa6", "#5eb89e", "#b0a0e0",
    "#d9946a", "#6fc5e3", "#b5c96a", "#e09bc0", "#7db8a0",
]


# ==============================================================================
# HEALTH CHECK
# ==============================================================================

@app.get("/api/health")
async def health_check():
    """Server status: GPU info, project count, model count.

    TRY: curl http://localhost:8001/api/health
    Or open http://localhost:8001/docs for interactive API explorer.
    """
    gpu_available = False
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    project_count = len([d for d in PROJECTS_DIR.iterdir() if d.is_dir()]) if PROJECTS_DIR.exists() else 0
    model_count = len([f for f in MODELS_DIR.iterdir() if f.suffix in ('.pt', '.onnx', '.engine')]) if MODELS_DIR.exists() else 0

    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": time.time(),
        "gpu": {"available": gpu_available, "name": gpu_name},
        "stats": {"projects": project_count, "models": model_count},
    }


# ==============================================================================
# STEP 1b: PROJECT CRUD
# ==============================================================================
#
# CRUD = Create, Read, Update, Delete
# REST mapping:  POST=Create, GET=Read, PUT=Update, DELETE=Delete
#
# FILE STRUCTURE PER PROJECT:
#   projects/{project_id}/
#       project.json       <- metadata: name, classes, settings
#       annotations.json   <- all annotations keyed by image filename
#       images/            <- original uploaded image files


@app.post("/api/projects")
async def create_project(request: Request):
    """Create a new annotation project.

    REQUEST BODY:
        {"name": "Street Signs", "classes": [{"name": "stop_sign", "color": "#e8a87c"}]}

    TRY: curl -X POST http://localhost:8001/api/projects \
              -H "Content-Type: application/json" \
              -d '{"name": "My First Project"}'
    """
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")

    project_id = make_project_id()
    project_dir = PROJECTS_DIR / project_id
    (project_dir / "images").mkdir(parents=True, exist_ok=True)

    # Assign colors to classes if not provided
    classes = body.get("classes", [])
    for i, cls in enumerate(classes):
        if "color" not in cls:
            cls["color"] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        if "id" not in cls:
            cls["id"] = uuid.uuid4().hex[:6]

    now = time.time()
    meta = {
        "id": project_id,
        "name": name,
        "classes": classes,
        "created": now,
        "modified": now,
        "settings": body.get("settings", {}),
    }

    write_json(project_dir / "project.json", meta)
    write_json(project_dir / "annotations.json", {})

    return {**meta, "image_count": 0, "annotation_count": 0}


@app.get("/api/projects")
async def list_projects():
    """List all projects with summary stats.

    Returns name, image count, annotation count, and thumbnail for each project.
    Sorted by most recently modified first.
    """
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    projects = []

    for d in sorted(PROJECTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = read_json(d / "project.json")
        if not meta:
            continue

        # Count images
        images_dir = d / "images"
        image_files = []
        if images_dir.exists():
            image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]

        # Count annotations
        annotations = read_json(d / "annotations.json", {})
        ann_count = sum(len(anns) for anns in annotations.values())

        projects.append({
            "id": meta.get("id", d.name),
            "name": meta.get("name", d.name),
            "classes": meta.get("classes", []),
            "created": meta.get("created", 0),
            "modified": meta.get("modified", 0),
            "image_count": len(image_files),
            "annotation_count": ann_count,
        })

    projects.sort(key=lambda p: p["modified"], reverse=True)
    return {"projects": projects}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    """Get full project details including image list."""
    meta = load_project_meta(project_id)
    project_dir = get_project_dir(project_id)

    images_dir = project_dir / "images"
    image_files = sorted([
        f.name for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    ]) if images_dir.exists() else []

    annotations = read_json(project_dir / "annotations.json", {})

    return {
        **meta,
        "images": image_files,
        "image_count": len(image_files),
        "annotation_count": sum(len(anns) for anns in annotations.values()),
    }


@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, request: Request):
    """Update project metadata (name, classes, settings).

    Send only the fields you want to change - others are preserved.
    """
    meta = load_project_meta(project_id)
    body = await request.json()
    project_dir = get_project_dir(project_id)

    if "name" in body:
        meta["name"] = body["name"].strip()
    if "classes" in body:
        meta["classes"] = body["classes"]
    if "settings" in body:
        meta["settings"] = body["settings"]

    meta["modified"] = time.time()
    write_json(project_dir / "project.json", meta)
    return meta


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project and ALL its data (images, annotations). Irreversible."""
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    shutil.rmtree(project_dir)
    return {"status": "deleted", "id": project_id}


# ==============================================================================
# STEP 1c: IMAGE MANAGEMENT
# ==============================================================================
#
# KEY DESIGN DECISION: HTTP file serving instead of base64
#
# Old (HTML app):  image stored as base64 string in IndexedDB
#   -> 33% larger, loaded into JS memory, no caching, RAM explosion at 2000+ imgs
#
# New (this API):  image served via <img src="/api/projects/{id}/images/{file}">
#   -> browser handles caching, lazy loading, streaming. Only visible images in memory.


@app.post("/api/projects/{project_id}/images")
async def upload_images(project_id: str, files: List[UploadFile] = File(...)):
    """Upload one or more images to a project.

    HOW FILE UPLOAD WORKS:
      Frontend:  const fd = new FormData(); fd.append('files', file); fetch(url, {method:'POST', body: fd})
      Backend:   files: List[UploadFile] -> FastAPI parses multipart form data automatically

    For each image we: validate format, sanitize filename, save to disk, read dimensions via PIL.
    """
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    images_dir = project_dir / "images"
    images_dir.mkdir(exist_ok=True)
    results = []

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            results.append({"filename": file.filename, "status": "skipped", "reason": f"Unsupported: {ext}"})
            continue

        # Sanitize filename (remove special chars, handle duplicates)
        safe_name = re.sub(r'[^\w\-.]', '_', file.filename)
        dest = images_dir / safe_name
        if dest.exists():
            stem = Path(safe_name).stem
            counter = 1
            while dest.exists():
                dest = images_dir / f"{stem}_{counter}{ext}"
                counter += 1
            safe_name = dest.name

        # Save file
        content = await file.read()
        with open(dest, "wb") as f:
            f.write(content)

        # Validate and get dimensions
        try:
            with Image.open(dest) as img:
                width, height = img.size
        except Exception:
            dest.unlink()
            results.append({"filename": file.filename, "status": "error", "reason": "Invalid image"})
            continue

        results.append({
            "filename": safe_name, "status": "ok",
            "width": width, "height": height, "size": len(content),
        })

    # Update modified time
    meta = read_json(project_dir / "project.json", {})
    meta["modified"] = time.time()
    write_json(project_dir / "project.json", meta)

    return {"uploaded": results, "total": len(results)}


@app.get("/api/projects/{project_id}/images")
async def list_images(
    project_id: str,
    page: int = Query(1, ge=1, description="Page number (starts at 1)"),
    limit: int = Query(50, ge=1, le=200, description="Images per page"),
    search: Optional[str] = Query(None, description="Search by filename"),
    status: Optional[str] = Query(None, description="Filter: annotated, unannotated, or all"),
):
    """List images with pagination, search, and annotation status filtering.

    PAGINATION: With 2000+ images, we send pages (e.g., 50 at a time).
      GET /api/projects/abc/images?page=1&limit=50  -> images 1-50
      GET /api/projects/abc/images?page=2&limit=50  -> images 51-100

    QUERY PARAMS: FastAPI extracts ?page=2&search=dog from the URL automatically.
    """
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    images_dir = project_dir / "images"
    if not images_dir.exists():
        return {"images": [], "total": 0, "page": page, "pages": 0}

    # Get all image files sorted
    all_files = sorted([f.name for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS])

    # Search filter
    if search:
        sl = search.lower()
        all_files = [f for f in all_files if sl in f.lower()]

    # Annotation status filter
    if status and status != "all":
        annotations = read_json(project_dir / "annotations.json", {})
        if status == "annotated":
            all_files = [f for f in all_files if len(annotations.get(f, [])) > 0]
        elif status == "unannotated":
            all_files = [f for f in all_files if len(annotations.get(f, [])) == 0]

    # Paginate
    total = len(all_files)
    pages = max(1, (total + limit - 1) // limit)
    start = (page - 1) * limit
    page_files = all_files[start:start + limit]

    # Build response — skip PIL dimension loading for speed (frontend reads dims on render)
    annotations = read_json(project_dir / "annotations.json", {})
    images = []
    for fname in page_files:
        fpath = images_dir / fname
        images.append({
            "filename": fname, "width": 0, "height": 0,
            "size": fpath.stat().st_size if fpath.exists() else 0,
            "annotation_count": len(annotations.get(fname, [])),
        })

    return {"images": images, "total": total, "page": page, "pages": pages, "limit": limit}


@app.get("/api/projects/{project_id}/images/{filename:path}")
async def serve_image(project_id: str, filename: str):
    """Serve an image file via HTTP (replaces base64 from old app).

    Used as: <img src="/api/projects/abc/images/photo.jpg">
    FileResponse streams the file from disk with proper Content-Type.
    Browser caches it automatically. Much more efficient than base64.
    """
    project_dir = get_project_dir(project_id)
    filepath = project_dir / "images" / filename

    # Security: prevent path traversal (e.g., filename="../../server.py")
    try:
        filepath.resolve().relative_to((project_dir / "images").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(filepath)



@app.delete("/api/projects/{project_id}/images/{filename:path}")
async def delete_image(project_id: str, filename: str):
    """Delete an image, its thumbnail, and its annotations."""
    project_dir = get_project_dir(project_id)
    filepath = project_dir / "images" / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    filepath.unlink()

    # Remove annotations
    ann_path = project_dir / "annotations.json"
    annotations = read_json(ann_path, {})
    if filename in annotations:
        del annotations[filename]
        write_json(ann_path, annotations)

    return {"status": "deleted", "filename": filename}


# ==============================================================================
# STEP 1d: ANNOTATIONS API
# ==============================================================================
#
# STORAGE FORMAT (annotations.json):
#   {
#     "photo1.jpg": [
#       {"id":"a1", "type":"bbox",    "classId":"c1", "x":0.1, "y":0.2, "w":0.3, "h":0.4},
#       {"id":"a2", "type":"polygon", "classId":"c2", "points":[{"x":0.1,"y":0.2}, ...]},
#       {"id":"a3", "type":"keypoint","classId":"c3", "keypoints":[{"x":0.5,"y":0.3,"v":2},...]}
#     ],
#     "photo2.jpg": [...]
#   }
#
# COORDINATES: All x/y/w/h are NORMALIZED to [0,1] relative to image dimensions.
# x=0.5 means "50% from left edge". Resolution-independent.
# Frontend converts pixel coords <-> normalized using image width/height.


@app.get("/api/projects/{project_id}/annotations")
async def get_all_annotations(project_id: str):
    """Get ALL annotations for a project. Used on project open."""
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    return {"annotations": read_json(project_dir / "annotations.json", {})}


@app.get("/api/projects/{project_id}/annotations/{filename:path}")
async def get_image_annotations(project_id: str, filename: str):
    """Get annotations for a single image."""
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")
    annotations = read_json(project_dir / "annotations.json", {})
    return {"filename": filename, "annotations": annotations.get(filename, [])}


@app.put("/api/projects/{project_id}/annotations/{filename:path}")
async def save_image_annotations(project_id: str, filename: str, request: Request):
    """Save annotations for a single image (auto-save endpoint).

    Called by frontend whenever user draws/edits/deletes an annotation.
    Updates only this image's entry, leaving all others untouched.
    """
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    body = await request.json()
    image_anns = body.get("annotations", [])

    ann_path = project_dir / "annotations.json"
    all_anns = read_json(ann_path, {})

    if image_anns:
        all_anns[filename] = image_anns
    elif filename in all_anns:
        del all_anns[filename]  # Clean up empty entries

    write_json(ann_path, all_anns)

    meta = read_json(project_dir / "project.json", {})
    meta["modified"] = time.time()
    write_json(project_dir / "project.json", meta)

    return {"status": "saved", "filename": filename, "count": len(image_anns)}


@app.put("/api/projects/{project_id}/annotations")
async def save_all_annotations(project_id: str, request: Request):
    """Bulk save annotations (for propagation, batch AI, import)."""
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail="Project not found")

    body = await request.json()
    annotations = body.get("annotations", {})
    write_json(project_dir / "annotations.json", annotations)

    meta = read_json(project_dir / "project.json", {})
    meta["modified"] = time.time()
    write_json(project_dir / "project.json", meta)

    total = sum(len(anns) for anns in annotations.values())
    return {"status": "saved", "images": len(annotations), "total_annotations": total}


# ==============================================================================
# STEP 1e: MODELS API
# ==============================================================================

@app.get("/api/models")
async def list_models():
    """List model files in the models/ directory with type hints and loaded model info."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for f in sorted(MODELS_DIR.iterdir()):
        if f.suffix.lower() in ('.pt', '.onnx', '.engine'):
            name_lower = f.name.lower()
            if "sam" in name_lower:
                model_type = "segment"
            elif "world" in name_lower:
                model_type = "world"
            else:
                model_type = "detect"
            models.append({
                "filename": f.name,
                "size": f.stat().st_size,
                "format": f.suffix.lstrip('.'),
                "type": model_type,
            })
    return {"models": models, "loaded": loaded_model_name, "loaded_type": loaded_model_type, "loaded_classes": loaded_model_classes}


@app.post("/api/models")
async def upload_model(file: UploadFile = File(...)):
    """Upload a YOLO model (.onnx/.pt/.engine) to the models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower()
    if ext not in ('.onnx', '.pt', '.engine'):
        raise HTTPException(status_code=400, detail="Only .onnx, .pt, .engine files allowed")
    safe_name = re.sub(r'[^\w\-.]', '_', file.filename)
    dest = MODELS_DIR / safe_name
    content = await file.read()
    with open(dest, 'wb') as f:
        f.write(content)
    return {"filename": safe_name, "size": len(content), "format": ext.lstrip('.')}


@app.get("/api/models/{filename:path}")
async def serve_model(filename: str):
    """Serve an ONNX model file for in-browser inference."""
    # Security: only alphanumeric, dots, dashes, underscores
    if not re.match(r'^[\w\-. ]+$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = MODELS_DIR / filename
    try:
        filepath.resolve().relative_to(MODELS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(filepath, media_type="application/octet-stream")


# ==============================================================================
# STEP 1f: GPU INFERENCE API
# ==============================================================================

@app.post("/api/models/load")
async def load_model_endpoint(request: Request):
    """Load a model into GPU memory (unloads previous model first)."""
    global loaded_model, loaded_model_name, loaded_model_type, loaded_model_classes
    body = await request.json()
    filename = body.get("filename", "")

    model_path = MODELS_DIR / filename
    if not model_path.is_file():
        raise HTTPException(404, "Model not found")

    # Unload previous model to free VRAM
    if loaded_model is not None:
        del loaded_model
        loaded_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    name_lower = filename.lower()
    try:
        if "sam" in name_lower:
            from ultralytics import SAM
            loaded_model = SAM(str(model_path))
            loaded_model_type = "segment"
        elif "world" in name_lower:
            from ultralytics import YOLOWorld
            loaded_model = YOLOWorld(str(model_path))
            loaded_model_type = "world"
        else:
            from ultralytics import YOLO
            loaded_model = YOLO(str(model_path))
            loaded_model_type = "detect"

        loaded_model_name = filename

        class_names = None
        try:
            if hasattr(loaded_model, 'names') and loaded_model.names:
                class_names = list(loaded_model.names.values()) if isinstance(loaded_model.names, dict) else list(loaded_model.names)
        except Exception:
            pass

        loaded_model_classes = class_names

        return {"status": "loaded", "filename": filename, "type": loaded_model_type, "classes": class_names}
    except Exception as e:
        loaded_model = None
        loaded_model_name = None
        loaded_model_type = None
        loaded_model_classes = None
        raise HTTPException(500, f"Failed to load model: {str(e)}")


@app.post("/api/models/unload")
async def unload_model_endpoint():
    """Unload the current model and free VRAM."""
    global loaded_model, loaded_model_name, loaded_model_type, loaded_model_classes
    if loaded_model is not None:
        del loaded_model
        loaded_model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    loaded_model_name = None
    loaded_model_type = None
    loaded_model_classes = None
    return {"status": "unloaded"}


def _infer_safe(model, img_path, **kwargs):
    """Run ultralytics model inference with automatic CPU fallback.

    ONNX models compiled for a different GPU architecture raise:
      cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device
    We catch that and retry on CPU so inference still works.
    """
    try:
        return model(str(img_path), **kwargs)
    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ('cuda', 'kernel image', 'no kernel', 'cudaerror', 'onnxruntime')):
            print(f"[Infer] CUDA/ONNX error — retrying on CPU. Error was: {e}")
            kwargs['device'] = 'cpu'
            return model(str(img_path), **kwargs)
        raise


@app.post("/api/infer")
async def run_inference(request: Request):
    """Single image inference. Handles detect, world, and segment (SAM) models."""
    global loaded_model, loaded_model_name, loaded_model_type
    if loaded_model is None:
        raise HTTPException(400, "No model loaded. Call POST /api/models/load first.")

    body = await request.json()
    project_id = body.get("project_id", "")
    filename = body.get("filename", "")

    if not project_id or not filename:
        raise HTTPException(400, "project_id and filename are required")

    project_dir = get_project_dir(project_id)
    image_path = project_dir / "images" / filename

    # Prevent path traversal
    try:
        image_path.resolve().relative_to((project_dir / "images").resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")

    if not image_path.is_file():
        raise HTTPException(404, "Image not found")

    try:
        if loaded_model_type in ("detect", "world"):
            conf = float(body.get("conf", 0.25))
            iou = float(body.get("iou", 0.45))
            classes = body.get("classes", None)  # list of names for YOLO-World

            if loaded_model_type == "world" and classes:
                loaded_model.set_classes(classes)

            results = _infer_safe(loaded_model, image_path, conf=conf, iou=iou)
            r = results[0]
            names = r.names  # dict {int_id: name}

            detections = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
                    cid = int(cls_ids[i])
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "score": float(confs[i]),
                        "class_id": cid,
                        "class_name": names.get(cid, f"class_{cid}"),
                    })

            return {"detections": detections, "model_type": loaded_model_type}

        elif loaded_model_type == "segment":
            points = body.get("points", None)   # [[x, y], ...]
            labels = body.get("labels", None)   # [1, 0, 1, ...]
            box = body.get("box", None)         # [x1, y1, x2, y2]

            kwargs = {}
            if points:
                kwargs["points"] = points
                kwargs["labels"] = labels
            if box:
                kwargs["bboxes"] = [box]

            results = _infer_safe(loaded_model, image_path, **kwargs)
            r = results[0]

            masks_out = []
            if r.masks is not None:
                for mask_xy in r.masks.xy:
                    pts = [[float(p[0]), float(p[1])] for p in mask_xy]
                    masks_out.append({"pts": pts, "score": 1.0})

            return {"masks": masks_out, "model_type": loaded_model_type}

        else:
            raise HTTPException(400, f"Unknown model type: {loaded_model_type}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {str(e)}")


@app.post("/api/infer/batch")
async def run_batch_inference(request: Request):
    """Batch detection for multiple images (detect/world models only)."""
    global loaded_model, loaded_model_name, loaded_model_type
    if loaded_model is None:
        raise HTTPException(400, "No model loaded.")
    if loaded_model_type == "segment":
        raise HTTPException(400, "Batch inference not supported for SAM. Use single /api/infer endpoint.")

    body = await request.json()
    project_id = body.get("project_id", "")
    filenames = body.get("filenames", [])
    conf = float(body.get("conf", 0.25))
    iou_thresh = float(body.get("iou", 0.45))
    classes = body.get("classes", None)

    if not project_id:
        raise HTTPException(400, "project_id required")

    project_dir = get_project_dir(project_id)

    if loaded_model_type == "world" and classes:
        loaded_model.set_classes(classes)

    results_out = []
    for fname in filenames:
        image_path = project_dir / "images" / fname
        # Security: prevent path traversal
        try:
            image_path.resolve().relative_to((project_dir / "images").resolve())
        except ValueError:
            continue
        if not image_path.is_file():
            results_out.append({"filename": fname, "detections": [], "error": "not found"})
            continue
        try:
            results = _infer_safe(loaded_model, image_path, conf=conf, iou=iou_thresh)
            r = results[0]
            names = r.names
            detections = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])
                    cid = int(cls_ids[i])
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "score": float(confs[i]),
                        "class_id": cid,
                        "class_name": names.get(cid, f"class_{cid}"),
                    })
            results_out.append({"filename": fname, "detections": detections})
        except Exception as e:
            results_out.append({"filename": fname, "detections": [], "error": str(e)})

    return {"results": results_out}


# ==============================================================================
# AGENT PIPELINE ROUTES
# ==============================================================================

@app.post("/api/agent/run")
async def run_agent_pipeline(request: Request):
    """
    Agentic auto-annotation pipeline.

    Steps:
    1. Load YOLO-World (or other detection) model
    2. Parse user description into class names
    3. Run detection on all/selected images at low conf to get all candidates
    4. Categorize by confidence: high → auto-accept, medium → flag, low → discard
    5. Optionally refine flagged detections with SAM
    6. Save accepted + flagged annotations (flagged marked with agent_status)
    7. Return summary with flagged image list
    """
    body = await request.json()
    project_id = body.get("project_id")
    description = body.get("description", "")
    accept_thresh = body.get("accept_threshold", 0.70)
    flag_thresh = body.get("flag_threshold", 0.30)
    refine_with_sam = body.get("refine_with_sam", False)
    scope = body.get("scope", "all")
    detection_model = body.get("detection_model", "")
    sam_model_name = body.get("sam_model", "")
    det_conf = body.get("detection_conf", 0.10)
    det_iou = body.get("detection_iou", 0.45)

    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(404, "Project not found")

    import time as _time
    start_time = _time.time()

    class_names = [c.strip() for c in description.split(",") if c.strip()]
    if not class_names:
        raise HTTPException(400, "Please provide object descriptions (comma-separated)")

    images_dir = project_dir / "images"
    all_images = sorted([f.name for f in images_dir.iterdir()
                         if f.suffix.lower() in IMAGE_EXTENSIONS])

    existing_annotations = read_json(project_dir / "annotations.json", {})

    if scope == "unannotated":
        target_images = [f for f in all_images if len(existing_annotations.get(f, [])) == 0]
    elif scope == "selected":
        selected = set(body.get("selected_images", []))
        target_images = [f for f in all_images if f in selected]
    else:
        target_images = all_images

    if not target_images:
        return {"status": "completed", "summary": {"total_images": 0, "processed_images": 0}}

    global loaded_model, loaded_model_name, loaded_model_type

    if loaded_model_name != detection_model:
        if loaded_model is not None:
            del loaded_model
            loaded_model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        det_path = MODELS_DIR / detection_model
        if not det_path.is_file():
            raise HTTPException(404, f"Detection model not found: {detection_model}")

        name_lower = detection_model.lower()
        try:
            if "world" in name_lower:
                from ultralytics import YOLOWorld
                loaded_model = YOLOWorld(str(det_path))
                loaded_model_type = "world"
            else:
                from ultralytics import YOLO
                loaded_model = YOLO(str(det_path))
                loaded_model_type = "detect"
            loaded_model_name = detection_model
        except Exception as e:
            raise HTTPException(500, f"Failed to load detection model: {e}")

    if loaded_model_type == "world":
        loaded_model.set_classes(class_names)

    auto_accepted = {}
    flagged = {}
    no_detections = []
    flagged_list = []
    class_counts = {}

    for img_name in target_images:
        img_path = images_dir / img_name
        try:
            results = _infer_safe(loaded_model, img_path, conf=det_conf, iou=det_iou, verbose=False)

            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                accepted_anns = []
                flagged_anns = []

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].tolist()
                    conf_val = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    cls_name = results[0].names.get(cls_id, f"class_{cls_id}")

                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                    ann = {"box": xyxy, "score": conf_val, "class_name": cls_name, "class_id": cls_id}

                    if conf_val >= accept_thresh:
                        accepted_anns.append(ann)
                    elif conf_val >= flag_thresh:
                        flagged_anns.append(ann)

                if accepted_anns:
                    auto_accepted[img_name] = accepted_anns
                if flagged_anns:
                    flagged[img_name] = flagged_anns
                    flagged_list.append(img_name)
                if not accepted_anns and not flagged_anns:
                    no_detections.append(img_name)
            else:
                no_detections.append(img_name)

        except Exception as e:
            print(f"[Agent] Error processing {img_name}: {e}")
            continue

    # Optionally refine flagged detections with SAM
    if refine_with_sam and flagged and sam_model_name:
        if loaded_model_name != sam_model_name:
            if loaded_model is not None:
                del loaded_model
                loaded_model = None
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            sam_path = MODELS_DIR / sam_model_name
            if sam_path.is_file():
                try:
                    from ultralytics import SAM
                    loaded_model = SAM(str(sam_path))
                    loaded_model_name = sam_model_name
                    loaded_model_type = "segment"
                except Exception as e:
                    print(f"[Agent] Failed to load SAM: {e}")

        if loaded_model_type == "segment":
            for img_name, dets in flagged.items():
                img_path = images_dir / img_name
                try:
                    for det in dets:
                        box = det["box"]
                        results = _infer_safe(loaded_model, img_path, bboxes=[box], verbose=False)
                        if results and results[0].masks is not None:
                            for mask_xy in results[0].masks.xy:
                                polygon_pts = mask_xy.tolist()
                                if len(polygon_pts) >= 3:
                                    det["polygon"] = polygon_pts
                                    det["refined"] = True
                except Exception as e:
                    print(f"[Agent] SAM refinement error on {img_name}: {e}")

    # Build final annotations
    project_meta = read_json(project_dir / "project.json", {})
    project_classes = project_meta.get("classes", [])
    existing_class_names = {c["name"].lower(): c for c in project_classes}

    for cls_name in class_names:
        if cls_name.lower() not in existing_class_names:
            new_cls = {
                "id": uuid.uuid4().hex[:6],
                "name": cls_name,
                "color": DEFAULT_COLORS[len(project_classes) % len(DEFAULT_COLORS)]
            }
            project_classes.append(new_cls)
            existing_class_names[cls_name.lower()] = new_cls

    project_meta["classes"] = project_classes
    project_meta["modified"] = _time.time()
    write_json(project_dir / "project.json", project_meta)

    def make_annotation(det, class_lookup):
        cls_obj = class_lookup.get(det["class_name"].lower())
        if not cls_obj:
            return None
        cid = cls_obj["id"]
        if det.get("polygon"):
            return {"id": uuid.uuid4().hex[:8], "tp": "pg", "pts": det["polygon"], "cid": cid}
        else:
            x1, y1, x2, y2 = det["box"]
            return {"id": uuid.uuid4().hex[:8], "tp": "bb", "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "cid": cid}

    total_auto = 0
    total_flagged_anns = 0
    all_new_annotations = {}

    for img_name, dets in auto_accepted.items():
        anns = []
        for det in dets:
            ann = make_annotation(det, existing_class_names)
            if ann:
                ann["agent_status"] = "accepted"
                ann["agent_score"] = det["score"]
                anns.append(ann)
        if anns:
            existing = existing_annotations.get(img_name, [])
            all_new_annotations[img_name] = existing + anns
            total_auto += len(anns)

    for img_name, dets in flagged.items():
        anns = []
        for det in dets:
            ann = make_annotation(det, existing_class_names)
            if ann:
                ann["agent_status"] = "flagged"
                ann["agent_score"] = det["score"]
                anns.append(ann)
        if anns:
            existing = all_new_annotations.get(img_name, existing_annotations.get(img_name, []))
            all_new_annotations[img_name] = existing + anns
            total_flagged_anns += len(anns)

    for img_name, anns in existing_annotations.items():
        if img_name not in all_new_annotations:
            all_new_annotations[img_name] = anns

    write_json(project_dir / "annotations.json", all_new_annotations)

    elapsed = _time.time() - start_time

    return {
        "status": "completed",
        "summary": {
            "total_images": len(all_images),
            "processed_images": len(target_images),
            "auto_accepted_images": len(auto_accepted),
            "flagged_images": len(flagged),
            "no_detection_images": len(no_detections),
            "total_annotations": total_auto + total_flagged_anns,
            "auto_accepted_annotations": total_auto,
            "flagged_annotations": total_flagged_anns,
            "classes_found": class_names,
            "processing_time_seconds": round(elapsed, 1)
        },
        "flagged": flagged_list,
        "no_detections": no_detections,
        "class_distribution": class_counts
    }


@app.get("/api/agent/flagged/{project_id}")
async def get_flagged_images(project_id: str):
    """Get list of images with flagged (uncertain) annotations from the agent pipeline."""
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(404, "Project not found")

    annotations = read_json(project_dir / "annotations.json", {})

    flagged = []
    accepted = []
    unprocessed = []

    images_dir = project_dir / "images"
    all_images = sorted([f.name for f in images_dir.iterdir()
                         if f.suffix.lower() in IMAGE_EXTENSIONS]) if images_dir.exists() else []

    for img_name in all_images:
        anns = annotations.get(img_name, [])
        has_flagged = any(a.get("agent_status") == "flagged" for a in anns)
        has_accepted = any(a.get("agent_status") == "accepted" for a in anns)
        has_manual = any("agent_status" not in a for a in anns)

        if has_flagged:
            flagged_count = sum(1 for a in anns if a.get("agent_status") == "flagged")
            flagged.append({"filename": img_name, "flagged_count": flagged_count, "total_count": len(anns)})
        elif has_accepted or has_manual:
            accepted.append(img_name)
        else:
            unprocessed.append(img_name)

    return {
        "flagged": flagged,
        "accepted_count": len(accepted),
        "unprocessed_count": len(unprocessed),
        "unprocessed": unprocessed
    }


@app.post("/api/agent/accept/{project_id}")
async def accept_flagged(project_id: str, request: Request):
    """Accept or reject flagged annotations for specific images.

    Request: {"filename": "img.jpg", "action": "accept_all" | "reject_all" | "selective",
              "keep_ids": ["ann_id1", "ann_id2"]}
    """
    body = await request.json()
    project_dir = get_project_dir(project_id)
    filename = body.get("filename")
    action = body.get("action", "accept_all")

    ann_path = project_dir / "annotations.json"
    all_anns = read_json(ann_path, {})

    if filename not in all_anns:
        raise HTTPException(404, "Image not found in annotations")

    anns = all_anns[filename]

    if action == "accept_all":
        for a in anns:
            a.pop("agent_status", None)
            a.pop("agent_score", None)
    elif action == "reject_all":
        anns = [a for a in anns if a.get("agent_status") != "flagged"]
        for a in anns:
            a.pop("agent_status", None)
            a.pop("agent_score", None)
    elif action == "selective":
        keep_ids = set(body.get("keep_ids", []))
        anns = [a for a in anns if a["id"] in keep_ids or a.get("agent_status") != "flagged"]
        for a in anns:
            a.pop("agent_status", None)
            a.pop("agent_score", None)

    all_anns[filename] = anns
    write_json(ann_path, all_anns)

    return {"status": "ok", "filename": filename, "remaining_annotations": len(anns)}


# ==============================================================================
# FRONTEND STATIC FILE SERVING (must be LAST - catch-all route)
# ==============================================================================
#
# WHY LAST? FastAPI checks routes in order. API routes (/api/*) are checked
# first. If none match, this catch-all serves the frontend HTML.
# This is the "SPA fallback" pattern - any unknown URL returns index.html,
# and React's client-side router handles the rest.

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve frontend files, with SPA fallback to index.html."""
    # Try exact file first
    if full_path:
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

    # Fallback to index.html
    index_path = FRONTEND_DIR / "index.html"
    if index_path.is_file():
        return HTMLResponse(index_path.read_text())

    return HTMLResponse("""
        <html><head><title>AntiGravity Annotator</title></head>
        <body style="background:#0e0d09;color:#c9953a;font-family:system-ui;
                     display:flex;align-items:center;justify-content:center;
                     height:100vh;margin:0;flex-direction:column;gap:16px;">
            <h1>AntiGravity Annotator</h1>
            <p style="color:#8a8279;">Server running. Frontend not found at ./frontend/index.html</p>
            <p style="color:#6b6358;font-size:0.9rem;">
                API docs: <a href="/docs" style="color:#c9953a;">/docs</a></p>
        </body></html>
    """, status_code=200)
