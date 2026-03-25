#!/usr/bin/env python3
"""
AntiGravity Annotator — Launcher
=================================
Usage:  python run.py
        python run.py --port 8080
        python run.py --no-browser

This script:
1. Creates required directories if they don't exist
2. Detects GPU availability
3. Prints server info (URL, network IP, GPU status)
4. Opens the browser automatically
5. Starts the uvicorn ASGI server

WHY A SEPARATE LAUNCHER?
-------------------------
You *could* put this logic inside server.py, but separating it keeps
server.py focused purely on API logic (routes, handlers). run.py handles
the "ops" side: environment checks, directory setup, browser opening.
This is a common pattern in production Python apps.
"""

import os
import sys
import socket
import argparse
import webbrowser
import subprocess
from pathlib import Path


# ── Directory Setup ──────────────────────────────────────────────────────────
# We define all paths relative to THIS file's location.
# Path(__file__).parent gives us the folder containing run.py.
# .resolve() turns it into an absolute path (no ../.. ambiguity).

BASE_DIR = Path(__file__).parent.resolve()
REQUIRED_DIRS = [
    BASE_DIR / "projects",    # Where project data lives
    BASE_DIR / "models",      # Where YOLO .pt/.onnx models go
    BASE_DIR / "frontend",    # Where the React UI lives
]


def ensure_directories():
    """Create required directories if they don't exist.
    
    WHY: First-time users shouldn't need to manually create folders.
    mkdir with parents=True, exist_ok=True is safe to call repeatedly —
    it won't error if the directory already exists.
    """
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        

def get_local_ip():
    """Get this machine's LAN IP address.
    
    HOW IT WORKS:
    We create a UDP socket and "connect" to a public IP (8.8.8.8).
    We never actually send data — we just use this trick to find out
    which network interface the OS would use, then read its IP.
    
    WHY: So we can print the URL that other people on your LAN can use.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def check_gpu():
    """Check if NVIDIA GPU + CUDA is available.
    
    Strategy: Try torch first (since ultralytics uses it).
    If torch isn't installed, fall back to nvidia-smi CLI.
    
    Returns dict: {available: bool, name: str, cuda_version: str}
    """
    gpu_info = {"available": False, "name": "None", "cuda_version": "N/A"}
    
    # Method 1: Try PyTorch (most reliable if installed)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["name"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_version"] = torch.version.cuda or "Unknown"
            return gpu_info
    except ImportError:
        pass
    
    # Method 2: Try nvidia-smi CLI (works even without torch)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info["available"] = True
            gpu_info["name"] = result.stdout.strip().split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    return gpu_info


def print_banner(host, port, gpu_info):
    """Print startup banner with connection info and GPU status."""
    local_ip = get_local_ip()
    gpu_icon = "✓" if gpu_info["available"] else "✗"
    gpu_text = f"{gpu_info['name']} (CUDA {gpu_info['cuda_version']})" if gpu_info["available"] else "No GPU detected — CPU mode"
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              ⬡ AntiGravity Annotator v1.0                     ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Local:     http://localhost:{port:<36}║
║  Network:   http://{local_ip}:{port:<36}║
║  GPU:       {gpu_icon} {gpu_text:<52}║
║                                                               ║
║  Projects:  ./projects/                                       ║
║  Models:    ./models/                                         ║
║                                                               ║
║  Press Ctrl+C to stop                                         ║
╚═══════════════════════════════════════════════════════════════╝
""")


def auto_select_gpu():
    """Auto-select the GPU with the most free memory.
    
    HOW IT WORKS:
    We call nvidia-smi to query free memory on each GPU, then pick
    the one with the most available. This avoids stepping on your
    teammate's GPU if they're using it heavily.
    
    WHY SET CUDA_VISIBLE_DEVICES?
    PyTorch uses this environment variable to decide which GPUs it can see.
    If we set CUDA_VISIBLE_DEVICES=1, then torch.cuda.device(0) actually
    points to physical GPU 1. This is the standard way to pin a process
    to a specific GPU without code changes.
    
    Returns: GPU index (int) or None if no GPUs found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        
        best_idx, best_free = None, 0
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split(",")
            if len(parts) == 2:
                idx = int(parts[0].strip())
                free_mb = int(parts[1].strip())
                if free_mb > best_free:
                    best_idx, best_free = idx, free_mb
        
        return best_idx
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def main():
    # ── Argument Parsing ─────────────────────────────────────────────────────
    # argparse lets users customize behavior from the command line.
    # Example: python run.py --port 9000 --no-browser
    
    parser = argparse.ArgumentParser(description="AntiGravity Annotator Server")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (0.0.0.0 = all interfaces, 127.0.0.1 = localhost only)")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port number (default: 8001, since 8000 is often taken)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device index to use (e.g., --gpu 1 for Tesla T4). "
                             "If not set, auto-selects GPU with most free memory.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser on start")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload on code changes (dev mode)")
    args = parser.parse_args()
    
    # ── GPU Selection ────────────────────────────────────────────────────────
    # CUDA_VISIBLE_DEVICES controls which GPUs PyTorch can see.
    # Setting it BEFORE importing torch is critical — torch reads it on import.
    #
    # WHY THIS MATTERS ON YOUR SERVER:
    # GPU 0 = Quadro K620 (2GB, weak) — used for display
    # GPU 1 = Tesla T4 (15GB, strong) — what we want for inference
    #
    # With --gpu 1:  CUDA_VISIBLE_DEVICES=1 → torch sees T4 as "cuda:0"
    # Without flag:  we auto-pick the GPU with the most free memory
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"  GPU: Using device {args.gpu} (set via --gpu flag)")
    else:
        # Auto-select: find GPU with most free memory
        best_gpu = auto_select_gpu()
        if best_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            print(f"  GPU: Auto-selected device {best_gpu} (most free memory)")
    
    # ── Pre-flight Checks ────────────────────────────────────────────────────
    ensure_directories()
    gpu_info = check_gpu()
    print_banner(args.host, args.port, gpu_info)
    
    # ── Auto-open Browser ────────────────────────────────────────────────────
    # We open BEFORE uvicorn.run() because that call blocks forever.
    # The browser shows "loading" briefly while the server finishes starting.
    if not args.no_browser:
        url = f"http://localhost:{args.port}"
        print(f"  → Opening browser at {url}\n")
        webbrowser.open(url)
    
    # ── Start Server ─────────────────────────────────────────────────────────
    # "server:app" tells uvicorn: import `app` from `server.py` in this dir.
    # host="0.0.0.0" means accept connections from any IP (LAN access).
    import uvicorn
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
