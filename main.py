"""
Simple RDD Predict - Road Damage Detection with WebSocket Streaming
Supports: Individual stream, Broadcast with Channel IDs (multiple rooms)

Environment Variables:
    MODELS: Model ID(s) to load, comma-separated (e.g., "1" or "1,2,3")
        1 = YOLOv8_Small_RDD (PyTorch) 
        2 = YOLO_TFRT_16 (TensorRT FP16)
        3 = YOLO_TFRT_32 (TensorRT FP32)
        4 = YOLO_TFLITE_16 (TFLite FP16)
        5 = YOLO_TFLITE_32 (TFLite FP32)
"""
import os
import logging
import time
import asyncio
from typing import Dict, Set
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()  # Load .env file

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import os, requests


# ============================================================================
# GPU and TensorFlow Detection
# ============================================================================
def detect_gpu():
    """Detect NVIDIA GPU availability and return device info."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        return True, {
            "available": True,
            "count": gpu_count,
            "name": gpu_name,
            "cuda_version": cuda_version,
            "device": "cuda"
        }
    return False, {"available": False, "device": "cpu"}

def detect_tensorflow():
    """Check if TensorFlow is available for TFLite models."""
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False

# Global GPU and TensorFlow availability flags
HAS_GPU, GPU_INFO = detect_gpu()
HAS_TENSORFLOW = detect_tensorflow()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "tfrt-32": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float32.engine",
        "local_path": "models/YOLOv8_Small_RDD_float32.engine",
        "description": "TensorRT Float32",
        "requires_gpu": True,
        "requires_tensorflow": False
    },
    "tfrt-16": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float16.engine",
        "local_path": "models/YOLOv8_Small_RDD_float16.engine",
        "description": "TensorRT Float16",
        "requires_gpu": True,
        "requires_tensorflow": False
    },
    "tflite-32": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float32.tflite",
        "local_path": "models/YOLOv8_Small_RDD_float32.tflite",
        "description": "TFLite Float32",
        "requires_gpu": False,
        "requires_tensorflow": True
    },
    "tflite-16": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD_float16.tflite",
        "local_path": "models/YOLOv8_Small_RDD_float16.tflite",
        "description": "TFLite Float16",
        "requires_gpu": False,
        "requires_tensorflow": True
    },
    "pytorch": {
        "url": "https://pub-0ccce103f38e4902912534cdb3973783.r2.dev/YOLOv8_Small_RDD.pt",  # Local file
        "local_path": "models/YOLOv8_Small_RDD.pt",
        "description": "PyTorch Original",
        "requires_gpu": False,  # Can run on CPU or GPU
        "requires_tensorflow": False
    }
}

# Map .env model IDs to model keys
# From .env: 1=pytorch, 2=tfrt-16, 3=tfrt-32, 4=tflite-16, 5=tflite-32
MODEL_ID_MAP = {
    "1": "pytorch",
    "2": "tfrt-16", 
    "3": "tfrt-32",
    "4": "tflite-16",
    "5": "tflite-32"
}

def get_enabled_models():
    """Parse MODELS env variable and return set of enabled model keys."""
    models_env = os.getenv("MODELS", "1")  # Default to pytorch
    model_ids = [m.strip() for m in models_env.split(",")]
    enabled = set()
    for model_id in model_ids:
        if model_id in MODEL_ID_MAP:
            enabled.add(MODEL_ID_MAP[model_id])
        else:
            logger.warning(f"[Config] Unknown model ID: {model_id}")
    if not enabled:
        logger.warning("[Config] No valid models specified, defaulting to pytorch")
        enabled.add("pytorch")
    return enabled

ENABLED_MODELS = get_enabled_models()
logger.info(f"[Config] MODELS env: {os.getenv('MODELS', '1')} -> Enabled: {ENABLED_MODELS}")

os.makedirs("models", exist_ok=True)

def download_file(url: str, filename: str) -> bool:
    """Download file with progress bar. Returns True if successful."""
    logger.info(f"[STEP] download_file() called for: {filename}")
    logger.debug(f"[STEP] Checking if file exists: {filename}")
    
    if os.path.exists(filename):
        logger.info(f"[Download] {filename} sudah ada, skip download.")
        return True
    
    try:
        logger.debug(f"[STEP] Starting HTTP request to: {url}")
        with requests.get(url, stream=True, timeout=30) as r:
            logger.debug(f"[STEP] HTTP response status: {r.status_code}")
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded_size = 0

            logger.info(f"[Download] Mengunduh {filename} dari {url}...")
            logger.debug(f"[STEP] Total file size: {total_size / (1024*1024):.2f} MB")
            
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                        # Log every 10% progress
                        if int(progress) % 10 == 0:
                            logger.debug(f"[Download] Progress: {progress:.2f}%")
                        print(f"\r[Download] Progres: {progress:.2f}% ({downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)", end="")
            
            print("\n[Download] Download selesai!")
            logger.info(f"[STEP] Download complete: {filename}")
            return True
    except requests.exceptions.Timeout as e:
        logger.error(f"[Download] Timeout error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False
    except requests.exceptions.HTTPError as e:
        logger.error(f"[Download] HTTP error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return False
    except Exception as e:
        logger.exception(f"[Download] Unexpected error downloading {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        return 

# Dictionary to store loaded models
models: dict = {}

def load_all_models():
    """Download and load models specified in .env MODELS variable."""
    logger.info("[STEP] " + "="*50)
    logger.info(f"[STEP] LOADING MODELS: {ENABLED_MODELS}")
    logger.info("[STEP] " + "="*50)
    
    for model_key, config in MODEL_CONFIGS.items():
        # Skip models not enabled in .env
        if model_key not in ENABLED_MODELS:
            logger.debug(f"[Model] SKIP: {model_key} not in ENABLED_MODELS")
            continue
            
        url = config["url"]
        local_path = config["local_path"]
        description = config["description"]
        requires_gpu = config.get("requires_gpu", False)
        requires_tensorflow = config.get("requires_tensorflow", False)
        
        logger.info(f"[STEP] Processing model: {model_key} ({description})")
        logger.debug(f"[STEP] Model config: url={url}, path={local_path}, gpu={requires_gpu}, tf={requires_tensorflow}")
        
        # Skip GPU-only models if no GPU available
        if requires_gpu and not HAS_GPU:
            logger.warning(f"[Model] SKIP: {model_key} requires GPU (TensorRT)")
            continue
        
        # Skip TFLite models if TensorFlow not available
        if requires_tensorflow and not HAS_TENSORFLOW:
            logger.warning(f"[Model] SKIP: {model_key} requires TensorFlow (TFLite)")
            continue
        
        # Download if URL is provided and file doesn't exist
        if url and not os.path.exists(local_path):
            logger.info(f"[STEP] Downloading model: {model_key}")
            success = download_file(url, local_path)
            if not success:
                logger.error(f"[Model] SKIP: Failed to download {model_key}")
                continue
        
        # Check if local file exists
        if not os.path.exists(local_path):
            logger.warning(f"[Model] SKIP: {local_path} not found")
            continue
        
        # Load model
        try:
            logger.info(f"[STEP] Loading YOLO model from: {local_path}")
            model = YOLO(local_path)
            models[model_key] = model
            logger.info(f"[Model] SUCCESS: {model_key} loaded from {local_path}")
        except Exception as e:
            # Handle TensorRT/CUDA initialization failures gracefully
            error_msg = str(e).lower()
            if "cuda" in error_msg or "tensorrt" in error_msg or "segmentation fault" in error_msg:
                logger.error(f"[Model] SKIP: {model_key} - CUDA/TensorRT initialization failed")
                logger.error(f"         This usually means GPU drivers are incompatible or missing.")
                logger.exception(f"         Error details: {e}")
            else:
                logger.exception(f"[Model] ERROR: Failed to load {model_key}: {e}")
    
    logger.info("[STEP] " + "="*50)
    logger.info(f"[STEP] LOADED MODELS: {list(models.keys())}")
    if HAS_GPU:
        logger.info("[STEP] GPU READY: TensorRT models available for acceleration")
    logger.info("[STEP] " + "="*50)

# Log GPU/Device information at startup
logger.info("=" * 60)
logger.info("🖥️  DEVICE CONFIGURATION")
logger.info("=" * 60)
if HAS_GPU:
    logger.info(f"🚀 NVIDIA GPU Detected: {GPU_INFO['name']}")
    logger.info(f"   CUDA Version: {GPU_INFO['cuda_version']}")
    logger.info(f"   GPU Count: {GPU_INFO['count']}")
    DEVICE = "cuda"
else:
    logger.warning("⚠️  No NVIDIA GPU detected - running on CPU")
    logger.warning("   TensorRT models will be skipped")
    DEVICE = "cpu"
logger.info(f"   TensorFlow Available: {HAS_TENSORFLOW}")
logger.info("=" * 60)

# Load all models at startup
load_all_models()

# Select default model from .env DEFAULT_MODEL
def get_default_model():
    """Get the default model to use for inference from .env."""
    # Check DEFAULT_MODEL env variable
    default_model_id = os.getenv("DEFAULT_MODEL", "1")
    
    if default_model_id in MODEL_ID_MAP:
        requested_model = MODEL_ID_MAP[default_model_id]
        if requested_model in models:
            logger.info(f"✅ Default model from .env: {requested_model} (ID={default_model_id})")
            return models[requested_model], requested_model
        else:
            logger.warning(f"[Config] DEFAULT_MODEL={default_model_id} ({requested_model}) not loaded, using fallback")
    else:
        logger.warning(f"[Config] Invalid DEFAULT_MODEL={default_model_id}, using fallback")
    
    # Fallback: first available model from priority order
    priority = ["pytorch", "tflite-32", "tflite-16", "tfrt-32", "tfrt-16"]
    for model_key in priority:
        if model_key in models:
            logger.info(f"✅ Default model (fallback): {model_key}")
            return models[model_key], model_key
    
    raise RuntimeError("No models available! Please ensure at least one model file exists.")

default_model, default_model_key = get_default_model()

# Create FastAPI app
app = FastAPI(title="RDD Predict - Simple")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================================
# Channel/Room Management
# ============================================================================
@dataclass
class Channel:
    """Represents a broadcast channel/room."""
    channel_id: str
    broadcaster: WebSocket = None
    viewers: Set[WebSocket] = field(default_factory=set)
    latest_frame: dict = None
    frame_count: int = 0
    created_at: float = field(default_factory=time.time)

# Store all active channels
channels: Dict[str, Channel] = {}


def get_or_create_channel(channel_id: str) -> Channel:
    """Get existing channel or create new one."""
    if channel_id not in channels:
        channels[channel_id] = Channel(channel_id=channel_id)
        logger.info(f"[Channel] Created: {channel_id}")
    return channels[channel_id]


def cleanup_channel(channel_id: str):
    """Remove channel if empty."""
    if channel_id in channels:
        channel = channels[channel_id]
        if channel.broadcaster is None and len(channel.viewers) == 0:
            del channels[channel_id]
            logger.info(f"[Channel] Deleted: {channel_id}")


# ============================================================================
# Utility functions
# ============================================================================
def decode_base64_to_frame(base64_str: str) -> np.ndarray:
    """Decode base64 string to OpenCV frame."""
    import base64
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',', 1)[1]
    image_bytes = base64.b64decode(base64_str)
    np_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")
    return frame


def encode_frame_to_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode OpenCV frame to base64 JPEG string."""
    import base64
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


def process_frame(frame: np.ndarray, model_key: str = None):
    """Run YOLO prediction on frame (GPU accelerated if available).
    
    Args:
        frame: OpenCV image array (BGR format)
        model_key: Optional specific model to use. If None, uses default_model.
    
    Returns:
        Tuple of (annotated_frame, detections_list)
    """
    # Select model and determine the model key
    if model_key and model_key in models:
        selected_model = models[model_key]
        current_model_key = model_key
    else:
        selected_model = default_model
        current_model_key = default_model_key
    
    # TFLite and TensorRT models are compiled with fixed input size (640x640)
    # Only PyTorch models can use dynamic input sizes
    if current_model_key.startswith(("tflite", "tfrt")):
        imgsz = 640  # Fixed size for compiled models
    else:
        imgsz = 480  # Smaller size for faster inference on PyTorch
    
    # Run inference with GPU if available
    results = selected_model(frame, device=DEVICE, verbose=False, imgsz=imgsz)
    result = results[0]
    annotated_frame = result.plot()
    
    detections = []
    for box in result.boxes:
        c = int(box.cls)
        class_name = selected_model.names[c]
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]
        detections.append({
            "class": class_name,
            "confidence": round(conf, 4),
            "bbox": [round(x, 2) for x in xyxy]
        })
    
    return annotated_frame, detections


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
def root():
    return {
        "message": "RDD Predict API is running",
        "device": DEVICE,
        "gpu": GPU_INFO if HAS_GPU else {"available": False},
        "loaded_models": list(models.keys()),
        "default_model": default_model_key,
        "endpoints": {
            "stream": "WS /ws/stream - individual streaming",
            "broadcast": "WS /ws/broadcast/{channel_id} - create/stream to channel",
            "watch": "WS /ws/watch/{channel_id} - watch a channel",
            "channels": "GET /channels - list active channels",
            "gpu_status": "GET /gpu - GPU status information"
        }
    }


@app.get("/gpu")
def gpu_status():
    """Get GPU status and information."""
    return {
        "has_gpu": HAS_GPU,
        "device": DEVICE,
        "gpu_info": GPU_INFO if HAS_GPU else None,
        "cuda_available": torch.cuda.is_available(),
        "loaded_models": {
            key: {
                "description": MODEL_CONFIGS[key]["description"],
                "requires_gpu": MODEL_CONFIGS[key]["requires_gpu"]
            }
            for key in models.keys()
        }
    }


@app.get("/channels")
def list_channels():
    """List all active channels."""
    return {
        "channels": [
            {
                "channel_id": ch.channel_id,
                "has_broadcaster": ch.broadcaster is not None,
                "viewer_count": len(ch.viewers),
                "frame_count": ch.frame_count,
                "created_at": ch.created_at
            }
            for ch in channels.values()
        ],
        "total": len(channels)
    }


# ============================================================================
# Individual Stream Mode
# ============================================================================
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Individual stream: each client sends frames and receives their own results."""
    await websocket.accept()
    frame_index = 0
    logger.info("[Stream] Client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()
            
            try:
                frame = decode_base64_to_frame(data)
                annotated_frame, detections = await asyncio.to_thread(process_frame, frame)
                processed_frame_b64 = encode_frame_to_base64(annotated_frame, quality=70)
                latency_ms = (time.time() - start_time) * 1000
                
                response = {
                    "status": "success",
                    "frame_index": frame_index,
                    "latency_ms": round(latency_ms, 2),
                    "processed_frame": processed_frame_b64,
                    "detections": detections,
                    "detection_count": len(detections)
                }
                
                if frame_index % 30 == 0:
                    logger.info(f"[Stream] Frame {frame_index:04d} | Latency: {latency_ms:.2f}ms")
                
                await websocket.send_json(response)
                frame_index += 1
                
            except ValueError as e:
                await websocket.send_json({"status": "error", "error": str(e)})
                
    except WebSocketDisconnect:
        logger.info(f"[Stream] Client disconnected after {frame_index} frames")


# ============================================================================
# Channel-based Broadcast Mode
# ============================================================================
@app.websocket("/ws/broadcast/{channel_id}")
async def websocket_broadcast(websocket: WebSocket, channel_id: str):
    """
    Broadcast source: create or join a channel to stream.
    
    Example: ws://localhost:8000/ws/broadcast/my-stream-123
    """
    channel = get_or_create_channel(channel_id)
    
    # Check if channel already has a broadcaster
    if channel.broadcaster is not None:
        await websocket.accept()
        await websocket.send_json({
            "status": "error",
            "error": f"Channel '{channel_id}' already has an active broadcaster"
        })
        await websocket.close()
        return
    
    await websocket.accept()
    channel.broadcaster = websocket
    logger.info(f"[Broadcast] Channel: {channel_id} - Broadcaster connected")
    
    # Send channel info to broadcaster
    await websocket.send_json({
        "status": "connected",
        "channel_id": channel_id,
        "message": f"Broadcasting to channel: {channel_id}"
    })
    
    try:
        while True:
            data = await websocket.receive_text()
            start_time = time.time()
            
            try:
                frame = decode_base64_to_frame(data)
                annotated_frame, detections = await asyncio.to_thread(process_frame, frame)
                processed_frame_b64 = encode_frame_to_base64(annotated_frame, quality=70)
                latency_ms = (time.time() - start_time) * 1000
                
                response = {
                    "status": "success",
                    "channel_id": channel_id,
                    "frame_index": channel.frame_count,
                    "latency_ms": round(latency_ms, 2),
                    "processed_frame": processed_frame_b64,
                    "detections": detections,
                    "detection_count": len(detections),
                    "viewer_count": len(channel.viewers)
                }
                
                # Store latest frame for new viewers
                channel.latest_frame = response
                channel.frame_count += 1
                
                # Send to broadcaster
                await websocket.send_json(response)
                
                # Broadcast to all viewers
                if channel.viewers:
                    await asyncio.gather(
                        *[viewer.send_json(response) for viewer in channel.viewers],
                        return_exceptions=True
                    )
                
                if channel.frame_count % 30 == 0:
                    logger.info(f"[Broadcast:{channel_id}] Frame {channel.frame_count:04d} | Viewers: {len(channel.viewers)}")
                
            except ValueError as e:
                await websocket.send_json({"status": "error", "error": str(e)})
                
    except WebSocketDisconnect:
        logger.info(f"[Broadcast:{channel_id}] Broadcaster disconnected after {channel.frame_count} frames")
    finally:
        channel.broadcaster = None
        channel.latest_frame = None
        cleanup_channel(channel_id)


@app.websocket("/ws/watch/{channel_id}")
async def websocket_watch(websocket: WebSocket, channel_id: str):
    """
    Watch a specific channel by ID.
    
    Example: ws://localhost:8000/ws/watch/my-stream-123
    """
    channel = get_or_create_channel(channel_id)
    
    await websocket.accept()
    channel.viewers.add(websocket)
    logger.info(f"[Watch:{channel_id}] Viewer connected. Total: {len(channel.viewers)}")
    
    # Send channel status
    await websocket.send_json({
        "status": "connected",
        "channel_id": channel_id,
        "has_broadcaster": channel.broadcaster is not None,
        "viewer_count": len(channel.viewers)
    })
    
    # Send latest frame if available
    if channel.latest_frame:
        await websocket.send_json(channel.latest_frame)
    
    try:
        # Keep connection alive
        while True:
            msg = await websocket.receive_text()
            # Handle ping or other messages
            if msg == "ping":
                await websocket.send_json({"status": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        channel.viewers.discard(websocket)
        logger.info(f"[Watch:{channel_id}] Viewer disconnected. Total: {len(channel.viewers)}")
        cleanup_channel(channel_id)