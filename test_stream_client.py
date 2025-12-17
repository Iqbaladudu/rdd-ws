#!/usr/bin/env python3
"""
Test client for Socket.IO video streaming endpoint.
Usage: python test_stream_client.py [video_file]

Migrated from WebSocket to Socket.IO for improved reliability and auto-reconnection.
"""
import asyncio
import base64
import json
import sys
import time
import cv2

try:
    import socketio
except ImportError:
    print("Please install python-socketio: pip install python-socketio[asyncio_client]")
    sys.exit(1)


async def stream_video(video_path: str, server_url: str = "http://localhost:8000", model: str = None):
    """Stream video frames to Socket.IO server and display results."""
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Connecting to {server_url}...")
    
    sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=5)
    frame_count = 0
    start_time = time.time()
    stop_event = asyncio.Event()
    
    @sio.event
    async def connect():
        print("Connected to Socket.IO server!")
    
    @sio.event
    async def session_info(data):
        print(f"Session info: {data}")
    
    @sio.event
    async def prediction(response):
        nonlocal frame_count
        
        if response.get("status") == "success":
            detection_count = response.get("detection_count", 0)
            latency = response.get("processing_latency_ms", 0)
            detections = response.get("detections", [])
            
            print(f"\n[Frame {frame_count}/{total_frames}] Latency: {latency:.1f}ms | Detections: {detection_count}")
            for det in detections:
                print(f"  - {det['class']}: {det['confidence']:.2%} @ {det['bbox']}")
            
            # Display the processed frame
            if response.get("processed_frame"):
                frame_data = response["processed_frame"]
                if frame_data.startswith("data:"):
                    frame_data = frame_data.split(",", 1)[1]
                img_bytes = base64.b64decode(frame_data)
                import numpy as np
                nparr = np.frombuffer(img_bytes, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imshow("Processed Stream", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
        else:
            print(f"\nError: {response.get('error', 'Unknown error')}")
        
        frame_count += 1
    
    @sio.event
    async def error(data):
        print(f"Server error: {data}")
    
    @sio.event
    async def disconnect():
        print("Disconnected from server")
        stop_event.set()
    
    try:
        # Connect with auth
        auth = {"device_id": f"python-client-{int(time.time())}", "model": model or "default"}
        await sio.connect(server_url, auth=auth)
        
        print("Streaming frames...")
        
        while cap.isOpened() and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame via Socket.IO
            await sio.emit('stream_frame', frame_base64)
            
            # Small delay to prevent overwhelming the server
            await asyncio.sleep(0.01)
        
        elapsed = time.time() - start_time
        print(f"\n\nCompleted! Processed {frame_count} frames in {elapsed:.2f}s")
        print(f"Average FPS: {frame_count / elapsed:.2f}")
        
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        await sio.disconnect()
        cap.release()
        cv2.destroyAllWindows()


async def stream_webcam(camera_device=0, server_url: str = "http://localhost:8000", model: str = None, save_video: str = None):
    """Stream webcam/virtual camera frames to Socket.IO server.
    
    Args:
        camera_device: Camera device index (0, 1, 2, ...) or path ('/dev/video0', '/dev/video2')
        save_video: Optional path to save output video (e.g., 'output.mp4')
    """
    
    # Try to open the camera device
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        # On Linux, try /dev/video paths
        if isinstance(camera_device, int):
            alt_path = f"/dev/video{camera_device}"
            print(f"Trying alternative path: {alt_path}")
            cap = cv2.VideoCapture(alt_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera device: {camera_device}")
            print("\nAvailable video devices on Linux:")
            import glob
            devices = glob.glob('/dev/video*')
            for d in sorted(devices):
                print(f"  {d}")
            print("\nTry: python test_stream_client.py --webcam 2")
            print("Or:  python test_stream_client.py --webcam /dev/video2")
            return
    
    # Setup video writer if save_video is specified
    video_writer = None
    if save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
        print(f"Recording to: {save_video} ({width}x{height} @ {fps}fps)")
    
    print(f"Camera opened: {camera_device}")
    print(f"Connecting to {server_url}...")
    
    sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=5)
    frame_count = 0
    stop_event = asyncio.Event()
    
    @sio.event
    async def connect():
        print("Connected! Streaming webcam (press 'q' to quit)...")
    
    @sio.event
    async def session_info(data):
        print(f"Session: {data.get('session_id', 'N/A')[:16]}...")
    
    @sio.event
    async def prediction(response):
        nonlocal frame_count
        
        if response.get("status") == "success":
            detection_count = response.get("detection_count", 0)
            latency = response.get("processing_latency_ms", 0)
            detections = response.get("detections", [])
            
            # Log JSON to console (exclude processed_frame for readability)
            log_response = {k: v for k, v in response.items() if k != "processed_frame"}
            print(f"\n{'='*60}")
            print(json.dumps(log_response, indent=2))
            print(f"{'='*60}")
            
            # Decode and display processed frame
            if response.get("processed_frame"):
                frame_data = response["processed_frame"]
                if frame_data.startswith("data:"):
                    frame_data = frame_data.split(",", 1)[1]
                img_bytes = base64.b64decode(frame_data)
                import numpy as np
                nparr = np.frombuffer(img_bytes, np.uint8)
                processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Save frame to video file
                if video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Add info overlay
                cv2.putText(processed_frame, f"Detections: {detection_count} | Latency: {latency:.1f}ms",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Real-time Detection", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
        
        frame_count += 1
    
    @sio.event
    async def error(data):
        print(f"Error: {data}")
    
    @sio.event
    async def disconnect():
        print("Disconnected")
        stop_event.set()
    
    try:
        auth = {"device_id": f"python-webcam-{int(time.time())}", "model": model or "default"}
        await sio.connect(server_url, auth=auth)
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send frame
            await sio.emit('stream_frame', frame_base64)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            await asyncio.sleep(0.01)
        
        print(f"\nProcessed {frame_count} frames")
        if save_video:
            print(f"Video saved to: {save_video}")
            
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        await sio.disconnect()
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


async def list_available_models(base_url: str = "http://localhost:8000"):
    """Fetch and display available models from the server."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/models") as response:
                data = await response.json()
                
        print("=" * 60)
        print("AVAILABLE MODELS")
        print("=" * 60)
        print(f"Total loaded: {data.get('total_loaded', 0)}")
        print(f"Default model: {data.get('default_model', 'N/A')}")
        print()
        
        for model in data.get('loaded_models', []):
            print(f"  [{model['key']}]")
            print(f"    Description: {model['description']}")
            print(f"    Endpoint: Socket.IO with auth={{model: '{model['key']}'}}")
            print()
            
    except ImportError:
        print("Please install aiohttp: pip install aiohttp")
        print("\nAlternatively, use curl to check models:")
        print("  curl http://localhost:8000/models")
    except Exception as e:
        print(f"Error fetching models: {e}")
        print("Make sure the server is running.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Parse global --model argument
        model = None
        args = sys.argv[1:]
        
        # Extract --model from args
        filtered_args = []
        i = 0
        while i < len(args):
            if args[i] == "--model" and i + 1 < len(args):
                model = args[i + 1]
                i += 2
            else:
                filtered_args.append(args[i])
                i += 1
        
        if not filtered_args:
            filtered_args = ["--help"]
        
        command = filtered_args[0]
        
        if command == "--webcam":
            # Parse arguments: --webcam [device] [--save output.mp4]
            device = 0
            save_video = None
            
            sub_args = filtered_args[1:]
            j = 0
            while j < len(sub_args):
                if sub_args[j] == "--save" and j + 1 < len(sub_args):
                    save_video = sub_args[j + 1]
                    j += 2
                else:
                    # Try to parse as device
                    try:
                        device = int(sub_args[j])
                    except ValueError:
                        device = sub_args[j]  # Keep as string path
                    j += 1
            
            print(f"Using model: {model or 'default'}")
            print(f"Protocol: Socket.IO")
            asyncio.run(stream_webcam(device, model=model, save_video=save_video))
            
        elif command == "--list":
            # List available video devices
            import glob
            print("Available video devices:")
            devices = glob.glob('/dev/video*')
            for d in sorted(devices):
                print(f"  {d}")
                
        elif command == "--list-models":
            # List available models from server
            asyncio.run(list_available_models())
            
        elif command in ["--help", "-h"]:
            print("Usage:")
            print("  python test_stream_client.py <video_file> [--model MODEL]        - Stream a video file")
            print("  python test_stream_client.py --webcam [device] [--save FILE] [--model MODEL] - Stream from webcam")
            print("  python test_stream_client.py --list                              - List available video devices")
            print("  python test_stream_client.py --list-models                       - List available models")
            print()
            print("Model options (use with --model):")
            print("  pytorch     : PyTorch Original")
            print("  tfrt-32     : TensorRT Float32")
            print("  tfrt-16     : TensorRT Float16")
            print("  tflite-32   : TFLite Float32")
            print("  tflite-16   : TFLite Float16")
            print()
            print("Examples:")
            print("  python test_stream_client.py --webcam                            - Use default camera + default model")
            print("  python test_stream_client.py --webcam --model pytorch            - Use PyTorch model")
            print("  python test_stream_client.py --webcam 2 --model tfrt-32          - cam 2 + TensorRT FP32")
            print("  python test_stream_client.py --webcam --save out.mp4 --model tfrt-16")
            print("  python test_stream_client.py video.mp4 --model tflite-32")
            print("  python test_stream_client.py --list-models                       - Show available models")
            print()
            print("Note: This client uses Socket.IO protocol (not raw WebSocket)")
        else:
            # Assume it's a video file
            video_file = command
            print(f"Using model: {model or 'default'}")
            print(f"Protocol: Socket.IO")
            asyncio.run(stream_video(video_file, model=model))
    else:
        print("Usage:")
        print("  python test_stream_client.py <video_file> [--model MODEL]        - Stream a video file")
        print("  python test_stream_client.py --webcam [device] [--save FILE] [--model MODEL] - Stream from webcam")
        print("  python test_stream_client.py --list                              - List available devices")
        print("  python test_stream_client.py --list-models                       - List available models")
        print()
        print("Model options (use with --model):")
        print("  pytorch     : PyTorch Original")
        print("  tfrt-32     : TensorRT Float32")
        print("  tfrt-16     : TensorRT Float16")
        print("  tflite-32   : TFLite Float32")
        print("  tflite-16   : TFLite Float16")
        print()
        print("Examples:")
        print("  python test_stream_client.py --webcam                            - Use default camera")
        print("  python test_stream_client.py --webcam 2 --model tfrt-32          - cam 2 + TensorRT")
        print("  python test_stream_client.py --webcam 2 --save output.mp4        - Stream & save video")
        print("  python test_stream_client.py --webcam --save result.mp4          - Default cam + save")
        print("  python test_stream_client.py --list-models                       - Show available models")
        print()
        print("Note: This client uses Socket.IO protocol (migrated from WebSocket)")
