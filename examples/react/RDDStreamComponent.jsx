/**
 * RDD Stream Component - Individual Mode
 * Each user has their own camera and receives their own detection results
 */
import React, { useEffect, useRef, useState } from 'react';
import { useRDDStream } from './useRDDStream';

export function RDDStreamComponent({ serverUrl }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const animationRef = useRef(null);

    const {
        isConnected,
        frame,
        detections,
        latency,
        fps,
        connect,
        sendFrame,
        disconnect
    } = useRDDStream(serverUrl);

    const startStream = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            videoRef.current.srcObject = stream;
            await videoRef.current.play();

            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;

            connect();
            setIsStreaming(true);
        } catch (err) {
            console.error('Failed to start stream:', err);
        }
    };

    const stopStream = () => {
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
        }
        disconnect();
        if (videoRef.current?.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(t => t.stop());
        }
        setIsStreaming(false);
    };

    // Capture and send frames
    useEffect(() => {
        if (!isConnected || !isStreaming) return;

        let pendingFrames = 0;
        const maxPending = 3;

        const captureLoop = () => {
            if (!isConnected || pendingFrames >= maxPending) {
                animationRef.current = requestAnimationFrame(captureLoop);
                return;
            }

            const ctx = canvasRef.current.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);
            const b64 = canvasRef.current.toDataURL('image/jpeg', 0.5);
            sendFrame(b64);
            pendingFrames++;

            // Decrease pending on next frame received
            setTimeout(() => pendingFrames--, 100);

            animationRef.current = requestAnimationFrame(captureLoop);
        };

        animationRef.current = requestAnimationFrame(captureLoop);

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isConnected, isStreaming, sendFrame]);

    return (
        <div style={{ fontFamily: 'system-ui', padding: '20px' }}>
            <h2>🛣️ Road Damage Detection - Individual Mode</h2>

            <div style={{ marginBottom: '16px' }}>
                <button onClick={startStream} disabled={isStreaming}>Start</button>
                <button onClick={stopStream} disabled={!isStreaming} style={{ marginLeft: '8px' }}>Stop</button>
                <span style={{ marginLeft: '16px', color: isConnected ? 'green' : 'red' }}>
                    {isConnected ? '● Connected' : '○ Disconnected'}
                </span>
            </div>

            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                <div>
                    <p>Camera</p>
                    <video ref={videoRef} style={{ width: '320px', background: '#000' }} muted />
                </div>
                <div>
                    <p>Detection Result</p>
                    {frame && <img src={frame} alt="Processed" style={{ width: '320px' }} />}
                </div>
            </div>

            <canvas ref={canvasRef} style={{ display: 'none' }} />

            <div style={{ marginTop: '16px' }}>
                <p><strong>FPS:</strong> {fps} | <strong>Latency:</strong> {latency}ms | <strong>Detections:</strong> {detections.length}</p>
            </div>

            {detections.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                    <h4>Detected Damage:</h4>
                    <ul>
                        {detections.map((d, i) => (
                            <li key={i}>{d.class} ({(d.confidence * 100).toFixed(1)}%)</li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}
