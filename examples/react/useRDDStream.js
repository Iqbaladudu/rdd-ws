/**
 * React Hook for RDD Stream - Individual mode
 * Each client sends their own camera frames
 */
import { useState, useRef, useCallback } from 'react';

export function useRDDStream(serverUrl = '') {
    const [isConnected, setIsConnected] = useState(false);
    const [frame, setFrame] = useState(null);
    const [detections, setDetections] = useState([]);
    const [latency, setLatency] = useState(0);
    const [fps, setFps] = useState(0);
    const wsRef = useRef(null);
    const fpsTimestampsRef = useRef([]);

    const connect = useCallback(() => {
        const wsUrl = serverUrl || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
        wsRef.current = new WebSocket(`${wsUrl}/ws/stream`);

        wsRef.current.onopen = () => {
            setIsConnected(true);
            fpsTimestampsRef.current = [];
        };

        wsRef.current.onclose = () => setIsConnected(false);
        wsRef.current.onerror = () => setIsConnected(false);

        wsRef.current.onmessage = (event) => {
            const res = JSON.parse(event.data);
            if (res.status === 'success') {
                setFrame(res.processed_frame);
                setDetections(res.detections || []);
                setLatency(res.latency_ms);

                // Calculate FPS
                const now = Date.now();
                const timestamps = fpsTimestampsRef.current;
                timestamps.push(now);
                while (timestamps.length > 0 && now - timestamps[0] > 2000) {
                    timestamps.shift();
                }
                if (timestamps.length > 1) {
                    setFps(Math.round((timestamps.length - 1) / ((now - timestamps[0]) / 1000)));
                }
            }
        };
    }, [serverUrl]);

    const sendFrame = useCallback((base64Frame) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            // Remove data URI prefix if present
            const data = base64Frame.includes(',') ? base64Frame.split(',')[1] : base64Frame;
            wsRef.current.send(data);
        }
    }, []);

    const disconnect = useCallback(() => {
        wsRef.current?.close();
        setIsConnected(false);
    }, []);

    return {
        isConnected,
        frame,
        detections,
        latency,
        fps,
        connect,
        sendFrame,
        disconnect
    };
}
