/**
 * React Hook for RDD Broadcast with Channel ID
 * - broadcast(channelId): Start broadcasting to a channel
 * - watch(channelId): Watch a specific channel
 */
import { useState, useRef, useCallback } from 'react';

export function useRDDChannel(serverUrl = '') {
    const [isConnected, setIsConnected] = useState(false);
    const [channelId, setChannelId] = useState(null);
    const [mode, setMode] = useState(null); // 'broadcast' | 'watch'
    const [frame, setFrame] = useState(null);
    const [detections, setDetections] = useState([]);
    const [latency, setLatency] = useState(0);
    const [viewerCount, setViewerCount] = useState(0);
    const [hasBroadcaster, setHasBroadcaster] = useState(false);
    const wsRef = useRef(null);
    const fpsTimestampsRef = useRef([]);
    const [fps, setFps] = useState(0);

    const getWsUrl = useCallback(() => {
        return serverUrl || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
    }, [serverUrl]);

    // Start broadcasting to a channel
    const broadcast = useCallback((id) => {
        const wsUrl = getWsUrl();
        wsRef.current = new WebSocket(`${wsUrl}/ws/broadcast/${id}`);
        setChannelId(id);
        setMode('broadcast');

        wsRef.current.onopen = () => {
            setIsConnected(true);
            fpsTimestampsRef.current = [];
        };

        wsRef.current.onclose = () => {
            setIsConnected(false);
            setMode(null);
        };

        wsRef.current.onerror = () => setIsConnected(false);

        wsRef.current.onmessage = (event) => {
            const res = JSON.parse(event.data);
            if (res.status === 'success') {
                setFrame(res.processed_frame);
                setDetections(res.detections || []);
                setLatency(res.latency_ms);
                setViewerCount(res.viewer_count || 0);

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
            } else if (res.status === 'error') {
                console.error('[RDD Channel] Error:', res.error);
            }
        };
    }, [getWsUrl]);

    // Watch a specific channel
    const watch = useCallback((id) => {
        const wsUrl = getWsUrl();
        wsRef.current = new WebSocket(`${wsUrl}/ws/watch/${id}`);
        setChannelId(id);
        setMode('watch');

        wsRef.current.onopen = () => setIsConnected(true);
        wsRef.current.onclose = () => {
            setIsConnected(false);
            setMode(null);
        };
        wsRef.current.onerror = () => setIsConnected(false);

        wsRef.current.onmessage = (event) => {
            const res = JSON.parse(event.data);
            if (res.status === 'connected') {
                setHasBroadcaster(res.has_broadcaster);
                setViewerCount(res.viewer_count || 0);
            } else if (res.status === 'success') {
                setFrame(res.processed_frame);
                setDetections(res.detections || []);
                setLatency(res.latency_ms);
                setViewerCount(res.viewer_count || 0);
                setHasBroadcaster(true);
            }
        };
    }, [getWsUrl]);

    // Send a frame (only works in broadcast mode)
    const sendFrame = useCallback((base64Frame) => {
        if (wsRef.current?.readyState === WebSocket.OPEN && mode === 'broadcast') {
            const data = base64Frame.includes(',') ? base64Frame.split(',')[1] : base64Frame;
            wsRef.current.send(data);
        }
    }, [mode]);

    // Disconnect from channel
    const disconnect = useCallback(() => {
        wsRef.current?.close();
        setIsConnected(false);
        setMode(null);
        setChannelId(null);
    }, []);

    return {
        isConnected,
        channelId,
        mode,
        frame,
        detections,
        latency,
        fps,
        viewerCount,
        hasBroadcaster,
        broadcast,
        watch,
        sendFrame,
        disconnect
    };
}

// Helper to fetch active channels
export async function fetchChannels(serverUrl = '') {
    const baseUrl = serverUrl || window.location.origin;
    const response = await fetch(`${baseUrl}/channels`);
    return response.json();
}
