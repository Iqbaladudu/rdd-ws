/**
 * RDD Channel Component - Broadcast & Watch with Channel IDs
 * Can act as broadcaster or viewer for a specific channel
 */
import React, { useEffect, useRef, useState } from 'react';
import { useRDDChannel, fetchChannels } from './useRDDBroadcast';

export function RDDChannelComponent({ serverUrl }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [inputChannelId, setInputChannelId] = useState('');
    const [channels, setChannels] = useState([]);
    const animationRef = useRef(null);

    const {
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
    } = useRDDChannel(serverUrl);

    // Fetch available channels
    const refreshChannels = async () => {
        try {
            const data = await fetchChannels(serverUrl);
            setChannels(data.channels || []);
        } catch (e) {
            console.error('Failed to fetch channels:', e);
        }
    };

    useEffect(() => {
        refreshChannels();
        const interval = setInterval(refreshChannels, 5000);
        return () => clearInterval(interval);
    }, []);

    // Start broadcasting
    const startBroadcast = async () => {
        if (!inputChannelId.trim()) {
            alert('Please enter a Channel ID');
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            videoRef.current.srcObject = stream;
            await videoRef.current.play();

            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;

            broadcast(inputChannelId.trim());
        } catch (err) {
            console.error('Failed to start broadcast:', err);
        }
    };

    // Start watching
    const startWatch = (id) => {
        watch(id || inputChannelId.trim());
    };

    // Stop everything
    const stop = () => {
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        disconnect();
        if (videoRef.current?.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(t => t.stop());
        }
    };

    // Capture and send frames when broadcasting
    useEffect(() => {
        if (!isConnected || mode !== 'broadcast') return;

        let pending = 0;
        const captureLoop = () => {
            if (!isConnected || pending >= 3) {
                animationRef.current = requestAnimationFrame(captureLoop);
                return;
            }

            const ctx = canvasRef.current.getContext('2d');
            ctx.drawImage(videoRef.current, 0, 0);
            const b64 = canvasRef.current.toDataURL('image/jpeg', 0.5);
            sendFrame(b64);
            pending++;
            setTimeout(() => pending--, 100);

            animationRef.current = requestAnimationFrame(captureLoop);
        };

        animationRef.current = requestAnimationFrame(captureLoop);
        return () => cancelAnimationFrame(animationRef.current);
    }, [isConnected, mode, sendFrame]);

    return (
        <div style={{ fontFamily: 'system-ui', padding: '20px', maxWidth: '900px' }}>
            <h2>🛣️ Road Damage Detection - Channels</h2>

            {/* Channel Input */}
            <div style={{ marginBottom: '16px', display: 'flex', gap: '8px', alignItems: 'center' }}>
                <input
                    type="text"
                    placeholder="Enter Channel ID (e.g., room-123)"
                    value={inputChannelId}
                    onChange={(e) => setInputChannelId(e.target.value)}
                    style={{ padding: '8px', width: '200px' }}
                    disabled={isConnected}
                />
                <button onClick={startBroadcast} disabled={isConnected}>📡 Broadcast</button>
                <button onClick={() => startWatch()} disabled={isConnected || !inputChannelId}>👁️ Watch</button>
                <button onClick={stop} disabled={!isConnected}>Stop</button>
                <button onClick={refreshChannels}>🔄 Refresh</button>
            </div>

            {/* Status */}
            <div style={{ marginBottom: '16px', padding: '10px', background: '#f0f0f0', borderRadius: '8px' }}>
                <strong>Status:</strong> {isConnected ? (
                    <span style={{ color: 'green' }}>
                        ● Connected as {mode === 'broadcast' ? 'Broadcaster' : 'Viewer'} to "{channelId}"
                    </span>
                ) : (
                    <span style={{ color: 'gray' }}>○ Not connected</span>
                )}
                {isConnected && (
                    <span style={{ marginLeft: '16px' }}>
                        Viewers: {viewerCount} | FPS: {fps} | Latency: {latency}ms
                    </span>
                )}
            </div>

            {/* Video Display */}
            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                {mode === 'broadcast' && (
                    <div>
                        <p>📷 Camera</p>
                        <video ref={videoRef} style={{ width: '320px', background: '#000' }} muted />
                    </div>
                )}
                <div>
                    <p>🔍 Detection Result</p>
                    {frame ? (
                        <img src={frame} alt="Processed" style={{ maxWidth: '400px' }} />
                    ) : (
                        <div style={{ width: '320px', height: '240px', background: '#222', color: '#666', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            {mode === 'watch' && !hasBroadcaster ? 'Waiting for broadcaster...' : 'No frames yet'}
                        </div>
                    )}
                </div>
            </div>

            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Active Channels List */}
            <div style={{ marginTop: '24px' }}>
                <h3>Active Channels ({channels.length})</h3>
                {channels.length === 0 ? (
                    <p style={{ color: '#666' }}>No active channels. Start broadcasting to create one!</p>
                ) : (
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                            <tr style={{ background: '#f0f0f0' }}>
                                <th style={{ padding: '8px', textAlign: 'left' }}>Channel ID</th>
                                <th>Status</th>
                                <th>Viewers</th>
                                <th>Frames</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
                            {channels.map((ch) => (
                                <tr key={ch.channel_id} style={{ borderBottom: '1px solid #ddd' }}>
                                    <td style={{ padding: '8px' }}><strong>{ch.channel_id}</strong></td>
                                    <td style={{ textAlign: 'center' }}>
                                        {ch.has_broadcaster ? '🟢 Live' : '⚫ No broadcaster'}
                                    </td>
                                    <td style={{ textAlign: 'center' }}>{ch.viewer_count}</td>
                                    <td style={{ textAlign: 'center' }}>{ch.frame_count}</td>
                                    <td style={{ textAlign: 'center' }}>
                                        <button
                                            onClick={() => startWatch(ch.channel_id)}
                                            disabled={isConnected}
                                        >
                                            Watch
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                )}
            </div>

            {/* Detections */}
            {detections.length > 0 && (
                <div style={{ marginTop: '16px' }}>
                    <h4>Detected Damage ({detections.length}):</h4>
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
