/**
 * RDD Viewer Component - Broadcast Watch Mode
 * View the stream from a broadcaster without sending any camera data
 */
import React, { useEffect } from 'react';
import { useRDDBroadcast } from './useRDDBroadcast';

export function RDDViewerComponent({ serverUrl }) {
    const {
        isConnected,
        frame,
        detections,
        latency,
        viewerCount,
        watch,
        disconnect
    } = useRDDBroadcast(serverUrl);

    useEffect(() => {
        return () => disconnect();
    }, [disconnect]);

    return (
        <div style={{ fontFamily: 'system-ui', padding: '20px' }}>
            <h2>📺 Road Damage Detection - Viewer Mode</h2>

            <div style={{ marginBottom: '16px' }}>
                <button onClick={watch} disabled={isConnected}>Watch Stream</button>
                <button onClick={disconnect} disabled={!isConnected} style={{ marginLeft: '8px' }}>Disconnect</button>
                <span style={{ marginLeft: '16px', color: isConnected ? 'green' : 'red' }}>
                    {isConnected ? '● Watching' : '○ Not connected'}
                </span>
            </div>

            {!isConnected && !frame && (
                <p style={{ color: '#666' }}>Click "Watch Stream" to view the broadcast...</p>
            )}

            {frame && (
                <div>
                    <img src={frame} alt="Broadcast" style={{ maxWidth: '640px', width: '100%' }} />
                </div>
            )}

            <div style={{ marginTop: '16px' }}>
                <p>
                    <strong>Viewers:</strong> {viewerCount} |
                    <strong> Latency:</strong> {latency}ms |
                    <strong> Detections:</strong> {detections.length}
                </p>
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
