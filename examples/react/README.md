# React Examples for RDD Predict

React components for Road Damage Detection streaming with Channel support.

## Components

| Component | Description |
|-----------|-------------|
| `RDDStreamComponent` | Individual mode - each user has camera |
| `RDDViewerComponent` | Simple viewer for one broadcast |
| `RDDChannelComponent` | **Full channel system with room IDs** |

## Usage

### Channel Mode (Recommended)

```jsx
import { RDDChannelComponent } from './RDDChannelComponent';

function App() {
  return <RDDChannelComponent serverUrl="ws://localhost:8000" />;
}
```

Features:
- Create channels with custom IDs (e.g., `room-123`)
- List all active channels
- Join any channel as viewer
- One broadcaster per channel, unlimited viewers

### Using Hooks

```jsx
import { useRDDChannel, fetchChannels } from './useRDDBroadcast';

const { broadcast, watch, sendFrame, disconnect, frame, detections } = useRDDChannel(serverUrl);

// Broadcast to channel
broadcast('my-channel-id');

// Watch a channel
watch('my-channel-id');

// Get active channels
const channels = await fetchChannels(serverUrl);
```

## Server Endpoints

| Endpoint | Description |
|----------|-------------|
| `WS /ws/stream` | Individual - each client sends own frames |
| `WS /ws/broadcast/{channel_id}` | Create & stream to channel |
| `WS /ws/watch/{channel_id}` | Watch specific channel |
| `GET /channels` | List active channels |

