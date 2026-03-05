---
name: reachyclaw
description: Give your OpenClaw AI agent a physical robot body with Reachy Mini. OpenClaw is the brain — it controls speech, movement, and vision. Works with physical robot OR simulator!
---

# ReachyClaw - Robot Body for OpenClaw

Give your OpenClaw agent a physical Reachy Mini robot body where OpenClaw is the actual brain.

## Overview

ReachyClaw embodies your OpenClaw AI agent in a Reachy Mini robot. Unlike typical setups where GPT-4o is the brain, ReachyClaw routes every message through OpenClaw and lets it control the robot body via action tags.

- **Hear**: Listen to voice commands via the robot's microphone
- **See**: View the world through the robot's camera
- **Speak**: Respond with natural voice through the robot's speaker
- **Move**: Control head movements, emotions, and dances from OpenClaw

## Architecture

```
You speak -> Reachy Mini microphone
                 |
          OpenAI Realtime API (STT only)
                 |
          OpenClaw (the actual brain)
                 |
     Response: "[EMOTION:happy] That's great!"
                 |
     ReachyClaw parses actions -> robot moves
     Clean text -> TTS -> robot speaks
```

## Requirements

### Option A: Physical Robot
- [Reachy Mini](https://github.com/pollen-robotics/reachy_mini) robot (Wireless or Lite)

### Option B: Simulator (No Hardware Required!)
- Any computer with Python 3.11+
- Install: `pip install "reachy-mini[mujoco]"`

### Software (Both Options)
- Python 3.11+
- OpenAI API key with Realtime API access
- OpenClaw gateway running on your network

## Installation

```bash
git clone https://github.com/shaunx/reachyclaw
cd reachyclaw
pip install -e .
```

## Configuration

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
OPENCLAW_GATEWAY_URL=http://your-host-ip:18789
OPENCLAW_TOKEN=your-gateway-token
```

## Usage

### With Simulator

```bash
# Terminal 1: Start simulator
reachy-mini-daemon --sim

# Terminal 2: Run ReachyClaw
reachyclaw --gradio
```

### With Physical Robot

```bash
reachyclaw

# With debug logging
reachyclaw --debug

# With Gradio web UI
reachyclaw --gradio
```

## Robot Actions

OpenClaw can include these action tags in its responses:

- `[LOOK:left/right/up/down/front]` — head movement
- `[EMOTION:happy/sad/surprised/curious/thinking/confused/excited]` — emotions
- `[DANCE:happy/excited/wave/nod/shake/bounce]` — dances
- `[CAMERA]` — capture what the robot sees
- `[FACE_TRACKING:on/off]` — toggle face tracking
- `[STOP]` — stop all movements

## Links

- [GitHub Repository](https://github.com/shaunx/reachyclaw)
- [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini)
- [OpenClaw Documentation](https://docs.openclaw.ai)

## License

Apache 2.0
