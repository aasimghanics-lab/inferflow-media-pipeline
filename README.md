# Inferflow Media Pipeline

An audio-visual generation pipeline built for Inferflow's AI research infrastructure.

## Overview

Programmatically generates synchronized audio-visual content using Python and FFmpeg for AI training data pipelines.

## What it does

- **Audio**: Multi-layered music track (bass, melody, drums) via numpy signal processing at 44.1kHz
- **Video**: Animated 1280x720 waveform visualizer synced frame-by-frame to audio
- **Pipeline**: Combines into production-ready MP4 via FFmpeg

## Tech Stack

- Python (numpy, Pillow)
- FFmpeg (libx264 + AAC)
- Signal processing (sine/square synthesis, envelope shaping, kick drum)
- Image compositing (layered animation, real-time waveform rendering)

## Output

`inferflow_demo.mp4` — 30 second 1280x720 audio-visual demo at 30fps

## Use Case

Generates synthetic media assets for AI model training and evaluation pipelines at Inferflow.
