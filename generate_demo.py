import numpy as np
import subprocess
import struct
import wave
import os

os.makedirs("/home/claude/demo", exist_ok=True)

# ── 1. AUDIO: generate a music track (bass + melody + drums) ──────────────────
SR = 44100
DURATION = 30  # seconds

def note_freq(semitone_offset, base=110.0):
    return base * (2 ** (semitone_offset / 12))

def sine(freq, duration, sr=SR, amp=0.4):
    t = np.linspace(0, duration, int(sr * duration), False)
    return amp * np.sin(2 * np.pi * freq * t)

def square(freq, duration, sr=SR, amp=0.2):
    t = np.linspace(0, duration, int(sr * duration), False)
    s = np.sign(np.sin(2 * np.pi * freq * t))
    return amp * s

def envelope(sig, attack=0.01, release=0.05, sr=SR):
    a = int(sr * attack)
    r = int(sr * release)
    env = np.ones(len(sig))
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    return sig * env

# Bass line pattern (semitones from A2=110Hz)
bass_pattern = [0, 0, 7, 0, 5, 0, 3, 0]
beat_dur = 60 / 120  # 120 BPM

audio = np.zeros(SR * DURATION)

for i in range(DURATION * 2):  # 2 beats per second at 120bpm
    start = int(i * beat_dur * SR)
    note = bass_pattern[i % len(bass_pattern)]
    seg = envelope(sine(note_freq(note), beat_dur * 0.8), attack=0.005, release=0.1)
    end = start + len(seg)
    if end < len(audio):
        audio[start:end] += seg

# Melody
melody = [12, 12, 14, 16, 14, 12, 10, 12]
for i in range(DURATION * 2):
    start = int(i * beat_dur * SR)
    note = melody[i % len(melody)]
    seg = envelope(square(note_freq(note, base=220), beat_dur * 0.6), attack=0.01, release=0.08)
    end = start + len(seg)
    if end < len(audio):
        audio[start:end] += seg * 0.5

# Hi-hat (noise bursts)
hat_interval = int(SR * beat_dur / 2)
for i in range(0, SR * DURATION, hat_interval):
    dur = int(SR * 0.04)
    noise = np.random.randn(dur) * 0.08
    env = np.linspace(1, 0, dur) ** 2
    end = i + dur
    if end < len(audio):
        audio[i:end] += noise * env

# Kick (sine sweep down)
kick_interval = int(SR * beat_dur)
for i in range(0, SR * DURATION, kick_interval):
    dur = int(SR * 0.15)
    t = np.linspace(0, 0.15, dur)
    freq = 180 * np.exp(-30 * t) + 40
    kick = 0.6 * np.sin(2 * np.pi * freq * t) * np.linspace(1, 0, dur) ** 1.5
    end = i + dur
    if end < len(audio):
        audio[i:end] += kick

# Normalize
audio = audio / np.max(np.abs(audio)) * 0.85

# Save as WAV
audio_int = (audio * 32767).astype(np.int16)
with wave.open("/home/claude/demo/track.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    wf.writeframes(audio_int.tobytes())

print("✅ Audio generated")

# ── 2. VIDEO: animated visual with FFmpeg ─────────────────────────────────────
from PIL import Image, ImageDraw, ImageFont
import math

frames_dir = "/home/claude/demo/frames"
os.makedirs(frames_dir, exist_ok=True)

FPS = 30
TOTAL_FRAMES = DURATION * FPS
W, H = 1280, 720

for f in range(TOTAL_FRAMES):
    t = f / FPS
    img = Image.new("RGB", (W, H), (10, 10, 20))
    draw = ImageDraw.Draw(img)

    # Animated background gradient circles
    for layer in range(3):
        phase = t * (0.5 + layer * 0.3) + layer * 2.1
        cx = int(W/2 + math.sin(phase) * 200)
        cy = int(H/2 + math.cos(phase * 0.7) * 150)
        r = 180 + layer * 60
        color = [
            (20, 80, 180),
            (140, 20, 160),
            (20, 160, 120)
        ][layer]
        for dr in range(r, 0, -20):
            alpha = int(40 * (dr / r))
            c = tuple(min(255, v + alpha) for v in color)
            draw.ellipse([cx-dr, cy-dr, cx+dr, cy+dr], fill=c)

    # Waveform visualizer
    wave_points = []
    for x in range(0, W, 4):
        progress = x / W
        sample_idx = int(progress * SR * DURATION)
        if sample_idx < len(audio):
            amp = audio[sample_idx]
        else:
            amp = 0
        y = int(H/2 + amp * 200 * math.sin(t * 3 + progress * 10))
        wave_points.append((x, y))

    for i in range(len(wave_points) - 1):
        x1, y1 = wave_points[i]
        x2, y2 = wave_points[i+1]
        brightness = int(128 + 127 * math.sin(t * 4 + i * 0.1))
        draw.line([x1, y1, x2, y2], fill=(brightness, 255-brightness, 200), width=2)

    # Title text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
        small = font

    draw.text((W//2, 60), "Inferflow — AI Media Pipeline", fill=(255,255,255), font=font, anchor="mm")
    draw.text((W//2, 100), "Audio-Visual Generation Demo", fill=(180,180,255), font=small, anchor="mm")

    # Beat pulse ring
    beat_phase = (t * 2) % 1.0
    ring_r = int(30 + beat_phase * 80)
    ring_alpha = int(255 * (1 - beat_phase))
    draw.ellipse([W//2-ring_r, H//2-ring_r, W//2+ring_r, H//2+ring_r],
                 outline=(ring_alpha, ring_alpha, 255), width=3)

    img.save(f"{frames_dir}/frame_{f:05d}.png")

print("✅ Frames generated")

# ── 3. COMBINE with FFmpeg ────────────────────────────────────────────────────
subprocess.run([
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", f"{frames_dir}/frame_%05d.png",
    "-i", "/home/claude/demo/track.wav",
    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
    "-c:a", "aac", "-b:a", "192k",
    "-shortest", "-pix_fmt", "yuv420p",
    "/home/claude/demo/inferflow_demo.mp4"
], check=True, capture_output=True)

print("✅ Video compiled: inferflow_demo.mp4")
