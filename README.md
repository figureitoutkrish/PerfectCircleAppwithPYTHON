# Perfect Circle 🟢

A gesture-controlled precision challenge where you draw a circle in the air using your index finger and get scored on how perfect it is. Built with MediaPipe, OpenCV, and Pygame.

---

## What It Does

- Uses your **webcam** to track your hand in real time
- Raise your **index finger** to start drawing
- Draw a circle around the center dot on screen
- Get scored on **shape accuracy** and **centering**
- Scores and player names saved to a local leaderboard

---

## Requirements

- Python **3.9 – 3.12** recommended (Python 3.13 may have issues with some MediaPipe versions)
- A working **webcam**
- **macOS, Windows, or Linux**

---

## Installation

### Step 1 — Make sure Python is installed

Open a terminal and run:

```bash
python --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/).

---

### Step 2 — (Recommended) Create a virtual environment

This keeps the project dependencies separate from your system Python.

```bash
python -m venv venv
```

Activate it:

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

---

### Step 3 — Install dependencies

> ⚠️ **Important:** Use `opencv-python-headless` and NOT `opencv-python`. On macOS especially, `opencv-python` bundles its own SDL2 which crashes when used alongside Pygame. The headless version avoids this.

```bash
pip install mediapipe opencv-python-headless pygame
```

If you already have `opencv-python` installed, remove it first:

```bash
pip uninstall opencv-python -y
pip install opencv-python-headless
```

---

### Step 4 — Run the game

```bash
python perfect_circle.py
```

The first time you run it, if you are on a newer version of MediaPipe it will automatically download a small hand landmark model file (~8 MB) into your system temp folder. This only happens once.

---

## How to Play

| Gesture | Action |
|---|---|
| ☝️ **Index finger only** | Start drawing |
| Draw in the air | Trace a circle around the green dot |
| Close the loop | Bring your finger back near the starting point to submit |
| ✌️ **Peace sign** | Play again / Cancel current drawing |
| 👎 **Thumb down** | Go back to the player name screen |

**Tips for a better score:**
- Draw slowly and steadily
- Try to keep the green dot inside your circle
- The more centered your circle is on the dot, the higher your centering score
- Good lighting makes a big difference — face a light source and avoid bright windows behind you

---

## Scoring System

Your final score is the product of two components:

**SHAPE** — How circular is your drawing?
- Radius consistency (are all points the same distance from the center?)
- Smoothness (is the path clean or jagged?)
- Closure (did you close the loop?)
- Fullness (did you draw a complete 360°?)

**CENTERING** — Is your circle centered on the green dot?
- Measured as the distance between your circle's center and the dot
- The further off-center, the more your shape score is multiplied down

```
Final Score = Shape Score × Centering Multiplier
```

### Tiers

| Score | Tier |
|---|---|
| 90 – 100% | Perfect Circle 🥇 |
| 78 – 89% | Legendary Circle |
| 65 – 77% | Master Circle |
| 50 – 64% | Great Circle |
| 30 – 49% | Decent Attempt |
| 0 – 29% | Keep Practicing |

---

## Files

```
perfect_circle.py          — main game file
perfect_circle_scores.json — leaderboard (auto-created on first score)
```

---

## Troubleshooting

**App crashes on startup with SDL2 errors**
You have `opencv-python` installed alongside Pygame. Fix:
```bash
pip uninstall opencv-python -y
pip install opencv-python-headless
```

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
Your MediaPipe version is 0.10+. The code handles this automatically, but make sure you have an internet connection on the first run so it can download the model file.

**Webcam not detected**
Make sure no other app (Zoom, FaceTime, etc.) is using the camera. On macOS, you may need to grant Terminal or your IDE camera permissions in System Settings → Privacy & Security → Camera.

**Hand not being detected**
- Improve lighting — face a lamp or window in front of you
- Keep your hand within arm's reach of the camera
- Make sure your background isn't too busy or dark

**Low FPS (below 20)**
- Close other applications
- Try reducing camera resolution by changing `1280` and `720` in `perfect_circle.py` to `640` and `480`

---

## Dependencies

| Package | Purpose |
|---|---|
| `mediapipe` | Hand landmark detection |
| `opencv-python-headless` | Webcam capture and frame processing |
| `pygame` | Window, rendering, and input |

---

*Made for IDSA subject project.*
