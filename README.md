# CricketLytics — AI-Powered Cricket Batting Analyser

A computer vision tool that analyses cricket batting technique from recorded video. It uses **Google MediaPipe** for real-time pose detection and **OpenCV** for video processing to measure joint angles, classify shots, track the ball, and give targeted posture feedback — all overlaid directly on the video frame.

## Features

- **Pose Estimation**: Detects 33 body landmarks per frame using MediaPipe Holistic, drawing a full skeletal overlay on the player.
- **Joint Angle Measurement**: Computes and displays live angles for the front knee, back knee, top-hand elbow, and body lean (shoulder–hip–knee).
- **Shot Classification**: Automatically identifies the batting shot being played (Drive, Pull/Hook, Sweep, Cut, Forward Defensive, Back Foot Punch) using a rule-based classifier smoothed over a 20-frame window.
- **Targeted Posture Coaching**: Select the shot you are attempting via `--shot` flag and receive specific, biomechanically-grounded feedback (e.g. "Drive: extend front leg — push into the ball").
- **Ball Tracking**: Detects a green cricket ball in the frame using HSV color masking and contour detection, marking it with a labelled circle.
- **Stance Width Analysis**: Measures horizontal pixel distance between both feet and between the leading hand and leading foot.
- **Left & Right Handed Support**: Analyses mirrored landmark sets for both batting orientations.
- **Front & Side View Support**: Works with both front-facing and side-profile camera angles.

---

## System Architecture

High-level overview of the CricketLytics pipeline:

```mermaid
graph TB
    subgraph "Input"
        Video[Recorded Video File<br/>MP4 / AVI]
        CLI[CLI Arguments<br/>option · hand · view · shot]
    end

    subgraph "Core - Cricket.py"
        Dispatch[Cricket Dispatcher<br/>Routes to correct class]
    end

    subgraph "Batting Analysis"
        LHB[LeftHandedBatting<br/>Right-body landmarks]
        RHB[RightHandedBatting<br/>Left-body landmarks]
    end

    subgraph "Computer Vision - OpenCV + MediaPipe"
        MP[MediaPipe Holistic<br/>33-point Pose Detection]
        Angles[Joint Angle Calculator<br/>Knee · Elbow · Body Lean]
        Ball[Ball Tracker<br/>HSV Color Masking]
        Dist[Distance Measurer<br/>Foot spread · Hand reach]
    end

    subgraph "Analysis Modules"
        SC[ShotClassifier<br/>Rule-based · 20-frame window]
        PA[PostureAdvisor<br/>Generic or Shot-specific]
        OV[Overlay Renderer<br/>Banner · Feedback Panel]
    end

    subgraph "Output"
        Window[OpenCV Display Window<br/>Annotated Video Feed]
    end

    CLI --> Dispatch
    Video --> Dispatch
    Dispatch --> LHB
    Dispatch --> RHB
    LHB --> MP
    RHB --> MP
    MP --> Angles
    MP --> Ball
    MP --> Dist
    Angles --> SC
    Angles --> PA
    SC --> OV
    PA --> OV
    Ball --> OV
    Dist --> OV
    OV --> Window

    style Video fill:#60a5fa,stroke:#2563eb,stroke-width:2px,color:#fff
    style CLI fill:#60a5fa,stroke:#2563eb,stroke-width:2px,color:#fff
    style MP fill:#a78bfa,stroke:#7c3aed,stroke-width:3px,color:#fff
    style SC fill:#34d399,stroke:#059669,stroke-width:2px,color:#fff
    style PA fill:#34d399,stroke:#059669,stroke-width:2px,color:#fff
    style OV fill:#fbbf24,stroke:#f59e0b,stroke-width:2px,color:#000
    style Window fill:#f87171,stroke:#dc2626,stroke-width:2px,color:#fff
```

---

## Analysis Pipelines

### 1. Pose & Joint Angle Pipeline

How each video frame is processed to extract biomechanical measurements:

```mermaid
graph TB
    Frame[Raw Video Frame] --> RGB[Convert BGR → RGB]
    RGB --> Holistic[MediaPipe Holistic<br/>min_confidence = 0.5]
    Holistic --> Landmarks[33 Pose Landmarks<br/>Normalised x · y · z]
    Landmarks --> Filter[Visibility Filter<br/>threshold = 0.5]
    Filter --> Coords[idx_to_coordinates<br/>Pixel coordinate map]

    Coords --> Knee[Front Knee Angle<br/>Hip → Knee → Ankle]
    Coords --> Elbow[Elbow Angle<br/>Wrist → Elbow → Shoulder]
    Coords --> Body[Body Lean<br/>Shoulder → Hip → Knee]
    Coords --> FeetSpread[Stance Width<br/>Left foot ↔ Right foot px]
    Coords --> HandReach[Hand Reach<br/>Leading hand ↔ Leading foot px]

    Knee --> AngleDict[angles_data dict]
    Elbow --> AngleDict
    Body --> AngleDict
    FeetSpread --> PosDict[positions_data dict]
    HandReach --> PosDict

    style Frame fill:#60a5fa,stroke:#2563eb,stroke-width:2px,color:#fff
    style Holistic fill:#a78bfa,stroke:#7c3aed,stroke-width:3px,color:#fff
    style AngleDict fill:#fbbf24,stroke:#f59e0b,stroke-width:2px,color:#000
    style PosDict fill:#fbbf24,stroke:#f59e0b,stroke-width:2px,color:#000
```

### 2. Shot Classification Pipeline

Rule-based classifier with temporal smoothing:

```mermaid
graph TB
    AngleDict[angles_data · positions_data] --> Rules

    subgraph Rules[Rule Engine - ordered priority]
        R1{front_knee < 125°?} -->|Yes| Sweep[Sweep Shot]
        R1 -->|No| R2
        R2{hand above shoulder<br/>+ back_knee > 148°?} -->|Yes| Pull[Pull / Hook Shot]
        R2 -->|No| R3
        R3{back_knee > 155°<br/>+ hand above shoulder?} -->|Yes| Cut[Cut Shot]
        R3 -->|No| R4
        R4{front_knee > 158°<br/>+ elbow > 125°?} -->|Yes| DriveCheck{hand above<br/>shoulder?}
        DriveCheck -->|Yes| Lofted[Lofted Drive]
        DriveCheck -->|No| Drive[Drive]
        R4 -->|No| R5
        R5{front_knee > 148°<br/>+ elbow < 120°?} -->|Yes| Def[Forward Defensive]
        R5 -->|No| R6
        R6{back_knee > 158°<br/>+ front_knee > 152°?} -->|Yes| BP[Back Foot Punch]
        R6 -->|No| Stance[Stance / Ready]
    end

    Sweep --> Deque[20-Frame Rolling Window<br/>deque maxlen=20]
    Pull --> Deque
    Cut --> Deque
    Lofted --> Deque
    Drive --> Deque
    Def --> Deque
    BP --> Deque
    Stance --> Deque

    Deque --> Counter[Counter.most_common]
    Counter --> Smoothed[Smoothed Shot Label]

    style AngleDict fill:#fbbf24,stroke:#f59e0b,stroke-width:2px,color:#000
    style Smoothed fill:#34d399,stroke:#059669,stroke-width:3px,color:#fff
    style Deque fill:#60a5fa,stroke:#2563eb,stroke-width:2px,color:#fff
```

### 3. Posture Coaching Pipeline

Two modes — generic auto-analysis, or targeted shot-specific coaching:

```mermaid
graph TB
    Input[angles_data · positions_data<br/>detected_shot · intended_shot] --> Mode{intended_shot<br/>== auto?}

    Mode -->|Yes - Auto| Generic[Generic Checks]
    Mode -->|No - Shot selected| Targeted[Shot-Specific Coaching]

    subgraph Generic[Generic Mode]
        G1[front_knee > 172° → Bend front knee more]
        G2[back_knee > 172° → Bend back knee]
        G3[elbow < 78° → Raise top elbow]
        G4[foot_spread / shoulder_width < 0.75 → Widen stance]
    end

    subgraph Targeted[Targeted Mode by Shot]
        Drive[Drive:<br/>Extend front leg · Raise elbow · Follow through]
        Def[Defensive:<br/>Bend front knee · Compact elbow · Keep hands low]
        Pull[Pull / Hook:<br/>Hands above shoulder · Pivot on back foot · Elbows up]
        Sweep[Sweep:<br/>Get lower · Extend back leg · Keep bat horizontal]
        Cut[Cut:<br/>Weight on back foot · Step back and across]
        BP[Back Foot Punch:<br/>Plant back foot · Punch through · Stay upright]
    end

    Generic --> Tips[Feedback List]
    Targeted --> Tips

    Tips --> Overlay[Overlay Panel]

    style Input fill:#fbbf24,stroke:#f59e0b,stroke-width:2px,color:#000
    style Tips fill:#34d399,stroke:#059669,stroke-width:2px,color:#fff
    style Overlay fill:#a78bfa,stroke:#7c3aed,stroke-width:2px,color:#fff
```

### 4. Ball Tracking Pipeline

Detects a green cricket ball using HSV color segmentation:

```mermaid
graph TB
    Frame[Video Frame BGR] --> HSV[Convert to HSV Colorspace]
    HSV --> Mask[Color Mask<br/>H:29-64 · S:86-255 · V:6-255]
    Mask --> Morph[Morphological Opening<br/>Remove noise]
    Morph --> Dilate[Dilate<br/>Fill gaps]
    Dilate --> Contours[Find Contours]
    Contours --> Check{Any contours<br/>found?}
    Check -->|No| Skip[Skip frame]
    Check -->|Yes| Largest[Select largest contour<br/>by area]
    Largest --> Circle[Minimum Enclosing Circle]
    Circle --> RadCheck{radius > 10px?}
    RadCheck -->|No| Skip
    RadCheck -->|Yes| Draw[Draw yellow circle + red centroid<br/>+ Ball label]

    style Frame fill:#60a5fa,stroke:#2563eb,stroke-width:2px,color:#fff
    style Draw fill:#34d399,stroke:#059669,stroke-width:3px,color:#fff
    style Skip fill:#f87171,stroke:#dc2626,stroke-width:2px,color:#fff
```

---

## MediaPipe Landmark Reference

Key landmark indices used by the analyser:

| Index | Landmark | Used For |
|-------|----------|----------|
| `0` | Nose | Player label position |
| `11` | Left Shoulder | RHB elbow angle, body lean |
| `12` | Right Shoulder | LHB elbow angle, shoulder width |
| `13` | Left Elbow | RHB elbow angle |
| `14` | Right Elbow | LHB elbow angle |
| `15` | Left Wrist | RHB elbow angle, hand height |
| `16` | Right Wrist | LHB elbow angle, hand height |
| `23` | Left Hip | RHB knee + body lean |
| `24` | Right Hip | LHB knee + body lean |
| `25` | Left Knee | RHB front knee angle |
| `26` | Right Knee | LHB front knee angle |
| `27` | Left Ankle | RHB knee angle |
| `28` | Right Ankle | LHB knee angle |
| `29` | Left Foot Index | Stance width, hand reach |
| `30` | Right Foot Index | Stance width, hand reach |

> **RHB** = Right-Handed Batsman (uses left-body landmarks)  
> **LHB** = Left-Handed Batsman (uses right-body landmarks, mirrored)

---

## Shot Coaching Reference

| Shot | `--shot` value | Key Requirements |
|------|----------------|-----------------|
| Drive | `drive` | Front knee extends (>155°), elbow raised (>100°), follow-through above shoulder |
| Forward Defensive | `defensive` | Front knee bent (140–165°), elbow compact (<120°), hands low |
| Pull / Hook | `pull` | Both hands above shoulder, back foot pivot, elbows up |
| Sweep | `sweep` | Front knee deeply bent (<135°), back leg extended, bat horizontal |
| Cut Shot | `cut` | Weight on back foot (>148°), step back-and-across, hands high |
| Back Foot Punch | `back-punch` | Back foot planted firmly, elbow extension through contact |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Pose Estimation | MediaPipe ≥0.10.30 (`mp.tasks` Tasks API · `PoseLandmarker`) |
| Video Processing | OpenCV ≥4.8 (`cv2`) |
| Numerical Ops | NumPy ≥1.24 |
| Shot Smoothing | Python `collections.deque` + `Counter` |
| Ball Detection | HSV masking + contour analysis (OpenCV) |
| Angle Geometry | Custom dot-product math (`utils.py` + `headless_analyzer.py`) |
| Web API | FastAPI ≥0.110 + Uvicorn |
| Dashboard | Vanilla HTML / CSS / JavaScript + Chart.js |

---

## Project Structure

```
sports-el-cricket-analysis/
├── Cricket.py                      # CLI entry point — argument parsing & dispatch
├── requirements.txt                # All dependencies
├── pose_landmarker_lite.task       # MediaPipe pose model (download separately)
│
├── api/                            # FastAPI web backend
│   ├── __init__.py
│   └── server.py                   # POST /api/analyze — video upload → JSON stats
│
├── dashboard/                      # Futuristic web dashboard (served by FastAPI)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
└── src/
    ├── utils.py                    # Geometry helpers: ang(), draw_ellipse(), convert_arc()
    ├── ThreadedCamera.py           # Legacy threaded camera reader (unused)
    ├── Analysis/
    │   ├── shot_classifier.py      # Rule-based shot detector with 20-frame smoothing
    │   ├── posture_advisor.py      # Generic + shot-specific posture coaching
    │   ├── headless_analyzer.py    # Headless pipeline → aggregated JSON stats (API)
    │   └── overlay.py              # Draws shot banner + feedback panel onto frames
    ├── Batting/
    │   ├── Batting.py              # Abstract base: bat(view) → front/side dispatch
    │   ├── LeftHandedBatting.py
    │   └── RightHandedBatting.py
    └── Bowling/
        ├── Bowling.py
        ├── LeftHandedBowling.py
        └── RightHandedBowling.py
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- A recorded cricket video file (MP4 or AVI)
- For ball tracking: the ball in the video should be **green** (HSV colour mask assumes a green training ball)

### 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install "mediapipe>=0.10.30" "opencv-python>=4.8.0" "numpy>=1.24.0" \
            "fastapi>=0.110.0" "uvicorn[standard]>=0.29.0" "python-multipart>=0.0.9"
```

### 2 — Download the Pose Model

MediaPipe ≥0.10.30 uses the Tasks API and requires a `.task` model file.
Run this once from the repo root:

```bash
curl -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task \
     -o pose_landmarker_lite.task
```

---

## Usage

### Option A — CLI (OpenCV display window)

```bash
# Basic auto-detection
python Cricket.py --option batting --hand right --view front --video path/to/video.mp4

# With intended shot coaching
python Cricket.py --option batting --hand right --view front \
  --video path/to/video.mp4 --shot drive

python Cricket.py --option batting --hand left --view side \
  --video path/to/video.mp4 --shot sweep
```

#### All CLI Arguments

| Argument | Required | Values | Description |
|----------|----------|--------|-------------|
| `--option` | ✅ | `batting` · `bowling` | Type of play to analyse |
| `--hand` | ✅ | `left` · `right` | Batting/bowling hand |
| `--view` | ✅ | `front` · `side` | Camera angle |
| `--video` | ✅ | File path | Path to the recorded MP4/AVI |
| `--shot` | ❌ | `auto` · `drive` · `defensive` · `pull` · `sweep` · `cut` · `back-punch` | Intended shot (default: `auto`) |

#### Keyboard Controls

| Key | Action |
|-----|--------|
| `ESC` | Exit the analyser |
| `P` | Pause / unpause playback |

---

### Option B — Web Dashboard (FastAPI + Browser UI)

A futuristic dark-mode dashboard that accepts a video upload and renders
aggregated stats — shot distribution chart, joint angle gauges, angle
timeline, and posture coaching tips.

**Start the server:**

```bash
# From the repo root
uvicorn api.server:app --port 8090 --reload
```

**Open your browser at:** `http://127.0.0.1:8090`

**Dashboard features:**
- 📤 Drag-and-drop video upload (MP4 / AVI)
- ⚙️ Configure hand, view, and intended shot
- ⏳ Live loading animation with step-by-step progress
- 🥧 Shot distribution donut chart (Chart.js)
- 📐 Average joint angle gauges with ideal-range colour coding
- 📈 Angle timeline chart across sampled frames
- 💡 Posture coaching tip cards
- 🏆 Technique score ring gauge

**API endpoint (JSON):**

```bash
curl -X POST http://127.0.0.1:8090/api/analyze \
  -F "video=@path/to/video.mp4" \
  -F "hand=right" \
  -F "view=front" \
  -F "intended_shot=drive"
```

Returns:

```json
{
  "total_frames": 450,
  "frames_analyzed": 390,
  "shot_distribution": { "Drive": 180, "Forward Defensive": 120, ... },
  "dominant_shot": "Drive",
  "average_angles": { "front_knee": 162.4, "front_elbow": 118.7 },
  "angle_timeline": [...],
  "coaching_tips": ["Drive: extend front leg — push into the ball", ...],
  "technique_score": 74
}
```

---

## On-Screen Display

```
┌──────────────────────────────────────────────────┐
│  Intended: Drive          Detected: Forward Def  │  ← coloured shot banner
├──────────────────────────────────────────────────┤
│                                                  │
│    [skeleton overlay + angle labels]             │
│    [yellow ball circle labelled "Ball"]          │
│    [yellow foot-to-hand measurement line]        │
│    [red foot-to-foot measurement line]           │
│                                                  │
├──────────────────────────────────────────────────┤
│  Drive Tips                                      │  ← feedback panel
│    Extend front leg — push into the ball         │
│    Follow through — hands above shoulder         │
└──────────────────────────────────────────────────┘
```

**Colour coding on skeleton:**
- 🔴 **Red lines** — joint angle guide lines
- 🟢 **Green text** — angle values in degrees
- 🟡 **Yellow lines** — distance measurements
- 🟡 **Yellow circle** — detected ball boundary
- 🔴 **Red dot** — ball centroid
