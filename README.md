# Smart Exam Proctoring System

## Overview
Smart Exam Proctoring System is a Python desktop application designed to explore AI-assisted exam monitoring using computer vision. It analyzes video footage to track students and identify potentially suspicious behaviors during exams.

## Features
- **Student Detection and Tracking:** Tracks students across video frames using YOLO Pose Estimation and BoT-SORT.
- **Head Pose Analysis:** Analyzes head orientation and downward-looking behavior.
- **Eye and Gaze Analysis:** Uses facial landmarks to estimate gaze direction.
- **Behavior Baseline Calibration:** Establishes an initial baseline for each tracked student to help distinguish normal behavior from changes in posture or gaze.
- **Object Detection:** Detects potential phone and book presence using an object detection model.
- **Desktop Interface:** Provides a graphical interface built with Python and Tkinter.
- **Video-Based Monitoring:** Processes uploaded video footage and displays detection results.

## Technologies Used
- Python
- OpenCV
- Ultralytics YOLO
- YOLO Pose Estimation
- BoT-SORT Tracking
- MediaPipe Face Mesh
- NumPy
- Tkinter
- SQLite (if enabled in the application)

## Installation

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Smart-Exam-Proctoring-System
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
```

On Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure model weights
Ensure the required YOLO model weights are available in a location recognized by the application's model-loading code. Follow the relevant model documentation for obtaining the weights.

### 5. Run the application
```bash
python Smart_Exam_Proctoring_baseline.py
```

## Current Limitations
- Detection performance depends on video quality, camera angle, lighting, and visibility of students.
- False positives may occur when students naturally look down or move their heads.
- The system is a prototype and requires further evaluation on representative test videos.
- Alerts indicate potentially suspicious behavior; they do not establish that cheating occurred.

## Future Improvements
- Reduce false-positive detections.
- Improve detection of subtle body movements toward neighboring students.
- Evaluate performance using labeled test videos and suitable metrics.
- Improve model robustness under different camera angles and lighting conditions.

## Responsible Use
This project is intended for educational and research purposes. Any real-world deployment should include appropriate privacy safeguards, human review of alerts, and evaluation for reliability and fairness.

## Author
Shada Medhat

