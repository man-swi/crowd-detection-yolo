# Crowd Detection System using YOLOv8

This project is a **real-time crowd detection system** that processes a video to identify groups of people based on spatial proximity and temporal persistence. Using the **YOLOv8** object detection model, it tracks groups of individuals across frames and flags a crowd when certain criteria are met.

## 📌 Features

- 🎯 Person detection using YOLOv8.
- 👥 Grouping people based on distance between their centroids.
- ⏱️ Tracks group consistency over time using the Hungarian algorithm.
- 🚨 Flags a **crowd** when:
  - At least 3 people are together,
  - They are within 120 pixels of each other,
  - They persist for 10+ consecutive frames.
- 📹 Visual output with bounding boxes and group IDs.
- 📁 Logs detected crowd events into a CSV file.
- 💾 Saves the annotated output video.

---

## Project Structure

```

crowd-detection/
│
├── crowd_detector.py                  # Main script with all functionality
├── yolov8s.pt               # YOLOv8 pre-trained model (downloaded automatically if not present)
├── output\_video\_crowd\_detection.mp4  # (Generated) Output video with annotations
├── crowd\_log.csv            # (Generated) CSV with crowd event logs
└── README.md                # Project documentation (this file)

````

---

## 🔧 Requirements

- Python 3.7+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV
- NumPy
- SciPy

Install dependencies:

```bash
pip install -r requirements.txt
````

> Alternatively:

```bash
pip install ultralytics opencv-python numpy scipy
```

---

## How It Works

### Step-by-Step:

1. **Load YOLOv8 model**

   * Detects only persons (`class_id = 0`).
2. **Read frames from video**

   * Video is processed frame-by-frame.
3. **Detect people using YOLO**

   * Bounding boxes of people are extracted.
4. **Form groups**

   * People within `MAX_DISTANCE_WITHIN_GROUP = 120` pixels are grouped using BFS on adjacency matrix.
5. **Track groups**

   * Groups are tracked across frames using centroid distance + Hungarian matching.
6. **Detect crowds**

   * A group is flagged as a crowd if it persists for `MIN_CONSECUTIVE_FRAMES_FOR_CROWD = 10` frames.
7. **Log and visualize**

   * Each crowd event is logged in `crowd_log.csv`.
   * Bounding boxes and metadata are drawn on the video output.

---

## Parameters

| Parameter                          | Description                                        | Value    |
| ---------------------------------- | -------------------------------------------------- | -------- |
| `MIN_PERSONS_IN_GROUP`             | Minimum people to form a group                     | `3`      |
| `MAX_DISTANCE_WITHIN_GROUP`        | Max pixel distance between people to be grouped    | `120` px |
| `MIN_CONSECUTIVE_FRAMES_FOR_CROWD` | Frames required for group to be considered a crowd | `10`     |
| `MAX_DISTANCE_GROUP_TRACKING`      | Max centroid move allowed between frames           | `300` px |
| `DETECTION_CONFIDENCE`             | YOLO confidence threshold                          | `0.25`   |

All parameters can be modified at the top of `main.py`.

---

##  Output

### 📹 Video Output

Annotated video is saved as:

```
output_video_crowd_detection.mp4
```

### 📄 CSV Log

Each row in `crowd_log.csv` contains:

```
[Frame Number], [Number of People], [Group ID]
```

---

## Visual Legend

* 🟩 Green Box: Individual person
* 🔲 Colored Box: Group member
* 🟡 "CROWD": Label shown for groups that meet crowd criteria
* 🆔 Group ID, People Count, Frame Count shown near group centroid

---

## Running the Code

Make sure to specify your input video path:

```python
VIDEO_PATH = r'D:\AI\Crowd detection\production_id_4196258 (720p).mp4'
```

Then run:

```bash
python main.py
```

> Press `q` during visualization to quit early.

---

## Model Note

The script uses the `yolov8s.pt` model (YOLOv8 Small). If not available, it will attempt to download it automatically via Ultralytics.

---

## Use Cases

* Crowd management & surveillance
* Public event safety analysis
* Smart city monitoring
* Social distancing enforcement

---

## Future Improvements

* Export JSON for deeper analytics
* Real-time camera support
* Web interface
* Multi-class crowd analysis (e.g., kids, adults)

---

## ⭐️ Show Your Support

If you found this helpful, consider giving a ⭐️ on GitHub!
