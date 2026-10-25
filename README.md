# 🧘 AsanaSenseAI — Real-Time Yoga Posture Coach

**AsanaSenseAI** is an AI-powered yoga posture coaching application designed to detect yoga poses, analyze body alignment, compare joint angles with a selected reference pose, and provide real-time visual feedback.

The application combines computer vision, machine learning, and an interactive dashboard to help users understand and improve their yoga posture.

---

## ✨ Features

* **Real-Time Pose Detection:** Detects body landmarks using MediaPipe.
* **AI Pose Classification:** Predicts the yoga pose performed by the user using a trained machine learning model.
* **Reference Skeleton Overlay:** Displays a reference skeleton aligned with the user's detected body.
* **Posture Similarity Score:** Estimates similarity between the user's measured joint angles and the selected reference pose.
* **Joint Angle Analysis:** Evaluates supported joints and their angular deviations.
* **Personalized Feedback:** Provides suggestions based on detected posture differences.
* **Multiple Target Poses:** Allows users to select different yoga poses for comparison.
* **Interactive Dashboard:** Displays AI confidence, posture score, joint analysis, and score history.
* **Responsive Interface:** Provides a modern dashboard designed for different screen sizes.

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Chart.js

### Backend

* Python
* Flask
* OpenCV

### Artificial Intelligence & Computer Vision

* MediaPipe
* NumPy
* Pandas
* Scikit-learn
* Joblib

---

## 🧘 Supported Yoga Poses

The current training dataset contains the following seven pose classes:

1. Bound Angle Pose
2. Cobra Pose
3. Downward-Facing Dog
4. Goddess Pose
5. Plank Pose
6. Tree Pose
7. Warrior II Pose

The available reference poses depend on the labels and valid samples in the dataset.

---

## 📂 Project Structure

```text
AsanaSenseAI/
│
├── app.py
├── collect_data.py
├── train_model.py
├── test_webcam.py
│
├── src/
│   ├── __init__.py
│   ├── angle_analyzer.py
│   ├── feature_extraction.py
│   ├── feedback_engine.py
│   ├── pose_classifier.py
│   ├── pose_detection.py
│   ├── reference_pose.py
│   └── skeleton_alignment.py
│
├── models/
│   ├── pose_classifier.joblib
│   └── pose_landmarker_lite.task
│
├── data/
│   └── train_landmarks.csv
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── dashboard.js
│
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*This structure describes the current intended project organization. Keep the file list synchronized with the files actually present in your repository.*

---

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the application:

* Python 3.11
* Git
* A webcam
* A modern web browser

Python 3.11 is the recommended environment for this project.

### 1. Clone the Repository


```bash
git clone https://github.com/KhushiKoranga/AsanaSenseAI.git
cd AsanaSenseAI
```

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv311
```

Activate it:

```powershell
.\.venv311\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult the Python documentation for environment activation options.

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root if your configuration requires one.

Example:

```env
CAMERA_INDEX=0
TARGET_POSE=Bound_Angle_Pose
HOST=127.0.0.1
PORT=5000
```

Keep secrets out of source control. Do not commit your private `.env` file.

### 5. Verify Required Files

Before starting the application, check that the following files are available:

* `models/pose_classifier.joblib`
* `models/pose_landmarker_lite.task`
* `data/train_landmarks.csv`

The classifier model must be trained and compatible with the feature extraction code.

### 6. Run the Application

```bash
python app.py
```

Open the dashboard in your browser:

**http://127.0.0.1:5000**

Allow camera access when prompted if the application uses browser-based camera access.

---

## 🖥️ How to Use

1. Start the Flask application.
2. Open the dashboard in your browser.
3. Select a target yoga pose from the dropdown.
4. Position yourself so that the required body landmarks are visible.
5. Perform the selected yoga pose.
6. Observe the AI's predicted pose and confidence.
7. Review the posture similarity score and joint analysis.
8. Follow the available coaching suggestions.
9. Observe the posture score history as the session progresses.

For more reliable measurements, use suitable lighting and position the camera so that the relevant body joints remain visible.

---

## 🧠 How It Works

### 1. Body Landmark Detection

MediaPipe detects body landmarks from the camera feed.

### 2. Pose Classification

The extracted landmark features are passed to a trained machine learning classifier, which predicts the yoga pose.

### 3. Reference Pose Selection

The selected target pose determines which reference samples are used for comparison.

### 4. Joint Angle Calculation

The application calculates supported joint angles from detected landmarks.

### 5. Posture Similarity Estimation

The measured joint angles are compared with the reference angles to estimate posture similarity.

### 6. Visual Feedback

The dashboard presents the predicted pose, AI confidence, posture score, joint deviations, and coaching suggestions.

---

## 📊 Understanding the Scores

### AI Confidence

Indicates the classifier's confidence in its predicted pose. It does not measure how accurately the user is performing the selected target pose.

### Posture Similarity

Estimates how closely the measured joint angles match the selected reference pose.

### Joint Deviation

Represents the angular difference between a measured joint angle and its reference value.

**Important:** These metrics are experimental estimates. They are not a certified measure of yoga technique or physical safety.

---

## 🔮 Future Improvements

Potential future improvements include:

* Expanding the yoga pose dataset.
* Evaluating classification performance on unseen webcam recordings.
* Improving reference pose alignment.
* Refining joint-specific feedback.
* Improving camera framing and landmark visibility handling.
* Adding session summaries and performance tracking.
* Deploying a publicly accessible version of the application.
* Optimizing inference speed and resource usage.

---

## 🚀 Deployment

The application currently runs locally using Flask and a connected camera.

Public deployment requires additional configuration. In particular, a cloud server cannot automatically access the webcam attached to a visitor's computer.

To support live webcam coaching on a public website, the camera pipeline must be designed to capture frames in the visitor's browser and process them through a suitable browser-based or backend inference workflow.

---

## ⚠️ Limitations and Disclaimer

* Pose detection depends on lighting, camera position, visibility, and framing.
* Classification confidence is not the same as posture similarity.
* Similarity scores depend on reference-data quality and the joints available for comparison.
* The system may produce incorrect classifications or feedback.
* The application is an experimental educational project, not a medical device.
* It does not replace guidance from a qualified yoga instructor or healthcare professional.

Perform exercises within your comfort level and stop if you experience pain or dizziness.

---

## 👩‍💻 Author

**Project:** AsanaSenseAI — Real-Time Yoga Posture Coach

Developed as an AI and computer vision project combining machine learning, pose detection, and an interactive web dashboard.

---

## 📄 License

Choose and add a license before distributing the project publicly. If no license is specified, others do not automatically receive permission to reuse the code.
