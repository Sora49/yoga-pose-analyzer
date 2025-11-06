# Yoga Pose Analyzer 🧘‍♀️

An AI-powered yoga pose analyzer that uses computer vision to detect and correct yoga poses.

## Features

- **Pose Detection**: Uses YOLOv8 for accurate pose keypoint detection
- **Pose Classification**: Custom trained CNN model for yoga pose recognition
- **Pose Correction**: Real-time feedback comparing your pose with reference poses
- **Interactive Interface**: Easy-to-use Streamlit web application

## Supported Poses

- Downward Dog
- Warrior II
- Tree Pose
- Plank
- Goddess Pose

## How to Use

1. Upload a clear image of your yoga pose
2. Click "Analyze Pose" to get AI predictions
3. Select the direction/variation
4. Compare with reference poses to get corrections

## Technology Stack

- **Frontend**: Streamlit
- **Pose Detection**: YOLOv8 (Ultralytics)
- **Pose Classification**: TensorFlow/Keras CNN
- **Image Processing**: OpenCV
- **Visualization**: Matplotlib

## Live Demo

[🚀 Try the App](your-streamlit-url-here)

## Local Setup

```bash
pip install -r requirements.txt
streamlit run yoga_pose_analyzer.py
```

---
*Built with ❤️ for yoga enthusiasts*"# yoga-pose-analyzer" 
