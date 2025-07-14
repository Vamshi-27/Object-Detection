# YOLO Web Object Detection

A simple and modern web-based object detection application using YOLOv8. Upload images or use your webcam for real-time object detection through an intuitive web interface.

## 🌟 Features

- **🌐 Modern Web Interface**: Clean, responsive web UI with drag-and-drop functionality
- **⚡ Real-time Detection**: Webcam support with live streaming
- **🎯 Multiple Formats**: Support for images, videos, and webcam input
- **📊 Live Statistics**: Real-time detection statistics and confidence metrics
- **🎨 Model Switching**: Support for all YOLOv8 variants (n/s/m/l/x)
- **📱 Mobile Friendly**: Works on desktop and mobile devices

## 🚀 Quick Start

### Start the Web Interface
```bash
python start_web.py
```

Then open your browser to `http://localhost:5000`

### Alternative Start Method
```bash
python web_interface.py
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, for real-time detection)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Object-Detection
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv yolo_env
   
   # Windows
   yolo_env\Scripts\activate
   
   # macOS/Linux
   source yolo_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python start_web.py
   ```

## 🖥️ Web Interface Features

- **📁 Drag & Drop Upload**: Simply drag images or videos to the upload area
- **📹 Live Webcam Feed**: Real-time detection with webcam integration
- **⚙️ Model Settings**: Switch between YOLOv8 variants (nano, small, medium, large, xlarge)
- **📊 Live Statistics**: View detection count, confidence scores, and processing time
- **🎯 Interactive Results**: Click on detections for detailed information
- **📱 Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
Object-Detection/
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt          # Python dependencies
├── 📄 start_web.py             # Quick web interface launcher
├── 🔍 yolo_detector.py         # Core detection logic
├── 🌐 web_interface.py         # Flask web backend
├── 📁 templates/               # Web interface templates
│   └── 🌐 index.html          # Main web interface
├── 📁 static/results/          # Detection outputs
├── 📁 uploads/                 # Uploaded files
└── 🗂️ yolov8n.pt              # YOLO model file
```

## 🎯 Supported YOLO Models

- **YOLOv8n** (Nano): Fastest, smallest model
- **YOLOv8s** (Small): Balanced speed and accuracy  
- **YOLOv8m** (Medium): Good accuracy
- **YOLOv8l** (Large): High accuracy
- **YOLOv8x** (XLarge): Maximum accuracy

## 📊 Performance

| Model | Size | mAP | Speed (CPU) | Speed (GPU) |
|-------|------|-----|-------------|-------------|
| YOLOv8n | 6.2MB | 37.3 | ~45ms | ~1.2ms |
| YOLOv8s | 21.5MB | 44.9 | ~65ms | ~1.4ms |
| YOLOv8m | 49.7MB | 50.2 | ~95ms | ~2.1ms |
| YOLOv8l | 83.7MB | 52.9 | ~120ms | ~2.8ms |
| YOLOv8x | 136.7MB | 53.9 | ~160ms | ~3.5ms |

## 🛠️ Usage Tips

### Confidence Threshold
- **0.20-0.30**: Good balance for most objects
- **0.15-0.25**: For detecting smaller or distant objects
- **0.40-0.60**: For high-precision detections

### Model Selection
- Use **YOLOv8n** for fast detection
- Use **YOLOv8m** or **YOLOv8l** for better accuracy with cars and small objects
- Use **YOLOv8x** for maximum detection accuracy

## 📝 License

This project is licensed under the MIT License.

---

**Made with ❤️ by [Vamshi-27](https://github.com/Vamshi-27)**