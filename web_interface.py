"""
Flask Web Interface for YOLO Object Detection
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_socketio import SocketIO, emit
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename
from pathlib import Path
import threading
import time
from yolo_detector import YOLODetector
import io
from PIL import Image
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolo_detection_secret'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
detector = None
webcam_active = False
webcam_thread = None

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

def initialize_detector():
    """Initialize YOLO detector"""
    global detector
    try:
        detector = YOLODetector()
        print("✅ YOLO detector initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not detector:
            return jsonify({'error': 'Detector not initialized'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Update detector confidence
        detector.confidence = confidence
        
        # Process image
        results = detector.detect_image(filepath, show=False)
        
        if results:
            # Get detection info
            detections = detector.get_detection_info(results)
            
            # Save result image
            result_img = results[0].plot()
            result_filename = f"detected_{filename}"
            result_path = os.path.join('static/results', result_filename)
            cv2.imwrite(result_path, result_img)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'result_image': url_for('static', filename=f'results/{result_filename}'),
                'total_objects': len(detections)
            })
        else:
            return jsonify({
                'success': True,
                'detections': [],
                'message': 'No objects detected',
                'total_objects': 0
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        confidence = float(request.form.get('confidence', 0.5))
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not detector:
            return jsonify({'error': 'Detector not initialized'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Update detector confidence
        detector.confidence = confidence
        
        # Process video in background
        result_filename = f"detected_{filename}"
        result_path = os.path.join('static/results', result_filename)
        
        def process_video():
            try:
                detector.detect_video(filepath, save_path=result_path, show=False)
                socketio.emit('video_complete', {
                    'success': True,
                    'result_video': url_for('static', filename=f'results/{result_filename}'),
                    'message': 'Video processing completed successfully!'
                })
            except Exception as e:
                socketio.emit('video_complete', {
                    'success': False,
                    'error': str(e)
                })
        
        threading.Thread(target=process_video, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': 'Video processing started. You will be notified when complete.',
            'processing': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('start_webcam')
def handle_start_webcam(data):
    """Start webcam detection"""
    global webcam_active, webcam_thread
    
    print(f"🎥 Webcam start request received: {data}")
    
    if not detector:
        print("❌ Detector not initialized")
        emit('webcam_error', {'error': 'Detector not initialized'})
        return
    
    if webcam_active:
        print("⚠️ Webcam already active")
        emit('webcam_error', {'error': 'Webcam already active'})
        return
    
    # Test webcam access first
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("❌ Cannot access webcam")
        emit('webcam_error', {'error': 'Cannot access webcam. Please check if it is connected and not being used by another application.'})
        test_cap.release()
        return
    test_cap.release()
    
    confidence = data.get('confidence', 0.5)
    detector.confidence = confidence
    print(f"🎯 Setting confidence to: {confidence}")
    
    def webcam_loop():
        global webcam_active
        webcam_active = True
        cap = cv2.VideoCapture(0)
        
        # Set webcam properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            socketio.emit('webcam_error', {'error': 'Could not open webcam'})
            webcam_active = False
            return
        
        socketio.emit('webcam_started', {'message': 'Webcam started successfully'})
        
        try:
            while webcam_active:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame from webcam")
                    socketio.emit('webcam_error', {'error': 'Failed to read frame from webcam'})
                    break
                
                try:
                    # Run detection
                    results = detector.model(frame, conf=detector.confidence)
                    annotated_frame = results[0].plot()
                    
                    # Convert to base64 for web display
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Get detection info
                    detections = []
                    if results[0].boxes is not None:
                        for box in results[0].boxes:
                            detection = {
                                'class_name': detector.model.names[int(box.cls[0])],
                                'confidence': float(box.conf[0])
                            }
                            detections.append(detection)
                    
                    socketio.emit('webcam_frame', {
                        'image': img_base64,
                        'detections': detections,
                        'total_objects': len(detections)
                    })
                    
                except Exception as e:
                    print(f"Error in detection: {e}")
                    socketio.emit('webcam_error', {'error': f'Detection error: {str(e)}'})
                    break
                
                time.sleep(0.1)  # Limit FPS to ~10 FPS
                
        except Exception as e:
            print(f"Webcam loop error: {e}")
            socketio.emit('webcam_error', {'error': f'Webcam error: {str(e)}'})
        finally:
            cap.release()
            webcam_active = False
            socketio.emit('webcam_stopped', {'message': 'Webcam stopped'})
    
    try:
        webcam_thread = threading.Thread(target=webcam_loop, daemon=True)
        webcam_thread.start()
        print("🚀 Webcam thread started successfully")
    except Exception as e:
        print(f"❌ Failed to start webcam thread: {e}")
        emit('webcam_error', {'error': f'Failed to start webcam thread: {str(e)}'})

@socketio.on('stop_webcam')
def handle_stop_webcam():
    """Stop webcam detection"""
    global webcam_active
    print("🛑 Webcam stop request received")
    webcam_active = False
    emit('webcam_stopped', {'message': 'Webcam stop signal sent'})

@app.route('/change_model', methods=['POST'])
def change_model():
    """Change YOLO model"""
    try:
        global detector
        model_name = request.json.get('model', 'yolov8n.pt')
        confidence = request.json.get('confidence', 0.5)
        
        detector = YOLODetector(model_path=model_name, confidence=confidence)
        
        return jsonify({
            'success': True,
            'message': f'Successfully switched to {model_name}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Initializing YOLO Web Interface...")
    
    if initialize_detector():
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize detector. Please check your installation.")
