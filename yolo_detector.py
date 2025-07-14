import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import argparse
import os

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt', confidence=0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path (str): Path to YOLO model weights
            confidence (float): Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        
    def detect_image(self, image_path, save_path=None, show=True):
        """
        Detect objects in a single image
        
        Args:
            image_path (str): Path to input image
            save_path (str): Path to save output image (optional)
            show (bool): Whether to display the result
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
            
        # Run inference
        results = self.model(image, conf=self.confidence)
        
        # Visualize results
        annotated_image = results[0].plot()
        
        if save_path:
            cv2.imwrite(save_path, annotated_image)
            print(f"Result saved to: {save_path}")
            
        if show:
            cv2.imshow('YOLO Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return results[0]
    
    def detect_video(self, video_path, save_path=None, show=True):
        """
        Detect objects in a video
        
        Args:
            video_path (str): Path to input video
            save_path (str): Path to save output video (optional)
            show (bool): Whether to display the result
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer for saving output
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            results = self.model(frame, conf=self.confidence)
            
            # Visualize results
            annotated_frame = results[0].plot()
            
            if save_path:
                out.write(annotated_frame)
                
            if show:
                cv2.imshow('YOLO Video Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
        
        cap.release()
        if save_path:
            out.release()
            print(f"Video saved to: {save_path}")
        cv2.destroyAllWindows()
        
    def detect_webcam(self, camera_index=0, save_path=None):
        """
        Real-time detection using webcam
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            save_path (str): Path to save output video (optional)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
            
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Video writer for saving output
        if save_path:
            fps = 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference
            results = self.model(frame, conf=self.confidence)
            
            # Visualize results
            annotated_frame = results[0].plot()
            
            if save_path:
                out.write(annotated_frame)
                
            cv2.imshow('YOLO Webcam Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if save_path:
            out.release()
            print(f"Video saved to: {save_path}")
        cv2.destroyAllWindows()
        
    def get_detection_info(self, results):
        """
        Extract detection information from results
        
        Args:
            results: YOLO detection results
            
        Returns:
            list: List of detection dictionaries
        """
        detections = []
        
        for box in results.boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': self.model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            }
            detections.append(detection)
            
        return detections

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Detection')
    parser.add_argument('--model', default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--source', required=True, help='Source: image path, video path, or "webcam"')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--save', help='Path to save output')
    parser.add_argument('--no-show', action='store_true', help='Don\'t display results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = YOLODetector(model_path=args.model, confidence=args.confidence)
    
    if args.source == 'webcam':
        detector.detect_webcam(save_path=args.save)
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        detector.detect_video(args.source, save_path=args.save, show=not args.no_show)
    else:
        # Assume it's an image
        results = detector.detect_image(args.source, save_path=args.save, show=not args.no_show)
        
        if results:
            detections = detector.get_detection_info(results)
            print(f"\nDetected {len(detections)} objects:")
            for i, det in enumerate(detections):
                print(f"{i+1}. {det['class_name']}: {det['confidence']:.2f}")

if __name__ == "__main__":
    main()
