#!/usr/bin/env python3
"""
Standalone script for ASL Gesture Detection Webcam Demo

This script is recommended for webcam demos as it handles camera resource
cleanup more reliably than notebook cells, especially on macOS.

Usage:
    python run_webcam_demo.py

Press 'q' in the OpenCV window to quit.
"""

import cv2
import numpy as np
import sys
import os
from contextlib import contextmanager

# Add the project directory to path to import model loading functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("Error: TensorFlow not installed. Please install it with: pip install tensorflow")
    sys.exit(1)


@contextmanager
def video_capture(camera_index=0):
    """
    Context manager for video capture that ensures proper cleanup.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise IOError(f"Cannot open webcam at index {camera_index}")
        yield cap
    finally:
        if cap is not None:
            try:
                cap.release()
            except:
                pass
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        import time
        time.sleep(0.1)  # Small delay for macOS to process release


def preprocess_frame(frame):
    """Preprocess webcam frame for model prediction."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28))
    # Normalize
    normalized = resized.astype('float32') / 255.0
    # Reshape for model
    preprocessed = normalized.reshape(28, 28, 1)
    return preprocessed


def predict_gesture(model, image, class_names, top_k=3):
    """
    Predict ASL gesture from an image.
    
    Args:
        model: Trained model
        image: Input image (28x28 grayscale or will be resized)
        class_names: List of class names
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and confidence scores
    """
    # Ensure image is the right shape
    if image.shape != (28, 28, 1):
        if len(image.shape) == 2:
            image = image.reshape(28, 28, 1)
        elif image.shape[:2] != (28, 28):
            image = cv2.resize(image, (28, 28))
            if len(image.shape) == 2:
                image = image.reshape(28, 28, 1)
    
    # Normalize if needed
    if image.max() > 1:
        image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image, axis=0)
    
    # Get predictions
    predictions = model.predict(image_batch, verbose=0)[0]
    
    # Get top k predictions
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    results = {
        'top_prediction': class_names[top_indices[0]],
        'confidence': float(predictions[top_indices[0]]),
        'top_k_predictions': [
            {'letter': class_names[idx], 'confidence': float(predictions[idx])}
            for idx in top_indices
        ]
    }
    
    return results


def load_model(model_path='asl_gesture_model_final.keras'):
    """Load the trained model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please train the model first or provide the correct path.")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def run_webcam_demo(model, class_names, camera_index=0):
    """
    Run real-time ASL gesture detection from webcam.
    """
    window_name = 'ASL Gesture Detection'
    
    try:
        print("="*60)
        print("ASL HAND GESTURE DETECTION - WEBCAM DEMO")
        print("="*60)
        print("Starting webcam demo...")
        print("Press 'q' in the OpenCV window to quit")
        print("Press Ctrl+C to interrupt")
        print("="*60)
        
        # Use context manager for reliable cleanup
        with video_capture(camera_index) as cap:
            print("Webcam opened successfully. Starting detection...\n")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Define ROI (Region of Interest)
                height, width = frame.shape[:2]
                roi_size = min(height, width) // 2
                roi_x = width // 2 - roi_size // 2
                roi_y = height // 2 - roi_size // 2
                
                # Draw ROI rectangle
                cv2.rectangle(frame, 
                             (roi_x, roi_y), 
                             (roi_x + roi_size, roi_y + roi_size), 
                             (0, 255, 0), 2)
                
                # Extract and preprocess ROI
                roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
                preprocessed = preprocess_frame(roi)
                
                # Make prediction (with error handling)
                try:
                    prediction = predict_gesture(model, preprocessed, class_names)
                    
                    # Display prediction on frame
                    cv2.putText(frame, 
                               f"Prediction: {prediction['top_prediction']}", 
                               (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                    
                    cv2.putText(frame, 
                               f"Confidence: {prediction['confidence']:.2%}", 
                               (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                    
                    # Print prediction every 30 frames (to reduce console spam)
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: {prediction['top_prediction']} ({prediction['confidence']:.2%})")
                    
                except Exception as e:
                    print(f"Prediction error: {e}")
                    cv2.putText(frame, "Prediction Error", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Exit on 'q' press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n'q' pressed - exiting...")
                    break
                
                frame_count += 1
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        print("Cleanup handled by context manager...")
    except IOError as e:
        print(f"\nError: {e}")
        print("Please check if your webcam is connected and not being used by another application.")
    except Exception as e:
        print(f"\nError during webcam demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Context manager handles cleanup, but ensure window is closed
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("Webcam demo ended. Camera released.")
        print("="*60)


def main():
    """Main function"""
    # Create label map for ASL letters (A-Y excluding J and Z)
    label_map = {i: chr(65+i+(i>=9)) for i in range(24)}
    class_names = [label_map[i] for i in range(24)]
    
    # Load model
    model = load_model()
    if model is None:
        print("\nTrying alternative model paths...")
        alt_paths = [
            'asl_gesture_model.keras',
            'asl_cnn_model_best.keras',
            'model.keras'
        ]
        for path in alt_paths:
            if os.path.exists(path):
                model = load_model(path)
                if model:
                    break
        
        if model is None:
            print("\nCould not find a trained model.")
            print("Please train the model first using the notebook or train_asl_model.py")
            sys.exit(1)
    
    # Run webcam demo
    run_webcam_demo(model, class_names, camera_index=0)


if __name__ == "__main__":
    main()

