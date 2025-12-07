"""
ASL Hand Gesture Detection - Prediction Script
Team 3 - AAI521 Applied Computer Vision in AI

This script loads a trained model and makes predictions on new images.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import json
from PIL import Image


class ASLPredictor:
    """Class for making ASL gesture predictions"""
    
    def __init__(self, model_path='asl_gesture_model.keras'):
        self.model_path = model_path
        self.model = None
        self.num_classes = 24
        
        # Create label map for ASL letters (excluding J and Z)
        self.label_map = {i: chr(65+i+(i>=9)) for i in range(24)}
        self.class_names = [self.label_map[i] for i in range(24)]
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            print(f"Model not found at {self.model_path}")
            print("Please train the model first using train_asl_model.py")
            return False
            
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        return True
    
    def preprocess_image(self, image):
        """
        Preprocess an image for prediction
        
        Args:
            image: Can be a file path, numpy array, or PIL Image
        
        Returns:
            Preprocessed image ready for prediction
        """
        # Handle different input types
        if isinstance(image, str):
            # File path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
                
        elif isinstance(image, Image.Image):
            # PIL Image
            image = np.array(image)
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            elif image.shape[2] == 3:  # RGB/BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        # Resize to 28x28
        if image.shape[:2] != (28, 28):
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Reshape for model input
        image = image.reshape(28, 28, 1)
        
        return image
    
    def predict(self, image, top_k=3):
        """
        Make prediction on a single image
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            print("Model not loaded. Please load a model first.")
            return None
            
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Add batch dimension
        image_batch = np.expand_dims(preprocessed, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image_batch, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'top_prediction': self.class_names[top_indices[0]],
            'confidence': float(predictions[top_indices[0]]),
            'top_k_predictions': [
                {
                    'letter': self.class_names[idx],
                    'confidence': float(predictions[idx])
                }
                for idx in top_indices
            ],
            'all_predictions': {
                self.class_names[i]: float(predictions[i])
                for i in range(self.num_classes)
            }
        }
        
        return results
    
    def predict_batch(self, images):
        """
        Make predictions on multiple images
        
        Args:
            images: List of images (file paths, numpy arrays, or PIL Images)
            
        Returns:
            List of prediction results
        """
        results = []
        for i, image in enumerate(images):
            try:
                result = self.predict(image)
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                results.append({'image_index': i, 'error': str(e)})
        return results
    
    def visualize_prediction(self, image, save_path=None):
        """
        Visualize prediction with confidence scores
        
        Args:
            image: Input image
            save_path: Optional path to save the visualization
        """
        # Get prediction
        result = self.predict(image, top_k=5)
        if result is None:
            return
            
        # Preprocess image for display
        display_image = self.preprocess_image(image)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display image
        axes[0].imshow(display_image.reshape(28, 28), cmap='gray')
        axes[0].set_title(f"Predicted: {result['top_prediction']} "
                         f"({result['confidence']:.2%})", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Top 5 predictions bar chart
        top_5 = result['top_k_predictions'][:5]
        letters = [p['letter'] for p in top_5]
        confidences = [p['confidence'] for p in top_5]
        
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(letters))]
        axes[1].barh(letters[::-1], confidences[::-1], color=colors[::-1])
        axes[1].set_xlabel('Confidence', fontsize=11)
        axes[1].set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1])
        
        for i, (letter, conf) in enumerate(zip(letters[::-1], confidences[::-1])):
            axes[1].text(conf + 0.01, i, f'{conf:.2%}', va='center', fontsize=9)
        
        # All predictions heatmap
        all_probs = [result['all_predictions'][letter] for letter in self.class_names]
        axes[2].bar(self.class_names, all_probs, color='purple', alpha=0.7)
        axes[2].set_xlabel('ASL Letter', fontsize=11)
        axes[2].set_ylabel('Confidence', fontsize=11)
        axes[2].set_title('All Class Probabilities', fontsize=12, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim([0, max(all_probs) * 1.1])
        
        plt.suptitle('ASL Gesture Prediction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
        
        return result
    
    def run_webcam_demo(self, camera_index=0):
        """
        Run real-time ASL gesture detection from webcam
        
        Args:
            camera_index: Camera index (0 for default camera)
        """
        if self.model is None:
            print("Model not loaded. Please load a model first.")
            return
            
        print("Starting webcam demo...")
        print("Position your hand in the green box")
        print("Press 'q' to quit, 's' to save current prediction")
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Define ROI (Region of Interest) in center
            roi_size = min(height, width) // 2
            roi_x = width // 2 - roi_size // 2
            roi_y = height // 2 - roi_size // 2
            
            # Draw ROI rectangle
            cv2.rectangle(frame,
                         (roi_x, roi_y),
                         (roi_x + roi_size, roi_y + roi_size),
                         (0, 255, 0), 2)
            
            # Extract ROI
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Make prediction
            try:
                result = self.predict(roi)
                
                if result:
                    # Display prediction
                    text = f"Prediction: {result['top_prediction']}"
                    confidence_text = f"Confidence: {result['confidence']:.2%}"
                    
                    cv2.putText(frame, text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, confidence_text, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Display top 3 predictions
                    y_offset = 100
                    for i, pred in enumerate(result['top_k_predictions'][:3]):
                        text = f"{i+1}. {pred['letter']}: {pred['confidence']:.1%}"
                        cv2.putText(frame, text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y_offset += 25
                        
            except Exception as e:
                cv2.putText(frame, f"Error: {str(e)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('ASL Gesture Detection - Real Time', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current prediction
                filename = f'capture_{saved_count}.png'
                cv2.imwrite(filename, roi)
                print(f"Saved capture to {filename}")
                saved_count += 1
                
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam demo ended")


def demo_single_image():
    """Demo function for single image prediction"""
    predictor = ASLPredictor()
    
    # Create a sample image for demonstration
    sample_image = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    
    print("\\n" + "="*50)
    print("SINGLE IMAGE PREDICTION DEMO")
    print("="*50)
    
    # Make prediction
    result = predictor.predict(sample_image)
    
    if result:
        print(f"\\nTop Prediction: {result['top_prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\\nTop 3 Predictions:")
        for i, pred in enumerate(result['top_k_predictions'][:3], 1):
            print(f"  {i}. {pred['letter']}: {pred['confidence']:.2%}")
        
        # Visualize
        predictor.visualize_prediction(sample_image, save_path='prediction_demo.png')


def demo_batch_prediction():
    """Demo function for batch prediction"""
    predictor = ASLPredictor()
    
    print("\\n" + "="*50)
    print("BATCH PREDICTION DEMO")
    print("="*50)
    
    # Create sample batch
    batch_images = [np.random.randint(0, 256, (28, 28), dtype=np.uint8) for _ in range(5)]
    
    # Make predictions
    results = predictor.predict_batch(batch_images)
    
    print(f"\\nProcessed {len(batch_images)} images:")
    for result in results:
        if 'error' not in result:
            print(f"  Image {result['image_index']}: {result['top_prediction']} "
                  f"({result['confidence']:.2%})")
        else:
            print(f"  Image {result['image_index']}: Error - {result['error']}")


def main():
    """Main function"""
    print("="*60)
    print("ASL HAND GESTURE DETECTION - PREDICTION MODULE")
    print("Team 3 - AAI521 Applied Computer Vision in AI")
    print("="*60)
    
    # Create predictor
    predictor = ASLPredictor()
    
    if predictor.model is None:
        print("\\nNo trained model found. Creating demo with synthetic model...")
        # For demo purposes, create a simple model if none exists
        from tensorflow.keras import layers, models
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(24, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.save('asl_gesture_model.keras')
        predictor = ASLPredictor()
    
    # Run demos
    demo_single_image()
    demo_batch_prediction()
    
    # Uncomment to run webcam demo
    # predictor.run_webcam_demo()
    
    print("\\n" + "="*60)
    print("PREDICTION MODULE DEMO COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
