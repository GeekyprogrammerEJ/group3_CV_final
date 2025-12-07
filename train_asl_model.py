"""
ASL Hand Gesture Detection - Training Script
Team 3 - AAI521 Applied Computer Vision in AI

This script trains a CNN model for ASL letter classification.
Dataset: Sign Language MNIST (Kaggle)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ASLGestureClassifier:
    """Main class for ASL Gesture Classification"""
    
    def __init__(self, train_path='sign_mnist_train.csv', test_path='sign_mnist_test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.model = None
        self.history = None
        self.num_classes = 24
        self.input_shape = (28, 28, 1)
        self.batch_size = 128
        self.epochs = 50
        
        # Create label map for ASL letters (excluding J and Z)
        self.label_map = {i: chr(65+i+(i>=9)) for i in range(24)}
        self.class_names = [self.label_map[i] for i in range(24)]
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        
        # Check if files exist, otherwise create synthetic data for demo
        if not os.path.exists(self.train_path):
            print("Dataset files not found. Creating synthetic data for demonstration...")
            self.X_train, self.y_train = self._create_synthetic_data(27455)
            self.X_test, self.y_test = self._create_synthetic_data(7172)
        else:
            # Load actual data
            train_data = pd.read_csv(self.train_path)
            test_data = pd.read_csv(self.test_path)
            
            # Preprocess
            self.X_train, self.y_train, _ = self._preprocess_data(train_data)
            self.X_test, self.y_test, _ = self._preprocess_data(test_data)
        
        # Create validation split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.15, random_state=42
        )
        
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Validation samples: {self.X_val.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        
    def _create_synthetic_data(self, n_samples):
        """Create synthetic data for demonstration"""
        labels = np.random.randint(0, self.num_classes, n_samples)
        pixels = np.random.randint(0, 256, (n_samples, 784))
        
        # Reshape and normalize
        images = pixels.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        labels_encoded = to_categorical(labels, num_classes=self.num_classes)
        
        return images, labels_encoded
    
    def _preprocess_data(self, data):
        """Preprocess the sign language MNIST data"""
        labels = data['label'].values
        pixels = data.drop('label', axis=1).values
        
        # Reshape and normalize
        images = pixels.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        labels_encoded = to_categorical(labels, num_classes=self.num_classes)
        
        return images, labels_encoded, labels
    
    def create_model(self):
        """Create the CNN model architecture"""
        print("\\nCreating CNN model...")
        
        self.model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=self.input_shape, name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1_2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2_2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3_2'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            # Dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(256, activation='relu', name='fc1'),
            layers.BatchNormalization(name='bn_fc1'),
            layers.Dropout(0.5, name='dropout_fc1'),
            
            layers.Dense(128, activation='relu', name='fc2'),
            layers.BatchNormalization(name='bn_fc2'),
            layers.Dropout(0.5, name='dropout_fc2'),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ], name='ASL_CNN')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        print(f"Model created with {self.model.count_params():,} parameters")
        
    def setup_data_augmentation(self):
        """Setup data augmentation for training"""
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        self.datagen.fit(self.X_train)
        print("Data augmentation configured")
        
    def create_callbacks(self):
        """Create training callbacks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return [
            keras.callbacks.ModelCheckpoint(
                filepath=f'asl_model_best_{timestamp}.keras',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=f'./logs/asl_{timestamp}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
    
    def train(self):
        """Train the model"""
        print("\\nStarting training...")
        
        # Setup
        self.setup_data_augmentation()
        callbacks = self.create_callbacks()
        
        # Train
        self.history = self.model.fit(
            self.datagen.flow(self.X_train, self.y_train, batch_size=self.batch_size),
            steps_per_epoch=len(self.X_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\\nTraining completed!")
        
    def evaluate(self):
        """Evaluate the model on test set"""
        print("\\nEvaluating model on test set...")
        
        # Get test metrics
        test_loss, test_accuracy, test_top3_accuracy = self.model.evaluate(
            self.X_test, self.y_test, 
            batch_size=self.batch_size,
            verbose=1
        )
        
        # Get predictions for detailed analysis
        y_pred_probs = self.model.predict(self.X_test, batch_size=self.batch_size)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Print results
        print("\\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f} ({test_top3_accuracy*100:.2f}%)")
        
        # Classification report
        print("\\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, digits=4))
        
        # Save results
        self._save_results(test_loss, test_accuracy, test_top3_accuracy)
        
        return y_true, y_pred
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Top-3 Accuracy
        axes[2].plot(self.history.history['top_3_accuracy'], label='Training Top-3')
        axes[2].plot(self.history.history['val_top_3_accuracy'], label='Validation Top-3')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Top-3 Accuracy')
        axes[2].set_title('Top-3 Accuracy')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.suptitle('Training Performance Metrics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix - ASL Gesture Classification', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def save_model(self, filepath='asl_gesture_model.keras'):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Also save in TensorFlow SavedModel format
        # In Keras 3, use model.export() for SavedModel format
        save_path = filepath.replace('.keras', '_saved')
        self.model.export(save_path)
        print(f"Model saved in SavedModel format at {save_path}/")
        
    def _save_results(self, test_loss, test_accuracy, test_top3_accuracy):
        """Save training results to JSON"""
        results = {
            'model_architecture': 'CNN with 3 convolutional blocks',
            'total_parameters': int(self.model.count_params()),
            'training_samples': int(self.X_train.shape[0]),
            'validation_samples': int(self.X_val.shape[0]),
            'test_samples': int(self.X_test.shape[0]),
            'number_of_classes': self.num_classes,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'test_top3_accuracy': float(test_top3_accuracy),
            'training_epochs': len(self.history.history['loss']) if self.history else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('training_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("Results saved to training_results.json")


def main():
    """Main training pipeline"""
    print("="*60)
    print("ASL HAND GESTURE DETECTION - TRAINING PIPELINE")
    print("Team 3 - AAI521 Applied Computer Vision in AI")
    print("="*60)
    
    # Initialize classifier
    classifier = ASLGestureClassifier()
    
    # Load data
    classifier.load_data()
    
    # Create model
    classifier.create_model()
    
    # Train model
    classifier.train()
    
    # Evaluate model
    y_true, y_pred = classifier.evaluate()
    
    # Visualizations
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(y_true, y_pred)
    
    # Save model
    classifier.save_model()
    
    print("\\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
