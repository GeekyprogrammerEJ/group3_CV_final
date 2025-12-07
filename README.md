# ASL Hand Gesture Detection with Deep Learning and Computer Vision

## Team 3 - AAI521 Applied Computer Vision in AI
**University of San Diego - Shiley-Marcos School of Engineering**

### Team Members
- **Yogesh Sangwikar** - Dataset preprocessing, EDA, Model evaluation
- **Evin Joy** - CNN architecture design, Loss/accuracy visualization
- **Eesha Kulkarni** - Model training, hyperparameter tuning, Real-time integration

### GitHub Repository
[https://github.com/EK77-mslabs/AAI521-Applied-CV-in-AI-Group-Project.git](https://github.com/EK77-mslabs/AAI521-Applied-CV-in-AI-Group-Project.git)

---

## 🎯 Project Overview

This project applies computer vision and deep learning techniques to recognize static American Sign Language (ASL) hand gestures representing alphabets A–Y (excluding J and Z). Our objective is to build a real-time gesture classification model to support accessibility-focused applications and enhance human-computer interaction.

### Why This Matters
- **Accessibility**: Bridges communication gaps for individuals who rely on sign language
- **Inclusion**: Promotes inclusive technology and equal access to digital services
- **Applications**: Can be integrated into assistive technologies, education tools, and communication apps

---

## 📊 Dataset

**Dataset**: Sign Language MNIST  
**Source**: [Kaggle - Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)

### Dataset Specifications
- **Training Images**: 27,455
- **Test Images**: 7,172
- **Image Dimensions**: 28×28 pixels (grayscale)
- **Format**: CSV files (784 pixel values + label per row)
- **Classes**: 24 ASL letters (A–Y, excluding J and Z)
- **Size**: ~100 MB total

### Why J and Z are Excluded
Letters J and Z require motion/dynamic gestures (tracing shapes in the air) and cannot be captured in static images. Future work could extend to video-based models for these letters.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- CUDA-compatible GPU (optional but recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/EK77-mslabs/AAI521-Applied-CV-in-AI-Group-Project.git
cd AAI521-Applied-CV-in-AI-Group-Project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**

Option A: Using Kaggle API
```bash
kaggle datasets download -d datamunge/sign-language-mnist
unzip sign-language-mnist.zip
```

Option B: Manual download
- Visit [Kaggle dataset page](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Download and extract the CSV files to the project directory

---

## 📁 Project Structure

```
ASL-Gesture-Detection/
│
├── data/
│   ├── sign_mnist_train.csv
│   └── sign_mnist_test.csv
│
├── models/
│   ├── asl_gesture_model.keras
│   └── asl_gesture_model_saved/
│
├── logs/
│   └── tensorboard_logs/
│
├── notebooks/
│   └── asl_gesture_detection_project.ipynb
│
├── scripts/
│   ├── train_asl_model.py
│   └── predict_asl_gesture.py
│
├── results/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── training_results.json
│
├── requirements.txt
└── README.md
```

---

## 🔧 Usage

### Option 1: Jupyter Notebook (Recommended for Exploration)
```bash
jupyter notebook asl_gesture_detection_project.ipynb
```
The notebook contains the complete pipeline with visualizations and step-by-step explanations.

### Option 2: Python Scripts

**Training the Model**
```bash
python train_asl_model.py
```

**Making Predictions**
```bash
python predict_asl_gesture.py
```

**Real-time Webcam Demo**
```python
from predict_asl_gesture import ASLPredictor

predictor = ASLPredictor('asl_gesture_model.keras')
predictor.run_webcam_demo()
```

---

## 🏗️ Model Architecture

### CNN Architecture Details
```
Input (28×28×1)
    ↓
Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    ↓
Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    ↓
Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
    ↓
Flatten
    ↓
Dense(256) → BN → Dropout(0.5)
    ↓
Dense(128) → BN → Dropout(0.5)
    ↓
Output: Dense(24, softmax)
```

### Key Features
- **Batch Normalization**: Improves training stability and speed
- **Dropout Regularization**: Prevents overfitting
- **Data Augmentation**: Rotation, shifting, zooming for better generalization
- **Learning Rate Scheduling**: Adaptive learning rate for optimal convergence

---

## 📈 Performance Metrics

### Expected Results (after 50 epochs)
- **Test Accuracy**: ~95-98%
- **Top-3 Accuracy**: ~99%
- **Training Time**: ~10-15 minutes on GPU, ~45-60 minutes on CPU

### Model Evaluation
The model is evaluated using:
- Confusion Matrix
- Per-class Accuracy Analysis
- Classification Report (Precision, Recall, F1-score)
- Misclassification Analysis

---

## 🔬 Technical Implementation

### Data Preprocessing
```python
# Reshape flattened pixels to images
images = pixels.reshape(-1, 28, 28, 1)

# Normalize pixel values
images = images.astype('float32') / 255.0

# One-hot encode labels
labels = to_categorical(labels, num_classes=24)
```

### Data Augmentation
```python
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)
```

### Training Callbacks
- **ModelCheckpoint**: Save best model based on validation accuracy
- **EarlyStopping**: Stop training when validation loss plateaus
- **ReduceLROnPlateau**: Reduce learning rate when metrics stop improving
- **TensorBoard**: Log training metrics for visualization

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

**Issue**: Out of memory error on GPU
```python
# Reduce batch size
BATCH_SIZE = 64  # Instead of 128
```

**Issue**: Slow training on CPU
```python
# Use Google Colab with GPU runtime
# Runtime → Change runtime type → GPU
```

**Issue**: Poor accuracy on certain letters
- Check confusion matrix for commonly confused pairs
- Increase training data for problematic classes
- Fine-tune model on misclassified samples

---

## 📊 Visualizations

The project includes comprehensive visualizations:
- Training/Validation curves
- Confusion Matrix
- Per-class accuracy analysis
- Sample predictions
- Misclassification analysis

---

## 🚧 Roadblocks & Solutions

1. **Limited GPU Runtime on Colab**
   - Solution: Use checkpoints to resume training
   - Implement efficient batch processing

2. **Dynamic Gestures (J, Z)**
   - Current limitation of static image dataset
   - Future work: Implement video-based classification

3. **Class Imbalance**
   - Solution: Use weighted loss or data augmentation
   - Monitor per-class metrics

---

## 🔮 Future Enhancements

1. **Expand to Dynamic Gestures**
   - Implement LSTM/GRU for letters J and Z
   - Use video sequences for motion capture

2. **Transfer Learning**
   - Use pre-trained models (MobileNet, EfficientNet)
   - Improve accuracy and reduce training time

3. **Mobile Deployment**
   - Convert to TensorFlow Lite
   - Create mobile app for real-time translation

4. **Word-Level Recognition**
   - Extend to recognize complete words
   - Implement sequence-to-sequence models

5. **Multi-Language Support**
   - Include other sign languages (BSL, ISL)
   - Create universal sign language translator

---

## 📚 References

1. [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
2. [TensorFlow Documentation](https://www.tensorflow.org/)
3. [Keras API Reference](https://keras.io/)
4. [OpenCV Documentation](https://opencv.org/)
5. [ASL Alphabet Reference](https://www.nidcd.nih.gov/health/american-sign-language)

---

## 📝 License

This project is developed for educational purposes as part of AAI521 course at University of San Diego.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

---

## 📧 Contact

For questions or collaboration:
- **Yogesh Sangwikar** - [email]
- **Evin Joy** - [email]
- **Eesha Kulkarni** - [email]

---

## 🙏 Acknowledgments

- University of San Diego, Shiley-Marcos School of Engineering
- Course Instructor for AAI521 Applied Computer Vision in AI
- Kaggle community for the dataset
- TensorFlow/Keras development team

---

**Project Status**: ✅ Active Development

*Last Updated: December 2024*
