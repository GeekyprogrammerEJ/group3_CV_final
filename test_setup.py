"""
ASL Hand Gesture Detection - Quick Test Script
Team 3 - AAI521 Applied Computer Vision in AI

This script performs quick tests to verify the project setup.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

def check_imports():
    """Check if all required libraries are installed"""
    print("\n" + "="*50)
    print("CHECKING REQUIRED LIBRARIES")
    print("="*50)
    
    libraries = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'cv2': 'OpenCV',
        'tensorflow': 'TensorFlow',
        'sklearn': 'Scikit-learn'
    }
    
    missing = []
    
    for lib, name in libraries.items():
        try:
            __import__(lib)
            print(f"✓ {name:20} - Installed")
        except ImportError:
            print(f"✗ {name:20} - Missing")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing libraries: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required libraries are installed!")
        return True

def check_tensorflow():
    """Check TensorFlow configuration"""
    print("\n" + "="*50)
    print("TENSORFLOW CONFIGURATION")
    print("="*50)
    
    try:
        import tensorflow as tf
        print(f"TensorFlow Version: {tf.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU Available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("⚠️  No GPU detected - Training will be slower")
            print("   Consider using Google Colab with GPU runtime")
        
        # Check if eager execution is enabled
        print(f"Eager Execution: {'Enabled' if tf.executing_eagerly() else 'Disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking TensorFlow: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("\n" + "="*50)
    print("TESTING DATA LOADING")
    print("="*50)
    
    try:
        import numpy as np
        import pandas as pd
        
        # Check if actual dataset exists
        if os.path.exists('sign_mnist_train.csv'):
            print("✅ Dataset files found!")
            train_data = pd.read_csv('sign_mnist_train.csv')
            print(f"   Training samples: {len(train_data):,}")
            print(f"   Features: {train_data.shape[1] - 1} pixels + 1 label")
        else:
            print("⚠️  Dataset files not found")
            print("   Creating synthetic data for testing...")
            
            # Create small synthetic dataset
            n_samples = 100
            pixels = np.random.randint(0, 256, (n_samples, 784))
            labels = np.random.randint(0, 24, n_samples)
            
            data = pd.DataFrame(pixels)
            data.insert(0, 'label', labels)
            
            print(f"   Created synthetic data: {data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("\n" + "="*50)
    print("TESTING MODEL CREATION")
    print("="*50)
    
    try:
        from tensorflow.keras import layers, models
        
        # Create a simple test model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(24, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model created successfully!")
        print(f"   Total parameters: {model.count_params():,}")
        print(f"   Input shape: (28, 28, 1)")
        print(f"   Output shape: (24,)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("\n" + "="*50)
    print("TESTING PREDICTION")
    print("="*50)
    
    try:
        import numpy as np
        from tensorflow.keras import layers, models
        
        # Create and compile a simple model
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(24, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Create a test image
        test_image = np.random.randn(1, 28, 28, 1)
        
        # Make prediction
        prediction = model.predict(test_image, verbose=0)
        predicted_class = np.argmax(prediction)
        
        # Map to ASL letter
        label_map = {i: chr(65+i+(i>=9)) for i in range(24)}
        predicted_letter = label_map[predicted_class]
        
        print("✅ Prediction test successful!")
        print(f"   Test image shape: (28, 28, 1)")
        print(f"   Predicted letter: {predicted_letter}")
        print(f"   Confidence: {prediction[0, predicted_class]:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction test: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("ASL HAND GESTURE DETECTION - SYSTEM CHECK")
    print("Team 3 - AAI521 Applied Computer Vision in AI")
    print("="*60)
    
    tests = [
        ("Library Import", check_imports),
        ("TensorFlow Setup", check_tensorflow),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Prediction Test", test_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} : {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your environment is ready.")
        print("\nNext steps:")
        print("1. Download the dataset from Kaggle")
        print("2. Run: python train_asl_model.py")
        print("3. Or open: asl_gesture_detection_project.ipynb")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("1. Install missing libraries: pip install -r requirements.txt")
        print("2. For GPU support, use Google Colab or install CUDA")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
