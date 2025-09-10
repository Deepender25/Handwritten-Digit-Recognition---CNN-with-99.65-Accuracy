#!/usr/bin/env python3
"""
Environment Check Script for MNIST Digit Recognition Project

This script checks if all required packages are installed and working correctly.
Run this before starting the main notebook to ensure everything is set up properly.
"""

import sys
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required. Please upgrade your Python version.")
        return False
    else:
        print("✅ Python version is compatible.")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            print(f"❌ {package_name} is not installed.")
            return False
        else:
            # Try to actually import it
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name} ({version}) is installed and working.")
            return True
    except ImportError as e:
        print(f"❌ {package_name} import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name} installed but there might be an issue: {e}")
        return False

def check_tensorflow_gpu():
    """Check if TensorFlow can access GPU"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            print(f"🚀 GPU support available! Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            print("💻 No GPU detected. Training will use CPU (slower but still works).")
        return True
    except Exception as e:
        print(f"⚠️  Could not check GPU status: {e}")
        return False

def main():
    """Main function to run all checks"""
    print("🔍 Checking Python ML Environment for MNIST Project")
    print("=" * 55)
    
    all_good = True
    
    # Check Python version
    all_good &= check_python_version()
    print()
    
    # Required packages
    packages = [
        ('tensorflow', 'tensorflow'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('scikit-learn', 'sklearn'),
        ('jupyter', 'jupyter'),
        ('pillow', 'PIL'),
        ('opencv-python', 'cv2'),
    ]
    
    print("📦 Checking required packages:")
    for package_name, import_name in packages:
        all_good &= check_package(package_name, import_name)
    
    print()
    
    # Check TensorFlow GPU support
    print("🎮 Checking GPU support:")
    check_tensorflow_gpu()
    
    print()
    print("=" * 55)
    
    if all_good:
        print("🎉 All checks passed! Your environment is ready for the MNIST project.")
        print("\n📝 Next steps:")
        print("   1. Start Jupyter: jupyter notebook")
        print("   2. Open: mnist_digit_recognition.ipynb")
        print("   3. Run all cells to train your model!")
    else:
        print("❌ Some packages are missing or have issues.")
        print("\n🔧 To fix missing packages, run:")
        print("   pip install -r requirements.txt")
        print("\n   Then run this script again to verify the installation.")
    
    return all_good

if __name__ == "__main__":
    main()
