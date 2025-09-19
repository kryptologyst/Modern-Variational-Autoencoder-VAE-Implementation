#!/usr/bin/env python3
"""Test script to verify installation and basic functionality."""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'tensorflow',
        'numpy', 
        'matplotlib',
        'sklearn',
        'PIL',
        'streamlit',
        'plotly',
        'seaborn',
        'pandas',
        'tqdm'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n🎉 All packages imported successfully!")
        return True

def test_tensorflow():
    """Test TensorFlow functionality."""
    try:
        import tensorflow as tf
        print(f"\n📊 TensorFlow version: {tf.__version__}")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🚀 GPU available: {len(gpus)} device(s)")
        else:
            print("💻 Using CPU (no GPU detected)")
        
        # Test basic tensor operations
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.matmul(x, x)
        print(f"✅ Basic tensor operations work")
        
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_project_structure():
    """Test if project files exist."""
    import os
    
    required_files = [
        'config.py',
        'models/vae.py',
        'data/data_loader.py',
        'utils/visualization.py',
        'train.py',
        'app.py',
        'README.md',
        'requirements.txt'
    ]
    
    print("\n📁 Checking project structure...")
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("\n🎉 All project files present!")
        return True

def main():
    """Run all tests."""
    print("🧪 VAE Project Installation Test")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("TensorFlow", test_tensorflow),
        ("Project Structure", test_project_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("📋 Test Summary:")
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! Your VAE project is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train a model: python3 train.py --dataset mnist --epochs 10")
        print("3. Launch UI: streamlit run app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
