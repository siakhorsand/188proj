"""
Test script to verify environment setup and dependencies
"""

import sys
import torch
import numpy as np
import gym
import matplotlib
import matplotlib.pyplot as plt
import cv2
from IPython.display import display
import jupyter_core
import tensorboard
import tqdm
import pandas as pd

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check MPS (Apple Silicon) availability
    print(f"MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("Apple Silicon GPU can be used for acceleration")
    
    print(f"NumPy version: {np.__version__}")
    print(f"Gym version: {gym.__version__}")
    print(f"Matplotlib version: {matplotlib.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Jupyter Core version: {jupyter_core.__version__}")
    print(f"Tensorboard version: {tensorboard.__version__}")
    print(f"tqdm version: {tqdm.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print("\nAll imports successful!")

def test_torch():
    """Test PyTorch functionality on both CPU and GPU"""
    print("\nTesting PyTorch...")
    
    # Test on CPU
    print("Testing CPU operations:")
    x_cpu = torch.randn(2, 2)
    y_cpu = torch.randn(2, 2)
    z_cpu = torch.matmul(x_cpu, y_cpu)
    print("PyTorch CPU matrix multiplication successful!")
    
    # Test on MPS if available
    if torch.backends.mps.is_available():
        print("\nTesting GPU (MPS) operations:")
        device = torch.device("mps")
        x_gpu = x_cpu.to(device)
        y_gpu = y_cpu.to(device)
        z_gpu = torch.matmul(x_gpu, y_gpu)
        print("PyTorch GPU (MPS) matrix multiplication successful!")
        print(f"Device used for GPU tensor: {z_gpu.device}")
        return z_cpu, z_gpu
    
    return z_cpu

def test_numpy():
    """Test NumPy functionality"""
    print("\nTesting NumPy...")
    x = np.random.randn(2, 2)
    y = np.random.randn(2, 2)
    z = np.dot(x, y)
    print("NumPy matrix multiplication successful!")
    return z

def test_matplotlib():
    """Test Matplotlib functionality"""
    print("\nTesting Matplotlib...")
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.title("Test Plot")
    plt.close()
    print("Matplotlib plotting successful!")

def test_opencv():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV...")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (0, 255, 0), -1)
    print("OpenCV image creation successful!")
    return img

def main():
    """Run all tests"""
    try:
        test_imports()
        test_torch()
        test_numpy()
        test_matplotlib()
        test_opencv()
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 