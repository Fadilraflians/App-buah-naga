"""
Script untuk testing API endpoint
Jalankan dengan: python test_api.py
"""

import requests
import os

API_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("🔍 Testing /api/health...")
    response = requests.get(f"{API_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_predict_vgg16(image_path):
    """Test VGG16 prediction endpoint"""
    print(f"🔵 Testing /api/predict/vgg16 dengan {image_path}...")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/predict/vgg16", files=files)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%\n")
    else:
        print(f"Error: {response.text}\n")

def test_predict_mobilenetv2(image_path):
    """Test MobileNetV2 prediction endpoint"""
    print(f"🟢 Testing /api/predict/mobilenetv2 dengan {image_path}...")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/predict/mobilenetv2", files=files)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Model: {result['model']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%\n")
    else:
        print(f"Error: {response.text}\n")

def test_predict_both(image_path):
    """Test both models prediction endpoint"""
    print(f"🔄 Testing /api/predict/both dengan {image_path}...")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{API_URL}/api/predict/both", files=files)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        if result.get('vgg16'):
            print(f"VGG16: {result['vgg16']['prediction']} ({result['vgg16']['confidence']:.2f}%)")
        if result.get('mobilenetv2'):
            print(f"MobileNetV2: {result['mobilenetv2']['prediction']} ({result['mobilenetv2']['confidence']:.2f}%)\n")
    else:
        print(f"Error: {response.text}\n")

if __name__ == "__main__":
    print("=" * 50)
    print("API Testing untuk Dragon Fruit Classification")
    print("=" * 50)
    print()
    
    # Test health check
    test_health()
    
    # Test dengan gambar (ganti path sesuai file Anda)
    test_image = "test_image.jpg"  # Ganti dengan path gambar Anda
    
    if os.path.exists(test_image):
        test_predict_vgg16(test_image)
        test_predict_mobilenetv2(test_image)
        test_predict_both(test_image)
    else:
        print(f"⚠️ File {test_image} tidak ditemukan. Skip testing prediksi.")
        print("Untuk test prediksi, siapkan file gambar dan update path di test_api.py")

