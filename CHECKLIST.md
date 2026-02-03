# ALAS AI/Perception - Week 1 Checklist

## Phase 1: Foundation Week

### Environment Setup & data_models.py

#### Jetson Nano Setup
```bash
# 1. Check CUDA & TensorRT
nvidia-smi
nvcc --version

# 2. Python environment
python3 --version  # Must be Python 3.8+
pip3 install --upgrade pip

# 3. Required libraries
pip3 install numpy opencv-python --break-system-packages
pip3 install onnxruntime-gpu --break-system-packages  # For Jetson
# OR (CPU fallback)
pip3 install onnxruntime --break-system-packages
```

**Checklist:**
- [ ] Jetson Nano CUDA is working
- [+] Git repo created
- [+] Folder structure ready
- [+] `src/data_models.py` added and tested
- [+] `data_models.py` shared with the team

---

### Model Preparation

#### Download and Test YOLOv8
```bash
# YOLOv8 library
pip3 install ultralytics --break-system-packages

# Test script
cd ~/alas_project
cat > test_yolo_download.py << 'EOF'
from ultralytics import YOLO
import cv2
import numpy as np

# Download model (auto-downloads on first run)
print("Downloading YOLOv8n model...")
model = YOLO('yolov8n.pt')

# Create test image
test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)

# Inference test
print("Running inference...")
results = model(test_img)

print("Model loaded successfully!")
print(f"   Speed: {results[0].speed}")
print(f"   Detections: {len(results[0].boxes)}")

# Check model file
import os
model_path = os.path.expanduser('~/.cache/ultralytics/yolov8n.pt')
if os.path.exists(model_path):
    print(f"Model file: {model_path}")
else:
    print("Model file not found!")
EOF

python3 test_yolo_download.py
```

#### ONNX Export
```bash
cat > export_onnx.py << 'EOF'
from ultralytics import YOLO

print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

print("Exporting to ONNX...")
model.export(
    format='onnx',
    simplify=True,  # ONNX simplification
    opset=12,       # ONNX opset version
)

print("Export complete!")
print("   Output: yolov8n.onnx")

# Move file to models/ folder
import shutil
shutil.move('yolov8n.onnx', 'models/yolov8n.onnx')
print("Moved to models/yolov8n.onnx")
EOF

python3 export_onnx.py
```

#### ONNX Runtime Test
```bash
cat > test_onnx.py << 'EOF'
import onnxruntime as ort
import numpy as np

# ONNX session
print("Loading ONNX model...")
session = ort.InferenceSession(
    'models/yolov8n.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

print("Model loaded!")
print(f"   Providers: {session.get_providers()}")

# Input/output info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"   Input: {input_name} - {input_shape}")

output_names = [out.name for out in session.get_outputs()]
print(f"   Outputs: {output_names}")

# Test inference
dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
print("\nRunning test inference...")
outputs = session.run(None, {input_name: dummy_input})

print("Inference successful!")
print(f"   Output shape: {outputs[0].shape}")
EOF

python3 test_onnx.py
```

**Checklist:**
- [ ] YOLOv8n model downloaded
- [ ] ONNX export successful
- [ ] `models/yolov8n.onnx` file exists
- [ ] ONNX Runtime test successful
- [ ] CUDA provider active (if available)

---

### Perceiver Implementation

#### Implement perceiver.py
```bash
# Place perception/perceiver.py file
cp /path/to/perceiver.py perception/

# Test
cd ~/alas_project
cat > test_perceiver.py << 'EOF'
from src.data_models import ModelConfig, generate_mock_frame
from perception.perceiver import EnvironmentPerceiver

# Config
config = ModelConfig(
    model_path='models/yolov8n.onnx',
    model_format='onnx',
    input_size=(640, 640),
    confidence_threshold=0.5,
    nms_threshold=0.4
)

# Perceiver
perceiver = EnvironmentPerceiver(config)

# Test 10 frames
print("Testing perceiver with mock frames...")
for i in range(10):
    frame = generate_mock_frame()
    frame.frame_id = i

    output = perceiver.perceive(frame)
    print(f"Frame {i}: {output.fps:.1f} FPS, "
          f"{len(output.obstacles)} obstacles")

# Stats
stats = perceiver.get_stats()
print(f"\nAverage FPS: {stats['avg_fps']:.1f}")
print(f"   Meets requirements: {stats['avg_fps'] >= 20}")
EOF

python3 test_perceiver.py
```

#### Real Image Test
```bash
# Download a test image
cd ~/alas_project/data
wget https://ultralytics.com/images/bus.jpg -O test_bus.jpg

# Real image test
cat > test_real_image.py << 'EOF'
import cv2
from src.data_models import Frame, ModelConfig
from perception.perceiver import EnvironmentPerceiver
import time

# Config
config = ModelConfig(
    model_path='models/yolov8n.onnx',
    model_format='onnx',
    input_size=(640, 640),
    confidence_threshold=0.5
)

# Perceiver
perceiver = EnvironmentPerceiver(config)

# Load real image
img = cv2.imread('data/test_bus.jpg')
frame = Frame(
    rgb=img,
    timestamp=time.time(),
    frame_id=0,
    metadata={'source': 'test_bus.jpg'}
)

# Perceive
print("Analyzing test image...")
output = perceiver.perceive(frame)

print("\nDetection results:")
print(f"   FPS: {output.fps:.1f}")
print(f"   Processing time: {output.processing_time_ms:.1f}ms")
print(f"   Obstacles: {len(output.obstacles)}")

for obs in output.obstacles:
    print(f"\n   {obs.object_type.value.upper()}")
    print(f"     Distance: {obs.distance:.1f}m")
    print(f"     Direction: {obs.direction}")
    print(f"     Severity: {obs.severity.name}")
    print(f"     Confidence: {obs.confidence:.2f}")

    # TTS message
    msg = obs.get_tts_message()
    if msg:
        print(f"     TTS: \"{msg}\"")
EOF

python3 test_real_image.py
```

**Checklist:**
- [ ] `perceiver.py` implemented
- [ ] Mock frame test successful
- [ ] FPS >= 20 (requirement met)
- [ ] Real image detection works
- [ ] ObstacleDescriptor output format correct

---

### Benchmark & Documentation

#### FPS Benchmark
```bash
cat > benchmark_fps.py << 'EOF'
from src.data_models import ModelConfig, generate_mock_frame
from perception.perceiver import EnvironmentPerceiver
import time
import numpy as np

# Config
config = ModelConfig(
    model_path='models/yolov8n.onnx',
    model_format='onnx',
    input_size=(640, 640),
    confidence_threshold=0.5
)

# Perceiver
perceiver = EnvironmentPerceiver(config)

# Benchmark
N_FRAMES = 100
fps_list = []
latency_list = []

print(f"Running benchmark with {N_FRAMES} frames...")
print("=" * 60)

for i in range(N_FRAMES):
    frame = generate_mock_frame()
    frame.frame_id = i

    start = time.time()
    output = perceiver.perceive(frame)
    latency = (time.time() - start) * 1000

    fps_list.append(output.fps)
    latency_list.append(latency)

    if i % 20 == 0:
        print(f"Frame {i}/{N_FRAMES}: {output.fps:.1f} FPS")

# Results
print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"Total frames: {N_FRAMES}")
print(f"Average FPS: {np.mean(fps_list):.2f}")
print(f"Min FPS: {np.min(fps_list):.2f}")
print(f"Max FPS: {np.max(fps_list):.2f}")
print(f"Std FPS: {np.std(fps_list):.2f}")
print()
print(f"Average latency: {np.mean(latency_list):.2f}ms")
print(f"Min latency: {np.min(latency_list):.2f}ms")
print(f"Max latency: {np.max(latency_list):.2f}ms")
print()

# Requirements check
avg_fps = np.mean(fps_list)
avg_latency = np.mean(latency_list)

print("REQUIREMENTS CHECK:")
print(f"  NFR-P16 (>=20 FPS): {'PASS' if avg_fps >= 20 else 'FAIL'}")
print(f"  NFR-P17 (<20ms preprocessing): {'PASS' if avg_latency < 20 else 'FAIL'}")
print("=" * 60)
EOF

python3 benchmark_fps.py
```

#### Module Documentation
```bash
cat > README_perception.md << 'EOF'
# ALAS Perception Module

## Overview
Real-time obstacle detection module based on YOLOv8.

## Files
- `perceiver.py`: Main perception class
- `preprocessing.py`: Image preprocessing (used by perceiver.py)

## Usage

### Basic Example
```python
from src.data_models import ModelConfig
from perception.perceiver import EnvironmentPerceiver

config = ModelConfig(
    model_path='models/yolov8n.onnx',
    input_size=(640, 640)
)

perceiver = EnvironmentPerceiver(config)
output = perceiver.perceive(camera_frame)

for obs in output.obstacles:
    print(f"{obs.object_type.value}: {obs.distance}m")
```

## Performance Metrics (Jetson Nano)
- Average FPS: [YOUR_RESULT] (requirement: >=20)
- Preprocessing latency: [YOUR_RESULT]ms (requirement: <20)

## Dependencies
- onnxruntime-gpu
- numpy
- opencv-python

## Test
```bash
python3 test_perceiver.py
python3 benchmark_fps.py
```

## Limitations
Distance estimation is currently heuristic-based (simplified).
For real implementation, a depth camera or monocular depth estimation is required.

## Next Steps
- [ ] Depth camera integration
- [ ] TensorRT optimization
- [ ] Fine-tuning with real-world dataset
EOF
```

**Checklist:**
- [ ] FPS benchmark completed
- [ ] Results documented
- [ ] Requirements met (>=20 FPS)
- [ ] README prepared
- [ ] Code pushed to GitHub

---

## End of Week Review

### Deliverables
1. `src/data_models.py` (shared with the team)
2. `perception/perceiver.py`
3. `models/yolov8n.onnx`
4. Test scripts
5. Benchmark report
6. README_perception.md

### Performance KPIs
| Requirement | Target | Your Result | Status |
|------------|--------|-------------|--------|
| NFR-P16 | >=20 FPS | _____ FPS | [ ] |
| NFR-P17 | <20ms | _____ ms | [ ] |
| FR-2.1 | Segmentation | PASS/FAIL | [ ] |

### Team Integration
- [ ] `data_models.py` synced with Camera team
- [ ] `data_models.py` synced with Navigation team
- [ ] `ObstacleDescriptor` format approved
- [ ] Interface documentation shared

---

## Next Steps (Week 2-3)

### Week 2: Optimization
- TensorRT conversion (optional, for FPS boost)
- Test with real dataset
- Improve distance estimation
- Integration prep

### Week 3: Integration
- Integration with Camera Manager
- Data flow to Navigation Manager
- End-to-end test
- Performance tuning

---

## Common Issues

### ONNX Runtime not found
```bash
# Jetson may require a special build
pip3 install onnxruntime-gpu --break-system-packages

# If you get errors:
pip3 install onnxruntime --break-system-packages  # CPU fallback
```

### CUDA Provider not active
```python
# Check providers
import onnxruntime as ort
print(ort.get_available_providers())

# If CUDA isn't available, CUDAExecutionProvider won't appear
# It will still run on CPU, but slower
```

### FPS too low (<10)
- Reduce input size: (640,640) -> (480,480)
- Increase confidence threshold: 0.5 -> 0.7
- Adjust NMS threshold: 0.4 -> 0.5
- Use TensorRT (faster)

### Memory error
```bash
# Increase swap on Jetson Nano
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---