# CUDA Quantum Simulator (AstraQ) ⚛️⏩


A Quantum computing simulator leveraging NVIDIA CUDA for accelerated circuit simulations.

## Features ✨

- Faster than CPU-based simulators When You Use Alot Of Shots
- Supports 7+ quantum gates
- Measurement sampling with configurable shots

## Requirements 📋

- NVIDIA GPU (Compute Capability 8.0+)
- CUDA Toolkit 11.8+
- Linux (Ubuntu 20.04+ recommended)
- Python 3.7+

## Installation 🛠️

### 1. Install CUDA Toolkit
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-11-8
```
### 2. Verify CUDA installation
```
nvcc --version  # Should show CUDA 11.8 or newer
nvidia-smi      # Verify GPU visibility
```
### 3. Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev python3-pip
### 4. Install  & Build AstraQ
```bash
git clone https://github.com/Hasv07/AstraQ.git
cd quantum-simulator
python3 setup.py build_ext --inplace
```

## Usage Example 🚀

Create `example.py`:
```python
from qsim import QuantumCircuit, Aer
import matplotlib.pyplot as plt

# Create 10-qubit circuit
qc = QuantumCircuit(10)
qc.h(0)          # Apply Hadamard
qc.cx(0, 1)      # CNOT gate
qc.swap(2, 3)    # SWAP operation

# Simulate with 1000 shots
result = Aer.get_backend('qsim_simulator')(qc, shots=1000)

# Visualize results
plt.bar(result.get_counts().keys(), result.get_counts().values())
plt.title('Quantum Measurement Results')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

```
You Can See More Examples in Simulation.ipynb

Run with:
```bash
python example.py 
```

## Troubleshooting 🔧

**Error: "CUDA driver version is insufficient"**
```bash
sudo apt-get install nvidia-driver-510
```

**Error: "Unsupported gpu architecture 'compute_86'"**
Update `setup.py` with your GPU's compute capability:
```python
# Change this line in setup.py
'-arch=compute_86'  # For RTX 3090/A100
```

Find your GPU's compute capability:
[NVIDIA CUDA GPUs Table](https://developer.nvidia.com/cuda-gpus)

## Contributing 🤝

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License 📄

MIT License 