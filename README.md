# D3 Audio Processing Project

This repository contains a collection of audio processing projects focusing on neural audio codecs, deepfake detection, and temporal reduction techniques. The project is organized into three main components: APCodec, SafeEar, and Temporal Reduction.

## Components

### 1. APCodec: Neural Audio Codec with Parallel Amplitude and Phase Spectrum Encoding

**Location:** `Codecs/workspace/APCodec/`

APCodec is a novel neural audio codec that treats audio amplitude and phase spectra as coding objects, achieving high-quality audio compression at low bitrates (e.g., 6 kbps for 48 kHz audio).

#### Environment Setup

The environment for APCodec is implemented using a Python virtual environment with specific package versions to ensure compatibility and reproducibility.

**Requirements:**
- Python 3.8+
- CUDA-compatible GPU (recommended for training/inference)

**Installation Steps:**
1. Create a conda environment:
   ```bash
   conda create -n apcodec python=3.8
   conda activate apcodec
   ```

2. Install PyTorch with CUDA support:
   ```bash
   pip install torch==1.8.1+cu111
   ```

3. Install other dependencies:
   ```bash
   pip install numpy==1.21.6 librosa==0.9.1 tensorboard==2.8.0 soundfile==0.10.3 matplotlib==3.1.3
   ```

**Implementation Notes:**
- The environment uses PyTorch 1.8.1 with CUDA 11.1 support for GPU acceleration
- NumPy 1.21.6 provides numerical computing capabilities
- Librosa 0.9.1 handles audio processing and feature extraction
- TensorBoard 2.8.0 enables training visualization and monitoring
- SoundFile 0.10.3 manages audio file I/O operations
- Matplotlib 3.1.3 supports plotting and visualization

#### Usage

**Training:**
```bash
cd Codecs/workspace/APCodec
CUDA_VISIBLE_DEVICES=0 python train.py
```

**Inference:**
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py
# For CPU inference:
CUDA_VISIBLE_DEVICES=CPU python inference.py
```

Configure data paths and model checkpoints in `config.json`.

### 2. SafeEar: Content Privacy-Preserving Audio Deepfake Detection

**Location:** `SafeEar_Improvements/workspace/`

SafeEar is a framework for detecting deepfake audio while preserving content privacy by separating semantic and acoustic information.

#### Environment Setup

SafeEar uses a comprehensive Python environment with deep learning frameworks and audio processing libraries.

**Requirements:**
- Python 3.9
- CUDA-compatible GPU
- Sufficient RAM for model training

**Installation Steps:**
1. Create conda environment:
   ```bash
   conda create -n safeear python=3.9
   conda activate safeear
   ```

2. Install PyTorch ecosystem:
   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```

3. Install project dependencies:
   ```bash
   cd SafeEar_Improvements/workspace/SafeEar
   pip install pip==24.0
   pip install -r requirements.txt
   ```

4. Download required models:
   ```bash
   mkdir model_zoos
   cd model_zoos
   wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt
   wget https://cloud.tsinghua.edu.cn/f/413a0cd2e6f749eea956/?dl=1 -O SpeechTokenizer.pt
   ```

**Implementation Notes:**
- Python 3.9 ensures compatibility with the latest PyTorch and related libraries
- PyTorch 1.13.1 with CUDA 11.6 provides stable deep learning operations
- The requirements.txt includes comprehensive dependencies for audio processing, machine learning, and evaluation
- Fairseq is included as an editable install for custom modifications
- Hubert and SpeechTokenizer models are pre-trained and downloaded for feature extraction

#### Data Preparation

1. Download ASVspoof datasets to `datas/datasets/`
2. Generate Hubert features:
   ```bash
   cd datas
   python dump_hubert_avg_feature.py datasets/ASVSpoof2019 datasets/ASVSpoof2019_Hubert_L9
   python dump_hubert_avg_feature.py datasets/ASVSpoof2021 datasets/ASVSpoof2021_Hubert_L9
   ```

#### Usage

**Training:**
```bash
cd SafeEar_Improvements/workspace
python train_gan_detector.py
```

**Evaluation:**
```bash
python eval_gan_detector.py
```

### 3. Temporal Reduction with L3AC

**Location:** `Temporal Reduction/workspace/`

This component uses L3AC (Lightweight Lossless Audio Codec) for temporal reduction in audio processing tasks.

#### Environment Setup

L3AC is implemented as a pip-installable package with minimal dependencies.

**Requirements:**
- Python 3.7+
- PyTorch
- Librosa (for examples)

**Installation Steps:**
1. Install L3AC:
   ```bash
   pip install l3ac
   ```

2. Install additional dependencies for examples:
   ```bash
   pip install librosa torch
   ```

**Implementation Notes:**
- L3AC is distributed as a wheel package for easy installation
- The codec supports multiple bitrate configurations (0.75kbps to 3kbps)
- PyTorch backend enables GPU acceleration
- Librosa provides audio loading and resampling utilities

#### Usage

**Example Usage:**
```python
import librosa
import torch
import l3ac

# Load model
codec = l3ac.get_model('1kbps')

# Load and process audio
audio, sr = librosa.load(librosa.example("libri1"))
audio = librosa.resample(audio, orig_sr=sr, target_sr=codec.config.sample_rate)
audio = torch.tensor(audio[None, :], dtype=torch.float32)

# Encode/decode
codec.network.eval()
with torch.inference_mode():
    q_feature, indices = codec.encode_audio(audio)
    reconstructed = codec.decode_audio(q_feature)
```

## Overall Environment Implementation

The project implements multiple isolated environments to prevent dependency conflicts between components:

1. **APCodec Environment:** Focused on audio codec development with specific PyTorch and audio libraries
2. **SafeEar Environment:** Comprehensive deep learning setup for detection tasks with extensive dependencies
3. **L3AC Environment:** Minimal setup for lightweight codec operations

Each environment is created using conda for package management and isolation. GPU support is implemented through CUDA-enabled PyTorch installations where applicable. Dependencies are pinned to specific versions to ensure reproducibility across different systems.

## System Requirements

- **OS:** Linux/Windows/macOS
- **GPU:** NVIDIA GPU with CUDA support (recommended)
- **RAM:** 16GB+ for training, 8GB+ for inference
- **Storage:** Sufficient space for datasets and model checkpoints

## Citation

If you use any component of this project, please cite the respective papers:

**APCodec:**
```
@article{ai2024apcodec,
  title={A{PC}odec: A Neural Audio Codec with Parallel Encoding and Decoding for Amplitude and Phase Spectra},
  author={Ai, Yang and Jiang, Xiao-Hang and Lu, Ye-Xin and Du, Hui-Peng and Ling, Zhen-Hua},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={32},
  pages={3256--3269},
  year={2024}
}
```

**SafeEar:**
Please refer to the SafeEar README for citation information.

**L3AC:**
```
@article{zhai2024l3ac,
  title={L3AC: Towards a Lightweight and Lossless Audio Codec},
  author={Zhai, LW},
  journal={arXiv preprint arXiv:2504.04949},
  year={2024}
}
```