#  Audio Deepfake Detection

## Overview

This repository showcases my main contributions in audio processing research, focusing on neural audio codecs for bitrate analysis, enhancements to the SafeEar deepfake detection framework through codec integrations, and implementations of temporal reduction techniques. The project demonstrates advancements in audio compression, deepfake detection, and temporal modeling for efficient audio processing.

## Main Contributions

### 1. Codec Implementations for Bitrate Analysis

I implemented three distinct neural audio codecs to analyze compression performance across various bitrates, enabling comparative studies of different encoding strategies.

#### APCodec: Neural Audio Codec with Parallel Amplitude and Phase Spectrum Encoding

**Location:** `Codecs/APCodec/`

APCodec treats audio amplitude and phase spectra as separate coding objects, achieving high-quality audio compression at low bitrates (e.g., 6 kbps for 48 kHz audio). The implementation includes parallel encoding of amplitude and phase components with specialized neural networks.

**Key Features:**
- Parallel amplitude and phase spectrum encoding
- Low-bitrate compression (6 kbps target)
- Neural network-based quantization
- Support for 48 kHz audio

#### MDCTC: Modified Discrete Cosine Transform Codec

**Location:** `Codecs/MDCTC/`

MDCTC implements a transform-based audio codec using Modified Discrete Cosine Transform (MDCT) with residual vector quantization (RVQ) for efficient compression. This traditional approach serves as a baseline for comparison with neural codecs.

**Key Features:**
- MDCT-based frequency domain transformation
- Residual vector quantization
- Frame-based processing with overlap
- Configurable frame lengths and hop sizes

#### NanoCodec: Lightweight Neural Audio Codec

**Location:** `Codecs/NanoCodec/`

NanoCodec is a compact neural audio codec utilizing finite scalar quantization (FSQ) with multiple codebooks for high compression ratios. The design focuses on minimal computational requirements while maintaining audio quality.

**Key Features:**
- Multi-codebook finite scalar quantization
- Lightweight encoder architecture
- Bitrate calculation and analysis tools
- Efficient inference for real-time applications

### 2. SafeEar Improvements: Enhanced Deepfake Detection with Codec Integration

**Location:** `SafeEar_Improvements/` and `SafeEar/`

I improved the original SafeEar framework by integrating audio codecs to enhance deepfake detection capabilities while preserving content privacy.

#### Original SafeEar Framework

SafeEar is a content privacy-preserving audio deepfake detection system that separates semantic and acoustic information to detect manipulations without compromising user privacy.

**Key Features:**
- Semantic-acoustic information separation
- Privacy-preserving detection
- Support for ASVspoof datasets
- GAN-based detector training

#### Integration with APCodec

Enhanced SafeEar by incorporating APCodec for feature extraction and compression-aware detection. This integration allows the detector to analyze compressed audio features, improving robustness against codec-based deepfake generation techniques.

**Improvements:**
- APCodec feature integration for compressed audio analysis
- Enhanced detection of codec-manipulated deepfakes
- Joint training with codec and detector components

#### Integration with MDCTC

Integrated MDCTC transform-based features into SafeEar for frequency-domain analysis of deepfake artifacts. This provides complementary detection capabilities to the neural codec integration.

**Improvements:**
- MDCT feature extraction for frequency analysis
- Transform-domain artifact detection
- Multi-modal feature fusion

### 3. Temporal Reduction Implementations

**Location:** `Temporal_Reduction/`

Implemented temporal reduction techniques for efficient audio processing and compression.

#### L3AC: Lightweight Lossless Audio Codec

**Location:** `Temporal_Reduction/L3AC/`

L3AC is a pip-installable lightweight lossless audio codec supporting multiple bitrate configurations (0.75kbps to 3kbps) for temporal reduction in audio tasks.

**Key Features:**
- Lossless compression
- Multiple bitrate options
- PyTorch-based implementation
- Easy integration with other audio processing pipelines

#### SNAC: Neural Audio Codec

**Location:** `Temporal_Reduction/snac/`

SNAC implements a neural audio codec with hierarchical encoding and residual vector quantization for temporal reduction and compression.

**Key Features:**
- Hierarchical encoder-decoder architecture
- Residual vector quantization
- Attention-based processing
- Configurable sampling rates and dimensions

## Environment Setup

The project implements multiple isolated Python environments to manage dependencies for different components and prevent conflicts. Each environment is created using conda for reproducibility and isolation.

### Environment Parameters

**Global Requirements:**
- Operating System: Linux/Windows/macOS
- Python Versions: 3.8+ (varies by component)
- GPU: CUDA-compatible (recommended for training)
- RAM: 16GB+ for training, 8GB+ for inference

**Component-Specific Environments:**

#### APCodec Environment
- **Python Version:** 3.8
- **Key Dependencies:**
  - PyTorch 1.8.1+cu111
  - NumPy 1.21.6
  - Librosa 0.9.1
  - TensorBoard 2.8.0
  - SoundFile 0.10.3
  - Matplotlib 3.1.3

#### MDCTC Environment
- **Python Version:** 3.8+
- **Key Dependencies:**
  - PyTorch (latest stable)
  - NumPy
  - SciPy (for signal processing)

#### NanoCodec Environment
- **Python Version:** 3.8+
- **Key Dependencies:**
  - PyTorch
  - NumPy

#### SafeEar Environment
- **Python Version:** 3.9
- **Key Dependencies:**
  - PyTorch 1.13.1+cu116
  - Torchaudio 0.13.1
  - Fairseq (custom fork)
  - Hubert models
  - SpeechTokenizer

#### Temporal Reduction Environment
- **Python Version:** 3.7+
- **Key Dependencies:**
  - PyTorch
  - Librosa (for L3AC examples)
  - NumPy

### Implementation Explanation

**Environment Isolation Strategy:**
Each component maintains its own conda environment to prevent dependency version conflicts. This approach ensures that updates to one component don't break others and allows for component-specific optimizations.

**Package Management:**
- Conda is used for environment creation and base package installation
- Pip handles Python-specific packages with version pinning
- GPU support is implemented through CUDA-enabled PyTorch installations
- Pre-trained models are downloaded separately to reduce repository size

**Reproducibility:**
- All package versions are pinned to specific releases
- Environment files (environment.yml or requirements.txt) are provided per component
- Installation scripts automate the setup process

**Setup Process:**
1. Create component-specific conda environment
2. Install PyTorch with appropriate CUDA version
3. Install component dependencies via pip
4. Download required pre-trained models
5. Configure data paths and model checkpoints

This structured approach ensures reliable reproduction of results across different systems and facilitates collaborative development.

## Usage

### Codec Training and Inference

**APCodec:**
```bash
cd Codecs/APCodec
conda activate apcodec
CUDA_VISIBLE_DEVICES=0 python train.py
CUDA_VISIBLE_DEVICES=0 python inference.py
```

**MDCTC:**
```bash
cd Codecs/MDCTC
# Training and inference scripts
```

**NanoCodec:**
```bash
cd Codecs/NanoCodec
python main.py
```

### SafeEar Training and Evaluation

```bash
cd SafeEar_Improvements
conda activate safeear
python train_gan_detector.py
python eval_gan_detector.py
```

### Temporal Reduction

**L3AC:**
```python
import l3ac
codec = l3ac.get_model('1kbps')
# Encode/decode audio
```

**SNAC:**
```python
from snac import SNAC
codec = SNAC()
# Process audio
```

## System Requirements

- **OS:** Linux (preferred), Windows, macOS
- **CPU:** Multi-core processor (8+ cores recommended)
- **GPU:** NVIDIA GPU with CUDA support (8GB+ VRAM)
- **RAM:** 32GB+ for training, 16GB+ for inference
- **Storage:** 500GB+ for datasets and models

## Citation

If you use this work in your research, please cite appropriately.

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
  title={L3AC: Towards a LightWeight and Lossless Audio Codec},
  author={Zhai, LW},
  journal={arXiv preprint arXiv:2504.04949},
  year={2024}
}
```  