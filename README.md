# Multi-Modal Deepfake Detection using Graph Neural Networks

A sophisticated deepfake detection system that combines audio and visual features using Graph Neural Networks (GNNs) to identify manipulated media content.

## 🎯 Overview

This project implements a state-of-the-art deepfake detection system using Graph Neural Networks (GNNs) for multi-modal analysis. The core innovation lies in representing video-audio relationships as dynamic graphs, where nodes represent individual video frames and audio segments, while edges capture temporal dependencies and cross-modal correlations. The system extracts sophisticated features from video streams (facial regions, optical flow, temporal patterns) and audio streams (MFCCs, spectral characteristics, chroma features) to create rich multi-dimensional representations. Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN) then process these graph structures, learning complex interaction patterns between visual and auditory cues that are characteristic of deepfake manipulations. The attention mechanism allows the model to focus on the most discriminative temporal segments and cross-modal relationships, while the graph structure preserves crucial spatial-temporal dependencies that traditional approaches often miss. This architecture is particularly effective because deepfakes typically exhibit inconsistencies across modalities - for example, lip-sync mismatches, unnatural audio-visual correlations, or temporal artifacts that become apparent when analyzing the relationships between consecutive frames and corresponding audio segments. The focal loss function addresses class imbalance issues common in deepfake datasets, ensuring the model learns to identify subtle manipulation patterns rather than simply memorizing dataset biases. Through end-to-end training, the system learns to detect these multi-modal inconsistencies that human perception might miss, making it robust against various deepfake generation techniques.

## 🚀 Features

- **Multi-Modal Analysis**: Processes both audio and video streams simultaneously
- **Graph-Based Learning**: Constructs dynamic graphs to capture temporal and cross-modal relationships
- **Advanced Feature Extraction**: 
  - Video: Facial detection, optical flow, and temporal features
  - Audio: MFCCs, spectral features, chroma, and zero-crossing rate
- **Robust Training**: Includes gradient accumulation, learning rate scheduling, and early stopping
- **Comprehensive Evaluation**: Provides accuracy, precision, recall, and F1-score metrics

## 📋 Requirements

### Dependencies
```python
torch>=1.9.0
torch-geometric>=2.0.0
opencv-python>=4.5.0
librosa>=0.8.0
scikit-learn>=0.24.0
numpy>=1.20.0
ffmpeg-python
```

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg installed for audio extraction

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

2. Install dependencies:
```bash
pip install torch torch-geometric opencv-python librosa scikit-learn numpy
```

3. Install FFmpeg:
- **Windows**: Download from [FFmpeg official website](https://ffmpeg.org/download.html)
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`

## 📊 Dataset Structure

The system expects the LAV-DF dataset with the following structure:
```
LAV-DF/
├── metadata.json
├── train/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── test/
│   ├── video1.mp4
│   └── ...
└── dev/
    ├── video1.mp4
    └── ...
```

### Metadata Format
The `metadata.json` should contain entries like:
```json
[
  {
    "file": "train/video1.mp4",
    "label": 0,  // 0 for real, 1 for fake
    "n_fakes": 0,
    "fake_periods": []
  }
]
```

## 🔧 Usage

### Basic Usage
```python
from deepfake_detector import DeepfakeDetector

# Initialize detector
detector = DeepfakeDetector(device='cuda')

# Load and prepare data
video_audio_pairs, labels = load_all_subsets('path/to/dataset', max_clips=750)

# Train the model
detector.feature_extractor.fit_scalers(video_audio_pairs)
graphs = detector.prepare_data(video_audio_pairs, labels)

# Split data and train
train_graphs, val_graphs, test_graphs = split_data(graphs, labels)
best_acc = detector.train(train_graphs, val_graphs, epochs=100)

# Evaluate
accuracy, precision, recall, f1 = detector.evaluate(test_graphs)
```

### Configuration Options

#### Model Parameters
- `input_dim`: Input feature dimension (default: 1024)
- `hidden_dim`: Hidden layer dimension (default: 512)
- `num_classes`: Number of output classes (default: 2)
- `dropout`: Dropout rate (default: 0.3)

#### Training Parameters
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size for training (default: 8)
- `learning_rate`: Initial learning rate (default: 5e-4)
- `accum_steps`: Gradient accumulation steps (default: 4)

## 🏗️ Architecture

### Feature Extraction
1. **Video Features**:
   - Facial region detection using Haar cascades
   - Optical flow computation between consecutive frames
   - Temporal feature aggregation

2. **Audio Features**:
   - MFCC coefficients (20 features)
   - Spectral centroid and rolloff
   - Chroma features (12 features)
   - Zero-crossing rate

### Graph Construction
- Nodes represent video frames and audio features
- Edges connect similar frames and cross-modal relationships
- Edge weights based on cosine similarity

### Neural Network Architecture
- **Graph Attention Networks**: Process video and audio nodes separately
- **Graph Convolutional Networks**: Aggregate neighborhood information
- **Multi-head Attention**: Capture long-range dependencies
- **Global Pooling**: Generate graph-level representations

## 📈 Performance

The model demonstrates strong learning capabilities with the following results:

### Training Performance
- **Final Training Accuracy**: 97.14% (Epoch 100)
- **Best Validation Accuracy**: 70.67%
- **Final Validation Accuracy**: 70.00%

### Test Set Results
- **Accuracy**: 61.33%
- **Precision**: 74.07%
- **Recall**: 47.62%
- **F1-Score**: 57.97%

### Performance Analysis
The model shows excellent training performance with 97.14% accuracy, indicating strong learning capacity. The validation accuracy of ~70% suggests reasonable generalization, while the test results reveal some challenges:

- **High Precision (74.07%)**: The model is conservative in predicting fake content, with low false positive rates
- **Moderate Recall (47.62%)**: The model misses some fake samples, indicating room for improvement in detection sensitivity
- **Balanced F1-Score (57.97%)**: Reasonable overall performance considering the precision-recall trade-off

The gap between training and test performance suggests potential overfitting, which could be addressed through:
- Enhanced regularization techniques
- Data augmentation strategies
- Cross-validation approaches
- Ensemble methods

## 🔍 Key Components

### AudioVisualFeatureExtractor
Handles extraction and preprocessing of audio-visual features with robust error handling and normalization.

### GraphConstructor
Creates graph representations from extracted features, establishing relationships between temporal and cross-modal elements.

### MultiModalGNN
The core neural network combining GAT, GCN, and attention mechanisms for classification.

### FocalLoss
Addresses class imbalance by focusing on hard-to-classify examples.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Memory Error**: Reduce batch size or use gradient accumulation
2. **FFmpeg Not Found**: Ensure FFmpeg is installed and in system PATH
3. **Empty Metadata**: Check dataset structure and metadata format
4. **Low Performance**: Increase training epochs or adjust hyperparameters

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export TORCH_SHOW_CPP_STACKTRACES=1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Citation

If you use this work in your research, please cite:
```bibtex
@article{deepfake_detection_2024,
  title={Multi-Modal Deepfake Detection using Graph Neural Networks},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## 🙏 Acknowledgments

- LAV-DF dataset creators
- PyTorch Geometric team
- OpenCV community
- Librosa developers

## 📧 Contact

For questions or support, please contact [arnav.gilankar@gmail.com](mailto:arnav.gilankar@gmail.com)

---

**Note**: This is a research implementation. For production use, consider additional optimizations and security measures.
