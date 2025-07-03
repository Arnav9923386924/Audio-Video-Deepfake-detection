# 🎭 DeepFake Detection using Audio-Visual Graph Neural Networks (GNNs) (Work in progress)

This repository provides a **multi-modal deep learning system** to detect deepfakes using both **audio and video features**. It leverages **Graph Neural Networks (GCN + GAT)** and **Cross-Attention** mechanisms for robust feature representation.

---

## 📌 Dataset: LAV-DF

This project uses the [**LAV-DF (Large-Scale Audio-Visual DeepFake Detection)**]([https://www.kaggle.com/datasets/xhlulu/lavdf-large-scale-audiovisual-deepfake-dataset](https://www.kaggle.com/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df)) dataset from Kaggle, which contains thousands of real and fake audio-video clips.

🔗 **Kaggle Dataset**:  
([https://www.kaggle.com/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df])

---

## ✨ Key Features

- 🎥 **Facial & Optical Flow Features** from video
- 🔊 **MFCC, Spectral, Chroma Features** from audio
- 🧠 **GNN with GAT & GCN layers** for multimodal fusion
- 🔄 **Cross-Attention mechanism** for inter-modal interaction
- 📊 **Custom Focal Loss** to handle imbalanced datasets
- 🔥 **Early stopping** and model checkpointing
- ⚙️ Audio extracted using **FFmpeg**
- 📈 Evaluation: Accuracy, Precision, Recall, F1

---

### 1. Clone the repo
```bash
git clone https://github.com/Arnav9923386924/Audio-Video-Deepfake-detection.git
cd Audio-Video-Deepfake-detection
```
### Contact Information: [arnav.gilankar@gmail.com]
