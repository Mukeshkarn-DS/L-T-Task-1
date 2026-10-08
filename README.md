# L&T EduTech – Deep Learning Task Assignment

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Tasks](https://img.shields.io/badge/Tasks-1--8-success)](#task-overview)

## About

This repository contains my practical implementation work for the **L&T EduTech Deep Learning Task Assignment**.

The assignment is organized into **8 independent tasks**, progressing from neural-network fundamentals to regularization, CNNs, transfer learning, computer vision, sequence modelling, representation learning, generative AI, and model optimization.

Each task has its own folder so the implementation can be opened and reviewed separately.

> **Academic Work:** This repository is maintained for learning, experimentation, evaluation, and academic demonstration.

---

## Task Overview

| Task | Topic | Main Work | Current Implementation |
|---|---|---|---|
| **Task 1** | Neural Network Fundamentals | Activations, optimizers, batch-size experiments, hyperparameter tuning, error analysis | Python scripts + results |
| **Task 2** | Regularization & Bias–Variance | L1/L2 regularization, Dropout, Batch Normalization, model analysis | Jupyter Notebook |
| **Task 3** | CNN Classification & Data Augmentation | CIFAR-10 CNN, augmentation, evaluation and comparison | Jupyter Notebook |
| **Task 4** | AlexNet & Transfer Learning | AlexNet-style CNN, VGG16 transfer learning, evaluation and comparison | 3 Jupyter Notebooks |
| **Task 5** | Object Detection & Segmentation | Object detection and image segmentation workflows | Jupyter Notebook |
| **Task 6** | Sequence Modelling | RNN, LSTM, sentiment classification and time-series modelling | Jupyter Notebook |
| **Task 7** | Representation Learning | Autoencoder, VAE, reconstruction and latent-space analysis | Jupyter Notebook |
| **Task 8** | Generative AI & Model Optimization | DCGAN, pruning, INT8 quantization and knowledge distillation | Jupyter Notebook |

---

# Task Details

## Task 1 — Neural Network Fundamentals

**Folder:** `L&T TASK 1/`

Task 1 implements and analyses the basic components of a neural network.

### Implemented work

- Dense neural-network training
- ReLU and Softmax activation comparison
- Optimizer comparison
- Batch-size comparison
- Single vs multi-configuration experiments
- Hyperparameter tuning
- Training-curve analysis
- Confusion matrix
- Misclassified-image analysis

### Main code files

- `train.py`
- `compare_activations.py`
- `compare_batch_sizes.py`
- `compare_optimizers.py`
- `single_vs_multi.py`
- `tune_hyperparameters.py`
- `visualize_errors.py`
- `nn/` — neural-network implementation files

### Results

The folder contains generated visualizations and experiment results, including:

- Activation comparison
- Optimizer comparison
- Batch-mode comparison
- Training curves
- Confusion matrix
- Misclassified images
- Hyperparameter results

---

## Task 2 — Regularization and Bias–Variance Analysis

**Folder:** `L&T TASK 2/`

**Notebook:** `L&T_TASK_2.ipynb`

Task 2 studies techniques used to control overfitting and improve model generalization.

### Implemented work

- Neural/CNN model experiments
- L1 regularization
- L2 regularization
- Dropout
- Batch Normalization
- Training and validation analysis
- Bias–variance analysis
- Model-performance comparison

The notebook is configured for GPU execution and uses TensorFlow/Keras with NumPy and Matplotlib.

---

## Task 3 — CNN Image Classification and Data Augmentation

**Folder:** `L&T TASK 3/`

**Notebook:** `L&T_TASK3.ipynb`

Task 3 focuses on image classification using a Convolutional Neural Network.

### Implemented work

- CIFAR-10 preprocessing
- Baseline CNN
- Convolution and pooling
- Batch Normalization
- Dropout
- Data augmentation
- Augmented CNN
- Training-history analysis
- Classification evaluation
- Confusion-matrix analysis
- Baseline vs augmented-model comparison

### Objective

To understand CNN-based image classification and demonstrate how data augmentation can improve model robustness and generalization.

---

## Task 4 — AlexNet and Transfer Learning

**Folder:** `L&T TASK 4/`

Task 4 contains three notebooks covering advanced CNN classification and transfer learning.

### Notebooks

1. `01_CIFAR_10_Class_Classifier_with_AlexNet_architecture.ipynb`
2. `Task_4_VGG16_Transfer_Learning_CIFAR10.ipynb`
3. `L&T_Task_4_Evaluation_and_Comparison.ipynb`

### Implemented work

- AlexNet-style CNN architecture
- CIFAR-10 classification
- VGG16 transfer learning
- Transfer-learning workflow
- Model evaluation
- Accuracy and loss analysis
- Precision, recall and F1-score analysis
- Confusion matrix
- Model comparison

> The README lists only files that are currently present in the repository.

---

## Task 5 — Object Detection and Image Segmentation

**Folder:** `L&T TASK 5/`

**Notebook:** `Task_5_Object_Detection_and_Image_Segmentation.ipynb`

Task 5 extends computer vision from image classification to object localization and image segmentation.

### Implemented work

- Object detection workflow
- Image segmentation workflow
- Detection/segmentation prediction visualization
- Model evaluation
- Computer-vision performance analysis

### Evaluation concepts

- mAP
- IoU
- Dice Score
- Prediction visualization

> GPU execution is recommended for computationally intensive experiments.

---

## Task 6 — Sequence Modelling with RNN and LSTM

**Folder:** `L&T TASK 6/`

**Notebook:** `L&T_TASK_6.ipynb`

Task 6 introduces recurrent neural networks for sequential and time-series data.

### Implemented work

- Sequence preprocessing
- Embedding
- Simple RNN
- LSTM
- Dropout
- Early stopping
- Sentiment-classification workflow
- Time-series modelling
- RNN vs LSTM comparison
- Model evaluation

### Main applications

- IMDB movie-review sentiment classification
- Airline passenger time-series prediction

---

## Task 7 — Autoencoder and Variational Autoencoder

**Folder:** `L&T TASK 7/`

**Notebook:** `L&T_TASK_7.ipynb`

Task 7 focuses on unsupervised representation learning.

### Implemented work

- Encoder and decoder architecture
- Latent representation learning
- Image reconstruction
- Autoencoder (AE)
- Variational Autoencoder (VAE)
- Reconstruction analysis
- Latent-space analysis
- AE vs VAE comparison
- Latent-space interpolation

### Dataset

**Fashion-MNIST**

### Objective

To understand how neural networks learn compact representations and how VAEs provide structured latent spaces for generative applications.

---

## Task 8 — Generative AI and Model Optimization

**Folder:** `L&T TASK 8/`

**Current notebook:** `L&t_Task_8.ipynb`

Task 8 combines generative modelling with practical deep-learning optimization techniques.

### Part A — DCGAN

The notebook implements a **Deep Convolutional Generative Adversarial Network (DCGAN)**.

Main components:

- Generator
- Discriminator
- Adversarial training
- Generator/discriminator loss tracking
- Generated-image visualization
- Generator/model saving

### Part B — Model Optimization

The notebook covers:

#### Pruning
Removes less-important parameters to reduce model complexity.

#### INT8 Quantization
Uses lower numerical precision to reduce model size and improve inference efficiency.

#### Knowledge Distillation
Transfers knowledge from a larger teacher model to a smaller student model.

### Comparison

The optimization experiments analyse:

- Parameter count
- Model size
- Accuracy
- Compression impact
- Efficiency

> **File status:** The current repository contains `L&t_Task_8.ipynb`. The README intentionally uses the exact filename currently present in GitHub.

---

# Repository Structure

```text
L_T_-Task-_Assignment/
│
├── L&T TASK 1/
│   ├── nn/
│   ├── train.py
│   ├── compare_activations.py
│   ├── compare_batch_sizes.py
│   ├── compare_optimizers.py
│   ├── single_vs_multi.py
│   ├── tune_hyperparameters.py
│   ├── visualize_errors.py
│   └── generated results and visualizations
│
├── L&T TASK 2/
│   └── L&T_TASK_2.ipynb
│
├── L&T TASK 3/
│   └── L&T_TASK3.ipynb
│
├── L&T TASK 4/
│   ├── 01_CIFAR_10_Class_Classifier_with_AlexNet_architecture.ipynb
│   ├── Task_4_VGG16_Transfer_Learning_CIFAR10.ipynb
│   └── L&T_Task_4_Evaluation_and_Comparison.ipynb
│
├── L&T TASK 5/
│   └── Task_5_Object_Detection_and_Image_Segmentation.ipynb
│
├── L&T TASK 6/
│   └── L&T_TASK_6.ipynb
│
├── L&T TASK 7/
│   └── L&T_TASK_7.ipynb
│
└── L&T TASK 8/
    └── L&t_Task_8.ipynb
```

---

# Technology Stack

### Programming and Development

- Python 3.x
- Jupyter Notebook
- Google Colab

### Deep Learning

- TensorFlow
- Keras
- PyTorch
- Torchvision

### Data Science and Visualization

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Computer Vision

- CNN
- AlexNet
- VGG16
- YOLO/object detection workflows
- U-Net/segmentation workflows

---

# Datasets Used Across the Tasks

| Dataset | Main Task / Use |
|---|---|
| **MNIST** | Neural networks and generative modelling |
| **Fashion-MNIST** | Representation learning |
| **CIFAR-10** | CNN and advanced image classification |
| **IMDB Movie Reviews** | RNN/LSTM sentiment classification |
| **Airline Passenger Dataset** | Time-series modelling |
| **Oxford-IIIT Pet** | Image segmentation |
| **Construction/PPE data** | Object detection experiments |
| **Road-sign data** | Object detection experiments |

The exact dataset and experiment depend on the notebook and task objective.

---

# Learning Progression

The assignment follows this progression:

1. **Neural Network Fundamentals**
2. **Regularization and Bias–Variance**
3. **CNN Classification and Data Augmentation**
4. **AlexNet and Transfer Learning**
5. **Object Detection and Image Segmentation**
6. **RNN and LSTM Sequence Modelling**
7. **Autoencoder and VAE Representation Learning**
8. **DCGAN and Model Optimization**

This provides a complete progression from basic neural-network concepts to modern deep-learning and generative-AI techniques.

---

# Evaluation

Depending on the task, the repository demonstrates:

- Training and validation loss
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Misclassified samples
- mAP
- IoU
- Dice Score
- Reconstruction quality
- Latent-space analysis
- Model-size comparison
- Parameter-count comparison
- Quantization/compression analysis

---

# How to Run

## Google Colab

1. Open the required notebook from the corresponding task folder.
2. Open it in Google Colab.
3. Select a GPU runtime when recommended.
4. Run the cells sequentially.
5. Review the outputs, metrics, visualizations, and results.

## Local Jupyter

Install the main dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow torch torchvision
```

Launch Jupyter:

```bash
jupyter notebook
```

Additional packages required by a particular task should be installed according to the instructions inside that notebook.

---

# Reproducibility

Where applicable, experiments use fixed random seeds to improve reproducibility.

A typical configuration is:

```python
SEED = 42
```

GPU acceleration is recommended for the more computationally intensive computer-vision, sequence, representation-learning, and generative-model experiments.

---

# Author

**Mukesh Kumar**  
B.Sc. Data Science

GitHub: [@Mukeshkarn-DS](https://github.com/Mukeshkarn-DS)

## Repository

[L&T EduTech – Deep Learning Task Assignment](https://github.com/Mukeshkarn-DS/L_T_-Task-_Assignment)

---

# Academic Note

This repository represents practical implementation and experimentation performed as part of the **L&T EduTech Deep Learning Task Assignment**.

The repository is organized so that **Tasks 1–8 can be reviewed independently**, with the README reflecting the current implementation files in each task folder.

---

## License

This repository is maintained for educational and academic purposes.
