# SimSiam: Self-Supervised Representation Learning with PyTorch

This repository contains an implementation of **SimSiam**, a self-supervised learning algorithm that trains an encoder to learn meaningful representations of input data without using labels. The goal of SimSiam is to achieve high-quality feature extraction, which can then be evaluated using downstream tasks like linear classification.

## Key Features
- Customizable SimSiam model with flexible options for the predictor and gradient flow.
- Fully implemented training pipeline with CIFAR-10 dataset support.
- Performance visualization through t-SNE plots and linear evaluation.
- Clear insights into learned representations and suggestions for improvement.

---

## How SimSiam Works
SimSiam leverages a Siamese network architecture with two identical branches. It trains the model by maximizing similarity between two augmented views of the same image. Key components include:
1. **Backbone Encoder**: Extracts feature representations.
2. **Projector**: A 3-layer MLP to project features into a space for contrastive learning.
3. **Predictor**: A 2-layer MLP to predict one view's representation from the other.
4. **Stop Gradient**: Prevents collapse by stopping gradients from flowing back through one branch.

---

## Results
- **Linear Evaluation Accuracy**: Achieved high classification accuracy on CIFAR-10 using features extracted by SimSiam.
- **t-SNE Visualization**: While classification accuracy is high, embedding clusters are not perfectly distinct. This behavior is typical of self-supervised learning.

---

## Installation

To run the code, ensure you have Python 3.8+ and install the required libraries:

```bash
pip install torch torchvision matplotlib scikit-learn
