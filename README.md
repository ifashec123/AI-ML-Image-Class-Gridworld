# 🧠 AI & Machine Learning Project  
### *Image Classification with CNNs & Reinforcement Learning in GridWorld*

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%2C%20CustomEnv-lightgrey" />
  <img src="https://img.shields.io/badge/Status-Completed-brightgreen" />
</p>

👋 Hi! I'm Christian, and this repository contains a dual-part project exploring two major branches of machine learning:

1. 🖼️ **Image Classification using Convolutional Neural Networks (CNNs)**  
2. 🚀 **Reinforcement Learning with Value Iteration & Q-Learning in a GridWorld environment**

The project was completed as part of my Applied Machine Learning coursework and is designed to demonstrate practical implementations of supervised and reinforcement learning models, including performance enhancements, model evaluation, and parameter optimization. Below is a summary however the full report can be accessed [here](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/MLreport.pdf)

---

## 📦 Contents
| Module | Description |
|--------|-------------|
| [`cnn_classification.py`](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/Task1_code.ipynb) | Image classification with CNNs using TensorFlow |
| [`cnn_regularized.py`](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/Task1_code.ipynb) | Improved CNN using data augmentation, L2 regularization, dropout, and early stopping |
| [`value_iteration.py`](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/Task2_code.ipynb) | Value Iteration implementation in a custom GridWorld |
| [`q_learning.py`](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/Task2_code.ipynb) | Q-Learning agent with policy learning and reward tracking |
| [`q_tuning.py](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/Task2_code.ipynb) | Hyperparameter tuning for Q-Learning (γ, ε grid search) |

---

## 🖼️ Part 1: Image Classification with CNNs

### 🧠 Objective
Classify objects such as parachutes, trucks, and oil boxes using a custom image dataset.

### 🔧 Core Features
- **Model Architecture:** Two convolutional layers with max pooling, followed by fully connected layers.
- **Input Shape:** RGB images resized to `(224x224x3)`
- **Evaluation:** Accuracy, loss curves, confusion matrix, and classification report
- **Enhancements (in `cnn_regularized.py`):**
  - ✅ Data Augmentation (flipping, rotation, zoom)
  - ✅ Dropout Layer (0.5) for regularization
  - ✅ L2 Regularization (λ=0.001)
  - ✅ Early Stopping to prevent overfitting

### 📊 Results Summary
- **Baseline CNN:** High training accuracy (97%) but low validation accuracy (~56%) due to overfitting.
- **Improved Model:** Better generalization (58% val accuracy), reduced overfitting, and stronger F1-scores on difficult classes.

### 🖼️ Sample Training Output
> *(Demo images, learning curves, and confusion matrices should be saved and inserted here)*

---

## 🧭 Part 2: Reinforcement Learning in GridWorld

### 🌍 Environment
A complex 2D maze-like grid with:
- `w`: Walls
- `o`: Obstacles
- `g`: Goal
- `a`: Agent starting point

Implemented using a custom `GridWorld` class with deterministic transitions.

---

### 📘 A. Value Iteration
- **Model-Based RL**
- Uses the Bellman Optimality Equation
- Fast convergence (0.08s), but resulted in a **suboptimal policy** due to limited reward structure

### 📘 B. Q-Learning
- **Model-Free RL**
- Epsilon-greedy exploration
- Learned policy converged after 10,000 episodes
- Achieved **average reward of +67** vs -125 for Value Iteration

### 📘 C. Hyperparameter Tuning
Tested 25 combinations of:
- **Discount Factor (γ):** [0.1, 0.5, 0.7, 0.9, 0.99]
- **Exploration Rate (ε):** [0.1, 0.3, 0.5, 0.7, 0.9]

**Best results** observed for `γ=0.9`, `ε=0.1` — strong success rate (~80%) and short episode lengths.

### 📈 Key Metrics Tracked
- Convergence Time
- Average Reward
- Success Rate
- Average Episode Length
- Q-Value Variance

### 🖼️ Visualisations
- Learned policies (screenshots)
- Cumulative reward over episodes
- Success rate and episode length across hyperparameter grids

---

## 📂 Directory Structure
- project report can be accessed [here](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/MLreport.pdf)
- Jupyter notebook for task 1 can be accessed [here](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/Task1_code.ipynb)
- Jupyter notebook for task 2 can be accessed [here](https://github.com/ifashec123/AI-ML-Image-Class-Gridworld/blob/main/Task2_code.ipynb)

---
## 💡 Lessons & Takeaways
- Supervised Learning: Regularization techniques like dropout and L2 can significantly improve generalization in CNNs.

- Reinforcement Learning: Model-free methods like Q-Learning are more robust in unknown or complex environments.

- Hyperparameter Tuning: Systematic experiments help uncover the best configurations for performance and stability.
---
🙋 About Me
This project was completed as part of my MSc Data Science degree. I enjoy building smart, scalable solutions and learning how to make machine learning models not just accurate—but useful.

🔗 LinkedIn | 📫 Email | 💼 Portfolio coming soon!


