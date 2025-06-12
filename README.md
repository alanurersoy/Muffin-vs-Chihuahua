# 🐶🧁 Chihuahua vs. Muffin Classifier (2-Layer CNN)

A binary image classifier built with TensorFlow/Keras to distinguish chihuahuas from muffins using a **2-layer convolutional neural network (CNN)**. Includes data augmentation, performance evaluation, and dynamic threshold optimization.

---

## 🚀 **Features**
- **Data Augmentation**: `RandomFlip` + `RandomRotation` layers for robust training.
- **CNN Architecture**:
  - **Layer 1**: 16 Conv2D filters → MaxPooling2D → BatchNorm
  - **Layer 2**: 24 Conv2D filters → MaxPooling2D → BatchNorm
  - Flatten → Dense(8 units) → Sigmoid output
- **Evaluation**:
  - Accuracy metrics
  - Confusion matrix + ROC curve
  - **Threshold Optimization**: Automatically finds the best decision boundary using `np.argmax(tpr - fpr)`.

---

## ⚙️ **Setup**
1. **Install dependencies**:
   ```bash
   pip install tensorflow scikit-learn matplotlib
