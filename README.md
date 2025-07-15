#  Federated Learning for Retinal Disease Detection

## 🔖 Overview

This project implements a **Federated Learning** pipeline using **ConvNeXt** and **ResNet50** models with attention mechanisms for **retinal disease detection**. The focus is on training privacy-preserving models across multiple clients using **TensorFlow Federated**, extracting global weights, and applying them to centralized models.

---

## 🔧 Project Components

### 1. 🔳 ConvNeXt-Based Model

**Techniques used:**

* Global Average Pooling
* Batch Normalization
* Dropout
* Dense layers with ReLU
* L2 Regularization
* Learning Rate Scheduling

**Performance:**

* Training Accuracy: 99.96%
* Validation Accuracy: 93.79%
* Test Accuracy: 99.37%

---

### 2. 🔳 ResNet50 with Channel Attention

**Techniques used:**

* ResNet50 Backbone
* Channel Attention Mechanism
* Global Average Pooling
* Dropout
* L2 Regularization
* Batch Normalization
* Learning Rate Scheduling

**Performance:**

* Training Accuracy: 92.89%
* Validation Accuracy: 84.29%
* Test Accuracy: 91.95%

---

## 🤝 Federated Learning Summary

**Framework:** TensorFlow Federated

**Technical Highlights:**

* TFRecord Parsing and Feature Dimension Inference
* Client-specific Local Datasets
* Centralized Federated Averaging Process
* Sparse Categorical Crossentropy Loss
* SGDM Optimizer for both client and server
* Federated Averaging across 3 clients

**Federated Rounds:** 200

**Final Round Performance:**

* Accuracy: 92.34%
* Validation Accuracy: 83.20%
* Loss: 0.34

**Global Model Storage:**

* Model parameters were extracted and saved post-training
* Used for transfer learning and evaluation in centralized settings

---

## 🚀 Key Features

* Distributed Federated Learning
* Attention-enhanced CNNs (ResNet + ConvNeXt)
* Weight Extraction and Reuse
* Privacy-preserving Training Architecture

---

## 📚 Tools & Libraries

* TensorFlow & TensorFlow Addons
* TensorFlow Federated
* NumPy
* Matplotlib

---

## 📊 Results Summary

| Model        | Training Accuracy | Validation Accuracy | Test Accuracy | Loss |
| ------------ | ----------------- | ------------------- | ------------- | ---- |
| ConvNeXt     | 99.96%            | 93.79%              | 99.37%        | 0.65 |
| ResNet+Attn  | 92.89%            | 84.29%              | 91.95%        | 0.28 |
| Federated FL | 92.34%            | 83.20%              | —             | 0.34 |

---

## 📁 Dataset Links 
for baseline models : https://drive.google.com/drive/folders/1OgXAVxoM4degJE9y4jByAJ6pkyPpf1im?usp=sharing
for federated learning model with 3 clinents :
https://drive.google.com/drive/folders/1jcsCVURrtbKzSBXzrcXYt8zvG--vsRDE?usp=drive_link

---

## 🌟 Conclusion

This pipeline demonstrates how federated learning can effectively combine deep learning models such as ConvNeXt and ResNet with attention mechanisms to enable high-performance, privacy-respecting medical image classification.

---

> 📅 Developed by Chandragi Sairam
> for queries : sairamchandragi@gmail.com 
• Powered by TensorFlow Federated
