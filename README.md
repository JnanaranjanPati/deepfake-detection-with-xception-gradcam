# Deepfake Detection and Manipulation Classification using Xception and GradCAM++

## 1. Problem Statement
With the rapid growth of artificial intelligence and deep learning technologies, creating realistic fake images and videos has become easier than ever. These manipulated media files, commonly known as deepfakes, can closely resemble real content, making them difficult for people to identify. As a result, deepfakes have become a serious concern in areas such as social media, news reporting, cybersecurity, digital forensics, and public trust, where misleading visual content can spread misinformation and cause significant harm.
Most existing deepfake detection systems are designed only to determine whether an image or video is genuine or manipulated. Although this is useful, such systems often do not reveal how the manipulation was created or which deepfake technique was used. In addition, many deep learning models operate as "black boxes," providing predictions without explaining the reasoning behind them. This lack of transparency can reduce confidence in the system's decisions, especially in applications that require reliable evidence and verification.
To address these limitations, this project proposes an explainable multi-task deepfake detection framework based on the Xception architecture. Instead of performing a single task, the model simultaneously carries out two related tasks:
1.	Binary Classification – Identifying whether a facial image is Real or Fake.
2.	Manipulation Classification – Determining the specific deepfake generation method used, such as DeepFakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter, or DeepFakeDetection.
To make the model's predictions more understandable, GradCAM++ is incorporated into the system. This technique generates visual explanations by highlighting the facial regions that have the greatest influence on the model's decisions. By combining effective facial preprocessing, multi-task learning, and explainable AI, the proposed framework not only detects deepfakes accurately but also provides meaningful insights into the detection process. This helps improve transparency, trustworthiness, and practical usability in digital media verification and forensic investigations.


## 2. Motivation

## 3. Dataset

## 4. Data Preprocessing Pipeline

## 5. Data Augmentation

## 6. Model Architecture

## 7. Multi-Task Learning Strategy

## 8. Training Methodology

## 9. Explainability with GradCAM++

## 10. Results

## 11. Limitations

## 12. Future Work
