# Deepfake Detection and Manipulation Classification using Xception and GradCAM++

## 1. Problem Statement
With the rapid growth of artificial intelligence and deep learning technologies, creating realistic fake images and videos has become easier than ever. These manipulated media files, commonly known as deepfakes, can closely resemble real content, making them difficult for people to identify. As a result, deepfakes have become a serious concern in areas such as social media, news reporting, cybersecurity, digital forensics, and public trust, where misleading visual content can spread misinformation and cause significant harm.
Most existing deepfake detection systems are designed only to determine whether an image or video is genuine or manipulated. Although this is useful, such systems often do not reveal how the manipulation was created or which deepfake technique was used. In addition, many deep learning models operate as "black boxes," providing predictions without explaining the reasoning behind them. This lack of transparency can reduce confidence in the system's decisions, especially in applications that require reliable evidence and verification.
To address these limitations, this project proposes an explainable multi-task deepfake detection framework based on the Xception architecture. Instead of performing a single task, the model simultaneously carries out two related tasks:
1.	Binary Classification – Identifying whether a facial image is Real or Fake.
2.	Manipulation Classification – Determining the specific deepfake generation method used, such as DeepFakes, FaceSwap, Face2Face, NeuralTextures, FaceShifter, or DeepFakeDetection.
To make the model's predictions more understandable, GradCAM++ is incorporated into the system. This technique generates visual explanations by highlighting the facial regions that have the greatest influence on the model's decisions. By combining effective facial preprocessing, multi-task learning, and explainable AI, the proposed framework not only detects deepfakes accurately but also provides meaningful insights into the detection process. This helps improve transparency, trustworthiness, and practical usability in digital media verification and forensic investigations.


## 2. Motivation
The increasing accessibility of deepfake generation tools has made the creation of realistic manipulated media easier than ever before. Modern deepfake techniques can generate highly convincing facial manipulations that are often difficult for humans to distinguish from authentic content. As a result, deepfakes have become a growing concern in areas such as social media, journalism, digital identity verification, cybersecurity, and digital forensics.
Most existing deepfake detection systems focus only on binary classification, determining whether content is real or fake. While such systems can identify manipulated media, they provide limited information about the nature of the forgery. In practical forensic investigations, understanding how an image was manipulated can be as important as detecting that it was manipulated.
This project was motivated by three key objectives:
1. Move Beyond Binary Deepfake Detection
Rather than only predicting whether an image is real or fake, the proposed system aims to identify the specific manipulation technique used to generate the forgery. This additional level of analysis provides more meaningful forensic information and encourages the model to learn richer manipulation-specific features.
2. Focus on Facial Manipulation Artifacts
Deepfake generation techniques primarily modify facial regions while leaving most background information unchanged. Training on complete video frames can cause a model to learn irrelevant patterns related to lighting, clothing, or scene context. To address this issue, face detection and cropping are incorporated into the preprocessing pipeline, ensuring that the model focuses on the most relevant visual evidence.
3. Improve Model Interpretability
Deep learning models often operate as black boxes, making it difficult to understand the reasoning behind their predictions. In security-sensitive applications, such lack of transparency can reduce trust in automated systems. To improve interpretability, GradCAM++ is integrated into the inference pipeline, enabling visualization of the facial regions that contribute most strongly to each prediction. These visual explanations help verify that the model is learning meaningful manipulation artifacts rather than relying on irrelevant image features.
Why Xception?
Xception was selected as the backbone architecture because of its strong performance in image forensics and deepfake detection tasks. Its depthwise separable convolution design enables efficient feature extraction while maintaining high representational power. Previous research has also demonstrated the effectiveness of Xception-based models on the FaceForensics++ benchmark, making it a suitable foundation for the proposed multi-task framework.
By combining facial preprocessing, multi-task learning, transfer learning, and explainable AI techniques, this project aims to develop a deepfake detection system that is not only accurate but also informative, interpretable, and practically useful for digital media verification.



## 3. Dataset

### FaceForensics++ Dataset

This project utilizes the FaceForensics++ (FF++) dataset, one of the most widely used benchmark datasets for deepfake detection and digital media forensics. The dataset contains both authentic and manipulated facial videos generated using multiple state-of-the-art face manipulation techniques, making it suitable for training and evaluating robust deepfake detection systems.

Unlike datasets that focus solely on binary deepfake classification, FaceForensics++ provides multiple manipulation categories, enabling the development of systems capable of identifying not only whether content is fake but also the specific technique used to generate the forgery.

### Dataset Categories

For this study, seven classes were used:

| Category          | Description                                                             |
| ----------------- | ----------------------------------------------------------------------- |
| Original          | Authentic, unaltered videos                                             |
| DeepFakes         | Identity swapping using deep learning-based face replacement            |
| FaceSwap          | Traditional face swapping technique                                     |
| Face2Face         | Facial expression transfer between individuals                          |
| NeuralTextures    | Neural rendering-based facial manipulation                              |
| FaceShifter       | Advanced face replacement using generative networks                     |
| DeepFakeDetection | Additional manipulated samples provided for deepfake detection research |

These categories allow the model to learn both general deepfake artifacts and manipulation-specific characteristics.

### Multi-Task Label Design

To support the proposed multi-task learning framework, each image is assigned two labels:

#### Binary Label

Used for deepfake detection:

| Class                   | Binary Label |
| ----------------------- | ------------ |
| Original                | 0            |
| All Manipulated Classes | 1            |

#### Manipulation Type Label

Used only for fake samples:

| Manipulation Technique | Type Label |
| ---------------------- | ---------- |
| DeepFakes              | 0          |
| FaceSwap               | 1          |
| Face2Face              | 2          |
| NeuralTextures         | 3          |
| FaceShifter            | 4          |
| DeepFakeDetection      | 5          |

This dual-label strategy enables the network to simultaneously learn whether an image is manipulated and identify the manipulation technique responsible for generating it.

### Dataset Preparation Strategy

The original FaceForensics++ dataset consists of videos rather than individual images. Training directly on raw videos introduces several challenges:

* High redundancy between adjacent frames
* Increased storage requirements
* Longer preprocessing and training times
* Potential bias toward specific facial poses and expressions

To address these issues, a dedicated preprocessing pipeline was designed to transform the raw videos into a balanced facial image dataset suitable for deep learning.

The preprocessing workflow includes:

1. Train-validation splitting at the video level (80:20 ratio)
2. Frame sampling from videos
3. Face detection using MediaPipe
4. Facial region extraction
5. Image resizing to 224 × 224 pixels
6. Dataset organization by manipulation category

This approach ensures that the model is trained on relevant facial information while maintaining diversity across identities, poses, and expressions.

### Why FaceForensics++?

FaceForensics++ was selected because it offers several advantages for deepfake detection research:

* Multiple manipulation techniques in a single dataset
* Large-scale collection of real and manipulated videos
* Widely adopted benchmark within the research community
* Availability of both classical and deep learning-based manipulations
* Suitable for both binary and multi-class forensic analysis

These characteristics make FaceForensics++ an ideal dataset for developing and evaluating an explainable multi-task deepfake detection framework.


## 4. Data Preprocessing Pipeline

The quality of a deepfake detection system depends not only on the model architecture but also on the quality of the data provided during training. Raw videos from FaceForensics++ contain redundant frames, varying lighting conditions, background clutter, and other factors that may negatively impact feature learning. Therefore, a dedicated preprocessing pipeline was designed to transform the raw video dataset into a structured collection of facial images suitable for deep learning.

The complete preprocessing workflow is illustrated below:

```text
Raw Videos
     ↓
Train/Validation Split
     ↓
Frame Sampling
(Every 30th Frame)
     ↓
MediaPipe Face Detection
     ↓
Face Cropping
     ↓
Image Resizing (224×224)
     ↓
Dataset Organization
```

### 4.1 Train-Validation Split

The original videos from each manipulation category were divided into training and validation sets using an 80:20 ratio.

This split was performed at the video level rather than the frame level to prevent data leakage. If frames from the same video appeared in both training and validation sets, the model could memorize video-specific patterns rather than learning generalized manipulation artifacts.

Benefits of video-level splitting:

* Prevents information leakage between datasets
* Produces a more realistic evaluation scenario
* Encourages better model generalization

### 4.2 Frame Sampling

Videos contain thousands of frames, many of which are nearly identical to their neighboring frames. Using every frame would significantly increase storage requirements and computational cost while contributing little additional information.

To address this issue, every 30th frame was extracted from each video.

This strategy offers several advantages:

* Reduces redundancy between consecutive frames
* Decreases preprocessing and training time
* Preserves diversity of facial expressions and poses
* Improves storage efficiency

By sampling frames at regular intervals, the dataset maintains visual diversity while eliminating unnecessary repetition.

### 4.3 Face Detection using MediaPipe

Deepfake manipulations primarily affect facial regions. Background objects, clothing, and environmental features typically remain unchanged and may introduce noise during training.

To focus the model on relevant forensic evidence, MediaPipe Face Detection was employed to locate facial regions within each sampled frame.

MediaPipe was selected because it provides:

* Fast real-time face detection
* High detection accuracy
* Robust performance across varying poses and lighting conditions
* Lightweight computational requirements

Only frames with successfully detected faces were retained for further processing.

### 4.4 Face Cropping

After face detection, the bounding box coordinates provided by MediaPipe were used to crop the facial region from the original frame.

This step serves several important purposes:

* Removes irrelevant background information
* Reduces the influence of scene-specific artifacts
* Forces the model to focus on facial manipulation traces
* Improves feature quality for deepfake detection

Training directly on facial regions enables the model to concentrate on subtle inconsistencies such as texture distortions, blending artifacts, unnatural boundaries, and synthetic facial patterns.

### 4.5 Image Resizing

The extracted facial regions were resized to a fixed resolution of 224 × 224 pixels.

This resolution was chosen because:

* It matches the expected input dimensions of the pretrained Xception network
* It reduces computational requirements
* It ensures consistency across all samples

Standardizing image dimensions allows efficient batch processing during training and inference.

### 4.6 Dataset Organization

The processed images were organized into category-specific directories corresponding to their manipulation type.

The final dataset structure consists of:

* Original
* DeepFakes
* FaceSwap
* Face2Face
* NeuralTextures
* FaceShifter
* DeepFakeDetection

Each image inherits both a binary label (Real/Fake) and a manipulation type label, supporting the multi-task learning framework used in this project.

### Impact of the Preprocessing Pipeline

The combination of video-level splitting, frame sampling, face detection, face cropping, and image normalization creates a high-quality training dataset that emphasizes relevant forensic information while minimizing noise and redundancy.

This preprocessing strategy provides several benefits:

* Improved training efficiency
* Reduced dataset redundancy
* Better feature learning
* Enhanced model generalization
* Stronger focus on manipulation artifacts

As a result, the model is able to learn more discriminative features for both deepfake detection and manipulation classification.


## 5. Data Augmentation

## 6. Model Architecture

## 7. Multi-Task Learning Strategy

## 8. Training Methodology

## 9. Explainability with GradCAM++

## 10. Results

## 11. Limitations

## 12. Future Work
