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

Deep learning models often suffer from overfitting when trained on limited variations of data. Although the FaceForensics++ dataset contains a large number of samples, real-world deepfake content can appear under different lighting conditions, compression levels, camera settings, and facial orientations. To improve the robustness and generalization capability of the proposed model, several data augmentation techniques were applied during training.

The augmentation pipeline was designed to simulate realistic variations that may be encountered in practical deployment scenarios while preserving the forensic artifacts necessary for deepfake detection.

### Augmentation Pipeline

The following transformations were applied to training images:

```python
transforms.Resize((224,224))
transforms.RandomHorizontalFlip()
transforms.ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1
)
transforms.ToTensor()
transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
)
```

### 5.1 Image Resizing

All facial images were resized to 224 × 224 pixels before being passed to the model.

This ensures:

* Consistent input dimensions
* Efficient batch processing
* Compatibility with the pretrained Xception backbone

Standardized image dimensions help maintain stable training and reduce computational overhead.

### 5.2 Random Horizontal Flipping

A random horizontal flip was applied during training.

Facial structures are largely symmetrical, meaning that a manipulated face remains manipulated regardless of whether it appears on the left or right side of an image.

Benefits include:

* Increased data diversity
* Reduced orientation bias
* Improved robustness to pose variations
* Better generalization to unseen samples

By exposing the model to mirrored versions of the same face, the network learns more invariant and generalized features.

### 5.3 Color Jittering

Real-world videos often experience changes in:

* Brightness
* Contrast
* Saturation
* Color balance

Additionally, deepfake videos distributed through social media platforms may undergo multiple rounds of compression and re-encoding, introducing visual variations not present in the original dataset.

To simulate these conditions, ColorJitter augmentation was applied with controlled variations in:

* Brightness
* Contrast
* Saturation
* Hue

This augmentation helps the model become less sensitive to environmental and recording conditions while maintaining its ability to identify manipulation artifacts.

### 5.4 Conversion to Tensor Format

Images were converted into PyTorch tensors to enable efficient GPU processing and compatibility with the deep learning training pipeline.

This transformation converts pixel values into a format suitable for neural network computations.

### 5.5 ImageNet Normalization

Since the Xception backbone was initialized using pretrained ImageNet weights, the input images were normalized using the standard ImageNet mean and standard deviation values:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

Normalization offers several advantages:

* Aligns the input distribution with pretrained weights
* Improves convergence speed
* Stabilizes training
* Enhances transfer learning effectiveness

Without proper normalization, the pretrained feature representations learned from ImageNet may not transfer effectively to the deepfake detection task.

### Validation Data Processing

Unlike the training dataset, validation images were not augmented.

Only the following transformations were applied:

* Resize to 224 × 224
* Conversion to tensor format
* ImageNet normalization

This ensures that validation performance reflects the model's true generalization ability rather than artificially modified samples.

### Impact of Data Augmentation

The augmentation strategy helps the model learn more robust and transferable features by exposing it to a wider range of visual variations during training.

Key benefits include:

* Reduced overfitting
* Improved generalization
* Better robustness to lighting variations
* Increased tolerance to compression artifacts
* Improved performance on unseen deepfake samples

Combined with face-based preprocessing and transfer learning, these augmentation techniques contribute significantly to the reliability and effectiveness of the proposed deepfake detection framework.


## 6. Model Architecture

The proposed deepfake detection framework is built upon the Xception architecture and follows a multi-task learning paradigm. Rather than training separate models for deepfake detection and manipulation classification, a single shared backbone network is used to learn common forensic features, while two specialized classification heads perform task-specific predictions.

This design improves feature utilization, reduces computational complexity, and encourages the model to learn richer representations of manipulated facial content.

### Architecture Overview

The overall architecture consists of three major components:

```text
Input Face Image (224 × 224)
            │
            ▼
    Xception Backbone
            │
            ▼
      Feature Vector
            │
    ┌───────┴────────┐
    ▼                ▼

Binary Head      Type Head

Real/Fake      DeepFakes
               FaceSwap
               Face2Face
               NeuralTextures
               FaceShifter
               DeepFakeDetection
```

### 6.1 Why Xception?

Xception (Extreme Inception) is a convolutional neural network architecture that replaces traditional convolutions with depthwise separable convolutions. This design allows the network to learn spatial and channel-wise features more efficiently while reducing the number of parameters.

Xception was selected for several reasons:

* Proven effectiveness in deepfake detection research
* Strong performance on FaceForensics++ benchmarks
* Efficient feature extraction through depthwise separable convolutions
* Availability of pretrained ImageNet weights
* Ability to capture subtle facial manipulation artifacts

Deepfake detection often relies on identifying extremely small inconsistencies in textures, boundaries, lighting, and facial details. Xception has consistently demonstrated strong capability in learning such fine-grained visual features.

### 6.2 Transfer Learning Strategy

Training a deep neural network entirely from scratch requires a massive amount of labeled data and computational resources.

To overcome this challenge, transfer learning was employed.

The model was initialized using pretrained ImageNet weights:

```python
self.backbone = timm.create_model(
    "xception",
    pretrained=True
)
```

The pretrained network already possesses a strong understanding of general visual patterns such as:

* Edges
* Textures
* Shapes
* Spatial relationships

These learned representations provide a strong starting point for deepfake detection and significantly accelerate convergence.

### 6.3 Feature Extraction Backbone

The original classification layer of Xception was removed:

```python
self.backbone.reset_classifier(0)
```

After removal, the backbone acts purely as a feature extractor.

For each input image, the network produces a high-dimensional feature representation containing forensic information learned from the facial image.

These extracted features are then shared across both downstream tasks.

### 6.4 Binary Classification Head

The first task is binary deepfake detection.

A fully connected layer is attached to the backbone output:

```python
self.binary_head = nn.Linear(
    in_features,
    1
)
```

This head predicts:

| Output | Meaning |
| ------ | ------- |
| 0      | Real    |
| 1      | Fake    |

A sigmoid activation is applied during inference to obtain the probability that an image is manipulated.

The binary branch learns generalized forensic features that distinguish authentic content from manipulated content.

### 6.5 Manipulation Classification Head

The second task identifies the manipulation technique responsible for generating a fake image.

A second fully connected layer is attached to the same backbone:

```python
self.type_head = nn.Linear(
    in_features,
    NUM_TYPES
)
```

The model predicts one of six manipulation categories:

| Class ID | Manipulation Type |
| -------- | ----------------- |
| 0        | DeepFakes         |
| 1        | FaceSwap          |
| 2        | Face2Face         |
| 3        | NeuralTextures    |
| 4        | FaceShifter       |
| 5        | DeepFakeDetection |

This branch encourages the backbone to learn manipulation-specific characteristics rather than only distinguishing real from fake.

### 6.6 Multi-Task Learning Framework

Traditional deepfake detectors perform only binary classification.

In contrast, the proposed architecture simultaneously learns:

#### Task 1: Deepfake Detection

```text
Real vs Fake
```

#### Task 2: Manipulation Classification

```text
DeepFakes
FaceSwap
Face2Face
NeuralTextures
FaceShifter
DeepFakeDetection
```

The shared backbone enables both tasks to benefit from common facial forensic features.

Advantages of this approach include:

* Better feature sharing
* Reduced model complexity
* Improved learning efficiency
* Stronger manipulation awareness
* Increased interpretability

By learning both tasks simultaneously, the network develops richer representations than a conventional binary classifier.

### 6.7 Forward Pass

For each input image:

1. The image is passed through the Xception backbone.
2. A feature vector is extracted.
3. The feature vector is simultaneously forwarded to:

   * Binary Classification Head
   * Manipulation Classification Head
4. Both outputs are produced in a single forward pass.

Implementation:

```python
def forward(self, x):

    features = self.backbone(x)

    binary_out = self.binary_head(features)

    type_out = self.type_head(features)

    return binary_out, type_out
```

### Architectural Advantages

The proposed architecture combines the strengths of transfer learning and multi-task learning to build an efficient and explainable deepfake detection system.

Key benefits include:

* Strong forensic feature extraction through Xception
* Simultaneous detection and manipulation classification
* Efficient parameter sharing between tasks
* Reduced training complexity
* Improved generalization capability
* Compatibility with GradCAM++ explainability methods

This architecture serves as the foundation of the proposed framework and enables both accurate deepfake detection and detailed manipulation analysis.


## 7. Multi-Task Learning Strategy


Traditional deepfake detection systems are typically designed as binary classifiers that determine whether an image is real or manipulated. While such approaches can achieve strong detection performance, they provide limited information about the nature of the forgery and often fail to capture manipulation-specific characteristics.

To overcome these limitations, this project adopts a multi-task learning strategy in which a single neural network is trained to perform two related tasks simultaneously:

### Task 1: Binary Deepfake Detection

The primary objective is to determine whether a facial image is authentic or manipulated.

```text
Real → 0
Fake → 1
```

This task enables the model to learn general forensic patterns that distinguish manipulated content from authentic content.

### Task 2: Manipulation Type Classification

The secondary objective is to identify the specific manipulation technique responsible for generating a fake image.

```text
DeepFakes          → 0
FaceSwap           → 1
Face2Face          → 2
NeuralTextures     → 3
FaceShifter        → 4
DeepFakeDetection  → 5
```

This task encourages the model to learn manipulation-specific artifacts and subtle differences between various deepfake generation methods.

### Why Multi-Task Learning?

A standard binary classifier only learns features necessary to separate real and fake samples. However, different deepfake generation techniques introduce different visual artifacts.

For example:

* DeepFakes often produce blending inconsistencies around facial boundaries.
* FaceSwap may introduce texture mismatches between source and target faces.
* Face2Face manipulates facial expressions while preserving identity.
* NeuralTextures generates synthetic facial details through neural rendering.

By forcing the model to recognize these manipulation types, the shared backbone learns richer and more discriminative forensic representations.

This provides several advantages:

* Better feature learning
* Improved generalization
* Reduced overfitting
* More informative predictions
* Increased forensic interpretability

### Shared Feature Learning

Both tasks utilize the same Xception backbone.

```text
Input Face
      │
      ▼
Xception Backbone
      │
 ┌────┴────┐
 ▼         ▼

Binary   Manipulation
Head       Head
```

The backbone learns facial forensic features only once, and these features are shared across both tasks.

This design is more efficient than training two independent models and allows both tasks to benefit from common learned representations.

### Loss Function Design

Training is performed using a combination of two loss functions:

#### Binary Classification Loss

The binary detection branch uses Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss):

```python
binary_loss = BCEWithLogitsLoss()
```

This loss measures the model's ability to distinguish between real and fake images.

#### Manipulation Classification Loss

The manipulation classification branch uses Cross Entropy Loss:

```python
type_loss = CrossEntropyLoss()
```

This loss evaluates the model's ability to correctly identify the manipulation technique.

### Selective Manipulation Loss

One of the key design decisions in this project is that manipulation classification loss is calculated only for fake images.

```python
fake_mask = binary_label == 1

if fake_mask.sum() > 0:
    type_loss = ce_loss(
        type_out[fake_mask],
        type_label[fake_mask]
    )
```

This approach was adopted because real images do not belong to any manipulation category.

Assigning manipulation labels to authentic images would introduce artificial relationships and could encourage the model to learn incorrect patterns.

By restricting manipulation classification to fake samples only:

* Real images contribute only to binary learning.
* Fake images contribute to both tasks.
* The model learns cleaner manipulation-specific features.
* Training becomes more stable and meaningful.

### Combined Objective Function

The final training objective is the sum of both task losses:

```text
Total Loss =
Binary Detection Loss
+
Manipulation Classification Loss
```

Conceptually:

```text
Ltotal = Lbinary + Ltype
```

The binary loss ensures accurate deepfake detection, while the manipulation loss encourages detailed forensic understanding of different forgery techniques.

### Benefits of the Proposed Strategy

The multi-task learning framework offers several advantages over traditional single-task deepfake detectors:

* Simultaneous detection and manipulation classification
* Improved feature sharing across tasks
* Better utilization of training data
* Stronger forensic feature extraction
* Reduced computational complexity compared to separate models
* More informative predictions for forensic analysis

By combining binary detection and manipulation classification within a unified framework, the proposed model learns both general and manipulation-specific forensic cues, resulting in a more robust and explainable deepfake detection system.

## 8. Training Methodology

The training process was designed to maximize the effectiveness of transfer learning while ensuring stable optimization, efficient GPU utilization, and strong generalization performance. The proposed multi-task framework was trained using the preprocessed FaceForensics++ dataset, enabling the model to simultaneously learn deepfake detection and manipulation classification.

### Training Configuration

The following configuration was used during training:

| Parameter                | Value                                |
| ------------------------ | ------------------------------------ |
| Backbone                 | Xception                             |
| Input Size               | 224 × 224                            |
| Batch Size               | 64                                   |
| Optimizer                | AdamW                                |
| Learning Rate            | 3 × 10⁻⁴                             |
| Maximum Epochs           | 80                                   |
| Loss Functions           | BCEWithLogitsLoss + CrossEntropyLoss |
| Mixed Precision Training | Enabled                              |
| Early Stopping           | Enabled                              |
| Device                   | GPU (CUDA)                           |

These settings were selected to balance training stability, convergence speed, and computational efficiency.

### Data Loading Strategy

The processed dataset was loaded using PyTorch DataLoaders.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)
```

The training dataset was shuffled during each epoch to prevent the model from learning ordering patterns, while the validation dataset remained fixed to ensure consistent evaluation.

### Optimization using AdamW

The AdamW optimizer was selected for model training.

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4
)
```

AdamW offers several advantages:

* Adaptive learning rates
* Faster convergence
* Better handling of sparse gradients
* Decoupled weight decay regularization
* Improved generalization performance

Compared to standard Adam, AdamW helps reduce overfitting while maintaining efficient optimization.

### Mixed Precision Training

To improve training efficiency, Automatic Mixed Precision (AMP) was employed.

```python
scaler = torch.cuda.amp.GradScaler()
```

Training was performed using:

```python
with torch.cuda.amp.autocast():
```

Mixed precision training provides several benefits:

* Reduced GPU memory consumption
* Faster computation
* Larger effective batch sizes
* Improved hardware utilization

This approach allows the model to train more efficiently without sacrificing predictive performance.

### Forward Pass

For each training batch:

1. Images are passed through the Xception backbone.
2. Shared feature representations are extracted.
3. Binary predictions are generated.
4. Manipulation-type predictions are generated.
5. Multi-task loss is computed.
6. Gradients are backpropagated.
7. Model parameters are updated.

Workflow:

```text
Input Images
      │
      ▼
Xception Backbone
      │
      ▼
Feature Representation
      │
 ┌────┴────┐
 ▼         ▼

Binary   Type
Head     Head
      │
      ▼
Combined Loss
      │
      ▼
Backpropagation
```

### Loss Optimization

The final loss consists of two components:

#### Binary Detection Loss

Measures performance on:

```text
Real vs Fake
```

using BCEWithLogitsLoss.

#### Manipulation Classification Loss

Measures performance on:

```text
DeepFakes
FaceSwap
Face2Face
NeuralTextures
FaceShifter
DeepFakeDetection
```

using CrossEntropyLoss.

The total objective function is:

```text
Total Loss = Binary Loss + Type Loss
```

This enables simultaneous optimization of both tasks during a single training process.

### Performance Monitoring

During training, three key metrics were monitored:

#### Training Loss

Measures how effectively the model is minimizing the objective function.

#### Binary Accuracy

Measures performance on:

```text
Real vs Fake
```

classification.

#### Manipulation Classification Accuracy

Measures performance on:

```text
Manipulation Type Prediction
```

for fake samples only.

Monitoring both tasks independently helps identify whether improvements in one task negatively affect the other.

### Model Checkpointing

The validation loss was used to determine whether the model improved during training.

Whenever validation loss decreased:

```python
torch.save(
    model.state_dict(),
    "best_model_multi.pth"
)
```

The best-performing model was automatically saved.

This ensures that the final deployed model corresponds to the strongest validation performance rather than simply the final training epoch.

### Early Stopping Strategy

To prevent overfitting, an early stopping mechanism was implemented.

```python
patience = 3
```

If validation loss failed to improve for three consecutive epochs:

```text
Training Stops
```

Benefits include:

* Reduced overfitting
* Faster experimentation
* Improved generalization
* Prevention of unnecessary computation

This strategy allows training to terminate automatically once the model stops learning meaningful patterns.

### Training Objectives

The training methodology was designed to achieve the following goals:

* Learn robust deepfake detection features
* Identify manipulation-specific artifacts
* Prevent overfitting
* Improve computational efficiency
* Maximize transfer learning benefits
* Produce a model suitable for explainable AI analysis

By combining transfer learning, multi-task optimization, mixed precision training, checkpointing, and early stopping, the proposed framework achieves efficient and stable training while learning both generalized and manipulation-specific forensic representations.


## 9. Explainability with GradCAM++

### Why Explainability Matters

Deep learning models have achieved remarkable success in deepfake detection; however, they often operate as "black boxes," providing predictions without explaining the reasoning behind them. In security-sensitive applications such as digital forensics, misinformation detection, and media verification, relying solely on prediction scores is often insufficient.

A deepfake detector should not only determine whether an image is manipulated but should also provide evidence supporting its decision. Understanding which facial regions influence the model's prediction helps validate the reliability of the system and increases trust in its outputs.

To address this challenge, GradCAM++ was integrated into the inference pipeline of the proposed framework.

### What is GradCAM++?

Gradient-weighted Class Activation Mapping Plus Plus (GradCAM++) is an explainable AI (XAI) technique that generates visual explanations for convolutional neural networks.

GradCAM++ works by:

1. Computing gradients flowing through a target convolutional layer.
2. Measuring the contribution of feature maps to the final prediction.
3. Generating a localization map that highlights important image regions.
4. Overlaying the heatmap on the original image for visual interpretation.

The resulting heatmap reveals which parts of the image the model considers most important when making a decision.

### Why GradCAM++ Instead of Standard GradCAM?

GradCAM++ was selected because it offers several improvements over standard GradCAM:

* Better localization of important regions
* Improved handling of multiple activation regions
* More precise heatmaps
* Stronger visual interpretability

These advantages make GradCAM++ particularly suitable for deepfake detection, where manipulation artifacts may appear in multiple facial regions simultaneously.

### Integration into the Proposed Framework

During inference, the trained Xception model generates a binary prediction indicating whether the image is real or fake.

To visualize the reasoning behind this prediction:

1. The input image is passed through the trained model.
2. Gradients are computed with respect to a selected convolutional layer.
3. A GradCAM++ heatmap is generated.
4. The heatmap is overlaid on the original facial image.

Inference workflow:

```text
Input Face Image
        │
        ▼
Xception Network
        │
        ▼
Prediction
(Real/Fake)
        │
        ▼
GradCAM++
        │
        ▼
Activation Heatmap
        │
        ▼
Visual Explanation
```

### Target Layer Selection

The final convolutional feature extraction stage of the Xception backbone was selected as the target layer:

```python
target_layer = model.backbone.conv4
```

This layer contains high-level semantic information while still preserving spatial localization, making it suitable for generating meaningful visual explanations.

### Binary Prediction Visualization

Since GradCAM++ is applied to the binary classification branch, the generated heatmaps highlight the facial regions that most strongly influence the model's real/fake decision.

For manipulated images, the network often focuses on:

* Mouth region
* Eye region
* Facial boundaries
* Skin texture inconsistencies
* Blending artifacts
* Synthetic texture patterns

These areas frequently contain visual clues introduced during deepfake generation.

### Interpretation of Heatmaps

The generated heatmaps provide valuable insight into the model's behavior.

#### High Activation Regions

Areas displayed with stronger intensity indicate regions that contribute significantly to the prediction.

#### Low Activation Regions

Areas with lower intensity have minimal influence on the model's decision-making process.

A reliable deepfake detector should primarily focus on manipulated facial regions rather than unrelated background elements.

### Benefits of Explainability

Integrating GradCAM++ provides several advantages:

* Improves model transparency
* Increases user trust
* Enables forensic analysis
* Assists in model debugging
* Helps verify meaningful feature learning
* Detects potential model biases

These visual explanations help ensure that the network is learning genuine manipulation artifacts rather than exploiting unintended correlations within the dataset.

### Example Analysis

In one of the generated predictions, the model classified an image as a manipulated Face2Face sample with a very high confidence score.

The corresponding GradCAM++ visualization highlighted regions around the mouth and lower facial area. This observation aligns with the nature of Face2Face manipulations, which primarily modify facial expressions while preserving identity.

Such behavior suggests that the model has learned meaningful forensic features rather than relying on irrelevant visual information.

### Contribution to the Overall Framework

The integration of GradCAM++ transforms the proposed system from a simple classification model into an explainable deepfake analysis framework.

Instead of providing only a prediction label, the system offers visual evidence supporting its decision. This additional layer of interpretability is particularly valuable in real-world applications where understanding the reasoning behind an automated decision is as important as the prediction itself.

By combining accurate classification with visual explainability, the proposed framework provides a more trustworthy and informative approach to deepfake detection and manipulation analysis.


## 10. Results

## 11. Limitations

## 12. Future Work
