# Deep Learning-Based Recognition of American Sign Language Hand Gestures Using Convolutional Neural Networks

**Running Head:** ASL GESTURE RECOGNITION WITH CNN

---

## Title Page

# Deep Learning-Based Recognition of American Sign Language Hand Gestures Using Convolutional Neural Networks

**Team 3 - AAI521 Applied Computer Vision in AI**

Yogesh Sangwikar, Evin Joy, Eesha Kulkarni

---

### Author Note

**Funding:** This research was conducted as part of the AAI521 Applied Computer Vision in AI course project. No external funding was received for this study.

**Conflicts of Interest:** The authors declare no conflicts of interest.

**Correspondence:** Correspondence concerning this article should be addressed to Team 3, AAI521 Applied Computer Vision in AI. Email: [Contact information to be provided]

---

## Abstract

American Sign Language (ASL) serves as a primary means of communication for individuals who are deaf or hard of hearing. This study presents a deep learning approach for recognizing static ASL hand gestures representing alphabets A–Y (excluding J and Z, which require dynamic motion). We developed a convolutional neural network (CNN) architecture to classify 24 distinct ASL letter gestures from grayscale images. The model was trained on the Sign Language MNIST dataset, comprising 27,455 training samples and 7,172 test samples. Our CNN architecture employed three convolutional blocks with batch normalization, dropout regularization, and data augmentation techniques. The model achieved perfect classification accuracy (100%) on the test set, with a test loss of 7.65 × 10⁻⁶ and top-3 accuracy of 100%. All 24 classes demonstrated perfect F1-scores of 1.0000, and Cohen's Kappa coefficient reached 1.0000, indicating perfect inter-rater agreement. The model architecture contained 620,920 trainable parameters. These results demonstrate the effectiveness of deep learning approaches for ASL gesture recognition, with potential applications in accessibility technology and human-computer interaction systems.

**Keywords:** American Sign Language, deep learning, convolutional neural networks, gesture recognition, computer vision, accessibility technology

---

## Introduction

American Sign Language (ASL) is a complete, natural language used by deaf and hard-of-hearing communities in the United States and parts of Canada (Liddell & Johnson, 1989). ASL employs hand gestures, facial expressions, and body movements to convey meaning, with static hand configurations representing individual letters of the alphabet. The development of automated systems capable of recognizing ASL gestures has significant implications for accessibility technology, enabling improved communication interfaces and educational tools (Kumar et al., 2019).

Recent advances in deep learning, particularly convolutional neural networks (CNNs), have shown remarkable success in image classification tasks (LeCun et al., 2015). CNNs are particularly well-suited for gesture recognition due to their ability to automatically learn hierarchical feature representations from raw pixel data (Krizhevsky et al., 2012). Previous research has demonstrated the effectiveness of CNNs for sign language recognition, though most studies have focused on limited gesture sets or video-based dynamic gestures (Rastgoo et al., 2021).

The present study addresses the recognition of static ASL letter gestures using a CNN-based approach. We focused on 24 letters (A–Y, excluding J and Z, which require motion) to create a comprehensive classification system. Our approach utilized the Sign Language MNIST dataset, which provides standardized grayscale images of ASL hand gestures, enabling reproducible research and fair comparison with existing methods.

The primary objectives of this research were: (a) to design an effective CNN architecture for ASL gesture classification, (b) to evaluate model performance using comprehensive metrics including accuracy, F1-score, and Cohen's Kappa, and (c) to develop a real-time prediction system capable of processing webcam input for practical applications.

---

## Method

### Participants and Materials

#### Dataset

The Sign Language MNIST dataset (Kaggle, n.d.) served as the primary data source for this study. This publicly available dataset contains grayscale images of ASL hand gestures, with each image sized at 28 × 28 pixels. The dataset includes:

- **Training set:** 27,455 samples
- **Test set:** 7,172 samples
- **Classes:** 24 ASL letters (A–Y, excluding J and Z)
- **Image format:** Grayscale, 28 × 28 pixels, normalized to [0, 1] range

The dataset was obtained from Kaggle (https://www.kaggle.com/datasets/datamunge/sign-language-mnist) under CC0-1.0 license, ensuring open access for research purposes.

#### Computational Environment

All experiments were conducted using Python 3.13 with the following key libraries:
- TensorFlow 2.20.0 (Abadi et al., 2016) for deep learning model development
- Keras API for high-level neural network construction
- OpenCV for image preprocessing and real-time video processing
- scikit-learn for evaluation metrics and data splitting
- NumPy and Pandas for data manipulation

Experiments were performed on a local computing environment. Random seeds were set to 42 for both NumPy and TensorFlow to ensure reproducibility across runs.

### Procedure

#### Data Preprocessing

The preprocessing pipeline involved several critical steps to prepare the data for model training. First, labels were remapped to exclude letters J (label 9) and Z (label 25, if present), resulting in 24 classes. Labels 0–8 (A–I) remained unchanged, while labels 10–24 (K–Y) were shifted down by one position to fill the gap left by excluding J.

Pixel values were normalized from the original [0, 255] range to [0, 1] by dividing by 255.0, which is standard practice for neural network training as it improves convergence stability (LeCun et al., 1998). Images were reshaped from flattened 784-pixel vectors to 28 × 28 × 1 tensors to match the CNN input requirements.

Labels were one-hot encoded to 24-dimensional vectors for categorical cross-entropy loss computation. The training set was further split into training (85%) and validation (15%) subsets using stratified sampling to maintain class distribution across splits.

#### Model Architecture

We designed a CNN architecture consisting of three convolutional blocks, each followed by batch normalization, max pooling, and dropout layers. The architecture can be described as follows:

**Block 1:**
- Two Conv2D layers with 32 filters (3 × 3 kernel, ReLU activation, same padding)
- Batch normalization after each convolutional layer
- Max pooling (2 × 2)
- Dropout (0.25)

**Block 2:**
- Two Conv2D layers with 64 filters (3 × 3 kernel, ReLU activation, same padding)
- Batch normalization after each convolutional layer
- Max pooling (2 × 2)
- Dropout (0.25)

**Block 3:**
- Two Conv2D layers with 128 filters (3 × 3 kernel, ReLU activation, same padding)
- Batch normalization after each convolutional layer
- Max pooling (2 × 2)
- Dropout (0.25)

**Classification Head:**
- Flatten layer
- Dense layer (256 units, ReLU activation) with batch normalization and dropout (0.5)
- Dense layer (128 units, ReLU activation) with batch normalization and dropout (0.5)
- Output layer (24 units, softmax activation)

The complete model contained 620,920 trainable parameters. Batch normalization was employed to stabilize training and reduce internal covariate shift (Ioffe & Szegedy, 2015), while dropout layers served to prevent overfitting by randomly deactivating neurons during training (Srivastava et al., 2014).

#### Training Procedure

The model was compiled using the Adam optimizer (Kingma & Ba, 2014) with a learning rate of 0.001. Categorical cross-entropy served as the loss function, appropriate for multi-class classification tasks. We tracked three metrics during training: accuracy, top-3 accuracy, and loss.

Data augmentation was applied to improve model generalization. The augmentation pipeline included:
- Random rotation (±10°)
- Random horizontal and vertical shifts (±10%)
- Random zoom (±10%)
- Random shear transformation (±10%)
- Nearest-neighbor fill mode for transformed pixels

Training was conducted for up to 50 epochs with a batch size of 128. Several callbacks were implemented to optimize training:

1. **Model Checkpoint:** Saved the best model based on validation accuracy
2. **Early Stopping:** Monitored validation loss with patience of 10 epochs
3. **Reduce Learning Rate on Plateau:** Reduced learning rate by factor of 0.5 when validation loss plateaued (patience: 5 epochs, minimum learning rate: 1 × 10⁻⁷)
4. **TensorBoard Logging:** Recorded training metrics for visualization

The training process utilized the augmented data generator, which created new training samples on-the-fly during each epoch, effectively increasing the training dataset size and improving model robustness.

#### Evaluation Metrics

Model performance was assessed using multiple metrics to provide comprehensive evaluation:

1. **Accuracy:** Overall classification accuracy on the test set
2. **Top-3 Accuracy:** Percentage of test samples where the correct class appeared in the top-3 predictions
3. **F1-Score:** Macro-averaged and weighted F1-scores across all classes
4. **Cohen's Kappa:** Inter-rater agreement coefficient, accounting for chance agreement
5. **Per-Class Metrics:** Individual precision, recall, and F1-score for each of the 24 classes
6. **Confusion Matrix:** Detailed breakdown of classification performance across all class pairs

These metrics were computed using scikit-learn's classification_report and confusion_matrix functions, providing both aggregate and class-specific performance insights.

---

## Results

### Model Performance

The trained CNN model demonstrated exceptional performance on the test set. As shown in Table 1, the model achieved perfect classification accuracy of 100% (1.0000), correctly classifying all 7,172 test samples. The test loss was 7.65 × 10⁻⁶, indicating near-perfect model confidence in its predictions. The top-3 accuracy also reached 100%, meaning that for every test sample, the correct class was always among the top three predicted classes.

**Table 1**

*Overall Model Performance Metrics on Test Set*

| Metric | Value |
|--------|-------|
| Test Accuracy | 1.0000 (100.00%) |
| Test Loss | 7.65 × 10⁻⁶ |
| Top-3 Accuracy | 1.0000 (100.00%) |
| F1-Score (Macro) | 1.0000 |
| F1-Score (Weighted) | 1.0000 |
| Cohen's Kappa | 1.0000 |

*Note.* All metrics indicate perfect classification performance. N = 7,172 test samples.

### Per-Class Performance

Analysis of individual class performance revealed perfect classification across all 24 ASL letter classes. Each class achieved precision, recall, and F1-score of 1.0000, indicating no misclassifications. The confusion matrix (not shown due to perfect diagonal structure) confirmed that all test samples were correctly classified to their true labels.

Class distribution in the test set ranged from 144 samples (letter R) to 498 samples (letter E), with a mean of 298.83 samples per class (SD = 95.47). Despite this variation in class representation, the model maintained perfect performance across all classes, suggesting robust generalization capabilities.

### Training Dynamics

The model training proceeded through 50 epochs, with the best model saved based on validation accuracy. Training and validation loss decreased consistently throughout training, with validation accuracy improving from initial random performance (approximately 4.17% for 24 classes) to perfect classification. The learning rate scheduler activated during training, reducing the learning rate when validation loss plateaued, which contributed to fine-tuning model parameters.

Data augmentation played a crucial role in model generalization. By applying random transformations to training samples, the model learned to recognize ASL gestures under various orientations and positions, contributing to its robust performance on the test set.

### Model Architecture Efficiency

The final model architecture contained 620,920 trainable parameters, distributed across convolutional layers (feature extraction) and dense layers (classification). The three-block convolutional design enabled the model to learn hierarchical features: low-level edge and texture patterns in early layers, mid-level shape patterns in middle layers, and high-level gesture-specific features in deeper layers.

Batch normalization layers contributed to training stability, allowing for consistent gradient flow and faster convergence. Dropout layers with rates of 0.25 (convolutional blocks) and 0.5 (dense layers) prevented overfitting, as evidenced by the minimal gap between training and validation performance.

---

## Discussion

### Interpretation of Results

The perfect classification accuracy achieved by our CNN model represents exceptional performance for ASL gesture recognition. Several factors likely contributed to this success. First, the Sign Language MNIST dataset provides standardized, high-quality images with consistent hand positioning and lighting conditions, reducing variability that might challenge real-world applications. Second, the comprehensive data augmentation strategy enhanced model robustness by exposing the network to varied representations of each gesture class.

The perfect performance across all 24 classes, regardless of class size, suggests that the model learned discriminative features that generalize well. The high Cohen's Kappa coefficient (κ = 1.0000) indicates perfect agreement between predicted and true labels, accounting for chance agreement, which is particularly meaningful given the 24-class classification task.

### Comparison with Previous Research

Our results compare favorably with previous ASL recognition studies. While direct comparison is challenging due to differences in datasets, evaluation protocols, and class sets, our 100% accuracy on a 24-class problem represents state-of-the-art performance for static ASL letter recognition. Previous studies using traditional machine learning approaches typically achieved accuracies in the 85–95% range (Kumar et al., 2019), while CNN-based methods have shown improvements, with recent work reporting accuracies above 95% (Rastgoo et al., 2021).

The three-block CNN architecture, while relatively simple compared to deeper networks like ResNet (He et al., 2016) or EfficientNet (Tan & Le, 2019), proved highly effective for this task. This suggests that the 28 × 28 image resolution and the relatively distinct nature of ASL letter gestures may not require extremely deep architectures, making our approach computationally efficient while maintaining high accuracy.

### Limitations and Future Directions

Several limitations should be acknowledged. First, the perfect accuracy on the test set, while impressive, may indicate potential overfitting to the specific dataset characteristics. The Sign Language MNIST dataset contains standardized images with consistent backgrounds and hand positioning, which may not reflect real-world variability in lighting, hand size, skin tone, and background complexity.

Second, our study focused exclusively on static gestures, excluding letters J and Z, which require motion. Future research should explore sequence-based models (e.g., LSTM, Transformer) or 3D CNNs to handle dynamic gestures, expanding the recognition capability to all 26 letters.

Third, the model was trained and evaluated on a single dataset. Cross-dataset validation would provide stronger evidence of generalizability. Additionally, real-world testing with diverse users, lighting conditions, and camera setups would validate practical applicability.

Future research directions include: (a) expanding to dynamic gestures using video-based models, (b) implementing transfer learning from large-scale image datasets to improve generalization, (c) developing mobile-optimized models for on-device deployment, (d) creating multi-word and sentence-level recognition systems, and (e) incorporating user-specific adaptation mechanisms to improve personalization.

### Practical Implications

Despite limitations, our results demonstrate the feasibility of accurate ASL gesture recognition using deep learning. The model architecture and training pipeline can serve as a foundation for accessibility applications, including:

1. **Educational Tools:** Interactive ASL learning applications that provide real-time feedback
2. **Communication Interfaces:** Systems that translate ASL gestures to text or speech
3. **Accessibility Technology:** Integration into assistive devices and applications
4. **Research Platforms:** Baseline models for further ASL recognition research

The real-time webcam integration capability developed in this project enables immediate practical applications, allowing users to interact with the system through standard webcam hardware.

### Conclusion

This study successfully developed and evaluated a CNN-based system for recognizing static ASL hand gestures. The model achieved perfect classification accuracy on a 24-class problem, demonstrating the effectiveness of deep learning approaches for ASL recognition. The comprehensive evaluation using multiple metrics, including F1-scores and Cohen's Kappa, provides robust evidence of model performance. While limitations exist regarding dataset specificity and static gesture focus, the results establish a strong foundation for future research and practical applications in accessibility technology.

The combination of effective architecture design, data augmentation, and careful training procedures resulted in a highly accurate classification system. Future work should focus on expanding to dynamic gestures, improving real-world robustness, and developing deployable applications that can benefit the deaf and hard-of-hearing communities.

---

## References

Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., ... Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16)*, 265–283. https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770–778. https://doi.org/10.1109/CVPR.2016.90

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *Proceedings of the 32nd International Conference on Machine Learning*, 37, 448–456. https://proceedings.mlr.press/v37/ioffe15.html

Kaggle. (n.d.). *Sign Language MNIST*. Retrieved from https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*. https://arxiv.org/abs/1412.6980

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems*, 25, 1097–1105. https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

Kumar, P., Gauba, H., Roy, P. P., & Dogra, D. P. (2019). A multimodal framework for sensor based sign language recognition. *Neurocomputing*, 259, 21–38. https://doi.org/10.1016/j.neucom.2016.08.132

LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444. https://doi.org/10.1038/nature14539

LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324. https://doi.org/10.1109/5.726791

Liddell, S. K., & Johnson, R. E. (1989). American Sign Language: The phonological base. *Sign Language Studies*, 64(1), 195–277. https://www.jstor.org/stable/26204428

Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep survey. *Expert Systems with Applications*, 164, 113794. https://doi.org/10.1016/j.eswa.2020.113794

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958. http://jmlr.org/papers/v15/srivastava14a.html

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning*, 97, 6105–6114. https://proceedings.mlr.press/v97/tan19a.html

---

## Figures and Tables

*Note: Figures and tables referenced in the text should be included here. Due to the markdown format, actual visualizations would need to be generated from the notebook outputs and inserted as image files.*
<img width="1789" height="515" alt="image" src="https://github.com/user-attachments/assets/cfc2c2d2-d789-4a19-b88f-d75dd45f8822" />
**Figure 1.** Training history showing loss and accuracy curves across 50 epochs.


<img width="1482" height="1390" alt="image" src="https://github.com/user-attachments/assets/853b8551-001d-4fb6-b63c-ce10453874cf" />
**Figure 2.** Confusion matrix displaying classification performance across 24 ASL letter classes.


<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/249316dc-f6eb-4d2d-8872-ffe841fb8ee7" />

**Figure 3.** Per-class accuracy visualization showing performance for each letter.

<img width="764" height="397" alt="image" src="https://github.com/user-attachments/assets/b38f3341-14c9-43f7-8bd3-7365a4dc9003" />

**Figure 4.** Sample images from each ASL letter class in the dataset.

---

*End of Report*

