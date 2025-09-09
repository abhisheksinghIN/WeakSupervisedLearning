# WeakSupervisedLearning
a deep adversarial model based on discrete wavelet transform (WT) to exploit weak/low-resolution label information for generating refined super-resolved weak reference maps (WRMs)

# Abstract
The unavailability of pixel-level detailed labels is a crucial challenge in the field of remote sensing image analysis. Deep learning (DL) models require a large number of labeled samples for an accurate estimation of a large number of trainable parameters. However, in remote sensing applications usually only a few reliable labeled data are available for the learning of a classifier, whereas often many weak/low-resolution unreliable labeled data can be collected from available land-cover maps. Accordingly, weak-supervised learning (WSL) may overcome the problems by using noisy and low-resolution labels in remote sensing. In this article, we propose a deep adversarial model based on discrete wavelet transform (WT) to exploit weak/low-resolution label information for generating refined super-resolved weak reference maps (WRMs). Our contribution includes the development of a discrete WT-based generator for enhancing the low-resolution labels to generate a refined high-resolution reference map. We also present an efficient framework for multisource image fusion that incorporates the refined super-resolved WRM, synthetic aperture radar (SAR) images, and corresponding low-resolution labels. Our findings highlight the effectiveness of the refined super-resolved WRM. In addition, we investigate the impact of the high-resolution reference maps on segmentation accuracy, which reveals their potential in improving the segmentation performance compared with other reference methods.

<img width="820" height="474" alt="image" src="https://github.com/user-attachments/assets/3b80f732-39e9-48a1-b41b-e6b656e9b192" />

Proposed framework for the weak-supervised-based multisource image classification.

# Contributions
We use a discrete-WT-inspired generator to produce refined super-resolved WRM from low-resolution labels using the characteristics of principal components of the multispectral images.

We develop an efficient framework for multisource image fusion which utilizes the characteristics of refined super-resolved WRM, multispectral images, SAR images and low-resolution labels.

We compare the effectiveness of the obtained feature map with those of other general generative models and study the impact of refined super-resolved WRM in terms of segmentation accuracy.

For deatailed Information: https://ieeexplore.ieee.org/document/11016942

<img width="820" height="545" alt="image" src="https://github.com/user-attachments/assets/776a2749-4fbc-4b82-96b5-be28a699ce0e" />

Proposed methodology of the proposed WRGAN for refined super-resolved WRM generation.
