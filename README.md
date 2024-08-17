# GBT: Geometric-oriented Brain Transformer for Autism Diagnosis
Our four brain-oriented submitted papers are all accepted by MICCAI 2024 (one Early Accept), where our GBT is the newest research on brain fMRI data analysis, focusing on prompting the representation learning development of the brain data analysis community.
# Abstract
*Human brains are typically modeled as networks of Regions of Interest (ROI) to comprehend brain functional Magnetic Resonance Imaging (fMRI) connectome for Autism diagnosis. Recently, various deep neural network-based models have been developed to learn the representation of ROIs, achieving impressive performance improvements. However, they ($i$) heavily rely on increasingly complex network architecture with an obscure learning mechanism, or ($ii$) solely utilize the cross-entropy loss to supervise the training process, leading to sub-optimal performance. To this end, we propose a simple and effective Geometric-oriented Brain Transformer (GBT) with the Attention Weight Matrix Approximation (AWMA)-based transformer module and the geometric-oriented representation learning module for brain fMRI connectome analysis. Specifically, the AWMA-based transformer module selectively removes the components of the attention weight matrix with smaller singular values, aiming to learn the most relevant and representative graph representation. The geometric-oriented representation learning module imposes low-rank intra-class compactness and high-rank inter-class diversity constraints on learned representations to promote that to be discriminative. Experimental results on the ABIDE dataset validate that our method GBT consistently outperforms state-of-the-art approaches.*

We appreciate it if you use this code and cite our related papers, which can be cited as follows,

> @inproceedings{peng2024gbt, <br>
>   title={GBT: Geometric-oriented Brain Transformer for Autism Diagnosis}, <br>
>   author={Peng, Zhihao, He, Zhibin, Jiang, Yu, Wang, Pengyu, Yuan, Yixuan .}, <br>
>   booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention}, <br>
>   year={2024}
> } <br>

Peng, Zhihao, et al. "GBT: Geometric-oriented Brain Transformer for Autism Diagnosis." International Conference on Medical Image Computing and Computer-Assisted Intervention, 2024.
