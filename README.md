# 4K-DMDNet
diffraction model-driven network for 4K computer-generated holography

Kexuan Liu, Jiachen Wu, Zehao He, and Liangcai Cao, “4K-DMDNet: diffraction model-driven network for 4K computer-generated holography,” Opto-Electronic Advances 6, 220135 (2023).
https://www.oejournal.org//article/doi/10.29026/oea.2023.220135

Deep learning offers a novel opportunity to achieve both high-quality and high-speed computer-generated holography (CGH). Current data-driven deep learning algorithms face the challenge that the labeled training datasets limit the training performance and generalization. The model-driven deep learning introduces the diffraction model into the neural network. It eliminates the need for the labeled training dataset and has been extensively applied to hologram generation. However, the existing model-driven deep learning algorithms face the problem of insufficient constraints. In this study, we propose a model-driven neural network capable of high-fidelity 4K computer-generated hologram generation, called 4K Diffraction Model-driven Network (4K-DMDNet). The constraint of the reconstructed images in the frequency domain is strengthened. And a network structure that combines the residual method and sub-pixel convolution method is built, which effectively enhances the fitting ability of the network for inverse problems. The generalization of the 4K-DMDNet is demonstrated with binary, grayscale and 3D images. High-quality full-color optical reconstructions of the 4K holograms have been achieved.


Before running, please download the following image datasets and networks for prediction or retraining. Matlab version: 2021a.

DIV2K_train_HR dataset download: https://cloud.tsinghua.edu.cn/f/ca0262cc418349848037/?dl=1

DIV2K_vaild_HR dataset download: https://cloud.tsinghua.edu.cn/f/c657099700464693bd00/?dl=1

networks download: https://cloud.tsinghua.edu.cn/f/4f8f09ffe25844809e9b/?dl=1

Contact: lkx20@mails.tsinghua.edu.cn; clc@tsinghua.edu.cn
