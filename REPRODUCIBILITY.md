# 4K-DMDNet Reproducibility and Compatibility Notes

This document supplements the original 4K-DMDNet release with practical instructions, MATLAB compatibility notes, and known limitations. It is intended to make the released code and pretrained network easier to reproduce and evaluate.

## 1. Software and hardware requirements

The original implementation and released network files were developed for MATLAB R2021a.

Required MATLAB products:

- MATLAB
- Deep Learning Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox

A CUDA-capable NVIDIA GPU is strongly recommended. Prediction and especially retraining at the released resolution require substantial memory and computation. CPU execution can be used for diagnostic purposes but is considerably slower.

## 2. Downloaded files and directory layout

Download the datasets and network files using the links in the main README. Extract or copy the required files into the repository root:

```text
4K-DMDNet/
|-- functions/
|-- DIV2K_train_HR/
|-- DIV2K_valid_HR/
|   `-- 0801.png
|-- Trained_4KDMDNet_Fresnel_30cm_520nm.mat
|-- training_code.m
`-- prediction_code.m
```

The correct validation directory name used by the code is `DIV2K_valid_HR`.

Only the files needed for the selected operation must be present:

| Operation | Script | Required files |
| --- | --- | --- |
| Prediction | `prediction_code.m` | Trained network and a test image |
| Retraining | `training_code.m` | A user-prepared untrained network and `DIV2K_train_HR` |

The released network file is primarily intended for testing the trained model with `prediction_code.m`. `training_code.m` can be used to train a prepared untrained network. Please refer to the paper for the detailed network architecture and design.

## 3. Quick start

Open MATLAB and change the current folder to the repository root.

### 3.1 Prediction

Run:

```matlab
prediction_code
```

The default script loads:

```text
Trained_4KDMDNet_Fresnel_30cm_520nm.mat
DIV2K_valid_HR/0801.png
```

### 3.2 Retraining

Run:

```matlab
training_code
```

Retraining requires an untrained network and `DIV2K_train_HR`. If GPU memory is insufficient, reduce `miniBatchSize`. Changing the mini-batch size, input preprocessing, crop, loss, or optical parameters can affect the reproduced result. Please refer to the paper for the detailed network architecture and design.

## 4. MATLAB R2023a/R2024a compatibility

The released `.mat` network was serialized with MATLAB R2021a. The original custom-layer interface works in the intended R2021a environment. When the legacy network is loaded in MATLAB R2023a or R2024a, MATLAB may report an error similar to:

```text
Layer 'Fcos' failed during execution.
Output argument "varargout{3}" was not assigned in nnet.layer.Layer/forward.
```

This error is caused by a change in the custom-layer API used while loading or executing the legacy serialized network. It does not indicate corrupted learned weights or a change in the Fourier diffraction equation.

For newer MATLAB releases, users may try adding the following method to `functions/fft2DLayer.m`:

```matlab
function [Z1,Z2,memory] = forward(~,X)
    Z = fft2(X);
    Z1 = real(Z);
    Z2 = imag(Z);
    memory = [];
end
```

This method performs the same calculation as the existing `predict` method. It does not modify the learned parameters and does not require retraining.

If a network using `ifft2DLayer` is used, apply the analogous interface update:

```matlab
function [Z1,Z2,memory] = forward(~,X)
    Z = ifft2(X);
    Z1 = real(Z);
    Z2 = imag(Z);
    memory = [];
end
```

For strict reproduction of the original release, MATLAB R2021a remains the reference environment.

## 5. Optical parameters

The default Fresnel configuration in `generation_code.m` is:

| Parameter | Value |
| --- | ---: |
| Wavelength | 520 nm |
| Propagation distance | 0.30 m |
| Hologram pixel pitch | 3.74 micrometres |
| Hologram resolution | 2160 by 3840 |
| Padded reconstruction grid | 7680 by 7680 |

The wavelength, propagation distance, pixel pitch, hologram size, padded FFT size, crop, and trained network form one consistent configuration. If any of these values is changed, the sampling pitch and crop must be recalculated. Changing only the wavelength variable does not by itself produce a validated model for a different optical system.

## 6. Full-colour reconstruction

The released example model and filename specify 520 nm. The full-colour optical experiment reported in the paper uses 450 nm, 520 nm, and 638 nm illumination with temporal multiplexing.

Please refer to the paper for implementation details.

## 7. Known limitations and edge cases

1. **MATLAB version:** the reference release targets R2021a; loading the serialized network in newer releases may require the custom-layer compatibility method described above.
2. **Fixed parameters:** example filenames, wavelength, propagation distance, pixel pitch, padded size, and reconstruction crop are hard-coded.
3. **Visualization normalization:** `imshow(Z,[])` and `mat2gray(Z)` are used primarily for display and rescale the reconstruction.
4. **Training numerical stability:** if non-finite losses or other numerical-stability problems occur during training, users may consider adding a small epsilon to the NPCC normalization denominator in `training_code.m`. A spatially constant target has zero variance and may produce a non-finite loss.
5. **Odd spatial dimensions:** the released backward method in `fftshiftLayer` applies `fftshift` again. This is correct for the released even spatial dimensions because `fftshift` is self-inverse for even sizes. Generalization to odd dimensions should use `ifftshift` in the backward method.

## 8. Citation

If this code or model is used, please cite:

Kexuan Liu, Jiachen Wu, Zehao He, and Liangcai Cao, "4K-DMDNet: diffraction model-driven network for 4K computer-generated holography," *Opto-Electronic Advances* 6, 220135 (2023). https://doi.org/10.29026/oea.2023.220135

## 9. Contact

For questions about the original implementation and released artifacts, contact: `clc@tsinghua.edu.cn`.
