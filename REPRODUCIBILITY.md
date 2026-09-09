# 4K-DMDNet Reproducibility and Compatibility Notes

This document supplements the original 4K-DMDNet release with practical instructions, MATLAB compatibility notes, numerical validation results, and known limitations. It is intended to make the released code and pretrained network easier to reproduce and evaluate.

## 1. Scope

4K-DMDNet generates a 2160-by-3840 phase-only hologram from a 1080-by-1920 grayscale input. The released Fresnel model appends a diffraction decoder that evaluates the hologram on a zero-padded 7680-by-7680 numerical reconstruction grid.

The 7680-by-7680 array is an oversampled diffraction plane used by the model-driven constraint. It is not an 8K hologram. The phase-only hologram displayed on the spatial light modulator has a resolution of 2160 by 3840.

## 2. Software and hardware requirements

The original implementation and released network files were developed for MATLAB R2021a.

Required MATLAB products:

- MATLAB
- Deep Learning Toolbox
- Image Processing Toolbox
- Parallel Computing Toolbox

A CUDA-capable NVIDIA GPU is strongly recommended. Prediction and especially retraining at the released resolution require substantial memory and computation. CPU execution can be used for diagnostic purposes but is considerably slower.

## 3. Downloaded files and directory layout

Download the datasets and network files using the links in the main README. Extract or copy the required files into the repository root:

```text
4K-DMDNet/
|-- functions/
|-- DIV2K_train_HR/
|-- DIV2K_valid_HR/
|   `-- 0801.png
|-- UNet.mat
|-- Untrained_4KDMDNet_Fresnel_30cm_520nm.mat
|-- Trained_4KDMDNet_Fresnel_30cm_520nm.mat
|-- generation_code.m
|-- training_code.m
`-- prediction_code.m
```

The correct validation directory name used by the code is `DIV2K_valid_HR`.

Only the files needed for the selected operation must be present:

| Operation | Script | Required files |
| --- | --- | --- |
| Prediction | `prediction_code.m` | Trained network and a test image |
| Network construction | `generation_code.m` | `UNet.mat` |
| Retraining | `training_code.m` | Untrained network and `DIV2K_train_HR` |

## 4. Quick start

Open MATLAB and change the current folder to the repository root.

### 4.1 Prediction

Run:

```matlab
prediction_code
```

The default script loads:

```text
Trained_4KDMDNet_Fresnel_30cm_520nm.mat
DIV2K_valid_HR/0801.png
```

It returns:

- a 2160-by-3840 phase-only hologram;
- a phase range of approximately `[-pi, pi]`;
- a 7680-by-7680 oversampled numerical reconstruction.

The call requesting the `tanh` output measures hologram generation. The subsequent call requesting the final network output also evaluates the appended numerical diffraction decoder. These timings should be reported separately.

### 4.2 Constructing an untrained network

Run:

```matlab
generation_code
```

The script loads `UNet.mat`, appends the Fresnel diffraction decoder, and saves:

```text
Untrained_4KDMDNet_Fresnel_30cm_520nm.mat
```

### 4.3 Retraining

Run:

```matlab
training_code
```

Retraining requires the untrained network and `DIV2K_train_HR`. If GPU memory is insufficient, reduce `miniBatchSize`. Changing the mini-batch size, input preprocessing, crop, loss, or optical parameters can affect the reproduced result and should be reported.

## 5. MATLAB R2023a/R2024a compatibility

The released `.mat` network was serialized with MATLAB R2021a. The original custom-layer interface works in the intended R2021a environment. When the legacy network is loaded in MATLAB R2023a or R2024a, MATLAB may report an error similar to:

```text
Layer 'Fcos' failed during execution.
Output argument "varargout{3}" was not assigned in nnet.layer.Layer/forward.
```

This error is caused by a change in the custom-layer API used while loading or executing the legacy serialized network. It does not indicate corrupted learned weights or a change in the Fourier diffraction equation.

For newer MATLAB releases, add the following method to `functions/fft2DLayer.m`:

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

## 6. Optical parameters and reconstruction sampling

The default Fresnel configuration in `generation_code.m` is:

| Parameter | Value |
| --- | ---: |
| Wavelength | 520 nm |
| Propagation distance | 0.30 m |
| Hologram pixel pitch | 3.74 micrometres |
| Hologram resolution | 2160 by 3840 |
| Padded reconstruction grid | 7680 by 7680 |

For the padded Fresnel transform, the reconstruction-plane sampling pitch is

```text
delta_out = wavelength * distance / (FFT_size * hologram_pixel_pitch)
          = 520 nm * 0.30 m / (7680 * 3.74 micrometres)
          = 5.431 micrometres.
```

Mapping the physical hologram aperture to this sampling pitch gives an approximately 1488-by-2644 centred signal region. The released training code therefore compares the target with rows `3097:4584` and columns `2519:5162` of the 7680-by-7680 reconstruction.

The wavelength, propagation distance, pixel pitch, hologram size, padded FFT size, crop, and trained network form one consistent configuration. If any of these values is changed, the sampling pitch and crop must be recalculated. Changing only the wavelength variable does not by itself produce a validated model for a different optical system.

## 7. Full-colour reconstruction

The released example model and filename specify 520 nm. The full-colour optical experiment reported in the paper uses 450 nm, 520 nm, and 638 nm illumination with temporal multiplexing.

Full colour is not obtained by interpreting a single 520 nm numerical reconstruction as an RGB image. Reproducing the colour experiment requires wavelength-consistent holograms and diffraction parameters, synchronized illumination timing, and optical calibration for all three wavelengths.

## 8. Local numerical validation

### 8.1 Environment

A local diagnostic run was performed with:

- MATLAB R2024a;
- Deep Learning Toolbox 24.1;
- Image Processing Toolbox 24.1;
- Parallel Computing Toolbox 24.1;
- NVIDIA Quadro GV100;
- the released `Trained_4KDMDNet_Fresnel_30cm_520nm.mat` model after applying the compatibility method above.

The SHA-256 value of the model file used in this diagnostic run was:

```text
07F996CCCD05688F01347417A663E16B9FDA7154EF47A42B14FCE7D4CDAA6F00
```

This checksum identifies the exact artifact tested here. Users should compare it only with a checksum published for the same released file.

### 8.2 Diffraction-path checks

The following numerical checks were performed:

- hologram size: 2160 by 3840;
- hologram range: `[-pi, pi]`;
- numerical reconstruction size: 7680 by 7680;
- relative difference between an independent FFT calculation and the network diffraction decoder: `3.85e-5`;
- Parseval energy ratio: `0.9999998`;
- measured translation between the target and the reconstruction of the checked example: 0 pixels;
- finite gradients in one training-step check: 19,710,552 of 19,710,552.

These results support the numerical consistency of the released default Fresnel propagation path. They do not replace an optical calibration or an independent reproduction of every experiment in the paper.

### 8.3 DIV2K validation-set protocol

All 100 images in `DIV2K_valid_HR` were evaluated. Each input was resized to 1080 by 1920 and converted to grayscale as in `prediction_code.m`.

For this software diagnostic, the centred 1488-by-2644 reconstruction crop and the resized target were independently min-max normalized before PCC, PSNR, and SSIM were calculated. The values below are therefore intended to document software behaviour. They must not be compared directly with results that use a different crop, normalization, propagation convention, or optical measurement protocol.

| Metric | Minimum | 5th percentile | Median | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| PCC | 0.6826 | 0.8668 | 0.9596 | 0.9497 | 0.9938 |
| PSNR (dB) | 9.8628 | 14.8276 | 20.8671 | 20.2745 | 27.9425 |
| SSIM | 0.3536 | 0.3973 | 0.6392 | 0.6363 | 0.8581 |

Observed execution behaviour:

- all 100 holograms and reconstructions contained finite values;
- the normal reconstruction orientation correlated with the target more strongly than horizontal or vertical flips for all 100 images;
- repeated GPU inference of the same input was identical in this environment;
- the relative difference between CPU and GPU phase outputs was approximately `1.18e-5`;
- 25% brightness, 150% brightness, intensity inversion, all-black, and all-white inputs all completed inference with finite outputs.

### 8.4 Observed generalization limitation

The weakest reconstructions occurred for images dominated by dense, high-spatial-frequency patterns, such as thin radiating lines, repetitive textures, and fine branches. Across the 100-image diagnostic set, mean image-gradient magnitude correlated with PCC at approximately `-0.73` and with SSIM at approximately `-0.89`.

The minimum-PCC example had PCC 0.6826, PSNR 15.82 dB, and SSIM 0.4682. The minimum-SSIM example had PCC 0.8536, PSNR 17.13 dB, and SSIM 0.3536. Visual inspection showed loss or smoothing of dense fine detail rather than a systematic flip, translation, or crop failure.

This behaviour should be treated as a generalization limitation of the released model. The network was optimized using an NPCC-based objective, so high correlation does not guarantee equally high pixel-wise PSNR or local-structure SSIM for every input.

## 9. Known limitations and edge cases

1. **MATLAB version:** the reference release targets R2021a; loading the serialized network in newer releases may require the custom-layer compatibility method described above.
2. **Fixed parameters:** example filenames, wavelength, propagation distance, pixel pitch, padded size, and reconstruction crop are hard-coded.
3. **Visualization normalization:** `imshow(Z,[])` and `mat2gray(Z)` rescale the reconstruction for display. Quantitative evaluation must state the crop and normalization explicitly.
4. **Constant-image training loss:** the NPCC denominator in `training_code.m` has no stabilizing epsilon. A spatially constant target has zero variance and produces a non-finite training loss. Constant images should be excluded or the denominator should be stabilized.
5. **Odd spatial dimensions:** the released backward method in `fftshiftLayer` applies `fftshift` again. This is correct for the released even spatial dimensions because `fftshift` is self-inverse for even sizes. Generalization to odd dimensions should use `ifftshift` in the backward method.
6. **High-frequency content:** dense fine textures may be smoothed or reconstructed less accurately than lower-frequency structures.
7. **Numerical versus optical reconstruction:** numerical validation does not include SLM phase response, pixel fill factor, optical aberrations, illumination nonuniformity, wavelength error, camera response, or system calibration.

## 10. Recommended reporting checklist

When reporting reproduced results, specify:

- MATLAB release and toolbox versions;
- GPU model and available memory;
- exact network filename and checksum;
- input preprocessing;
- wavelength, propagation distance, and pixel pitch;
- hologram and padded FFT dimensions;
- reconstruction crop coordinates;
- normalization used for visualization and metrics;
- whether the result is numerical or optical;
- whether timing includes only hologram generation or also numerical propagation.

## 11. Citation

If this code or model is used, please cite:

Kexuan Liu, Jiachen Wu, Zehao He, and Liangcai Cao, "4K-DMDNet: diffraction model-driven network for 4K computer-generated holography," *Opto-Electronic Advances* 6, 220135 (2023). https://doi.org/10.29026/oea.2023.220135

## 12. Contact

For questions about the original implementation and released artifacts, contact: `clc@tsinghua.edu.cn`.
