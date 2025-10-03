# ERA4 Assignment 7 - Target 🎯
- Work on CIFAR-10 Dataset
  - has the architecture to C1C2C3C40 (No MaxPooling, but convolutions, where the last one has a stride of 2 instead) (NO restriction on using 1x1) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
  - total RF must be more than 44
  - One of the layers must use Depthwise Separable Convolution
  - One of the layers must use Dilated Convolution
  - use GAP (compulsory):- add FC after GAP to target #of classes (optional)
  - Use the albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
  - achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.

## Data Augmentation 🛠️
- Horizontal flip - Randomly flips the image left-to-right to simulate mirrored versions of objects.
- ShiftScaleRotate - Randomly shifts, scales, and rotates the image to make the model invariant to position, size, and orientation changes.
- CoarseDropout - Randomly masks out small rectangular regions of the image (filled with dataset mean) to improve robustness against occlusions.

Example : 
<img width="1166" height="286" alt="image" src="https://github.com/user-attachments/assets/56a86e29-8cc1-4142-9c89-c4ae60242497" />

  
## Model Comparison Summary :
| **Feature** | **Model - 1** | **Model - 2** | **Model - 3** |
| :--- | :--- | :--- | :--- |
| **Receptive Field (RF)** | $\mathbf{21 \times 21}$ | $\mathbf{39 \times 39}$ | $\mathbf{41 \times 41}$ |
| **Number of Parameters** | $\mathbf{181,050}$ | $\mathbf{365,322}$ | $\mathbf{163,954}$ |
| **Optimizer** | $\mathbf{Adam}$ (LR=0.001) | $\mathbf{Adam}$ (LR=0.001) |Adam (LR=0.001) + StepLR (step\_size=15, gamma=0.1)|
| **Dropout** | **None** | $\mathbf{Dropout2d}$ ($\mathbf{p=0.25}$) | $\mathbf{Dropout2d}$ ($\mathbf{p=0.1}$) |
| **Batch Normalization** | Present in all blocks | Present in all blocks | Present in all blocks |
| **Train Accuracy** | $\mathbf{87.85}$ | $\mathbf{83.30}$ |  $\mathbf{84.34}$ | 
| **Validation Accuracy** | $\mathbf{82.46}$ | $\mathbf{86.36}$ |  $\mathbf{85.01}$ | 
