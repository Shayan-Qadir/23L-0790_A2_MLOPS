# Task 2.2: Baseline CNN Layer-by-Layer Architecture

| Layer Name     | Layer Type             | Output Shape         |   Param Count |
|:---------------|:-----------------------|:---------------------|--------------:|
| input_image    | InputLayer             | (None, 224, 224, 3)  |             0 |
| rescaling_norm | Rescaling              | (None, 224, 224, 3)  |             0 |
| conv1          | Conv2D                 | (None, 224, 224, 32) |           896 |
| bn1            | BatchNormalization     | (None, 224, 224, 32) |           128 |
| pool1          | MaxPooling2D           | (None, 112, 112, 32) |             0 |
| conv2          | Conv2D                 | (None, 112, 112, 64) |        18,496 |
| bn2            | BatchNormalization     | (None, 112, 112, 64) |           256 |
| pool2          | MaxPooling2D           | (None, 56, 56, 64)   |             0 |
| conv3          | Conv2D                 | (None, 56, 56, 128)  |        73,856 |
| bn3            | BatchNormalization     | (None, 56, 56, 128)  |           512 |
| pool3          | MaxPooling2D           | (None, 28, 28, 128)  |             0 |
| gap            | GlobalAveragePooling2D | (None, 128)          |             0 |
| dense1         | Dense                  | (None, 64)           |         8,256 |
| dropout        | Dropout                | (None, 64)           |             0 |
| output_sigmoid | Dense                  | (None, 1)            |            65 |

**Total Parameters:** 102,465
**Trainable Parameters:** 102,017
**Non-trainable Parameters:** 448
