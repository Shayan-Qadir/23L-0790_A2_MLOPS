
### Stage 1 (Frozen Backbone): Trainable Layers (6 layers)

| Scope   | Layer             | Type                   |
|:--------|:------------------|:-----------------------|
| Head    | input_image       | InputLayer             |
| Head    | transfer_gap      | GlobalAveragePooling2D |
| Head    | transfer_dense128 | Dense                  |
| Head    | transfer_bn       | BatchNormalization     |
| Head    | transfer_dropout  | Dropout                |
| Head    | transfer_sigmoid  | Dense                  |

### Stage 2 (Fine-Tuned Top Block): Trainable Layers (25 layers)

| Scope                | Layer                   | Type                   |
|:---------------------|:------------------------|:-----------------------|
| Head                 | input_image             | InputLayer             |
| mobilenetv2_1.00_224 | block_14_expand         | Conv2D                 |
| mobilenetv2_1.00_224 | block_14_expand_relu    | ReLU                   |
| mobilenetv2_1.00_224 | block_14_depthwise      | DepthwiseConv2D        |
| mobilenetv2_1.00_224 | block_14_depthwise_relu | ReLU                   |
| mobilenetv2_1.00_224 | block_14_project        | Conv2D                 |
| mobilenetv2_1.00_224 | block_14_add            | Add                    |
| mobilenetv2_1.00_224 | block_15_expand         | Conv2D                 |
| mobilenetv2_1.00_224 | block_15_expand_relu    | ReLU                   |
| mobilenetv2_1.00_224 | block_15_depthwise      | DepthwiseConv2D        |
| mobilenetv2_1.00_224 | block_15_depthwise_relu | ReLU                   |
| mobilenetv2_1.00_224 | block_15_project        | Conv2D                 |
| mobilenetv2_1.00_224 | block_15_add            | Add                    |
| mobilenetv2_1.00_224 | block_16_expand         | Conv2D                 |
| mobilenetv2_1.00_224 | block_16_expand_relu    | ReLU                   |
| mobilenetv2_1.00_224 | block_16_depthwise      | DepthwiseConv2D        |
| mobilenetv2_1.00_224 | block_16_depthwise_relu | ReLU                   |
| mobilenetv2_1.00_224 | block_16_project        | Conv2D                 |
| mobilenetv2_1.00_224 | Conv_1                  | Conv2D                 |
| mobilenetv2_1.00_224 | out_relu                | ReLU                   |
| Head                 | transfer_gap            | GlobalAveragePooling2D |
| Head                 | transfer_dense128       | Dense                  |
| Head                 | transfer_bn             | BatchNormalization     |
| Head                 | transfer_dropout        | Dropout                |
| Head                 | transfer_sigmoid        | Dense                  |
