# Session 2 - MobileNets and ShuffleNets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MnZKMhE4fLNth7WJ2iZakXgUF1uABrt3?usp=sharing)

The goal of this assignment is to train MobileNet V2 on a custom dataset that conatins the classes small quadcopters, large quadcopters, flying birds and winged drones using transfer learning. After training, the trained model has to be deployed to AWS Lambda.

The model reaches a test accuracy of **89.20%** and was trained for **12 Epochs**.

## Parameters and Hyperparameters

- Loss Function: Cross Entropy Loss (combination of `nn.LogSoftmax` and `nn.NLLLoss`)
- Optimizer: SGD
- Learning Rate: 0.01
- Data Augmentation
  - Random Resize Crop
- Reduce LR on Plateau
  - Decay factor: 0.1
  - Patience: 2
  - Min LR: 1e-6
- Batch Size: 256
- Epochs: 12

## Results

The model achieves the highest validation accuracy of **89.20%**

### Accuracy Change vs Number of Epochs

![accuracy_change](images/accuracy_change.png)

### Predictions

#### Correctly Classified Images

![correct](images/correct_predictions.jpg)

#### Misclassified Images

![correct](images/incorrect_predictions.jpg)

### Deployment

The deployed model gave the following results  
<img src="images/postman.png">

Image used to test the model:  
<img src="images/image_input.jpg" width="300">

_Note:_ It might happen that during the first call the API might give a timed out error. If such error happens, call the API again.

## Testing on Custom Image

API Link: [https://5a7jq62zm2.execute-api.ap-south-1.amazonaws.com/dev/classify](https://5a7jq62zm2.execute-api.ap-south-1.amazonaws.com/dev/classify)

Send a POST request to the link above with the image that needs to be classified.

## Dataset

The dataset contains images of the below classes:

Original dataset count:

- flying_birds - 8338
- large_quadcopters - 4169
- small_quadcopters - 3623
- winged_drones - 5675

The dataset initially had a many irrelevant images, that is why data cleaning was done. Following techniques were applied:

- Remove all the image that were not in .jpeg, .jpg and .png format
- Convert .png to .jpeg
- Convert images that had channel 1 or 4 to 3 channel image (RGB)
- Removed all the image that had resolution below than 150x150 (In our experiments, we found that image with resolution below 150x150 were getting distorted upon resizing to 224x224)

Cleaned dataset count:

- flying_birds - 8115
- large_quadcopters - 4097
- small_quadcopters - 3137
- winged_drones - 5540

### Resizing Strategy: RandomResizedCrop

```[python]
transforms.RandomResizedCrop(224)
```

Crop the given image to random scale and aspect ratio. This is popularly used to train the Inception networks.

This will extract a patch of size (224, 224) from input image randomly. So, it might pick this path from topleft, bottomright or anywhere in between.

1. **Utilizing entire 224x224 (each pixel)** to represent input image unlike other techqniues e.g. center aligned on mask which add black border/padding.
2. In this part, we are also doing **data augmentation** to increase the variability of the input images, as a result, model has higher robustness to the images obtained from different environments which is much better than simple resizing `transforms.Resize(size=224)`

### Sampler: Class Imbalance Handler

To solve class Imbalance problem, a widely adopted technique is called resampling. It consists of less samples from the majority class and/or adding more examples from the minority class.

<center>
    <img src="https://img.techpowerup.org/200730/40677251-b08f504a-63af-11e8-9653-f28e973a5664.png" width="400px" alt="centered image" />
</center>

```[python]
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size, sampler=sampler)
```

## MobilenetV2 Architecture

### Inverted Residuals

The original residual block follows a wide->narrow->wide approach, on the other hand, MobileNetV2 follows a narrow->wide->narrow approach. The first step widens the network using a 1x1 convolution because the following 3x3 depthwise convolution already greatly reduces the number of parameters. Afterwards another 1x1 convolution squeezes the network in order to match the initial number of channels.

<center>
    <img src="images/res_vs_inverse_res.png" width="400px" alt="centered image" />
</center>

### ReLU6

MobileNetV2 use ReLU6 instead of ReLU, which limits the value of activations to a maximum of 6. The activation is linear as long as it’s between 0 and 6.

```
def relu6(x):
    return min(max(0, x), 6)
```

### Linear Bottleneck

The last convolution of a residual block has a linear output before it’s added to the initial activations.

```
def bottleneck_block(x, expand=64, squeeze=16):
    m = Conv2D(expand, (1,1))(x)
    m = BatchNormalization()(m)
    m = Activation('relu6')(m)
    m = DepthwiseConv2D((3,3))(m)
    m = BatchNormalization()(m)
    m = Activation('relu6')(m)
    m = Conv2D(squeeze, (1,1))(m)
    m = BatchNormalization()(m)
    return Add()([m, x])
```

### Architecture

<center>
    <img src="images/mobilenetv2.png" width="350px" alt="centered image" />
</center>

t: expansion rate of the channels  
c: number of input channels  
n: how often the block is repeated  
s: the first repetition of a block used a stride of 2 for the downsampling process
