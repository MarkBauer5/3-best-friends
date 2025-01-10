# Deep Learning to identify AI generated images
Task: Using Deep Learning to Develop Explainable Models for Detection of AI generated images


With the ever growing capabilities of generative AI, it can be hard to tell truth from fiction, as deep-fake images and videos become more realistic and prevalent. To help solve this problem in an explainable way, we implement a series of deep learning models capable of identifying AI generated from real human faces and utilized the GradCAM library to visualize what parts of an AI image to look for when trying to find these deep-fake images. Our models proved quite effective, achieving nearly 97% accuracy on the test set.

Read our full report [here](https://github.com/MarkBauer5/3-best-friends/blob/86e26996493b11e3c127b0214799416e00b0691c/Project%20Report.pdf)!

## Model Architectures

In addition to more popular transformer-based architectures like [SWIN](https://arxiv.org/abs/2103.14030) and [ViT](https://arxiv.org/abs/2010.11929), we implement a variety of CNN-based architectures and optimizations that seek to make these more traditional model types competitive with their more modern counterparts.

<img width="867" alt="Pasted image 20250108150754" src="https://github.com/user-attachments/assets/dbeaf914-d297-4ede-b492-b159ae642a4c" />

- A diagram of our residual pooling layer. This layer incorporates the popular residual connection as introduced in the [ResNet Paper](https://arxiv.org/abs/1512.03385) with the ability to downsample the incoming features while still retaining the residual connection. We achieve this by performing a standard max pooling operation (gray) and a learned downsampling function (blue). These two feature activations are then added and normalized to produce a downsampled feature map (blue-gray) which maintains a residual connection to previous layers allowing for longer gradient propagation.

![1_o3mKhG3nHS-1dWa_plCeFw](https://github.com/user-attachments/assets/157242b8-820f-4ae4-8cc6-a96a18bc8570)

- Another feature we implement are spatially separable convolutions. Instead of performing a 3x3 convolution, we separate it into a 3x1 and a 1x3 convolution which is more efficient computationally. This enables our models to train faster with similar performance.


![gradcam++_cam](https://github.com/user-attachments/assets/ce3bfb8c-d4e6-4410-be9e-d84579cfc2d0)

- GradCAM visualizations showing what part of the image each layer looks at most. This is an AI generated image, and in the 3rd layer, we see the model pays attention to the eyes and mouth which may be an indicator of a deep-fake image.


## Findings

![image](https://github.com/user-attachments/assets/d159bbfd-2b40-45cd-8af3-9e16f30788e4)

![image](https://github.com/user-attachments/assets/590123a3-1ba6-4464-a2bd-944d9dd117b9)

- With our custom CNN implementation and optimized data augmentation strategies, we are able to compete with more advanced architectures like SWIN in terms of raw classification capability although SWIN and ViT are still more efficient architectures in terms of parameter counts.

While our models proved quite effective, certain kinds of augmentations severely impact model validation and test accuracy. In particular, a simple Gaussian blur dramatically harms our SWIN and CNN performance metrics, indicating that these models —while capable at the task at hand— mainly utilize small image artifacts as a key feature for discrimination instead of looking at macroscopic features like distorted environments or facial features.

![image](https://github.com/user-attachments/assets/16a04f64-3ed6-4a86-8bb7-bb659a464447)

- A table showing the performance of different model architectures and how they are degraded when tested on certain data augmentations like grayscale application, Gaussian blurs, or simple downsampling.


## Repository Structure
README.md: Overview of the project

requirements.txt: Required libraries to run the code

train.py: Code to train the models, contains parameters to change the model and super-parameter values

testModel.py: Code to test the trained models, which loads the model and test the model on the test data

models.py: Code to define the models, which includes the `CNN`s, `ViT`, `Swin Transformer` models

swin_transformer_v2.py: Code to define the Swin Transformer model, which is compatible with GradCam

swinT_xai.py: Explainability code for Swin Transformer with GradCam

vit_xai.py: Explainability code for ViT with GradCam

cnn_xai.py: Explainability code for CNN with GradCam

## Trained Models
The trained models can be found in the following link: [Trained Models](https://drive.google.com/drive/folders/1M4pZIkd9ctpEjZeOEp4uwB1Oow6niAY7?usp=sharing)
