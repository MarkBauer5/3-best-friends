# Deep Learning to identify AI generated images
Task: Using Deep Learning to identify AI generated images

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

