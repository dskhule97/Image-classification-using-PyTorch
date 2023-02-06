#First, we import the 'models' module from the torchvision library.
from torchvision import models

#We will set the pretrained parameter to True, indicating that the model should be initialized with pre-trained weights.
#By using a pre-trained model, the model can be fine-tuned to solve a new problem,
# with less data and computational resources required, compared to training the model from scratch.
resnet = models.resnet101(pretrained=True)
resnet

#We will then use 'transforms' which resizes the image to 256x256, then crops it to a size of 224x224,
#followed by converting it to a tensor and normalizing it using the mean and standard deviation values specified.

from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

#We will use the PIL library to open the image we need to classify and use the 'preprocess' pipeline
# we created to transform the image into standard form

from PIL import Image
img = Image.open("cat.jpeg")
img
img_t = preprocess(img)

#Unsqueeze adds a batch dimension to the image tensor img_t, so that it can be used as input to the model.
# batch_t = torch.unsqueeze(img_t, 0): This line  adds a singleton dimension (of size 1) at the specified position (0 in this case).
# This is necessary because the ResNet101 model expects input to be in the form of a batch of images, where each image is a 3D tensor of size (C, H, W),
# and the batch is a 4D tensor of size (B, C, H, W). By adding a batch dimension of size 1, the image tensor is transformed from a 3D tensor to a 4D tensor,
# making it suitable for use as input to the model.

#resnet.eval(): This line sets the model in evaluation mode, which disables some behaviors (such as dropout) that are used during training but not during evaluation.

#out = resnet(batch_t): This line passes the input image through the ResNet101 model and stores the output in the out variable.
# The output is a tensor that represents the model's predictions for the input image. The dimensions of the tensor depend on the number of classes the model was trained on,
# but typically the last dimension will represent the predicted class scores for each class in the classification problem.

import torch
batch_t = torch.unsqueeze(img_t, 0)
resnet.eval()
out = resnet(batch_t)
out

#The code then loads a list of ImageNet class labels from a text file and uses the torch.max function to find the class with the highest predicted probability.
#Finally, the predicted class and its probability are printed to the console.

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())