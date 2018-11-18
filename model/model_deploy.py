import os, os.path
import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image
from random import randint
import cv2
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

classes = ['bicycles', 'golf_clubs', 'pants', 'forks']


def create_model(model_path):
	model = models.resnet50()
	model.fc = nn.Linear(2048, 4)
	model.load_state_dict(torch.load(model_path))
	return model 


def predict(image, model):
	image = np.expand_dims(image, 0)
	image = np.concatenate((image, image, image))
	image = np.resize(image, (224, 224, 3))
	image = Image.fromarray(image, 'RGB')
	image = transforms.ToTensor()(image).float().div(255.0)
	image = torch.unsqueeze(image, 0)
	output = model(image)
	output = np.argmax(output.detach().numpy(), axis=1)[0]
	return output

