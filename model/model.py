import os, os.path
import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image
from random import randint

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labels = os.listdir("../data/images")
for i in range(len(labels)):
	if labels[i] == ".DS_Store": 
		del labels[i]
		i += 1
print(labels)

class DoodleDataset(Dataset):
	def __init__(self, data_dir):
		self.data_dir = data_dir

	def __len__(self):
		return len(labels)

	def __getitem__(self, label):
		label_name = labels[label]
		fruit_list = os.listdir(self.main_dir + self.root_dir + label_name)

		r = randint(0, len(fruit_list) - 1)
		img_id = fruit_list[r]
		img_name = os.path.join(self.main_dir + self.root_dir, label_name + "/" + img_id)

		image = io.imread(img_name)
		image = Image.fromarray(image, "RGB")
		image = transforms.Resize(224)(image)
		image = transforms.ToTensor()(image).float().div(255.0)

		sample = (image, label)
		return sample

	def showImage(self, idx):
		image = self[idx][0]
		image = image.mul(255.0)
		image = transforms.ToPILImage()(image)
		image.show()

class DataLoader:
	def __init__(self, dataset, batch_size):
		self.dataset = dataset
		self.batch_size = batch_size

	def __call__(self):
		images_load = []
		labels_load = []
		for i in range(self.batch_size):
			r = randint(0, len(self.dataset) - 1)
			sample = self.dataset[r]
			images_load.append(sample[0])
			labels_load.append(sample[1])
		labels_load = torch.Tensor(labels_load).long()
		images_load = torch.stack(images_load)
		return images_load, labels_load

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

learning_rate = .001
n_epochs = 12
batch_size = 10

train_set = FruitDataset("fruits-360/", "Training/")
train_loader = DataLoader(train_set, batch_size=batch_size)

model = models.resnet18()
model.fc = nn.Linear(512, 78)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
	for i in range(len(train_set)):
		images, actuals = train_loader()
		images = images.to(device)
		actuals = actuals.to(device)
		outputs = model.forward(images)

		 # Forward pass
		loss = criterion(outputs, actuals)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 10 == 0:
			_, predicted = torch.max(outputs, 1)
			print("Fruits: ")
			print(actuals.tolist())
			print("Predictions: ")
			print(predicted.tolist())
			print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, loss.item()))