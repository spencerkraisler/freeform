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

PATH_TO_IMAGES = "../data/images/"

def get_labels():
	proto_labels = os.listdir(PATH_TO_IMAGES)
	labels = []
	for label in proto_labels:
		if label != ".DS_Store": 
			labels.append(label)
	return labels

labels = get_labels()


class DoodleDataset(Dataset):

	def __len__(self):
		return len(labels)

	def __getitem__(self, label_idx):
		label = labels[label_idx]
		list_of_image_names = os.listdir(PATH_TO_IMAGES + label + "/")
		r = randint(0, len(list_of_image_names) - 1)
		image_name = list_of_image_names[r]
		image_path = PATH_TO_IMAGES + label + "/" + image_name
		image = cv2.imread(image_path)
		cv2.imshow('test', image)
		image = np.expand_dims(image, 0)
		image = np.concatenate((image, image, image))
		image = np.resize(image, (224, 224, 3))
		image = Image.fromarray(image, 'RGB')
		image = transforms.ToTensor()(image).float().div(255.0)
		sample = (image, label_idx)
		return sample

	def showImage(self, idx):
		image = self[idx][0]
		image = image.mul(255.0)
		image = transforms.ToPILImage()(image)
		image.show()

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

learning_rate = .001
n_epochs = 1sd00
batch_size = 5

train_set = DoodleDataset()
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
print("Creating model...")
model = models.resnet50()
model.fc = nn.Linear(2048, 4)

train_set[3]
criterion = nn.CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def getOneHotEncoded(idxes):
	OHV = np.zeros((len(idxes), 4))
	i = 0
	for idx in idxes:
		OHV[i][idx] = 1
		i += 1
	OHV = torch.Tensor(OHV)
	return OHV

print("Begin training...")
for epoch in range(n_epochs):
	for i, samples in enumerate(train_loader):
		
		images, ground_truths = samples
		ground_truths = getOneHotEncoded(ground_truths)
		outputs = model.forward(images)

		 # Forward pass
		loss = nn.MSELoss()(outputs, ground_truths)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch % 10 == 0:
			_, predictions = torch.max(outputs, 1)
			print("Ground truths: ")
			print(np.argmax(ground_truths, axis=1).tolist())
			print("Predictions: ")
			print(predictions.tolist())
			print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, n_epochs, loss.item()))
			print()


torch.save(model.state_dict(), './model.pth')


print("Training has ended...")
