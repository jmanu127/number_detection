
import yaml
import argparse
import time
import copy
import os

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2


import mat73
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# import torchvision.transforms as transforms
from torchvision import datasets, transforms, models


def setup_data():

  # Define the data transforms
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  # Load the SVHN dataset
  svhn_train = datasets.SVHN('./svhn', split='train', download=True, transform=transform)
  print(len(svhn_train))
  svhn_train, svhn_val = torch.utils.data.random_split(svhn_train, [0.8,0.2])

  svhn_test = datasets.SVHN('./svhn', split='test', download=True, transform=transform)

  cifar = datasets.CIFAR10('./cifar10', train=True, download=True, transform=transform)
  cifar.targets[:] = [10]*len(cifar)

  cifar_train, cifar_val, cifar_test, _ = torch.utils.data.random_split(cifar, [0.2,0.1,.1,.6])

  #   print(len(svhn_train))
  #   print(len(svhn_val))
  #   print(len(svhn_test))
  #   print(len(cifar_train))
  #   print(len(cifar_val))
  #   print(len(cifar_test))

  train_set = torch.utils.data.ConcatDataset([svhn_train, cifar_train])
  val_set = torch.utils.data.ConcatDataset([svhn_val, cifar_val])
  test_set = torch.utils.data.ConcatDataset([svhn_test, cifar_test])

  # Create a PyTorch DataLoader for the dataset
  dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True) 

  dataloader_val = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True) 

  dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True) 



  return dataloader_train, dataloader_val, dataloader_test 