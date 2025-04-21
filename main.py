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

from setup_data import setup_data
from my_model import MyModel
from train import train, validate, accuracy, adjust_learning_rate


parser = argparse.ArgumentParser(description='params for model')
parser.add_argument('--config', default='./config_mymodel.yaml')


def find_digit(image, delta, min_area, max_area,distance,model):
    result = image.copy()
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    # gray = cv2.Canny(gray, 180, 200)

    mser = cv2.MSER_create(delta,min_area,max_area)#
    regions, bounding = mser.detectRegions(gray)
    #https://answers.opencv.org/question/19015/how-to-use-mser-in-python/


    grouped_bounding, weights = cv2.groupRectangles(bounding, groupThreshold=1, eps=.5)

    new_bounding=[]
    for i,box in enumerate(grouped_bounding):
        x, y, w, h = box
        
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)

    
    digit_bounding = []
    check = []
    for box in grouped_bounding:
        x, y, w, h = box

        for box2 in grouped_bounding:
            x1, y1, w1, h1 = box2

            # Check if the boxes are adjacent horizontally
            if x == x1 and y == y1:
                continue

            if (abs((x+w) -x1) < distance or (abs((x1+w1) - x)<distance)) and abs(y-y1) < distance:
                if [x, y, w, h] not in digit_bounding:
                    digit_bounding.append([x, y, w, h])

            if (abs((y+h) - y1) < distance or (abs((y1+h1)-y))<distance) and abs(x-x1) < distance:
                digit_bounding.append([x, y, w, h])


    grouped_bounding=digit_bounding
    

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    # Extract the digits from the image
    rois = []
    for bound in grouped_bounding:
        x, y, w, h = bound

        new_bounding.append([x, y, w, h])
        roi = image[y:y+h, x:x+w]
        # https://pytorch.org/vision/main/generated/torchvision.transforms.functional.to_pil_image.html
        # https://pytorch.org/vision/main/auto_examples/others/plot_repurposing_annotations.html#sphx-glr-auto-examples-others-plot-repurposing-annotations-py
        roi = transforms.functional.to_pil_image(roi)  
        roi = transform(roi)
        rois.append(roi)


    rois = torch.stack(rois)

    with torch.no_grad():
        outputs = model(rois)
        _, predicted = torch.max(outputs.data, 1)


    for i, bound in enumerate(grouped_bounding):
        x, y, w, h = bound
        
        cv2.putText(image, str(predicted[i].item()), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2)

    #visualization
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image



def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    # dataloader_train, dataloader_val, dataloader_test = setup_data()
    
    if args.model == 'vgg16':
        if args.train == True:
            model = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT') 
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            num_classes = 11
            model.classifier[-1] = nn.Linear(4096, num_classes)
        else:
            model = torchvision.models.vgg16()
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            num_classes = 11
            model.classifier[-1] = nn.Linear(4096, num_classes)
            model.load_state_dict(torch.load('vgg16.pth', map_location=device))
    

    elif args.model == 'MyModel':
        if args.train == True:
            model = MyModel()
        else:
            model = MyModel()
            model.load_state_dict(torch.load('mymodel.pth', map_location=device))

    
    if torch.cuda.is_available():
        model = model.cuda()


    criterion = nn.CrossEntropyLoss()
  

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.reg)
    if args.train == True:
        dataloader_train, dataloader_val, dataloader_test = setup_data()
        best = 0.0
        Tbest = 0.0
        best_cm = None
        best_model = None
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args)

            # train loop
            train(epoch, dataloader_train, model, optimizer, criterion)

            # validation loop
            acc, cm = validate(epoch, dataloader_val, model, criterion)

            #test loop
            tacc, tcm = validate(epoch, dataloader_test, model, criterion)

            if acc > best:
                Tbest = tacc
                best = acc
                best_cm = cm
                best_model = copy.deepcopy(model)

        #modified and influenced from DL class project
        print('Best Prec @1 Acccuracy Valadation: {:.4f}'.format(best))
        print('Best Prec @1 Acccuracy Test: {:.4f}'.format(Tbest))
        per_cls_acc = best_cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

        if args.train == True:
            torch.save(best_model.state_dict(), './' + args['model'].lower() + '.pth')
    
    
    output = './images'

    # ** Future work **
    
    image1 = cv2.imread('12524.jpeg')
    out = find_digit(image1, 14, 380, 500,10, model) #delta, min_area, max_area
    cv2.imwrite(os.path.join(output, '1.png'), out)

    image2 = cv2.imread('10137.jpg')
    out = find_digit(image2, 7, 37, 140,7, model)
    cv2.imwrite(os.path.join(output, 'test.png'), out)

    image2 = cv2.imread('15700.png')
    out = find_digit(image2, 5, 50, 180,10, model)
    cv2.imwrite(os.path.join(output, '2.png'), out)

    image3 = cv2.imread('50505080.png')
    out = find_digit(image3, 8, 100, 800,10, model)
    cv2.imwrite(os.path.join(output, '3.png'), out)

    image4 = cv2.imread('238.jpeg')
    out = find_digit(image4, 10, 165, 195, 10, model)
    cv2.imwrite(os.path.join(output, '4.png'), out)

    image5 = cv2.imread('211.jpg')
    out =find_digit(image5, 10, 30, 70,10, model)
    cv2.imwrite(os.path.join(output, '5.png'), out)

if __name__ == '__main__':
    main()