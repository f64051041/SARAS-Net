import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import shutil
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
from torchvision.utils import save_image
import model as models
import glob    
import torchvision.transforms.functional as TF

#show the result for each image pairs.
show_result = True

# load model and weight
model = models.Change_detection()
model = nn.DataParallel(model)
pretrain_deeplab_path = "./model_weight_LEVIR.pth" 
checkpoint = torch.load(pretrain_deeplab_path,map_location='cuda:0')
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda()
model.eval()

#init parameters
patch = 256
TP, TN, FP, FN , ii= 0, 0, 0, 0, 0


for file in tqdm.tqdm(glob.glob('./test_dataset/A/*')):
    filename = file.split('/')
    filename = filename[-1]
    testCase1_01 = './test_dataset/A/' + filename 
    testCase1_02 = './test_dataset/B/' + filename 
    gt           = './test_dataset/label/' +  filename  
    img1 = Image.open(testCase1_01)
    img2 = Image.open(testCase1_02)
    gt = Image.open(gt)

    # show result
    if show_result:
        plt.subplot(1,4,1)
        plt.imshow(img1)
        plt.title('image 1')
        plt.subplot(1,4,2)
        plt.imshow(img2)
        plt.title('image 2')
        plt.subplot(1,4,3)
        plt.imshow(gt)
        plt.title('ground truth') 
    

    h1, w1 = img1.size
    gt_show = np.zeros((1024 * 2, 1024 * 2))
    show_image_0 = np.zeros((1024 * 2, 1024 * 2))
    show_image_1 = np.zeros((1024 * 2, 1024 * 2))
    for i in range(0, h1, patch):
        for j in range(0, w1, patch):
            temp_img1 = img1.crop((i, j, i+patch, j + patch )) 
            temp_img2 = img2.crop((i, j, i+patch, j + patch ))
            temp_gt   = gt.crop((i, j, i+patch, j + patch ))
            temp_img1 = temp_img1.resize((512,512))
            temp_img2 = temp_img2.resize((512,512))
            temp_gt   = temp_gt.resize((512,512))           
            temp_gt_acc = np.array(temp_gt,dtype= np.uint8)
            temp_gt_acc[temp_gt_acc > 0] = 1 
            height,width,_ = np.array(temp_img1,dtype= np.uint8).shape
            temp_img1 = np.array(temp_img1,dtype= np.uint8)
            temp_img2 = np.array(temp_img2,dtype= np.uint8)
            temp_gt = np.array(temp_gt,dtype= np.uint8)
            label = np.zeros((height,width,3),dtype=np.uint8)
            temp_img1 = TF.to_tensor(temp_img1)                                          
            temp_img2 = TF.to_tensor(temp_img2)                                          
            label = torch.from_numpy(label).long()                                       
            temp_img1 = TF.normalize(temp_img1, mean=[0.44758545, 0.44381796,  0.37912835],std=[0.21713617, 0.20354738, 0.18588887])   
            temp_img2 = TF.normalize(temp_img2, mean=[0.34384388, 0.33675833, 0.28733085],std=[0.1574003, 0.15169171, 0.14402839])    
            inputs1,inputs2, targets = temp_img1, temp_img2, label
            inputs1,inputs2,targets = inputs1.cuda(),inputs2.cuda(), targets.cuda()
            inputs1,inputs2,targets = Variable(inputs1.unsqueeze(0), volatile=True),Variable(inputs2.unsqueeze(0),volatile=True) ,Variable(targets)

            output_map = model(inputs1,inputs2)
            
            output_map = output_map.detach()
            
            param = 1  # This parameter is balance precision and recall to get higher F1-score
            output_map[:,1,:,:] = output_map[:,1,:,:] + param 
                        
            

            pred = output_map.argmax(dim=1, keepdim=True)
            pred = pred.cpu().detach().numpy()
            pred_acc = pred 
            pred = (pred)*255
            pred = pred.squeeze()
            gt_show[int((j//256) * 512):int((j//256) * 512)+(patch*2),int((i//256) * 512) :int((i//256) * 512) +(patch*2)] = pred
            pred_acc = pred_acc.squeeze()

            confmatrix_TP = pred_acc * temp_gt_acc
            confmatrix_TN = (pred_acc + 1) * (temp_gt_acc + 1)
            confmatrix_FP = (pred_acc) * (temp_gt_acc + 1)
            confmatrix_FN = (pred_acc + 1) * (temp_gt_acc)
            TP = TP + len(np.where(confmatrix_TP == 1)[0])  
            TN = TN + len(np.where(confmatrix_TN == 1)[0]) 
            FP = FP + len(np.where(confmatrix_FP == 1)[0])  
            FN = FN + len(np.where(confmatrix_FN == 1)[0])   

    if show_result:
        plt.subplot(1,4,4)
        plt.imshow(gt_show)
        plt.title('prediction')
        plt.show()

precision = TP/(TP+FP)
recall = TP/(TP+FN)
print("precision:", TP/(TP+FP))       
print("recall   :", TP/(TP+FN))
print("F1       :", 2/(1/precision + 1/recall))
print("IoU      :", TP/(TP+FP+FN))
print("OA       :", (TP+TN)/(TP+FP+FN+TN)) 
            

