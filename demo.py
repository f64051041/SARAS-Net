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

import glob    
import torchvision.transforms.functional as TF


# dataset
data_name = 'LEVIR'     

# image              
img1_file = './sample/LEVIR_t1_1.png'
img2_file = './sample/LEVIR_t2_1.png'
lbl_file =  './sample/LEVIR_gt_1.png'

# load model and weight
import model as models
pretrain_deeplab_path = "./model_weight_LEVIR.pth"


device = torch.device("cuda:0")
model = models.Change_detection()
model = nn.DataParallel(model, device_ids = [0])
checkpoint = torch.load(pretrain_deeplab_path)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()


img1 = Image.open(img1_file)
img2 = Image.open(img2_file)
gt = Image.open(lbl_file)
gt_show = cv2.imread(lbl_file, 0)
temp_img1 = img1.resize((512, 512))
temp_img2 = img2.resize((512, 512))
temp_gt   = gt.resize((512, 512))
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
if data_name == 'LEVIR': 
    temp_img1 = TF.normalize(temp_img1, mean=[0.44758545, 0.44381796,  0.37912835],std=[0.21713617, 0.20354738, 0.18588887])   
    temp_img2 = TF.normalize(temp_img2, mean=[0.34384388, 0.33675833, 0.28733085],std=[0.1574003, 0.15169171, 0.14402839])  
elif data_name == 'wuhan':  
    temp_img1 = TF.normalize(temp_img1, mean=[0.48533902,0.44470758, 0.38696629],std=[0.17376693,0.16997644,0.1731293 ])    
    temp_img2 = TF.normalize(temp_img2, mean=[0.48430226,0.48375396, 0.46155609],std=[0.21371132,0.20393126,0.22009166])  
inputs1,input2, targets = temp_img1, temp_img2, label
inputs1,input2,targets = inputs1.to(device),input2.to(device), targets.to(device)
inputs1,inputs2,targets = Variable(inputs1.unsqueeze(0), volatile=True),Variable(input2.unsqueeze(0),volatile=True) ,Variable(targets)

# model
output_map = model(inputs1,inputs2)
output_map = output_map.detach()

output_map[:,1,:,:] = output_map[:,1,:,:] 
pred = output_map.argmax(dim=1, keepdim=True)
pred = pred.cpu().detach().numpy()
pred_acc = pred 
pred = pred.squeeze()
pred = pred*255
gt_show = cv2.resize(gt_show,(512,512))
gt_show = gt_show * 255
final_output = np.zeros((height,width,3),dtype=np.uint8)


for i in range(512):
    for j in range(512):
        if(gt_show[i][j] == 255 and pred[i][j] == 255):
            final_output[i][j] = [255, 255, 255]
        elif(gt_show[i][j] == 255 and pred[i][j] == 0):
            final_output[i][j] = [0, 255, 0]
        elif(gt_show[i][j] == 0 and pred[i][j] == 255):
            final_output[i][j] = [0, 0, 255]

# output
cv2.imwrite("./sample/pred.jpg",pred)
cv2.imwrite("./sample/gt.jpg",gt_show)
cv2.imwrite("./sample/compare_pred_gt.jpg",final_output)
       

 
