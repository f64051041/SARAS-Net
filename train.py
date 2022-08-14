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
import tqdm
from einops.einops import rearrange
import math
from torchvision import transforms as transforms1
from torch.optim import lr_scheduler
import cfgs.config as cfg
import dataset.CD_dataset as dates



def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

ab_test_dir = cfg.SAVE_PATH
check_dir(ab_test_dir)



def main():

    train_transform_det = dates.Compose([
        dates.Scale(cfg.TRANSFROM_SCALES),
    ])
    val_transform_det = dates.Compose([
        dates.Scale(cfg.TRANSFROM_SCALES),
    ])

    train_data = dates.Dataset(cfg.TRAIN_DATA_PATH,cfg.TRAIN_LABEL_PATH,
                                cfg.TRAIN_TXT_PATH,'train',transform=True,
                                transform_med = train_transform_det)
    train_loader = Data.DataLoader(train_data,batch_size=cfg.BATCH_SIZE,
                                    shuffle= True, num_workers= 4, pin_memory= True)
    val_data = dates.Dataset(cfg.VAL_DATA_PATH,cfg.VAL_LABEL_PATH,
                            cfg.VAL_TXT_PATH,'val',transform=True,
                            transform_med = val_transform_det)
    val_loader = Data.DataLoader(val_data, batch_size= cfg.BATCH_SIZE,
                                shuffle= False, num_workers= 4, pin_memory= True)
    # build model
    import model as models
    device = torch.device("cuda:0")
    model = models.Change_detection()

    model= nn.DataParallel(model, device_ids = cfg.gpu_ids)
    model.to(device)

    # Cross entropy loss
    MaskLoss = torch.nn.CrossEntropyLoss().to(device)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters()
                                ,lr=cfg.INIT_LEARNING_RATE,momentum=cfg.MOMENTUM,weight_decay=cfg.DECAY)

    # Scheduler, For each 50 epoch, decay 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


    print("train_loader", len(train_loader))
    print("val_loader", len(val_loader))
    loss_pre = 100000

    
    for epoch in range(cfg.MAX_ITER):
        print("epoch", epoch, "learning rate: ", optimizer.param_groups[0]['lr'])
        model.train()
        # Start to train
        for batch_idx, batch in tqdm.tqdm(enumerate(train_loader)):
            step = epoch * len(train_loader) + batch_idx
            img1_idx,img2_idx,label_idx, filename,height,width = batch
            

            img1 = img1_idx.to(device)
            img2  = img2_idx.to(device)
            label = label_idx.to(device)      
            
            
            output_map = model(img1, img2)
            b_num = output_map.shape[0]
            gt = Variable(dates.resize_label(label.data.cpu().numpy(), \
                                                    size=output_map.data.cpu().numpy().shape[2:]).to(device)) 

            loss = MaskLoss(output_map, gt.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_idx) % 100 == 0:
                print(" Epoch [%d/%d] Loss: %.4f " % (epoch, batch_idx,loss.item()))
        loss_total = 0

        # Start to validate
        for batch_idx, batch in tqdm.tqdm(enumerate(val_loader)):
            with torch.no_grad():
                img1_idx,img2_idx,label_idx, filename,height,width = batch
                img1 = Variable(img1_idx.to(device))
                img2  = Variable(img2_idx.to(device))
                label = Variable(label_idx.to(device))
                output_map = model(img1, img2)
                gt = Variable(dates.resize_label(label.data.cpu().numpy(), \
                                                    size=output_map.data.cpu().numpy().shape[2:]).to(device))
                loss = MaskLoss(output_map, gt.long())
                #loss = MaskLoss(output_map, gt.float())
                loss_total = loss_total + loss
        scheduler.step()
        print("loss_total", loss_total)
        print("loss_pre", loss_pre)
        if loss_total < loss_pre:  
            loss_pre = loss_total
            torch.save({'state_dict': model.state_dict()}, os.path.join(ab_test_dir, 'model_best.pth'))

        if epoch % 10 == 0:
            torch.save({'state_dict': model.state_dict()}, os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
        

if __name__ == '__main__':
    main()

