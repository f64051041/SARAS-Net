# SARAS-Net
**Scale And Relation Aware Siamese Network for Change Detection ** 
**Target:** Change detection aims to find the difference between two images at different times and output a change map.  

**Overview of SARAS-Net:** 
![image](https://github.com/f64051041/SARAS-Net/blob/main/image/model.jpg)

**Visualize each module by Gradcam:**   
![image](https://github.com/f64051041/SARAS-Net/blob/main/image/structure_heatmap.jpg)

## Requirements
```ruby
cuda: 11.0  
python: 3.6.9  
pytorch: 1.7.0  
torchvision: 0.8.1 
```
## Installation
```ruby
git clone https://github.com/f64051041/SARAS-Net.git  
cd SARAS-Net  
```

## Quick start

## Train
You can find `cfgs/config.py` to set the training parameter.
```ruby
python train.py
```

## Data structure
```ruby
train_dataset  
  |- train_dataset 
      |- image1, image2, gt  
  |- val_dataset  
      |- image1, image2, gt  
  |- train.txt
  |- val.txt
```
