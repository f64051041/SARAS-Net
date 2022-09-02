import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import scipy.io
import scipy.misc as m
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import cfgs.config as cfg
from torchvision import transforms as transforms1
import random
import torchvision.transforms.functional as TF
from PIL import Image
from PIL import ImageFilter
import collections
import torch.nn as nn
from torch.autograd import Variable

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class Scale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

def resize_label(label, size):

    label = np.expand_dims(label,axis=0)
    label_resized = np.zeros((1,label.shape[1],size[0],size[1]))
    interp = nn.Upsample(size=(size[0], size[1]),mode='bilinear')

    labelVar = Variable(torch.from_numpy(label).float())  
    label_resized[:, :,:,:] = interp(labelVar).data.numpy()
    label_resized = np.array(label_resized, dtype=np.int32)
    return torch.from_numpy(np.squeeze(label_resized,axis=0)).float()

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

palette = [0, 0, 0,255,255,255]
'''''''''
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
'''''''''

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_pascal_labels():
    return np.asarray([[0,0,0],[255,255,255]])

def decode_segmap(temp, plot=False):

    label_colours = get_pascal_labels()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    #rgb = np.resize(rgb,(321,321,3))
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

    
def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)

def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top+ch, cont_left, cont_left+cw, img_top, img_top+ch, img_left, img_left+cw

def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype)*default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype)*default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)

#### source dataset is only avaiable by sending an email request to author #####
#### upon request is shown in http://ghsi.github.io/proj/RSS2016.html ####
#### more details are presented in http://ghsi.github.io/assets/pdfs/alcantarilla16rss.pdf ###
class Dataset(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=True, transform_med = None, normal = None):
        
        self.label_path = label_path
        self.img_path = img_path
        #self.img2_path = img2_path
        self.img_txt_path = file_name_txt_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        self.flag = split_flag
        self.transform = transform
        self.transform_med = transform_med
        self.normal = normal
        self.img_label_path_pairs = self.get_img_label_path_pairs()
        self.with_scale_random_crop = True
        
    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        #print("self.img_txt_path", self.img_txt_path) /root/notebooks/SceneChangeDet/val.txt
        if self.flag =='train':
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name,image2_name,mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                #print("image1_name", image1_name)
                #print("image2_name", image2_name)
                #print("mask_name", mask_name)
                #print("mask_name", mask_name)
                #extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]  #其他dataset 需要這行
                #print extract_name
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                #print("img1_file",img1_file, img2_file,lbl_file)
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,image2_name])

        if self.flag == 'val':
            self.label_ext = '.png'
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                #extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')] #其他dataset 需要這行
                #print extract_name
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path , mask_name)
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,image2_name])

        if self.flag == 'test':

            for idx, did in enumerate(open(self.img_txt_path)):
              image1_name, image2_name = did.strip("\n").split(' ')
              img1_file = os.path.join(self.img_path, image1_name)
              img2_file = os.path.join(self.img_path, image2_name)
              img_label_pair_list.setdefault(idx, [img1_file, img2_file,None,image2_name])

        return img_label_pair_list


    def __getitem__(self, index):
        
        img1_path,img2_path,label_path,filename = self.img_label_path_pairs[index]
        ####### load images #############
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        #print filename
        height,width,_ = np.array(img1,dtype= np.uint8).shape
        if self.transform_med != None:
           img1 = self.transform_med(img1)
           img2 = self.transform_med(img2)
        #img1 = np.array(img1,dtype= np.uint8)
        #img2 = np.array(img2,dtype= np.uint8)
        ####### load labels ############
        if self.flag == 'train' or self.flag == 'val':
            #print("label_path", label_path)
            label = Image.open(label_path)
            #print("label", label.size, np.max(label), label_path)
            
            if self.transform_med != None:
                label = self.transform_med(label)
            
            #label = np.array(label,dtype=np.int32)
            #label[label>0] = 1
        else:
            label = np.zeros((height,width,3),dtype=np.uint8)
        
        #img1 = TF.to_pil_image(img1)   
        #img2 = TF.to_pil_image(img2)
        #label = TF.to_pil_image(label)
        img_size = img1.size[0]
        random_base = 0.5
        if random.random() > 0.5:
            img1 = TF.hflip(img1) 
            img2 = TF.hflip(img2) 
            label = TF.hflip(label) 
            
        if random.random() > 0.5:
            img1 = TF.vflip(img1) 
            img2 = TF.vflip(img2) 
            label = TF.vflip(label)
            
        if random.random() > 0.5:
            #angles = [90, 180, 270]
            #index = 
            angle = random.randint(0, 359)
            
            img1 = TF.rotate(img1, angle) 
            img2 = TF.rotate(img2, angle) 
            label = TF.rotate(label, angle)     
            
        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            img1 = pil_rescale(img1, target_scale, order=3) 
            img2 = pil_rescale(img2, target_scale, order=3) 
            label = pil_rescale(label, target_scale, order=0) 
            # crop
            imgsize = img1.size  # h, w
            
            box = get_random_crop_box(imgsize=imgsize, cropsize=img_size)
            img1 = pil_crop(img1, box, cropsize=img_size, default_value=0)
            img2 = pil_crop(img2, box, cropsize=img_size, default_value=0)
            label = pil_crop(label, box, cropsize=img_size, default_value=0)


        if random.random() > 0:
            radius = random.random()
            img1 = img1.filter(ImageFilter.GaussianBlur(radius=radius))
            img2 = img2.filter(ImageFilter.GaussianBlur(radius=radius))   
        
        img1 = TF.to_tensor(img1) 
        img2 = TF.to_tensor(img2) 
        label = torch.from_numpy(np.array(label, np.uint8)) #.unsqueeze(dim=0)
                

        #levir
        if cfg.dataset_name == 'LEVIR_CD':
            img1 = TF.normalize(img1, mean=[0.44758545, 0.44381796,  0.37912835],std=[0.21713617, 0.20354738, 0.18588887]) 
            img2 = TF.normalize(img2, mean=[0.34384388, 0.33675833, 0.28733085],std=[0.1574003, 0.15169171, 0.14402839])  
        #DFINF
        elif cfg.dataset_name == 'DSIFN_CD':
            img1 = TF.normalize(img1, mean=[0.39297381,0.40958949,0.37175044],std=[0.20037114,0.19629362,0.19664517]) 
            img2 = TF.normalize(img2, mean=[0.38990542,0.38161003,0.38456258],std=[0.19449101,0.17802028,0.17139462]) 
        #wuhan
        elif cfg.dataset_name == 'WHU_CD':
            img1 = TF.normalize(img1, mean=[0.48533902,0.44470758, 0.38696629],std=[0.17376693,0.16997644,0.1731293 ])
            img2 = TF.normalize(img2, mean=[0.48430226,0.48375396,0.46155609],std=[0.21371132,0.20393126,0.22009166])
        #CCD 
        elif cfg.dataset_name == 'CCD_CD': 
            img1 = TF.normalize(img1, mean=[0.35406487, 0.39149388, 0.34328843],std=[0.21638811, 0.23465545, 0.20915913])
            img2 = TF.normalize(img2, mean=[0.47369619, 0.49932158, 0.46938096],std=[0.24345056, 0.26037342, 0.25711795]) 
        #OSCD
        elif cfg.dataset_name == 'OSCD_CD': 
            img1 = TF.normalize(img1, mean=[0.30269907, 0.29237894, 0.31021307],std=[0.20468995, 0.14207161, 0.12051828])
            img2 = TF.normalize(img2, mean=[0.2926714, 0.28220245, 0.30294364],std=[0.19940712, 0.14064144, 0.12057147]) 
        else:
            print("Your dataset_name in cfgs/config.py has error!")

        label[label>0]=1    
        

        
        return img1,img2,label,str(filename),int(height),int(width)

    def __len__(self):

        return len(self.img_label_path_pairs)


