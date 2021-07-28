import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os.path
import csv
import math
import collections
from tqdm import tqdm
import datetime
import numpy as np
import pandas as pd


class OneShot_Imagenet(data.Dataset):
    def __init__(self,path,type_='train',ways=5,shots=1,test_num=1,epoch=100,gallery_img=100):
        self.ways=ways
        self.shots=shots
        self.test_num=test_num# test samples in each class
        self.size=epoch
        self.gallery_num=gallery_img
        self.gallery_transform=transforms.Compose([transforms.ToPILImage(),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        def loadData(path):
            dictlabels=dict()
            data=pd.read_csv(path)
            for index,row in data.iterrows():
                label=row['label']
                if label in dictlabels:
                    dictlabels[label].append(int(row['filename'][:-4].replace(label,'')))
                else:
                    dictlabels[label]=[int(row['filename'][:-4].replace(label,''))]
            return dictlabels
        self.path=path
        self.train_path=os.path.join(self.path,'train.pkl')
        self.image_path=os.path.join(self.path,type_+'.pkl')
        self.train_csv=os.path.join(self.path,'train.csv')
        self.csv_path=os.path.join(self.path,type_+'.csv')
        self.data=loadData(self.csv_path)

        #loading gallery images
        if self.image_path==self.train_path:
            self.train_data=self.data.copy()
        else:
            self.train_data=loadData(self.train_csv)
        self.gallery=[]
        for class_ in self.train_data.keys():
            files=np.random.choice(self.train_data[class_],self.gallery_num,False)
            self.gallery=self.gallery+list(files)

        self.all_keys=dict()
        key_count=0
        for key in self.data.keys():
            self.all_keys[key]=key_count
            key_count=key_count+1
        key_count=0
        for key in self.train_data.keys():
            self.all_keys[key]=key_count
            key_count=key_count+1
            
        '''
        loading image data
        '''
        self.image_data=pd.read_pickle(self.image_path)
        if self.image_path==self.train_path: 
            self.gallery_image_data=self.image_data
        else:
            self.gallery_image_data=pd.read_pickle(self.train_path)
       
    def get_features(self,model,batch_size=128):
        '''
        this function is used to extract features from gallery images and these
        features are used to calculated the distance between support and gallery features.
        Based on the distance features with greater distance are selected as gallery and support images.
        We could have used the main function but we dont want to store the gradients of this operation
        '''
        print('Gallery size',len(self.gallery))
        num_of_batches=(len(self.gallery)+batch_size-1)//batch_size
        feature_beginning=True
        for b in range(num_of_batches):
            image_beginning=True
            for j in range(b*batch_size,min((b+1)*batch_size,len(self.gallery))):                                
                img=self.gallery_image_data['image_data'][self.gallery[j]]
                image=self.transform(img)
                image=image.unsqueeze(0)
                if image_beginning:
                    image_beginning=False
                    all_images=image
                else:
                    all_images=torch.cat((all_images,image),0)

            
            with torch.no_grad():
                features=model(Variable(all_images,requires_grad=False),mode='feature_extraction')
                if feature_beginning:
                    feature_beginning=False
                    all_features=features
                else:
                    all_features=torch.cat((all_features,features),dim=0)
                    
        return all_features
                              
    def get_gallery_images(self,img): # used in selecting gallery images in "aug_images_basedOnDistance" function
        image=self.gallery_transform(self.gallery_image_data['image_data'][img])
        return image
    
    
    def __getitem__(self,index):
        
        classes=np.random.choice(list(self.data.keys()),self.ways)
        supportFirst,testFirst = True,True
        support_group = torch.LongTensor(self.ways*self.shots,1)
        support_class = torch.LongTensor(self.ways*self.shots,1)

        test_group = torch.LongTensor(self.ways*self.test_num,1)
        test_class = torch.LongTensor(self.ways*self.test_num,1)
        for way in range(self.ways):
            images=np.random.choice(self.data[classes[way]],self.shots)
            for j in range(self.shots):
                image=self.transform(self.image_data['image_data'][images[j]])
                image=image.unsqueeze(0)
                if supportFirst:
                    supportFirst=False
                    supportImages=image
                else:
                    supportImages=torch.cat((supportImages,image),0)
                
                support_group[way*self.shots+j,0]=way
                support_class[way*self.shots+j,0]=self.all_keys[classes[way]]
                
                
            images=np.random.choice(self.data[classes[way]],self.test_num)
            for j in range(self.test_num):
                image=self.transform(self.image_data['image_data'][images[j]])
                image=image.unsqueeze(0)
                
                if testFirst:
                    testFirst=False
                    testImages=image
                else:
                    testImages=torch.cat((testImages,image),0)
                test_group[way*self.test_num+j,0]=way
                test_class[way*self.test_num+j,0]=self.all_keys[classes[way]]
                
             # remeber these are finalized as the gallery is just an another copy of training images
                                     # we will finalized and augment the training the data based on shortest distance between gallery and train
        return supportImages,support_group,support_class,testImages,test_group,test_class
    def __len__(self):
        return self.size
                
        
        
                
        
