

import argparse
import os

class Parameters():
    def __init__(self):### 
        # Training settings
        self.LR=0.001
        self.clsLR=0.001
        self.batch_size=30
        self.nthreads=8
        self.tensorname='IDeMNet'
        self.ways=5
        self.shots=5
        self.test_num=15
        self.augnum=5
        self.data='miniImageEmbedding'
        self.network='None'
        self.gallery_img=30
        self.stepSize=10
        self.patch_size=3
        self.epoch=600
        self.trainways=5
        self.fixScale=0
        self.GNet='none'
        self.train_from_scratch=True
        self.fix_deform=True
        self.fix_emb=True
        self.chooseNum=15