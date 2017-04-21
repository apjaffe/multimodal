import numpy as np
import h5py 
f = h5py.File('flickr30k_ResNet50_pool5_train.mat','r') 
feats = f.get('feats') 
data = np.array(feats)
