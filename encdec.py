import numpy as np
import h5py
from nltk.tokenize import word_tokenize
import argparse
import json
import os

def get_imgs(fname):
  f = h5py.File(fname,'r') 
  feats = f.get('feats') 
  return np.array(feats)

def get_captions(fname, num):
  captions = list()
  for cnum in xrange(1,num+1):
    with open(fname+str(num)) as f:
      for i, line in enumerate(f):
        if len(captions) <= i:
          captions.append([])
        captions[i].append(word_tokenize(line.lower().strip()))





def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_src', default='mmt_task2/en/train/train.')
  parser.add_argument('--train_tgt', default='mmt_task2/de/train/de_train.')
  parser.add_argument('--captions_src', default='captions_src.json')
  parser.add_argument('--num_captions', default=5)
  parser.add_argument('--train_img', default='flickr30k_ResNet50_pool5_train.mat')
  args = parser.parse_args()

  if os.path.isfile(args.captions_src):
    with open(args.captions_src) as cp:
      captions_src = json.loads(cp)
  else:
    captions_src = get_captions(args.train_src, args.num_captions)
    json.dump(captions_src, open(args.captions_src,"w"))
  imgs = get_imgs(args.train_img)
  
  training = zip(imgs, list(captions_src))
  print(training[0:10])

main()
