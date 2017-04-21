import numpy as np
import h5py
from nltk.tokenize import word_tokenize
import argparse
import json
import os
import dynet as dy
import random
import mt_util

def get_imgs(fname):
  f = h5py.File(fname,'r') 
  feats = f.get('feats') 
  return np.array(feats)

def get_captions(fname, num):
  captions = list()
  for cnum in xrange(num):
    with open(fname+str(cnum+1)) as f:
      for i, line in enumerate(f):
        if len(captions) <= i:
          captions.append([])
        captions[i].append(word_tokenize(line.lower().strip()))
  return captions

def lstm_builder(layer_depth, emb_size, hidden_size, model):
  return dy.VanillaLSTMBuilder(layer_depth, emb_size, hidden_size, model)

class EncoderDecoder:
  def __init__(self, model, imgs, captions_src, model_file, token_file, min_freq, embed_size, hidden_size, dropout):
    self.model = model
    self.src_freqs = mt_util.word_freqs(captions_src)
    self.training = zip(imgs, list(captions_src))
    self.src_vocab_size = len(self.src_token_to_id)
    self.embed_size = int(embed_size)
    self.hidden_size = int(hidden_size)
    self.layers = 1
    if os.path.isfile("src"+token_file):
      self.src_token_to_id = mt_util.defaultify(json.load(open("src"+token_file)))
      self.src_id_to_token = mt_util.invert_ids(self.src_token_to_id)
    else:
      self.src_token_to_id, self.src_id_to_token = mt_util.word_ids(self.src_freqs, int(min_freq))

    if os.path.isfile(model_file):
      self.src_lookup, self.dec_builder, self.W_y, self.b_y = model.load(model_file)
    else:
      self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.self.embed_size))
      self.dec_builder = builder(self.layers, self.embed_size, self.hidden_size, model)
      self.W_y = model.add_parameters((self.src_vocab_size, self.hidden_size))
      self.b_y = model.add_parameters((self.src_vocab_size))

    self.params = [self.src_lookup, self.dec_builder, self.W_y, self.b_y]
    self.dec_builder.set_dropout(float(dropout))

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_src', default='mmt_task2/en/train/train.')
  parser.add_argument('--train_tgt', default='mmt_task2/de/train/de_train.')
  parser.add_argument('--captions_src', default='captions_src.json')
  parser.add_argument('--num_captions', default=5)
  parser.add_argument('--train_img', default='flickr30k_ResNet50_pool5_train.mat')
  parser.add_argument('--batch_size',default=16)
  parser.add_argument('--min_freq',default=10)
  parser.add_argument('--model_file', default = "encdec1.mdl")
  parser.add_argument('--token_file', default = "tokens10.json")
  parser.add_argument('--vocab_freq', default = 2)
  parser.add_argument('--embed_size', default = 256)
  parser.add_argument('--hidden_size', default = 256)
  parser.add_argument('--dropout', default = 0.02)
  parser.add_argument('--image_size', default = 2048) #fixed
  args = parser.parse_args()

  if os.path.isfile(args.captions_src):
    with open(args.captions_src) as cp:
      captions_src = json.loads(cp)
  else:
    captions_src = get_captions(args.train_src, args.num_captions)
    json.dump(captions_src, open(args.captions_src,"w"))
  imgs = get_imgs(args.train_img)
  

  model = dy.Model()
  trainer = dy.AdamTrainer(model)
  encdec = EncoderDecoder(model, imgs, captions_src, args.model_file, args.token_file, args.vocab_freq, args.embed_size, args.hidden_size, args.dropout, args.image_size)
  batches = []
  for cnum in xrange(args.num_captions):
    batches.append(mt_util.make_batches(encdec.training, args.batch_size, cnum, 3))

  num_epochs = 100
  cnums = list(xrange(args.num_captions))
  min_perp = 100000
  for eidx, epoch in enumerate(range(num_epochs)):
    train_loss, train_words = 0, 0
    partial_loss, partial_words = 0, 0
    random.shuffle(cnums)
    for cnum in cnums:
      random.shuffle(batches[cnum])
      for tidx, batch in enumerate(batches[cnum]):
        loss, words = encdec.step_batch(batch)

        if loss is not None:
          lv = loss.value()
          train_loss += lv
          partial_loss += lv
          train_words += words
          partial_words += words
          loss.backward()
          trainer.update()
        
        if tidx % 100 == 0:
          print("Batch %d for caption set %d with loss %f" % (tdix, cnum, partial_loss / partial_words))
          partial_loss = 0
          partial_words = 0

    # TODO: dev perplexity 
    dev_perp = 0
    print("EPOCH %d COMPLETED" % epoch)
    print("Train loss: %f" % (train_loss / train_words))
    print("Dev Perplexity: %f" % dev_perp)
    if dev_perp <= min_perp:
      min_perp = dev_perp
      print("start saving")
      model.save(model_file,attention.params)
      print("done saving")
    trainer.update_epoch(1.0)
  

main()
