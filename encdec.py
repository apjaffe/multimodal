import numpy as np
import h5py
from nltk.tokenize import word_tokenize
import argparse
import json
import os
import dynet as dy
import random
import mt_util
import math

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

def rnn_builder(layer_depth, emb_size, hidden_size, model):
  return dy.SimpleRNNBuilder(layer_depth, emb_size, hidden_size, model)

class EncoderDecoder:
  def __init__(self, model, imgs, captions_src, model_file, token_file, min_freq, embed_size, hidden_size, dropout, builder):
    self.model = model
    self.src_freqs = mt_util.word_freqs(captions_src)
    self.training = zip(imgs, list(captions_src))
    self.embed_size = int(embed_size)
    self.hidden_size = int(hidden_size)
    #self.image_size = int(image_size)
    self.layers = 1
    if os.path.isfile("src"+token_file):
      self.src_token_to_id = mt_util.defaultify(json.load(open("src"+token_file)))
      self.src_id_to_token = mt_util.invert_ids(self.src_token_to_id)
    else:
      self.src_token_to_id, self.src_id_to_token = mt_util.word_ids(self.src_freqs, int(min_freq))
    
    self.src_vocab_size = len(self.src_token_to_id)

    if os.path.isfile(model_file):
      self.src_lookup, self.dec_builder, self.W_y, self.b_y = model.load(model_file)
    else:
      self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
      self.dec_builder = builder(self.layers, self.embed_size, self.hidden_size, model)
      self.W_y = model.add_parameters((self.src_vocab_size, self.hidden_size))
      self.b_y = model.add_parameters((self.src_vocab_size))
      #self.W_img = model.add_parameters((self.hidden_size, self.image_size))

    self.params = [self.src_lookup, self.dec_builder, self.W_y, self.b_y]
    #self.dec_builder.set_dropout(float(dropout))
 
  def make_caption(self, img, max_len = 50):
    dy.renew_cg()
    W_y = dy.parameter(self.W_y)
    b_y = dy.parameter(self.b_y)
    img_vec = dy.inputVector(img)

    trans_sentence = ['<S>']
    cw = trans_sentence[0]
    dec_state = self.dec_builder.initial_state([img_vec])
    while len(trans_sentence) < max_len:
        embed_t = dy.lookup(self.src_lookup, self.src_token_to_id[cw])
        dec_state = dec_state.add_input(embed_t)
        y_star =  W_y*dec_state.output() + b_y
        # Get probability distribution for the next word to be generated
        p = dy.softmax(y_star)
        p_val = p.npvalue() #vec_value
        amax = np.argmax(p_val)

        # Find the word corresponding to the best id
        cw = self.src_id_to_token[amax]
        if cw == '</S>':
            break
        trans_sentence.append(cw)

    return ' '.join(trans_sentence[1:])

  def step_batch(self, batch, cnum):
    dy.renew_cg()
    W_y = dy.parameter(self.W_y)
    b_y = dy.parameter(self.b_y)
    src_batch = [["<S>"]+x[1][cnum]+["</S>"] for x in batch]
    losses = []
    total_words = 0
    src_cws = []
    masks = []
    src_len = [len(sent) for sent in src_batch]
    max_len_src = max(src_len)
    total_words = sum(src_len)
    num_batches = len(src_batch)
    for c in xrange(max_len_src):
      row = []
      mask = []
      for r in xrange(len(src_batch)):
        if c < len(src_batch[r]):
          row.append(src_batch[r][c])
          mask.append(1)
        else:
          row.append("</S>")
          mask.append(0)
      src_cws.append(row)
      masks.append(dy.reshape(dy.inputVector(mask), (1,), batch_size = num_batches))

    img_batch = [dy.inputVector(x[0]) for x in batch]
    img_vec = dy.reshape(dy.concatenate(img_batch), (self.hidden_size,), batch_size = num_batches)
    #img_state = dy.inputVector(img_batch)#dy.reshape(dy.inputVector(img_batch), (self.hidden_size,), batch_size = num_batches)
    dec_state = self.dec_builder.initial_state([img_vec])
    
    for i, (cws, nws, mask) in enumerate(zip(src_cws, src_cws[1:], masks)):
        cwids = [self.src_token_to_id[cw] for cw in cws] 
        embed_t = dy.lookup_batch(self.src_lookup, cwids)
        dec_state = dec_state.add_input(embed_t)
        nwids = [self.src_token_to_id[nw] for nw in nws]
        y_star =  W_y*dec_state.output() + b_y
        loss = dy.pickneglogsoftmax_batch(y_star, nwids)
        mask_loss = dy.cmult(loss, mask)
        losses.append(mask_loss)

    return (dy.sum_batches(dy.esum(losses))), total_words

def dev_perplexity(dev_batches, encdec, num_captions):
    dev_loss = 0
    dev_words = 0
    for cnum in xrange(num_captions):
      random.shuffle(dev_batches[cnum])
      for tidx, batch in enumerate(dev_batches[cnum]):
        loss,words = encdec.step_batch(batch, cnum) 
        if loss is not None:
          lv = loss.value()
          dev_loss += lv
          dev_words += words
        if tidx >= 1:
          print("Dev cnum %d" % cnum)
          break

    return math.exp(dev_loss / dev_words)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_src', default='mmt_task2/en/train/train.')
  parser.add_argument('--train_tgt', default='mmt_task2/de/train/de_train.')
  parser.add_argument('--captions_src', default='captions_src.json')
  parser.add_argument('--num_captions', default=5)
  parser.add_argument('--train_img', default='flickr30k_ResNet50_pool5_train.mat')
  parser.add_argument('--valid_src', default='mmt_task2/en/val/val.')
  parser.add_argument('--valid_tgt', default='mmt_task2/de/val/de_val.')
  parser.add_argument('--valid_img', default='flickr30k_ResNet50_pool5_val.mat')
  parser.add_argument('--batch_size',default=16)
  parser.add_argument('--min_freq',default=10)
  parser.add_argument('--model_file', default = "encdec1.mdl")
  parser.add_argument('--token_file', default = "tokens10.json")
  parser.add_argument('--vocab_freq', default = 2)
  parser.add_argument('--embed_size', default = 256)
  parser.add_argument('--hidden_size', default = 2048)
  parser.add_argument('--dropout', default = 0.02)
  parser.add_argument('--lstm', action='store_true')
  #parser.add_argument('--image_size', default = 2048) #fixed
  parser.add_argument('--dynet-mem')
  parser.add_argument('--dynet-gpu')
  args = parser.parse_args()

  if os.path.isfile(args.captions_src):
    with open(args.captions_src) as cp:
      captions_train_src = json.load(cp)
  else:
    captions_train_src = get_captions(args.train_src, args.num_captions)
    json.dump(captions_src, open(args.captions_src,"w"))
  
  train_imgs = get_imgs(args.train_img)
  valid_imgs = get_imgs(args.valid_img)
 
  captions_valid_src = get_captions(args.valid_src, args.num_captions)
  dev = zip(valid_imgs, list(captions_valid_src))
  dev_batches = []
  for cnum in xrange(args.num_captions):
    dev_batches.append(mt_util.make_batches( dev, args.batch_size, cnum,3))
  
  builder = lstm_builder if args.lstm else rnn_builder

  model = dy.Model()
  trainer = dy.AdamTrainer(model)
  encdec = EncoderDecoder(model, train_imgs, captions_train_src, args.model_file, args.token_file, args.vocab_freq, args.embed_size, args.hidden_size, args.dropout, builder)
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
        loss, words = encdec.step_batch(batch, cnum)

        if loss is not None:
          lv = loss.value()
          train_loss += lv
          partial_loss += lv
          train_words += words
          partial_words += words
          loss.backward()
          trainer.update()
        
        if tidx % 100 == 0:
          print(encdec.make_caption(valid_imgs[0]))
          print(encdec.make_caption(valid_imgs[1]))
          print("Batch %d for caption set %d with loss %f" % (tidx, cnum, partial_loss / partial_words))
          partial_loss = 0
          partial_words = 0

    # TODO: dev perplexity 
    dev_perp = dev_perplexity(dev_batches, encdec, args.num_captions)
    print("EPOCH %d COMPLETED" % epoch)
    print("Train loss: %f" % (train_loss / train_words))
    print("Dev Perplexity: %f" % dev_perp)
    if dev_perp <= min_perp:
      min_perp = dev_perp
      print("start saving")
      model.save(args.model_file,encdec.params)
      print("done saving")
    trainer.update_epoch(1.0)
  

main()
