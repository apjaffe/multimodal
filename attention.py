import numpy as np
from nltk.tokenize import word_tokenize
import argparse
import json
import os
import dynet as dy
import random
import mt_util
import math
import heapq
from nltk.translate import bleu_score

ENCDECPIPELINE="encdecpipeline"
ATTPIPELINE="attpipeline"
FORK="fork"
MRT="mrt"
DUALATT="dualatt"

def sample(p_val):
  spot = -1
  rnd = random.random() * sum(p_val)
  while rnd > 0:
    spot += 1
    rnd -= p_val[spot]
  return spot


def get_imgs(fname):
  return np.load(fname)


def get_captions(fname, num, tokenizer):
  captions = list()
  for cnum in xrange(num):
    with open(fname+str(cnum+1)) as f:
      for i, line in enumerate(f):
        if len(captions) <= i:
          captions.append([])
        captions[i].append(tokenizer(line.decode("utf-8").lower().strip()))
  return captions

def lstm_builder(layer_depth, emb_size, hidden_size, model):
  return dy.VanillaLSTMBuilder(layer_depth, emb_size, hidden_size, model)

def rnn_builder(layer_depth, emb_size, hidden_size, model):
  return dy.SimpleRNNBuilder(layer_depth, emb_size, hidden_size, model)

class Attention:
  def __init__(self, model, imgs, captions_src, captions_tgt, model_file, token_file, min_freq, embed_size, hidden_size, image_size, image_points, attention_size, dropout, builder, multilang, multilangmode, pipeline_candidates, sample_embeds, unk_penalty):
    self.model = model
    self.src_freqs = mt_util.word_freqs(captions_src)
    self.tgt_freqs = mt_util.word_freqs(captions_tgt)
    self.training = zip(imgs, list(captions_src), list(captions_tgt))
    self.embed_size = int(embed_size)
    self.hidden_size = int(hidden_size)
    self.image_size = int(image_size)
    self.image_points = int(image_points)
    self.attention_size = int(attention_size)
    self.layers = 1
    self.multilang = multilang
    self.multilangmode = multilangmode
    self.unk_penalty = float(unk_penalty)
    self.max_len = 20 # for MRT, not for generation
    if os.path.isfile("tokens/src"+token_file):
      self.src_token_to_id = mt_util.defaultify(json.load(open("tokens/src"+token_file)))
      self.src_id_to_token = mt_util.invert_ids(self.src_token_to_id)
      self.tgt_token_to_id = mt_util.defaultify(json.load(open("tokens/tgt"+token_file)))
      self.tgt_id_to_token = mt_util.invert_ids(self.tgt_token_to_id)
    else:
      self.src_token_to_id, self.src_id_to_token = mt_util.word_ids(self.src_freqs, int(min_freq))
      self.tgt_token_to_id, self.tgt_id_to_token = mt_util.word_ids(self.tgt_freqs, int(min_freq))
      #json.dump(dict(self.src_token_to_id), open("tokens/src"+token_file,"w"))
      #json.dump(dict(self.tgt_token_to_id), open("tokens/tgt"+token_file,"w"))
    
    self.src_vocab_size = len(self.src_token_to_id)
    self.tgt_vocab_size = len(self.tgt_token_to_id)
    print("Src vocab size: %d" % self.src_vocab_size)
    print("Tgt vocab size: %d" % self.tgt_vocab_size)

    if os.path.isfile(model_file):
      if multilang:
        if multilangmode == FORK:
          self.src_lookup, self.dec_builder, self.W_y, self.b_y, self.W1_att_img, self.W1_att_src, self.w2_att, self.tgt_lookup, self.W_tgt, self.b_tgt = model.load(model_file)
        elif multilangmode == ENCDECPIPELINE:
           self.src_lookup, self.dec_builder, self.W_y, self.b_y, self.W1_att_img, self.W1_att_src, self.w2_att, self.tgt_lookup, self.W_tgt, self.b_tgt, self.tgt_enc_builder, self.tgt_dec_builder = model.load(model_file)
        elif multilangmode == ATTPIPELINE or multilangmode == MRT or multilangmode == DUALATT:
           self.src_lookup, self.dec_builder, self.W_y, self.b_y, self.W1_att_img, self.W1_att_src, self.w2_att, self.tgt_lookup, self.W_tgt, self.b_tgt, self.W1_patt_src, self.W1_patt_tgt, self.w2_patt, self.tgt_dec_builder = model.load(model_file)
       
      else:
        self.src_lookup, self.dec_builder, self.W_y, self.b_y, self.W1_att_img, self.W1_att_src, self.w2_att = model.load(model_file)
    else:
      self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
      self.dec_builder = builder(self.layers, self.embed_size + self.image_size, self.hidden_size, model)
      self.W_y = model.add_parameters((self.src_vocab_size, self.hidden_size))
      self.b_y = model.add_parameters((self.src_vocab_size))
      self.W1_att_img = model.add_parameters((self.attention_size, self.image_size))
      self.W1_att_src = model.add_parameters((self.attention_size, self.hidden_size))
      self.w2_att = model.add_parameters((1,self.attention_size))
      if multilang:
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.W_tgt = model.add_parameters((self.tgt_vocab_size, self.hidden_size))
        self.b_tgt = model.add_parameters((self.tgt_vocab_size))
      if multilangmode == ENCDECPIPELINE:
        self.tgt_enc_builder = builder(self.layers, self.embed_size, self.hidden_size, model)
        self.tgt_dec_builder = builder(self.layers, self.embed_size, self.hidden_size, model)
      if multilangmode == ATTPIPELINE or multilangmode == MRT:
        self.W1_patt_src = model.add_parameters((self.attention_size, self.embed_size))
        self.W1_patt_tgt = model.add_parameters((self.attention_size, self.hidden_size))
        self.w2_patt = model.add_parameters((1,self.attention_size))
        self.tgt_dec_builder = builder(self.layers, self.embed_size + self.embed_size, self.hidden_size, model) # attention + current word
      if multilangmode == DUALATT:
        self.W1_patt_src = model.add_parameters((self.attention_size, self.embed_size+self.image_size))
        self.W1_patt_tgt = model.add_parameters((self.attention_size, self.hidden_size))
        self.w2_patt = model.add_parameters((1,self.attention_size))
        self.tgt_dec_builder = builder(self.layers, self.embed_size + self.embed_size+ self.image_size, self.hidden_size, model) # attention + current word + current image


    self.pipeline_candidates = int(pipeline_candidates)
    self.params = [self.src_lookup, self.dec_builder, self.W_y, self.b_y, self.W1_att_img, self.W1_att_src, self.w2_att]
    if multilang:
      self.params += [self.tgt_lookup, self.W_tgt, self.b_tgt]
    if multilangmode == ENCDECPIPELINE:
      self.params += [self.tgt_enc_builder, self.tgt_dec_builder]
    if multilangmode == ATTPIPELINE or multilangmode == MRT:
      self.params += [self.W1_patt_src, self.W1_patt_tgt, self.w2_patt, self.tgt_dec_builder]
      
    if sample_embeds == "argmax":
      self.sampler = np.argmax
    elif sample_embeds == "random":
      self.sampler = sample
    elif sample_embeds == "uniform":
      self.sampler = np.random.choice
    else:
      print("Invalid sampler %s" % sampler_embeds)

    self.dec_builder.set_dropout(float(dropout))
 
  # Calculates the context vector using a MLP
  # h_fs: matrix of embeddings for the source words
  # h_e: hidden state of the decoder
  # Partly inspired by examples at https://github.com/clab/dynet/blob/master/examples/
  def __attention_mlp(self, h_fs_matrix, h_src, w1, W1_att_src, w2_att):
      w2 = W1_att_src*h_src #(attention_size X hidden_size) * (hidden_size * 1) 
      #h_fs_matrix is (image_size) x (image_points)
      #h_src is hidden_size x 1
      #w1 is (attention_size x image_points)
      tanned = dy.tanh(dy.colwise_add(w1,w2)) 
      a_t = dy.transpose(w2_att * tanned)
      alignment = dy.softmax(a_t)
      
      # alignment is image_points x 1
      c_t = h_fs_matrix * alignment
      return c_t, alignment # image_size x 1, image_points x 1
  
  def do_make_beam_caption(self, img, src_lookup, src_token_to_id, src_id_to_token, src_vocab_size, W_y, b_y, max_len, show_attention, beam_size, show_candidates):
    if beam_size == 1:
      return self.do_make_caption(img, src_lookup, src_token_to_id, src_id_to_token, W_y, b_y, max_len, show_attention, src_vocab_size)

    W1_att_img = dy.parameter(self.W1_att_img) # attention_size * image_size
    img_vec = dy.inputTensor(img) #image_points * image_size
    h_fs_matrix = dy.transpose(img_vec)
    trans_sentence = ['<S>']
    w1 = W1_att_img * h_fs_matrix
    cw = trans_sentence[0]
    c_t = dy.vecInput(self.image_size)
    start = dy.concatenate([dy.lookup(src_lookup, src_token_to_id['<S>']), c_t])
    dec_state = self.dec_builder.initial_state().add_input(start)
 
    candidates = []

    candidates.append((trans_sentence, dec_state, 0))
    position = 0
    W1_att_src = dy.parameter(self.W1_att_src)
    w2_att = dy.parameter(self.w2_att)

    while position < max_len:
      next_candidates = []
      for (trans_sentence, dec_state, prob) in candidates:
        cw = trans_sentence[-1]
        if cw == '</S>':
          next_candidates.append((trans_sentence, dec_state, prob))
          continue

        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_att_src, w2_att)

        embed_t = dy.lookup(src_lookup, src_token_to_id[cw])
        x_t = dy.concatenate([embed_t, c_t])
        
        dec_state = dec_state.add_input(x_t)
        y_star =  W_y*dec_state.output() + b_y
        # Get probability distribution for the next word to be generated
        p = dy.softmax(y_star)
        p_val = p.npvalue() #vec_value
        p_val[0] *= self.unk_penalty
        
        candidate_ids = xrange(src_vocab_size)
        amaxs = heapq.nlargest(beam_size, candidate_ids, lambda id: p_val[id])
        for next_id in amaxs:
          next_prob = prob + math.log(p_val[next_id])
          nw = src_id_to_token[next_id]
          next_sentence = trans_sentence + [nw]
          next_candidates.append((next_sentence, dec_state, next_prob))


      candidates = heapq.nlargest(beam_size, next_candidates, lambda x: x[2])
  
      position += 1

    #print(list([cand[0] for cand in candidates]))
    if show_candidates:
      out = ""
      for cand in candidates:
        out += " ".join(cand[0][1:-1]) + " | "
      return out, [], []
      

    return " ".join(candidates[0][0][1:-1]), [], []
    #sent = ' '.join(trans_sentence[1:])
    #return sent

   
  def do_make_dual_beam_caption(self, img, src_lookup, src_token_to_id, src_id_to_token, src_vocab_size, w_y, b_y, max_len, show_attention, beam_size, show_candidates):
    if beam_size == 1:
      return self.do_make_caption(img, src_lookup, src_token_to_id, src_id_to_token, W_y, b_y, max_len, show_attention, src_vocab_size)

    W1_att_img = dy.parameter(self.W1_att_img) # attention_size * image_size
    img_vec = dy.inputTensor(img) #image_points * image_size
    h_fs_matrix = dy.transpose(img_vec)
    trans_sentence = ['<S>']
    w1 = W1_att_img * h_fs_matrix
    cw = trans_sentence[0]
    c_t = dy.vecInput(self.image_size)
    start = dy.concatenate([dy.lookup(src_lookup, src_token_to_id['<S>']), c_t])
    dec_state = self.dec_builder.initial_state().add_input(start)
 
    candidates = []

    candidates.append((trans_sentence, dec_state, 0))
    position = 0
    W1_att_src = dy.parameter(self.W1_att_src)
    w2_att = dy.parameter(self.w2_att)

    while position < max_len:
      next_candidates = []
      for (trans_sentence, dec_state, prob) in candidates:
        cw = trans_sentence[-1]
        if cw == '</S>':
          next_candidates.append((trans_sentence, dec_state, prob))
          continue

        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_att_src, w2_att)

        embed_t = dy.lookup(src_lookup, src_token_to_id[cw])
        x_t = dy.concatenate([embed_t, c_t])
        
        dec_state = dec_state.add_input(x_t)
        y_star =  W_y*dec_state.output() + b_y
        # Get probability distribution for the next word to be generated
        p = dy.softmax(y_star)
        p_val = p.npvalue() #vec_value
        p_val[0] *= self.unk_penalty
        
        candidate_ids = xrange(src_vocab_size)
        amaxs = heapq.nlargest(beam_size, candidate_ids, lambda id: p_val[id])
        for next_id in amaxs:
          next_prob = prob + math.log(p_val[next_id])
          nw = src_id_to_token[next_id]
          next_sentence = trans_sentence + [nw]
          next_candidates.append((next_sentence, dec_state, next_prob))


      candidates = heapq.nlargest(beam_size, next_candidates, lambda x: x[2])
  
      position += 1

    #print(list([cand[0] for cand in candidates]))
    if show_candidates:
      out = ""
      for cand in candidates:
        out += " ".join(cand[0][1:-1]) + " | "
      return out, [], []
      

    return " ".join(candidates[0][0][1:-1]), [], []

  def do_make_caption(self, img, src_lookup, src_token_to_id, src_id_to_token, W_y, b_y, max_len, show_attention, src_vocab_size):
    W1_att_img = dy.parameter(self.W1_att_img) # attention_size * image_size
    img_vec = dy.inputTensor(img) #image_points * image_size
    h_fs_matrix = dy.transpose(img_vec)
    trans_sentence = ['<S>']
    w1 = W1_att_img * h_fs_matrix
    cw = trans_sentence[0]
    c_t = dy.vecInput(self.image_size)
    start = dy.concatenate([dy.lookup(src_lookup, src_token_to_id['<S>']), c_t])
    dec_state = self.dec_builder.initial_state().add_input(start)
    att = []
    embeds = []
    W1_att_src = dy.parameter(self.W1_att_src)
    w2_att = dy.parameter(self.w2_att)
    while len(trans_sentence) < max_len:
        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_att_src, w2_att)
        if show_attention:
          att.append(a_t.value().tolist())

        embed_t = dy.lookup(src_lookup, src_token_to_id[cw])
        x_t = dy.concatenate([embed_t, c_t])
        
        dec_state = dec_state.add_input(x_t)
        y_star =  W_y*dec_state.output() + b_y
        # Get probability distribution for the next word to be generated
        p = dy.softmax(y_star)
        p_val = p.npvalue() #vec_value
        p_val[0] *= self.unk_penalty
        amax = np.argmax(p_val)

        if self.multilangmode == ENCDECPIPELINE or self.multilangmode == ATTPIPELINE:
          if self.pipeline_candidates > 1:
            avg_embed = None
            for i in xrange(self.pipeline_candidates):
              i = random.randint(0, src_vocab_size - 1)
              embed = dy.lookup(src_lookup, i) * dy.pick(p, i)
              if avg_embed is None:
                avg_embed = embed
              else:
                avg_embed += embed
            embeds.append(avg_embed)
          else:
            embeds.append(dy.lookup(src_lookup, amax) * dy.pick(p, amax))
        elif self.multilangmode == MRT:
          embeds.append(dy.lookup(src_lookup, amax))

        # Find the word corresponding to the best id
        cw = src_id_to_token[amax]
        if cw == '</S>':
            break
        trans_sentence.append(cw)

    sent = ' '.join(trans_sentence[1:])
    return sent, att, embeds

  def make_caption(self, img, max_len = 30, show_attention = False, is_src = True, beam_size = 1, show_candidates = False):
    dy.renew_cg()

    if is_src:
      W_y = dy.parameter(self.W_y)
      b_y = dy.parameter(self.b_y)
      sent, att, emb = self.do_make_beam_caption(img, self.src_lookup, self.src_token_to_id,  self.src_id_to_token, self.src_vocab_size, W_y, b_y, max_len, show_attention, beam_size, show_candidates)
      return sent, att
    elif self.multilangmode == FORK:
      W_tgt = dy.parameter(self.W_tgt)
      b_tgt = dy.parameter(self.b_tgt)
      sent, att, emb = self.do_make_beam_caption(img, self.tgt_lookup, self.tgt_token_to_id,  self.tgt_id_to_token, self.tgt_vocab_size, W_tgt, b_tgt, max_len, show_attention, beam_size, show_candidates)
      return sent, att
    elif self.multilangmode == ENCDECPIPELINE or self.multilangmode == ATTPIPELINE or self.multilangmode == MRT: # beam size must be 1
      if beam_size > 1:
        return "beam search not supported for target language", []
      W_y = dy.parameter(self.W_y)
      b_y = dy.parameter(self.b_y)
      sent, att, emb = self.do_make_beam_caption(img, self.src_lookup, self.src_token_to_id,  self.src_id_to_token, self.src_vocab_size, W_y, b_y, max_len, show_attention, beam_size, show_candidates)
      W_tgt = dy.parameter(self.W_tgt)
      b_tgt = dy.parameter(self.b_tgt)

      if self.multilangmode == ENCDECPIPELINE:
        sent_tgt = self.make_encdec_caption(emb, self.tgt_lookup, self.tgt_token_to_id, self.tgt_id_to_token, self.tgt_vocab_size, W_tgt, b_tgt, self.tgt_enc_builder, self.tgt_dec_builder, max_len)
        return sent_tgt, []
      elif self.multilangmode == ATTPIPELINE or self.multilangmode == MRT:
        W1_patt_src = dy.parameter(self.W1_patt_src)
        W1_patt_tgt = dy.parameter(self.W1_patt_tgt)
        w2_patt = dy.parameter(self.w2_patt)
        return self.make_att_caption(emb, self.tgt_lookup, self.tgt_token_to_id, self.tgt_id_to_token, self.tgt_vocab_size, W_tgt, b_tgt, W1_patt_src, W1_patt_tgt, w2_patt, self.tgt_dec_builder, max_len, show_attention)


  def make_encdec_caption(self, avg_embeds, tgt_lookup, tgt_token_to_id, tgt_id_to_token, tgt_vocab_size, W_tgt, b_tgt, tgt_enc_builder, tgt_dec_builder, max_len):
    enc_state = tgt_enc_builder.initial_state()
    for embed in avg_embeds:
      enc_state = enc_state.add_input(embed)
   
    encoded = enc_state.output()
    dec_state = tgt_dec_builder.initial_state([dy.vecInput(self.hidden_size), encoded])

    trans_sentence = ['<S>']
    cw = trans_sentence[0]

    while len(trans_sentence) < max_len:
        embed_t = dy.lookup(tgt_lookup, tgt_token_to_id[cw])
        dec_state = dec_state.add_input(embed_t)
        
        y_star =  W_tgt*dec_state.output() + b_tgt
        p = dy.softmax(y_star)
        p_val = p.npvalue()
        amax = np.argmax(p_val)

        # Find the word corresponding to the best id
        cw = tgt_id_to_token[amax]
        if cw == '</S>':
            break
        trans_sentence.append(cw)

    sent = ' '.join(trans_sentence[1:])
    return sent

  def make_att_caption(self, avg_embeds, tgt_lookup, tgt_token_to_id, tgt_id_to_token, tgt_vocab_size, W_tgt, b_tgt, W1_patt_src, W1_patt_tgt, w2_patt, tgt_dec_builder, max_len, show_attention):
    h_fs_matrix = dy.concatenate_cols(avg_embeds) #embed_size * sentence_len 
    trans_sentence = ['<S>']
    w1 = W1_patt_src * h_fs_matrix
    cw = trans_sentence[0]
    c_t = dy.vecInput(self.embed_size)
    start = dy.concatenate([dy.lookup(tgt_lookup, tgt_token_to_id['<S>']), c_t])
    dec_state = tgt_dec_builder.initial_state().add_input(start)
    att = []
    embeds = []
    while len(trans_sentence) < max_len:
        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_patt_tgt, w2_patt)
        if show_attention:
          att.append(a_t.value().tolist())

        embed_t = dy.lookup(tgt_lookup, tgt_token_to_id[cw])
        x_t = dy.concatenate([embed_t, c_t])
        
        dec_state = dec_state.add_input(x_t)
        y_star =  W_tgt*dec_state.output() + b_tgt
        # Get probability distribution for the next word to be generated
        p = dy.softmax(y_star)
        p_val = p.npvalue() #vec_value
        amax = np.argmax(p_val)


        # Find the word corresponding to the best id
        cw = tgt_id_to_token[amax]
        if cw == '</S>':
            break
        trans_sentence.append(cw)

    sent = ' '.join(trans_sentence[1:])
    return sent, att
    

  def att_losses(self, avg_embeds, tgt_batch, tgt_lookup, tgt_token_to_id, W_tgt, b_tgt, W1_patt_src, W1_patt_tgt, w2_patt, tgt_dec_builder, prob):
    h_fs_matrix = dy.concatenate_cols(avg_embeds) #embed_size * sentence_len 
    c_t = dy.vecInput(self.embed_size)
    total_words, att_cws, masks, num_batches = self.get_masks(tgt_batch)

    losses = []
    start_tokens = ["<S>"] * num_batches
    start_ids = [tgt_token_to_id[st] for st in start_tokens] 
    start = dy.concatenate([dy.lookup_batch(tgt_lookup, start_ids), c_t]) 
    dec_state = tgt_dec_builder.initial_state().add_input(start)
    w1 = W1_patt_src * h_fs_matrix
    avg_embeds = list()
    for i, (cws, nws, mask) in enumerate(zip(att_cws, att_cws[1:], masks)):
        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_patt_tgt, w2_patt)
        cwids = [tgt_token_to_id[cw] for cw in cws] 
        embed_t = dy.lookup_batch(tgt_lookup, cwids)

        x_t = dy.concatenate([embed_t, c_t])
        dec_state = dec_state.add_input(x_t)

        y_star =  W_tgt*dec_state.output() + b_tgt #y_star is src_vocab_size * 1
        
        nwids = [tgt_token_to_id[nw] for nw in nws]
        loss = dy.pickneglogsoftmax_batch(y_star, nwids)
        mask_loss = dy.cmult(loss, mask) * prob
        losses.append(mask_loss)
    return losses, total_words

  def compute_embeds(self, src_batch, src_lookup, src_token_to_id):
    total_words, src_cws, masks, num_batches = self.get_masks(src_batch)
    embeds = list()
    for cws in src_cws:
      cwids = [src_token_to_id[cw] for cw in cws] 
      embed_t = dy.lookup_batch(src_lookup, cwids)
      embeds.append(embed_t)
    return embeds

  def dual_att_losses(self, avg_embeds, batch, tgt_batch, tgt_lookup, tgt_token_to_id, W_tgt, b_tgt, W1_patt_src, W1_patt_tgt, w2_patt, tgt_dec_builder, W1_att_img, W1_att_src, w2_att):
    h_fs_matrix = dy.concatenate_cols(avg_embeds) #embed_size * sentence_len 
    c_t = dy.vecInput(self.embed_size)
    total_words, att_cws, masks, num_batches = self.get_masks(tgt_batch)
    
    img_vec = dy.inputTensor([x[0] for x in batch], batched = True)
    h_fs_matrixi = dy.transpose(img_vec) #image_size * image_points
    c_ti = dy.vecInput(self.image_size)

    losses = []
    start_tokens = ["<S>"] * num_batches
    start_ids = [tgt_token_to_id[st] for st in start_tokens] 
    start = dy.concatenate([dy.lookup_batch(tgt_lookup, start_ids), c_t, c_ti]) 
    dec_state = tgt_dec_builder.initial_state().add_input(start)
    w1 = W1_patt_src * h_fs_matrix
    w1i = W1_att_img * h_fs_matrixi
    avg_embeds = list()
    for i, (cws, nws, mask) in enumerate(zip(att_cws, att_cws[1:], masks)):
        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_patt_tgt, w2_patt)
        c_ti, a_ti = self.__attention_mlp(h_fs_matrixi, h_e, w1i, W1_att_src, w2_att)
        cwids = [tgt_token_to_id[cw] for cw in cws] 
        embed_t = dy.lookup_batch(tgt_lookup, cwids)

        x_t = dy.concatenate([embed_t, c_ti])
        dec_state = dec_state.add_input(x_t)

        y_star =  W_tgt*dec_state.output() + b_tgt #y_star is src_vocab_size * 1
        
        nwids = [tgt_token_to_id[nw] for nw in nws]
        loss = dy.pickneglogsoftmax_batch(y_star, nwids)
        mask_loss = dy.cmult(loss, mask) * prob
        losses.append(mask_loss)
    return losses, total_words

  def get_masks(self, tgt_batch):
    total_words = 0
    tgt_cws = []
    masks = []
    tgt_len = [len(sent) for sent in tgt_batch]
    max_len_tgt = max(tgt_len)
    total_words = sum(tgt_len)
    num_batches = len(tgt_batch)
    for c in xrange(max_len_tgt):
      row = []
      mask = []
      for r in xrange(len(tgt_batch)):
        if c < len(tgt_batch[r]):
          row.append(tgt_batch[r][c])
          mask.append(1)
        else:
          row.append("</S>")
          mask.append(0)
      tgt_cws.append(row)
      masks.append(dy.reshape(dy.inputVector(mask), (1,), batch_size = num_batches))
    return total_words, tgt_cws, masks, num_batches


  def encdec_losses(self, avg_embeds, tgt_batch, tgt_lookup, tgt_token_to_id, W_tgt, b_tgt, tgt_enc_builder, tgt_dec_builder):
# note issues with avg_embed having different lengths between different sentences in batch
    enc_state = tgt_enc_builder.initial_state()
    for embed in avg_embeds:
      enc_state = enc_state.add_input(embed)
   
    encoded = enc_state.output()
    
    losses = []
    
    dec_state = tgt_dec_builder.initial_state([dy.vecInput(self.hidden_size), encoded])
    total_words, tgt_cws, masks, num_batches = self.get_masks(tgt_batch)

    for i, (cws, nws, mask) in enumerate(zip(tgt_cws, tgt_cws[1:], masks)):
        cwids = [tgt_token_to_id[cw] for cw in cws] 
        embed_t = dy.lookup_batch(tgt_lookup, cwids)
        dec_state = dec_state.add_input(embed_t)
        
        nwids = [tgt_token_to_id[nw] for nw in nws]
        y_star =  W_tgt*dec_state.output() + b_tgt
        loss = dy.pickneglogsoftmax_batch(y_star, nwids)
        mask_loss = dy.cmult(loss, mask)
        losses.append(mask_loss)

    return losses, total_words

  def compute_minrisk(self, batch, src_batch, src_lookup, src_token_to_id, W_y, b_y):
    W1_att_img = dy.parameter(self.W1_att_img) # attention_size * image_size
    losses = []
    total_words = 0


    total_words, src_cws, masks, num_batches = self.get_masks(src_batch)

    img_vec = dy.inputTensor([x[0] for x in batch], batched = True)
    h_fs_matrix = dy.transpose(img_vec) #image_size * image_points 
    c_t = dy.vecInput(self.image_size)
    start_tokens = ["<S>"] * num_batches
    start_ids = [src_token_to_id[st] for st in start_tokens] 
    start = dy.concatenate([dy.lookup_batch(src_lookup, start_ids), c_t]) 
    dec_state = self.dec_builder.initial_state().add_input(start)
    w1 = W1_att_img * h_fs_matrix
    W1_att_src = dy.parameter(self.W1_att_src)
    w2_att = dy.parameter(self.w2_att)
    cwids = start_tokens
    position = 0
    prob = dy.inputVector([1] * num_batches, batched = True)

    sent_ended = [0] * num_batches
    while position < self.max_len:
        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_att_src, w2_att)
        #cwids = [src_token_to_id[cw] for cw in cws] 
        embed_t = dy.lookup_batch(src_lookup, cwids)


        x_t = dy.concatenate([embed_t, c_t])
        dec_state = dec_state.add_input(x_t)

        y_star =  W_y*dec_state.output() + b_y #y_star is src_vocab_size * 1

        p = dy.softmax(y_star) #p is src_vocab_size * 1
        p_val_batch = p.npvalue()
        #print(p_val_batch.shape)
        q = p_val_batch.shape
        embed_avg = None
        if len(q) == 1: # batch size of one messes up dimensionality
          amaxs = [self.sampler(p_val_batch)]
        else:
          amaxs = []
          for p_val in p_val_batch.T:
            amax = self.sampler(p_val)
            amaxs.append(amax)
        #embeds = dy.lookup_batch(src_lookup, amaxs) # embeds is embed_size * 1
        pick_p = dy.pick_batch(p, amaxs) # 1x1
        prob *= pick_p
        
        cw_ids = amaxs
        position += 1
        #nwids = [src_token_to_id[nw] for nw in nws]
        #loss = dy.pickneglogsoftmax_batch(y_star, nwids)
        #mask_loss = dy.cmult(loss, mask)
        #losses.append(mask_loss)
    
    #return losses, total_words, avg_embeds

  def compute_losses(self, batch, src_batch, src_lookup, src_token_to_id, W_y, b_y):
    W1_att_img = dy.parameter(self.W1_att_img) # attention_size * image_size
    losses = []
    total_words = 0


    total_words, src_cws, masks, num_batches = self.get_masks(src_batch)

    img_vec = dy.inputTensor([x[0] for x in batch], batched = True)
    h_fs_matrix = dy.transpose(img_vec) #image_size * image_points 
    c_t = dy.vecInput(self.image_size)
    start_tokens = ["<S>"] * num_batches
    start_ids = [src_token_to_id[st] for st in start_tokens] 
    start = dy.concatenate([dy.lookup_batch(src_lookup, start_ids), c_t]) 
    dec_state = self.dec_builder.initial_state().add_input(start)
    w1 = W1_att_img * h_fs_matrix
    avg_embeds = list()
    W1_att_src = dy.parameter(self.W1_att_src)
    w2_att = dy.parameter(self.w2_att)
    prob = dy.inputTensor([1] * num_batches, batched = True)
    
    for i, (cws, nws, mask) in enumerate(zip(src_cws, src_cws[1:], masks)):
        h_e = dec_state.output()
        c_t, a_t = self.__attention_mlp(h_fs_matrix, h_e, w1, W1_att_src, w2_att)
        cwids = [src_token_to_id[cw] for cw in cws] 
        embed_t = dy.lookup_batch(src_lookup, cwids)


        x_t = dy.concatenate([embed_t, c_t])
        dec_state = dec_state.add_input(x_t)

        y_star =  W_y*dec_state.output() + b_y #y_star is src_vocab_size * 1

        if self.multilangmode == ENCDECPIPELINE or self.multilangmode == ATTPIPELINE:
          p = dy.softmax(y_star) #p is src_vocab_size * 1
          p_val_batch = p.npvalue()
          #print(p_val_batch.shape)
          q = p_val_batch.shape
          embed_avg = None
          for i in xrange(self.pipeline_candidates):  
            if len(q) == 1: # batch size of one messes up dimensionality
              amaxs = [self.sampler(p_val_batch)]
            else:
              amaxs = []
              for p_val in p_val_batch.T:
                amax = self.sampler(p_val)
                amaxs.append(amax)
            embeds = dy.lookup_batch(src_lookup, amaxs) # embeds is embed_size * 1
            pick_p = dy.pick_batch(p, amaxs) # 1x1
            if embed_avg is None:
              embed_avg = embeds*pick_p
            else:
              embed_avg += embeds * pick_p
          avg_embeds.append(embed_avg)
        elif self.multilangmode == MRT or self.multilangmode == DUALATT:
          p = dy.softmax(y_star) #p is src_vocab_size * 1
          p_val_batch = p.npvalue()
          q = p_val_batch.shape
          if len(q) == 1: # batch size of one messes up dimensionality
            amaxs = [self.sampler(p_val_batch)]
          else:
            amaxs = []
            for p_val in p_val_batch.T:
              amax = self.sampler(p_val)
              amaxs.append(amax)
          embeds = dy.lookup_batch(src_lookup, amaxs) # embeds is embed_size * 1
          pick_p = dy.pick_batch(p, amaxs)
          prob *= pick_p
          avg_embeds.append(embeds)

          #candidate_ids = xrange(src_vocab_size)
          #amaxs = heapq.nlargest(self.pipeline_candidates, candidate_ids, lambda id: p_val[id])
          #embeds = dy.lookup_batch(src_lookup, amaxs) #
          #p_sum = dy.sum_elems(p) 
          #p_norm = p / p_sum
          #embed_avg = embeds * p_norm

          #avg_embeds.append(dy.transpose(src_lookup) * p)
        
        nwids = [src_token_to_id[nw] for nw in nws]
        loss = dy.pickneglogsoftmax_batch(y_star, nwids)
        mask_loss = dy.cmult(loss, mask)
        losses.append(mask_loss)
    return losses, total_words, avg_embeds, prob

  def step_batch(self, batch, cnum, cnum2):
    #print("Batch size %d" % len(batch))
    dy.renew_cg()
    W_y = dy.parameter(self.W_y)
    b_y = dy.parameter(self.b_y)
    src_batch = [["<S>"]+x[1][cnum]+["</S>"] for x in batch]
    tgt_batch = [["<S>"]+x[2][cnum2]+["</S>"] for x in batch]

    losses, total_words, avg_embeds, prob = self.compute_losses(batch, src_batch, self.src_lookup, self.src_token_to_id, W_y, b_y)
    sum1 = dy.sum_batches(dy.esum(losses))

    if self.multilang:
      W_tgt = dy.parameter(self.W_tgt)
      b_tgt = dy.parameter(self.b_tgt)
      if self.multilangmode == FORK:
        losses_tgt, total_words_tgt, avg_embeds_tgt = self.compute_losses(batch, tgt_batch, self.tgt_lookup, self.tgt_token_to_id, W_tgt, b_tgt)
      elif self.multilangmode == ENCDECPIPELINE:
        tgt_enc_builder = self.tgt_enc_builder
        tgt_dec_builder = self.tgt_dec_builder
        losses_tgt, total_words_tgt = self.encdec_losses(avg_embeds, tgt_batch, self.tgt_lookup, self.tgt_token_to_id, W_tgt, b_tgt, tgt_enc_builder, tgt_dec_builder)
      #losses = losses + losses_tgt
      elif self.multilangmode == ATTPIPELINE or self.multilangmode == MRT:
        W1_patt_src = dy.parameter(self.W1_patt_src)
        W1_patt_tgt = dy.parameter(self.W1_patt_tgt)
        w2_patt = dy.parameter(self.w2_patt)
        tgt_dec_builder = self.tgt_dec_builder

        losses_tgt, total_words_tgt = self.att_losses(avg_embeds, tgt_batch, self.tgt_lookup, self.tgt_token_to_id, W_tgt, b_tgt, W1_patt_src, W1_patt_tgt, w2_patt, tgt_dec_builder,prob)
      elif self.multilangmode == DUALATT:
        W1_patt_src = dy.parameter(self.W1_patt_src)
        W1_patt_tgt = dy.parameter(self.W1_patt_tgt)
        w2_patt = dy.parameter(self.w2_patt)
        tgt_dec_builder = self.tgt_dec_builder
        W1_att_img = dy.parameter(self.W1_patt_img)
        W1_att_src = dy.parameter(self.W1_patt_src)
        w2_att = dy.parameter(self.w2_patt)

        losses_tgt, total_words_tgt = self.dual_att_losses(avg_embeds, tgt_batch, self.tgt_lookup, self.tgt_token_to_id, W_tgt, b_tgt, W1_patt_src, W1_patt_tgt, w2_patt, tgt_dec_builder,W1_att_img, W1_att_src, w2_att)


      sum2 = dy.sum_batches(dy.esum(losses_tgt))
      total_words += total_words_tgt
      return sum1 + sum2, total_words

    return sum1, total_words#(dy.sum_batches(dy.esum(losses))), total_words

def dev_perplexity(dev_batches, encdec, num_captions):
    dev_loss = 0
    dev_words = 0
    for cnum in xrange(num_captions):
      random.shuffle(dev_batches[cnum])
      for tidx, batch in enumerate(dev_batches[cnum]):
        loss,words = encdec.step_batch(batch, cnum, cnum) 
        if loss is not None:
          lv = loss.value()
          dev_loss += lv
          dev_words += words
        if tidx >= 3:
          print("Dev cnum %d" % cnum)
          break

    return math.exp(dev_loss / dev_words)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_src', default='mmt_task2/en/train/train.')
  parser.add_argument('--train_tgt', default='mmt_task2/de/train/de_train.')
  parser.add_argument('--captions_src', default='captions_src.json')
  parser.add_argument('--captions_tgt', default='captions_tgt.json')
  parser.add_argument('--num_captions', default=5)
  parser.add_argument('--train_img', default='flickr30k_ResNets50_blck4_train.fp16.npy')
  parser.add_argument('--valid_src', default='mmt_task2/en/val/val.')
  parser.add_argument('--valid_tgt', default='mmt_task2/de/val/de_val.')
  parser.add_argument('--valid_img', default='flickr30k_ResNets50_blck4_val.fp16.npy')
  parser.add_argument('--batch_size',default=16)
  parser.add_argument('--min_freq',default=10)
  parser.add_argument('--model_file', default = "encdec1.mdl")
  parser.add_argument('--token_file', default = "tokens10.json")
  parser.add_argument('--vocab_freq', default = 2)
  parser.add_argument('--embed_size', default = 256)
  parser.add_argument('--hidden_size', default = 512)
  parser.add_argument('--attention_size', default = 256)
  parser.add_argument('--dropout', default = 0.02)
  parser.add_argument('--lstm', action='store_true')
  parser.add_argument('--image_size', default = 1024) #fixed
  parser.add_argument('--image_points', default = 196) #fixed
  parser.add_argument('--dynet-mem')
  parser.add_argument('--dynet-gpu', action='store_true')
  parser.add_argument('--eval', nargs='*')
  parser.add_argument('--output', default='out')
  parser.add_argument('--show_attention', action='store_true')
  parser.add_argument('--multilang', action='store_true')
  parser.add_argument('--pipeline_candidates', default = 5) # ignored, always 1
  
  parser.add_argument('--multilangmode', default=FORK)
    # fork: one decoder, separate word embeddings
    # encdecpipeline: output of src used to generate tgt (enc-dec)
  
  parser.add_argument('--beam_size', default = 1)
  parser.add_argument('--show_candidates', action='store_true') # if set, print the top [beam_size] candidates
  parser.add_argument('--sample_embeds', default = 'argmax')
  parser.add_argument('--simple_tokenize', action='store_true')
  parser.add_argument('--unk_penalty', default=1.0)
  args = parser.parse_args()

  tokenizer = mt_util.simple_tokenize if args.simple_tokenize else word_tokenize

  if os.path.isfile(args.captions_src):
    with open(args.captions_src) as cp:
      captions_train_src = json.load(cp)
  else:
    captions_train_src = get_captions(args.train_src, args.num_captions, tokenizer)
    json.dump(captions_train_src, open(args.captions_src,"w"))
  
  if os.path.isfile(args.captions_tgt):
    with open(args.captions_tgt) as cp:
      captions_train_tgt = json.load(cp)
  else:
    captions_train_tgt = get_captions(args.train_tgt, args.num_captions, tokenizer)
    json.dump(captions_train_tgt, open(args.captions_tgt,"w"))

  # save time when train imgs won't be used 
  if args.eval and len(args.eval) > 0:
    train_imgs = [0] * len(captions_train_src)
  else: 
    train_imgs = get_imgs(args.train_img)

  valid_imgs = get_imgs(args.valid_img)

  beam_size = int(args.beam_size)
  num_captions = int(args.num_captions)
 
  captions_valid_src = get_captions(args.valid_src, args.num_captions, tokenizer)
  captions_valid_tgt = get_captions(args.valid_tgt, args.num_captions, tokenizer)
  dev = zip(valid_imgs, list(captions_valid_src), list(captions_valid_tgt))
  dev_batches = []
  for cnum in xrange(args.num_captions):
    dev_batches.append(mt_util.make_batches(dev, int(args.batch_size), cnum,3))
  
  builder = lstm_builder if args.lstm else rnn_builder

  model = dy.Model()
  trainer = dy.AdamTrainer(model)
  encdec = Attention(model, train_imgs, captions_train_src, captions_train_tgt, args.model_file, args.token_file, args.vocab_freq, args.embed_size, args.hidden_size, args.image_size, args.image_points, args.attention_size, args.dropout, builder, args.multilang, args.multilangmode, args.pipeline_candidates, args.sample_embeds, args.unk_penalty)
  
  if args.eval and len(args.eval) > 0:
    for idx, eval_file in enumerate(args.eval):
      with open("attout/"+eval_file+"."+args.output + ".txt","w") as fa:
        with open("output/"+eval_file+"."+args.output + ".src","w") as f:
          with open("output/"+eval_file+"."+args.output + ".tgt","w") as ft:
            test_imgs = get_imgs(eval_file)
            for img in test_imgs:
              sent, att = encdec.make_caption(img, show_attention = args.show_attention, is_src = True, beam_size = beam_size, show_candidates = args.show_candidates)
              print(sent)
              if(args.show_attention):
                fa.write(json.dumps(att)+"\n")
              f.write(sent.encode("utf-8")+"\n")
              if args.multilang:
                sent, att = encdec.make_caption(img, show_attention = args.show_attention, is_src = False, beam_size = beam_size, show_candidates = args.show_candidates)
                print(sent)
                ft.write(sent.encode("utf-8")+"\n")

    return
  
  batches = []
  for cnum in xrange(args.num_captions):
    cbatches = mt_util.make_batches(encdec.training, int(args.batch_size), cnum, 3)
    print(len(cbatches))
    for cbatch in cbatches:
      batches.append((cbatch,cnum))

  print("Batches: %d" % len(batches))

  num_epochs = 100
  cnums = list(xrange(args.num_captions))
  min_perp = 100000
  for eidx, epoch in enumerate(range(num_epochs)):
    train_loss, train_words = 0, 0
    partial_loss, partial_words = 0, 0
    random.shuffle(batches)
    #random.shuffle(cnums)
    #for cnum in cnums:
    #  random.shuffle(batches[cnum])
    dev_perp = dev_perplexity(dev_batches, encdec, num_captions)
    print("Dev Perplexity: %f" % dev_perp)
    offset = random.randint(0, num_captions - 1)
    for tidx, (batch, cnum) in enumerate(batches):
      #for tidx, batch in enumerate(batches[cnum]):
        loss, words = encdec.step_batch(batch, cnum,(cnum+offset) % num_captions)

        if loss is not None:
          lv = loss.value()
          train_loss += lv
          partial_loss += lv
          train_words += words
          partial_words += words
          loss.backward()
          trainer.update()
        
        if tidx % 100 == 0:
          print(encdec.make_caption(valid_imgs[0], is_src = True)[0])
          print(encdec.make_caption(valid_imgs[1], is_src = True)[0])
          if args.multilang:
            print(encdec.make_caption(valid_imgs[0], is_src = False)[0])
            print(encdec.make_caption(valid_imgs[1], is_src = False)[0])
          print("Batch %d with loss %f" % (tidx, partial_loss / partial_words))
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
    else:
      print("start saving overfit")
      model.save("of."+args.model_file,encdec.params)
      print("done saving")
    trainer.update_epoch(1.0)
  

main()
