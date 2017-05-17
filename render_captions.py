import sys
import argparse
import random

#python render_captions.py --captions output/flickr30k_ResNets50_blck4_val.fp16.npy.att1d_beam10.src

#python render_captions.py --captions output/flickr30k_ResNets50_blck4_val.fp16.npy.att1d.src --captions2 output/flickr30k_ResNets50_blck4_val.fp16.npy.att2.src

#python render_captions.py --captions mmt_task2/de/val/de_val.1 mmt_task2/de/val/de_val.2 mmt_task2/de/val/de_val.3 mmt_task2/de/val/de_val.4 mmt_task2/de/val/de_val.5 --out_html captions_de.html

#python render_captions.py --captions mmt_task2/en/val/val.1 mmt_task2/en/val/val.2 mmt_task2/en/val/val.3 mmt_task2/en/val/val.4 mmt_task2/en/val/val.5 --out_html captions_en.html



#python render_captions.py --captions output/task2_ResNet50_res4fx_test2017.fp16.npy.encdecpipe1nof.de_beam10.src output/task2_ResNet50_res4fx_test2017.fp16.npy.encdecpipe1nof.de_beam20.src --base images_flickr.task2/task2 --val_images image_ids_flickr.task2

#python render_captions.py --captions output/task2_ResNet50_res4fx_test2017.fp16.npy.encdecpipe1nof.de_beam10.src output/task2_ResNet50_res4fx_test2017.fp16.npy.encdecpipe1nof.de_beam20.src --val_images splits/test_images.txt

#python render_captions.py --captions output/task2_ResNet50_res4fx_test2017.fp16.npy.encdecpipe1nof.de_beam10.src  --base images_flickr.task2/task2 --val_images image_ids_flickr.task2

#python render_captions.py --captions std_full/flickr30k_ResNets50_blck4_val.fp16.npy.multi1d.tgt std_full/flickr30k_ResNets50_blck4_val.fp16.npy.encdecpipe1b.de_beam10.src.src

#python render_captions.py --captions std_full/flickr30k_ResNets50_blck4_val.fp16.npy.multi1d.tgt std_full/flickr30k_ResNets50_blck4_val.fp16.npy.attpipe2gp_beam10.de.src

#python render_captions.py --captions std_full/flickr30k_ResNets50_blck4_val.fp16.npy.att1.de_beam10.src std_full/flickr30k_ResNets50_blck4_val.fp16.npy.att4.de_beam10.src std_full/flickr30k_ResNets50_blck4_val.fp16.npy.fork1_beam10.tgt std_full/flickr30k_ResNets50_blck4_val.fp16.npy.encdecpipe1nof.de_beam10.src std_full/flickr30k_ResNets50_blck4_val.fp16.npy.attpipe2gp_beam10.de.src std_full/flickr30k_ResNets50_blck4_val.fp16.npy.attpipeavg1b.de_beam10.src std_full/flickr30k_ResNets50_blck4_val.fp16.npy.dualatt1c_beam10.tgt --out_html captions_compare_val.html 

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--val_images', default='splits/val_images.txt')
  parser.add_argument('--captions', nargs='+')
  #parser.add_argument('--captions2')
  parser.add_argument('--out_html', default="captions.html")
  parser.add_argument('--out_text', default="captions.txt")
  parser.add_argument('--base', default="flickr30k-images/flickr30k-images")
  args = parser.parse_args()
  print(args.captions)
  with open(args.out_text,"w") as ot:
    with open(args.val_images) as vf:
        with open(args.out_html,"w") as hf:
          caplines = list()
          for cap in args.captions:
            with open(cap) as cf:
              caplines.append(list(cf))
          
          lines1 = list(vf)
          #for i in xrange(min(len(lines1), len(caplines[0]))):
          caps = random.sample(list(xrange(len(lines1))),4)
          for i in caps:
            hf.write("<img src='%s/%s' width=200/><br>" % (args.base,lines1[i]))
            for capset in caplines:
              if i < len(capset):
                hf.write("%s<br>" % capset[i])
                ot.write("NeuralEncoderDecoder\t%s\t%s\t2\tC\n" % (lines1[i].strip(), capset[i].strip()))
            hf.write("<br>\n")
    

main()