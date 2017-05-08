import sys
import argparse

#python render_captions.py --captions output/flickr30k_ResNets50_blck4_val.fp16.npy.att1d_beam10.src

#python render_captions.py --captions output/flickr30k_ResNets50_blck4_val.fp16.npy.att1d.src --captions2 output/flickr30k_ResNets50_blck4_val.fp16.npy.att2.src

#python render_captions.py --captions mmt_task2/de/val/de_val.1 mmt_task2/de/val/de_val.2 mmt_task2/de/val/de_val.3 mmt_task2/de/val/de_val.4 mmt_task2/de/val/de_val.5 --out_html captions_de.html

#python render_captions.py --captions mmt_task2/en/val/val.1 mmt_task2/en/val/val.2 mmt_task2/en/val/val.3 mmt_task2/en/val/val.4 mmt_task2/en/val/val.5 --out_html captions_en.html


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--val_images', default='splits/val_images.txt')
  parser.add_argument('--captions', nargs='+')
  #parser.add_argument('--captions2')
  parser.add_argument('--out_html', default="captions.html")
  parser.add_argument('--base', default="flickr30k-images/flickr30k-images")
  args = parser.parse_args()
  print(args.captions)
  with open(args.val_images) as vf:
      with open(args.out_html,"w") as hf:
        caplines = list()
        for cap in args.captions:
          with open(cap) as cf:
            caplines.append(list(cf))
        
        lines1 = list(vf)
        for i in xrange(min(len(lines1), len(caplines[0]))):
          hf.write("<img src='%s/%s' width=200/><br>" % (args.base,lines1[i]))
          for capset in caplines:
            if i < len(capset):
              hf.write("%s<br>" % capset[i])
          hf.write("<br>\n")
    
main()