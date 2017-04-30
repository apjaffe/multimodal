import sys
import argparse

#python render_captions.py --captions output/flickr30k_ResNets50_blck4_val.fp16.npy.att1d_beam10.src

#python render_captions.py --captions output/flickr30k_ResNets50_blck4_val.fp16.npy.att1d.src --captions2 output/flickr30k_ResNets50_blck4_val.fp16.npy.att2.src

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--val_images', default='splits/val_images.txt')
  parser.add_argument('--captions')
  parser.add_argument('--captions2')
  parser.add_argument('--out_html', default="captions.html")
  parser.add_argument('--base', default="flickr30k-images/flickr30k-images")
  args = parser.parse_args()
  
  with open(args.val_images) as vf:
    with open(args.captions) as cf:
      with open(args.out_html,"w") as hf:
        if args.captions2:
          with open(args.captions2) as cf2:
            lines3 = list(cf2)
        else:
          lines3=[]
        lines1 = list(vf)
        lines2 = list(cf)
        for i in xrange(min(len(lines1), len(lines2))):
          hf.write("<img src='%s/%s' width=200/><br>" % (args.base,lines1[i]))
          hf.write("%s<br>" % lines2[i])
          if i < len(lines3):
            hf.write("%s<br>" % lines3[i])
          hf.write("<br>\n")
    
main()