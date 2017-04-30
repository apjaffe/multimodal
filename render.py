import json
from PIL import Image, ImageDraw
from PIL.ImageDraw import Draw
import sys

linenum = int(sys.argv[1])
attnum = int(sys.argv[2])

attfile="attout/flickr30k_ResNets50_blck4_val.fp16.npy.att1c.txt"

ratio = 50
num=14
im = Image.new("RGB", (num*ratio, num*ratio), "white")
dr = ImageDraw.Draw(im)
with open(attfile) as f:
  lines = list(f)
  line0 = json.loads(lines[linenum])
  att0 = line0[attnum]
  dr.rectangle([0,0,num*ratio,num*ratio],fill=(255,255,255))
  for r in xrange(num):
    for c in xrange(num):
      atq = att0[r*num+c][0]
      atqi = 255-(int(atq*255)*5)
      dr.rectangle([(r*ratio, c*ratio),((r+1)*ratio, (c+1)*ratio)],fill=(atqi,atqi,atqi))

im.save("atttest.png", "PNG")
