import json
from PIL import Image, ImageDraw
from PIL.ImageDraw import Draw
import sys

linenum = int(sys.argv[1])
attnum = int(sys.argv[2])

img_src = "flickr30k/"+list(open("splits/val_images.txt"))[linenum].strip()

attfile="attout/flickr30k_ResNets50_blck4_val.fp16.npy.att1d.txt"

ratioX = 500/14.0
ratioY = 375/14.0
num=14
#im = Image.new("RGB", (num*ratio, num*ratio), "white")
im = Image.open(img_src)
dr = ImageDraw.Draw(im,'RGBA')
with open(attfile) as f:
  lines = list(f)
  line0 = json.loads(lines[linenum])
  att0 = line0[attnum]
  #dr.rectangle([0,0,num*ratio,num*ratio],fill=(255,255,255))
  for r in xrange(num):
    for c in xrange(num):
      atq = att0[c*num+r][0]
      atqi = 255-(int(atq*255)*5)
      dr.rectangle([(r*ratioX, c*ratioY),((r+1)*ratioX, (c+1)*ratioY)],fill=(0,0,0,atqi))

im.save("atttest.png", "PNG")
