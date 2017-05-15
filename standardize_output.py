import sys
import os
from mt_util import simple_tokenize
tgt_dir = sys.argv[1]
for fname in sys.argv[2:]:
  fn = os.path.basename(fname)
  with open(fname) as inf:
    print(os.path.join(tgt_dir, fn))
    with open(os.path.join(tgt_dir, fn),"w") as outf:
      for line in inf:
        outf.write(" ".join(simple_tokenize(line.decode("utf-8"))).encode("utf-8"))
