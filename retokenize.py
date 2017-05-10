import sys
from mt_util import simple_tokenize
for line in sys.stdin:
  print(" ".join(simple_tokenize(line.decode("utf-8").strip())).encode("utf-8"))
