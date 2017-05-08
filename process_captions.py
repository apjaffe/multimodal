from nltk.tokenize import word_tokenize
import sys
import mt_util

with open(sys.argv[1]) as f:
  for line in f:
    decoded = line.strip().decode("utf-8")
    lowered = decoded.lower()
    tokenized = mt_util.simple_tokenize(lowered)
    joined = " ".join(tokenized)
    print(joined.encode("utf-8"))
