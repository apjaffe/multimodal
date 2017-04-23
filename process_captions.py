from nltk.tokenize import word_tokenize
import sys

with open(sys.argv[1]) as f:
  for line in f:
    decoded = line.strip().decode("utf-8")
    lowered = decoded.lower()
    tokenized = word_tokenize(lowered)
    joined = " ".join(tokenized)
    print(joined.encode("utf-8"))
