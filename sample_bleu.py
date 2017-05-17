import argparse
import random
import subprocess
from bleu_util import get_bleu, read_file

def read_file(fname):
    with open(fname) as f:
      lines = f.read().split("\n")
      if len(lines[-1]) <= 1: # blank last line
        return lines[:-1]
      else:
        return lines


def bootstrap(aref, ahyp1, ahyp2, samples):
  refs = list()
  for ref in aref:
    refs.append(read_file(ref))
  hyp1 = read_file(ahyp1)
  hyp2 = read_file(ahyp2)
  for ref in refs:
    if len(ref) != len(hyp1) or len(ref) != len(hyp2):
      print("Length mismatch %d vs %d vs %d" % (len(ref), len(hyp1), len(hyp2)))
      return -1.0

  numl = len(ref)
  bwins = 0
  sum1 = 0
  sum2 = 0
  sumdiff = 0
  for i in xrange(samples):
    r = []
    h1 = []
    h2 = []

    trs = list()
    for rf in refs:
      trs.append(list())

    for j in xrange(numl):
      if random.randint(0,1) == 1:
        for k in xrange(len(trs)):
          trs[k].append(refs[k][j])
        h1.append(hyp1[j])
        h2.append(hyp2[j])

    refnames = list()
    for k in xrange(len(trs)):
      refnames.append("/tmp/ref%d" % k)
      with open("/tmp/ref%d" % k,"w") as rff:
        rff.write("\n".join(trs[k]))

    raw1 = subprocess.Popen(["perl","multi-bleu.perl"]+refnames, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate("\n".join(h1))[0]
    raw2 = subprocess.Popen(["perl","multi-bleu.perl"]+refnames, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate("\n".join(h2))[0]
    bleu1 = get_bleu(raw1)
    bleu2 = get_bleu(raw2)
    if bleu1 > bleu2:
      bwins += 1
    sum1 += bleu1
    sum2 += bleu2
    sumdiff += (bleu1-bleu2)
    #print(i)

  return float(bwins)/samples
  #print("%f" % (float(bwins)/samples))
  #print("The first hypothesis won %d/%d times (%f) with avg diff of %f. %f vs %f" % (bwins, samples, float(bwins) / samples, float(sumdiff) / samples, sum1/samples, sum2/samples))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ref', nargs='+')
  parser.add_argument('--hyp1')
  parser.add_argument('--hyp2')
  parser.add_argument('--samples', default = 100)

  args = parser.parse_args()


  pval = bootstrap(args.ref, args.hyp1, args.hyp2, int(args.samples))
  print(pval)

if __name__ =='__main__': main()

