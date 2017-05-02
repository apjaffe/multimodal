import subprocess
import argparse

DEVNULL = open("/dev/null","w")

# perform sequence-level interpolation from http://aclweb.org/anthology/D/D16/D16-1139.pdf

def read_file(fname):
    with open(fname) as f:
      lines = f.read().split("\n")
      if len(lines[-1]) <= 1: # blank last line
        return lines[:-1]
      else:
        return lines


def get_bleu(raw):
  return float(raw.split(",")[0].split(" = ")[1])

def compute_bleu(sent, refs):
  for i, ref in enumerate(refs):
    with open("/tmp/ref%d" % i,"w") as rff:
      rff.write(ref)

  ref_files = ["/tmp/ref%d" % i for i in xrange(len(refs))]
  bleu = 0.0
  try:
    raw1 = subprocess.Popen(["perl","multi-bleu.perl"] + ref_files, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr = DEVNULL ).communicate(sent)[0]
    bleu = get_bleu(raw1)
  except Exception, e:
    print(e)

  if bleu == 0.0: 
    # fallback to Jaccard Similarity if bleu score can't be computed
    st1 = set(sent.split(" "))
    st2 = set()
    for rf in refs:
      st2 |= set(rf.split(" "))
    return float(len(st1 & st2)) / len(st1 | st2) / 100.0
  else:
    return bleu

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--ref_base')
  parser.add_argument('--num_refs',default = 5)
  parser.add_argument('--hyps')
  parser.add_argument('--output')
  args = parser.parse_args()

  num_refs = int(args.num_refs)
  refs = list()
  for i in xrange(num_refs):
    refs.append(read_file(args.ref_base + str(i+1)))

  hyps = read_file(args.hyps)
  trefs = list()
  for i in xrange(len(hyps)):
    trefs.append(list())
    for j in xrange(num_refs):
      trefs[i].append(refs[j][i])


  with open(args.output,"w") as out:
    for i in xrange(len(hyps)):
      cands = hyps[i].split("|")[:-1]
      best_bleu = 0
      best_cand = None
      for cand in cands:
        cnd = cand.strip()
        bleu = compute_bleu(cnd, trefs[i])
        if bleu > best_bleu or best_cand is None:
          best_bleu = bleu
          best_cand = cnd
      print(best_cand, best_bleu)
      out.write(best_cand + "\n")

main() 
