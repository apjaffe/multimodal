import sys
import argparse
import subprocess
from bleu_util import get_bleu, read_file, get_meteor
from sample_bleu import bootstrap

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--refs',nargs='+')
  parser.add_argument('--hyps',nargs='+')
  parser.add_argument('--baseline',default=None)
  parser.add_argument('--samples',default=100)
  parser.add_argument('--meteor_refs',default=None)
  args = parser.parse_args()

  DEVNULL = open("/dev/null","w")
  
  for hyp in args.hyps:
    hyplines = read_file(hyp)
    raw1 = subprocess.Popen(["perl","multi-bleu.perl"]+args.refs, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=DEVNULL).communicate("\n".join(hyplines))[0]
    bleu1 = get_bleu(raw1)
    pval = -1.0
    if args.baseline is not None and bleu1 > 0:
      pval = bootstrap(args.refs, args.baseline, hyp, int(args.samples))
    meteor = -1.0
    if args.meteor_refs is not None and bleu1 > 0:
      raw2 = subprocess.check_output(["java","-Xmx2G","-jar", "/mnt/home/ubuntu/meteor-1.5/meteor-1.5.jar",hyp,args.meteor_refs,"-l", "de", "-norm", "-r", "5"],stderr=DEVNULL)
      meteor = get_meteor(raw2)
    print("%s\t%f\t%f\t%f" % (hyp, bleu1,pval,meteor))
    
  

main()

