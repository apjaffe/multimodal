from nltk.translate import bleu_score
import sys

def calc_bleu_score(ref, hyp):
  return bleu_score.sentence_bleu([ref], hyp, smoothing_function = bleu_score.SmoothingFunction().method1)

def pick_best(choices):
  besti = None
  bestsum = 0
  for i in xrange(len(choices)):
    sums = 0
    for j in xrange(len(choices)):
      if j != i:
        sums += calc_bleu_score(choices[j],choices[i])

    if sums > bestsum or besti is None:
      besti = i
      bestsum = sums
  return choices[besti]


def main():
  ins = sys.argv[1:]
  infs = [open(f) for f in ins]
  linefs = [list(inf) for inf in infs]
  closed = [f.close() for f in infs]
  for j in xrange(len(linefs[0])):
    choices = list()
    for i in xrange(len(linefs)):
      choices.append(linefs[i][j])
    best_choice = pick_best(choices)
    print(best_choice.strip())


main()
