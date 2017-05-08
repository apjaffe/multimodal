import sys
fnames = sys.argv[1:]
lines = [list(open(f)) for f in fnames]
for i in xrange(len(lines[0])):
  for j in xrange(len(lines)):
    print(lines[j][i].strip())
