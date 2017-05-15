def get_bleu(raw):
  try:
    return float(raw.split(",")[0].split(" = ")[1])
  except:
    print(raw)
    return -1

def get_meteor(raw):
  try:
    return float(raw.split("Final score:")[1])
  except:
    print(raw)
    return -1

def read_file(fname):
    with open(fname) as f:
      lines = f.read().split("\n")
      if len(lines[-1]) <= 1: # blank last line
        return lines[:-1]
      else:
        return lines
