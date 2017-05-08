mkdir -p tokenized/en/val
mkdir -p tokenized/en/train
mkdir -p tokenized/de/val
mkdir -p tokenized/de/train
mkdir -p translate

for i in `seq 1 5`;
do
  python process_captions.py mmt_task2/en/train/train.${i} > tokenized/en/train/train.${i}
  python process_captions.py mmt_task2/de/train/de_train.${i} > tokenized/de/train/de_train.${i}
  python process_captions.py mmt_task2/en/val/val.${i} > tokenized/en/val/val.${i}
  python process_captions.py mmt_task2/de/val/de_val.${i} > tokenized/de/val/de_val.${i}
done

cat tokenized/en/train/train.* > translate/train.en
cat tokenized/de/train/de_train.* > translate/train.de
cat tokenized/en/val/val.* > translate/valid.en
cat tokenized/de/val/de_val.* > translate/valid.de

python join_refs.py tokenized/en/train/train.* > refs/train.en
python join_refs.py tokenized/de/train/de_train.* > refs/train.de
python join_refs.py tokenized/en/val/val.* > refs/valid.en
python join_refs.py tokenized/de/val/de_val.* > refs/valid.de
