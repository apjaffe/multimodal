for i in `seq 1 5`;
do
  python process_captions.py mmt_task2/en/train/train.${i} > tokenized/en/train/train.${i}
  python process_captions.py mmt_task2/de/train/de_train.${i} > tokenized/de/train/de_train.${i}
  python process_captions.py mmt_task2/en/val/val.${i} > tokenized/en/val/val.${i}
  python process_captions.py mmt_task2/de/val/de_val.${i} > tokenized/de/val/de_val.${i}
done
