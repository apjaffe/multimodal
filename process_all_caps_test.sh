TGT_DIR=test_toks
REF_DIR=test_refs
SRC_DIR=test
mkdir -p $TGT_DIR/en/val
mkdir -p $TGT_DIR/en/train
mkdir -p $TGT_DIR/de/val
mkdir -p $TGT_DIR/de/train
mkdir -p $REF_DIR

for i in `seq 1 5`;
do
  python process_captions.py $SRC_DIR/en/test.${i} > $TGT_DIR/en/test.${i}
  python process_captions.py $SRC_DIR/de/de_test.${i} > $TGT_DIR/de/de_test.${i}
done

#cat $TGT_DIR/en/train/train.* > translate/train.en
#cat $TGT_DIR/de/train/de_train.* > translate/train.de
#cat $TGT_DIR/en/val/val.* > translate/valid.en
#cat $TGT_DIR/de/val/de_val.* > translate/valid.de

python join_refs.py $TGT_DIR/en/test.* > $REF_DIR/test.en
python join_refs.py $TGT_DIR/de/de_test.* > $REF_DIR/test.de
#python join_refs.py $TGT_DIR/en/val/val.* > $REF_DIR/valid.en
#python join_refs.py $TGT_DIR/de/val/de_val.* > $REF_DIR/valid.de
