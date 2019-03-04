python preprocess.py -train_src data/gigaword/train.article.txt \
                     -train_tgt data/gigaword/train.title.txt \
                     -valid_src data/gigaword/valid.article.filter.txt \
                     -valid_tgt data/gigaword/valid.title.filter.txt \
                     -save_data data/gigaword/GIGA \
                     -src_seq_length 10000 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 100000
