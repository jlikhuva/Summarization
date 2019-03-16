src_path_train="../data/gigaword/train.article.txt"
tgt_path_train="../data/gigaword/train.title.txt"
src_path_valid="../data/gigaword/valid.article.filter.txt"
tgt_path_valid="../data/gigaword/valid.title.filter.txt"
echo "creating validation set"
python preprocess_copy.py -src $src_path_valid\
                          -tgt $tgt_path_valid\
                          -output ../data/gigaword/bottom_up_valid\
                          -prune 400\
                          -num_examples 100000
echo "creating training set"
python preprocess_copy.py -src $src_path_train\
                          -tgt $tgt_path_train\
                          -output ../data/gigaword/bottom_up_train\
                          -prune 400\
                          -num_examples 100000

