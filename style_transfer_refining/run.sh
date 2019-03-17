#!/bin/bash
if [ "$1" = "train" ]; then
	python train_style_transfer.py
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=../data/gigaword/train.article.txt\
					--train-tgt=../data/gigaword/valid.article.filter.txt \
					vocab.json
else
	echo "Invalid Option Selected"
fi
