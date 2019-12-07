#!/usr/bin/env bash
out='/out'
data='/Users/cuent/Downloads/processed_new/mv/out/cv1/train'
num_topics=50
model='/model'
max_iter_train=100
max_iter_pred=500
save_every=100

python mixehr/main.py train -every $save_every -it $max_iter_train $num_topics $data $out
#python mixehr/main.py predict --model $model -it $max_iter $num_topics $data $out