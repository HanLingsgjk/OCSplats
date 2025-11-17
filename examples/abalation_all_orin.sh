#!/bin/bash
NAME=sls
DATA_DIR=/home/lh/all_datasets/RobustScene/on-the-go
RES_DIR=/home/lh/SpotLessSplats/examples/results_orin3DGS


# crab patio_high mountain fountain and-bot yoda corner spot train_station tree
for SCENE in crab patio_high and-bot corner spot
do
	CUDA_HOME=/usr/local/cuda-11.8/;CUDA_VISIBLE_DEVICES=1 python spotless_orinorin.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--data_factor 8 \
		--result_dir="${RES_DIR}/${SCENE}_${NAME}" \
		--loss_type l1 --no-semantics --no-cluster --train_keyword "clutter" --test_keyword "extra"
done
#patio
for SCENE in patio
do
	CUDA_HOME=/usr/local/cuda-11.8/;CUDA_VISIBLE_DEVICES=1 python spotless_orinorin.py \
		--data_dir="${DATA_DIR}/${SCENE}" \
		--data_factor 4 \
		--result_dir="${RES_DIR}/${SCENE}_${NAME}" \
		--loss_type l1 --no-semantics --no-cluster --train_keyword "clutter" --test_keyword "extra"
done


