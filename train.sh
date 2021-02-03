export CUDA_VISIBLE_DEVICES=0
python train_code_finder.py configs/autoencoder.yaml stylegan-256px-new.model  --images dataset/train.json  --val-images dataset/val.json  -s 1 --neural-rendering
