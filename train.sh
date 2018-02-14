srun -c 1 --mem 8G --gres=gpu:1,gmem:12G python train.py -data data/ASLG-PC12/aslg-pc12.atok.low.pt -save_model weights/aslg-pc12.trained -save_mode best -proj_share_weight
