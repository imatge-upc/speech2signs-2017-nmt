srun -c 1 --mem 8G --gres=gpu:1,gmem:12G python translate.py -model weights/aslg-pc12.trained.chkpt -vocab data/ASLG-PC12/aslg-pc12.atok.low.pt -src data/ASLG-PC12/ENG-ASL_test.en.atok
