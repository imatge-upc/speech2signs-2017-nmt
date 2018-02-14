for l in en asl; do for f in data/ASLG-PC12/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en asl; do for f in data/ASLG-PC12/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
srun -c 1 --mem 8G --gres=gpu:1,gmem:12G python preprocess.py -train_src data/ASLG-PC12/ENG-ASL_Train.en.atok -train_tgt data/ASLG-PC12/ENG-ASL_Train.asl.atok -valid_src data/ASLG-PC12/ENG-ASL_Dev.en.atok -valid_tgt data/ASLG-PC12/ENG-ASL_Dev.asl.atok -save_data data/ASLG-PC12/aslg-pc12.atok.low.pt
