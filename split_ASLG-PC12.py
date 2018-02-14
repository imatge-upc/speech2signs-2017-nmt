
import numpy as np
import argparse


def splitBDD(eng_path, asl_path, split=0.3):

    eng_data = open(eng_path, 'r').readlines()
    asl_data = open(asl_path, 'r').readlines()

    eng_test = []
    eng_train = []
    asl_test = []
    asl_train = []
    

    for i in range(len(eng_data)):
        
        randValue = np.random.uniform (0, 1)

        if randValue < split:

            eng_test.append(eng_data[i])
            asl_test.append(asl_data[i])

        else:

            eng_train.append(eng_data[i])
            asl_train.append(asl_data[i])

    print("Train")
    print(len(eng_train))
    print(len(asl_train))
    print("Test")
    print(len(eng_test))
    print(len(asl_test))

    with open(eng_path.replace("sample-corpus-asl-en", "ENG-ASL_Train_"+split.__str__()), 'w') as f:
        for line in eng_train:
            f.write(line)

    with open(asl_path.replace("sample-corpus-asl-en", "ENG-ASL_Train_"+split.__str__()), 'w') as f:
        for line in asl_train:
            f.write(line)

    with open(eng_path.replace("sample-corpus-asl-en", "ENG-ASL_Test_"+split.__str__()), 'w') as f:
        for line in eng_test:
            f.write(line)

    with open(asl_path.replace("sample-corpus-asl-en", "ENG-ASL_Test_"+split.__str__()), 'w') as f:
        for line in asl_test:
            f.write(line)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split a database in train and test')

    parser.add_argument('eng_corpora', help='english file')
    parser.add_argument('asl_corpora', help='asl file')
    parser.add_argument('--split_percentage_test', type=float, default=0.3, help='percentage of the database to split for the test dataset (default 30%)')
    
    args = parser.parse_args()
    
    splitBDD(args.eng_corpora, args.asl_corpora, args.split_percentage_test)
                            
