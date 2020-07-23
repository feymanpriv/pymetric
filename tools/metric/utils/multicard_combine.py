import os
import pickle

OUTFILE = 'DBfea.pickle'
OUTPUT_DIR = 'eval_outputs/'
COMBINE_DIR='eval_outputs/combine_results/'
def main():
    all_dict = {}
    files = os.listdir(COMBINE_DIR)
    for file in files:
        tmppath = os.path.join(COMBINE_DIR, file)
        print(tmppath)
        with open(tmppath,'rb') as fin:
            tmpres = pickle.load(fin)
            all_dict.update(tmpres)
    print(len(all_dict.keys()))
    with open(OUTPUT_DIR+OUTFILE,'wb') as fout:
        pickle.dump(all_dict, fout, protocol=2)    
            

if __name__ == '__main__':
    main()
