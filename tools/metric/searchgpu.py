import time
import os
import sys
import numpy as np
import faiss
import pickle

def loadFeaFromPickle(feafile):
    feadic = pickle.load(open(feafile,'rb'))
    fea_items = feadic.items()
    names = [fea[0] for fea in fea_items]
    feas = [fea[1].reshape(-1) for fea in fea_items]
    feas = np.array(feas)
    return feas, names

def search_gpu(workroot, output, topk=100):
    query_path = os.path.join(workroot, "queryfea.pickle")
    refer_path = os.path.join(workroot, 'DBfea.pickle')
    queryfeas, queryconts = loadFeaFromPickle(query_path)
    referfeas, referconts = loadFeaFromPickle(refer_path)
    assert(queryfeas.shape[1] == referfeas.shape[1])
    dim = int(queryfeas.shape[1])
    print("=> query feature shape: {}".format(queryfeas.shape), file=sys.stderr)
    print("=> refer feature shape: {}".format(referfeas.shape), file=sys.stderr)
    
    start = time.time()
    ngpus = faiss.get_num_gpus()
    print("=> search use gpu number of GPUs: {}".format(ngpus), file=sys.stderr)
    cpu_index = faiss.IndexFlat(dim, faiss.METRIC_INNER_PRODUCT)   # build the index
    gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
            )
    gpu_index.add(referfeas)                  # add vectors to the index print(index.ntotal)
    print("=> building gpu index success, \
           total index number: {}".format(gpu_index), file=sys.stderr)
    distance, ind = gpu_index.search(queryfeas, int(topk))
    assert(distance.shape == ind.shape)
    end = time.time()
    print("=> searching total use time {}".format(end - start), file=sys.stderr)
    outdic = {}
    for key_id in range(queryfeas.shape[0]):
        querycont = queryconts[key_id]
        searchresult = [(referconts[ind[key_id][i]], distance[key_id][i]) \
                         for i in range(len(distance[key_id]))]
        outdic[querycont] = searchresult
    print("=> convert search gpu result to output format success")
    pickle.dump(outdic, open(output,"wb"), protocol=2)

def main():

    workroot = 'eval_outputs/'
    output = 'eval_outputs/searchresult.pickle'
    topk = 100 
    
    search_gpu(workroot, output, topk=topk)
    

if  __name__ == '__main__':
    main()

