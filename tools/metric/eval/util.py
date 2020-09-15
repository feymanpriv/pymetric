# -*- coding: utf-8 -*-
"""utility functions"""
import sys, os
import struct
import numpy as np
import pickle


def readkv(f):
    """read kv from file handle"""
    keylendata = f.read(4)
    if len(keylendata) != 4:
        print("wrongkey", file=sys.stderr)
        return None
    keylen = struct.unpack('I', keylendata)[0]
    if keylen > 5000:
        raise Exception('wrong key len %s' % (keylen))
    key = f.read(keylen)
    valuelen = struct.unpack('I', f.read(4))[0]
    if valuelen > 1000000000:
        raise Exception('wrong value len %s' % (valuelen))
    value = f.read(valuelen)
    return key, value


def writekv(f, k, v, flush=True):
    """writekv to file handle"""
    f.write(struct.pack('I', len(k)))
    f.write(k)
    f.write(struct.pack('I', len(v)))
    f.write(v)
    if flush:
        f.flush()


def loopkv(fin):
    """iter the kv file """
    while True:
        r = readkv(fin)
        if r is None:
            break
        key, value = r
        yield key, value


def getasciikey(key):
    #return key
    """convert 8bytes key to 'a,b' string contsign"""
    if len(key) == 8:
        key = struct.unpack('II', key)
        key = '%s,%s' % (key[0], key[1])
    return key


def l2_norm(fea):
    """l2 norm feature get cos dis"""
    if fea is None:
        return fea
    if len(fea.shape) > 1:
        fea = fea / np.linalg.norm(fea, axis=1)[:, None]
    else:
        fea = fea / np.linalg.norm(fea)
    return fea


def norm(x):
    n = np.sqrt(np.sum(x ** 2,1)).reshape(-1,1)
    return x / (n + 0.000001)


def walkfile(spath):
    """get files in input spath """
    files = os.listdir(spath)
    for file in files:
        tmppath = os.path.join(spath, file)
        if not os.path.isdir(tmppath):
            yield tmppath
        else:
            for lowfile in walkfile(tmppath):
                yield lowfile


def conmbineFeaPickle(COMBINE_DIR, OUTFILE):
    all_dict = {}
    files = os.listdir(COMBINE_DIR)
    for file in files:
        tmppath = os.path.join(COMBINE_DIR, file)
        print(tmppath)
        with open(tmppath,'rb') as fin:
            tmpres = pickle.load(fin)
            all_dict.update(tmpres)
    print(len(all_dict.keys()))
    with open(OUTFILE,'wb') as fout:
        pickle.dump(all_dict, fout, protocol=2)    
   
def visualize(onebookpath):
    fullpath = os.path.abspath(onebookpath)
    searchfile = os.path.join(onebookpath, "searchresults.pickle")
    outhtml = os.path.join(onebookpath, "searchresults.html")
    with open(searchfile, "rb") as f:
        resdic = pickle.load(f)
    platform = '/home/yangmin09'
    suffix = 'http://10.88.149.44:8903'
    #bookpath = fullpath.replace(platform, suffix)
    bookpath = 'http://yq01-sys-hic-k8s-v100-box-a225-0492.yq01.baidu.com:8902/weikai/data/huawei/test_data_A/'
    print(bookpath)

    queryurl = ""
    referurl = ""

    html_file = open(outhtml, "w")
    html_file.write('<html><meta charset=\"utf-8\"><body>\n')
    html_file.write('<p>\n')
    html_file.write('<table border="1">\n')

    for i, (k,v) in enumerate(sorted(resdic.items())):
        html_file.write('<td><img src="%s" width="150" height="150" /></td>' % (bookpath + "query/" + k))
        html_file.write("<td> %s %s</td>\n" % (i, k))

        recall = v[:10] # top10
        #dis = v['distance']

        if len(recall) == 0:
            html_file.write('</tr>\n')
            continue
        for i in range(len(recall)):
            referhtml = bookpath + 'gallery/' + recall[i][0]
            html_file.write('<td><img src="%s" width="150" height="150" /><br/><color="green">score: %s<br/> </td>' % (referhtml, str(recall[i][1])))

        html_file.write('</tr>\n')

    html_file.write("</table>\n</p>\n</body>\n</html>")
    print(onebookpath, "draw html finished ")

def recall_topk(fea, lab, k = 1): 
    print(fea.shape)
    print(lab.shape)
    fea = np.array(fea)
    fea = fea.reshape(fea.shape[0], -1) 
    n = np.sqrt(np.sum(fea**2, 1)).reshape(-1, 1)
    fea = fea/n
    a = np.sum(fea ** 2, 1).reshape(-1, 1)
    b = a.T 
    ab = np.dot(fea, fea.T)
    d = a + b - 2*ab
    d = d + np.eye(len(fea)) * 1e8 
    sorted_index = np.argsort(d, 1)
    res = 0 
    for i in range(len(fea)):
        pred = lab[sorted_index[i][0]]
        if lab[i] == pred:
            res += 1.0 
    res = res/len(fea)
    return res 


def eval(searchfile, gtfile):
    resdic = dict()
    with open(searchfile, "rb") as f:
        resdic = pickle.load(f)
    gtdict = dict()
    with open(gtfile, "r") as f:
        for line in f:
            line = line.strip().split()
            filename = os.path.basename(line[0])
            #print(filename)
            label = int(line[1])
            gtdict[filename] = label
    file_list = resdic.keys()
    ff = []
    ll = []
    for filename in file_list:
        f = resdic[filename]
        l = gtdict[filename]
        ff.append(f)
        ll.append(l)

    ff = np.vstack(ff)
    ll = np.vstack(ll)
    recall = recall_topk(ff, ll, k=1)
    print("recall ~", recall)

if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('tools.py command', file=sys.stderr)
