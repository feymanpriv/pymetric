# -*- coding: utf-8 -*-
"""utility functions"""
import sys, os
import struct
import numpy as np


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
   


if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('tools.py command', file=sys.stderr)
