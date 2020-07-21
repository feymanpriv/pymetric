import os
import sys
import cv2
import datetime
import numpy as np
import pickle

import torch
import metric.core.config as config
import metric.datasets.transforms as transforms
import metric.core.builders as builders
from metric.core.config import cfg
from utils.linear_head import LinearHead


_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

INFER_DIR = '../../data/eval/query'
MODEL_WEIGHTS = 'saved_models/resnest_arc/model_epoch_0043.pyth'


class MetricModel(torch.nn.Module):
    def __init__(self):
        super(MetricModel, self).__init__()
        self.backbone = builders.build_model()
        self.head = LinearHead()
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def preprocess(im):
    im = transforms.scale(cfg.TEST.IM_SIZE, im)
    im = transforms.center_crop(cfg.TRAIN.IM_SIZE, im)
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = transforms.color_norm(im, _MEAN, _SD)
    return [im]

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def extract(imgpath, model):
    im = cv2.imread(imgpath)
    im = im.astype(np.float32, copy=False)
    im = preprocess(im)
    im_array = np.asarray(im, dtype=np.float32)
    input_data = torch.from_numpy(im_array)
    if torch.cuda.is_available(): 
        input_data = input_data.cuda() 
    fea = model(input_data, targets=None)
    embedding = to_numpy(fea)
    #print("fea_shape: ", embedding.shape)     
    return embedding


def main(spath):
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    model = builders.build_arch()
    print(model)
    load_checkpoint(MODEL_WEIGHTS, model)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    
    feadic = {}
    for index, imgfile in enumerate(walkfile(spath)):
        ext = os.path.splitext(imgfile)[-1]
        name = os.path.basename(imgfile)
        if ext.lower() in ['.jpg', '.jpeg', '.bmp', '.png', '.pgm']:
            embedding = extract(imgfile, model)
            feadic[name] = embedding
            print(feadic)
            if index%5000 == 0:
                print(index, embedding.shape)
    
    with open(spath.split("/")[-1]+"fea.pickle", "wb") as fout:
        pickle.dump(feadic, fout, protocol=2)
    

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


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()
    
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.'.format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))
    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint

if __name__ == '__main__':
    main(INFER_DIR)

