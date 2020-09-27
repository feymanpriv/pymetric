# encoding: utf-8
import sys, os

import torch
import metric.core.config as config
import metric.core.builders as builders
from metric.core.config import cfg

import metric.datasets.transforms as transforms
from linear_head import LinearHead, LinearHead_cat_with_pred

from tools.convert.tf_export import export_tf_model
import numpy as np

_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]


class MetricModel(torch.nn.Module):
    def __init__(self):
        super(MetricModel, self).__init__()
        self.backbone = builders.build_model()
        #self.head = LinearHead()
        self.head = LinearHead_cat_with_pred()
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

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
    #ms = model.module if cfg.NUM_GPUS > 1 else model
    ms = model
    model_dict = ms.state_dict()

    #state_dict = {'backbone.'+k : v for k, v in state_dict.items()}
    #state_dict = {k[7:] : v for k, v in state_dict.items()}
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
    #ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    #return checkpoint["epoch"]
    return checkpoint


def preprocess(im):
    im = transforms.scale(cfg.TEST.IM_SIZE, im)
    im = transforms.center_crop(cfg.TRAIN.IM_SIZE, im)
    im = im.transpose([2, 0, 1])
    im = im / 255.0
    im = transforms.color_norm(im, _MEAN, _SD)
    return [im]

def process(im):
    #import cv2
    #im = cv2.imread(imgpath)
    im = im.astype(np.float32, copy=False)
    im = preprocess(im)
    im_array = np.asarray(im, dtype=np.float32)
    input_data = torch.from_numpy(im_array)
    return input_data


def export():
    config.load_cfg_fom_args("Convert a metric model.")
    config.assert_and_infer_cfg()
    cfg.freeze()

    model = MetricModel()
    print(model)
    #model.load_state_dict(torch.load(cfg.CONVERT_MODEL_FROM)['model_state'], strict=False)
    load_checkpoint(cfg.TRAIN.WEIGHTS, model)
    model.eval()
    dummy_inputs = torch.randn(1, 3, 224, 224)
    print(dummy_inputs.size())
    fea = model(dummy_inputs)
    print(fea.size())
    export_tf_model(model, dummy_inputs)

    
def infer():
    config.load_cfg_fom_args("Infer a metric model.")
    config.assert_and_infer_cfg()
    cfg.freeze()

    model = MetricModel()
    print(model)
    #model.load_state_dict(torch.load(cfg.CONVERT_MODEL_FROM)['model_state'], strict=False)
    load_checkpoint(cfg.TRAIN.WEIGHTS, model)
    model.eval()
    
    new_input = np.ones((1,3,224,224), dtype='float32')
    new_input = torch.from_numpy(new_input)
    #new_input = process(new_input)
    '''
    img = np.ones([224,224,3])
    b = np.ones([224,224])
    g = b * 2
    r = b * 3
    img[:, :, 0]  = b
    img[:, :, 1]  = g
    img[:, :, 2]  = r
    new_input = process(img)
    '''
    fea = model(new_input)
    fea_numpy = fea.detach().numpy()
    print(fea_numpy[0][:10])

   

if __name__ == '__main__':
    export() 
    #infer()
