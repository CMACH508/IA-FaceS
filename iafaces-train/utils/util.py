import torch
import json
import numpy as np
import random
from pathlib import Path
from collections import OrderedDict


def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def float_tuple(s):
    return tuple(float(i) for i in s.split(','))


def str_tuple(s):
    return tuple(s.split(','))


def bool_flag(s):
    if s == '1' or s == 'True' or s == 'true':
        return True
    elif s == '0' or s == 'False' or s == 'false':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0/1 or True/False or true/false)'
    raise ValueError(msg % s)


def set_seed(seed, base=0, is_set=True):
    seed += base
    assert seed >= 0, '{} >= {}'.format(seed, 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_node_box(fs, ori_size, boxes):
    scale = ori_size / fs
    rescaled_boxes = torch.ceil((boxes - scale / 2) / scale).int()
    rescaled_boxes = torch.clamp(rescaled_boxes, 0, fs - 1).to(torch.int32)
    return rescaled_boxes


def get_node_feats(raw_feature, boxes, layers):
    all_nodes = []
    for idx, box in enumerate(boxes):
        layer = layers[max(0, idx - 1)]
        node_feature = raw_feature[:, :, box[1]:box[3], box[0]:box[2]].to(raw_feature)
        all_nodes.append(layer(torch.flatten(node_feature, 1)).unsqueeze(1))
    return torch.cat(all_nodes, dim=1)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
