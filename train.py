from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import os
import pprint

import torch
import numpy as np
import random

import datetime
from path import Path

from logger import init_logger

#####Seed setting for reproductivility#########
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
###############################################


options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":

    from ipdb import set_trace
    set_trace()
    # store files day by day
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    opts.save_root = Path('./checkpoints/ht_dcmnet/outputs') / curr_time[:6] / curr_time[6:]
    opts.save_root.makedirs_p()

    # init logger
    _log = init_logger(log_dir=opts.save_root, filename=curr_time[6:] + '.log')
    _log.info('=> will save everything to {}'.format(opts.save_root))

    # show configurations
    cfg_str = pprint.pformat(opts)
    _log.info('=> configurations \n ' + cfg_str)

    trainer = Trainer(opts, _log)
    trainer.train()
