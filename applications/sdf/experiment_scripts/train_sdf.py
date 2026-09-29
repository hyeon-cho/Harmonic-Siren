'''Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
'''

# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, meta_modules, utils, training, loss_functions, modules

from torch.utils.data import DataLoader
import configargparse

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=1400)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')

p.add_argument('--model_type', type=str, choices=('sine', 'resin'), default='sine',
               help='SIREN baseline or recurrent sinusoidal model')
p.add_argument('--point_cloud_path', type=str, default='point_cloud/armadillo.xyz',
               help='Armadillo point cloud in XYZ+normal format')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')

# resin (recurrent sinusoidal) architecture options
p.add_argument('--hidden_features', type=int, default=500)
p.add_argument('--num_hidden_layers', type=int, default=1)
p.add_argument('--recurrent_steps', type=int, default=3)
p.add_argument('--recurrent_idx', type=int, default=None)
p.add_argument('--seed', type=int, default=None, help='fix random seed for reproducibility')
opt = p.parse_args()

if opt.seed is not None:
    import random
    import torch
    random.seed(opt.seed)
    import numpy as _np
    _np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)


import numpy as np

_point_cloud = np.genfromtxt(opt.point_cloud_path)
num_on_surface_points = _point_cloud.shape[0]
print(f"Number of on-surface points: {num_on_surface_points}")
# import pdb; pdb.set_trace()
sdf_dataset = dataio.PointCloud(opt.point_cloud_path, on_surface_points=num_on_surface_points) #opt.batch_size)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

# Define the model.
if opt.model_type == 'resin':
    model = modules.SingleBVPNet(out_features=1,
                                 type='resin',
                                 in_features=3,
                                 hidden_features=opt.hidden_features,
                                 num_hidden_layers=opt.num_hidden_layers,
                                 recurrent_steps=opt.recurrent_steps,
                                 recurrent_idx=opt.recurrent_idx,
)
else:
    model = modules.SingleBVPNet(type=opt.model_type, in_features=3, )
                                #  hidden_features=512, num_hidden_layers=1)
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf
summary_fn = utils.write_sdf_summary

root_path = os.path.join(opt.logging_root, opt.experiment_name)

training.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, double_precision=False,
               clip_grad=True)
