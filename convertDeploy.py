import torch
from nets.repvgg import repvgg_model_convert
from nets.efficient_reatinanet import retinanet

#   Converting training weights to inference weights
# load model
model = retinanet(num_classes=1)
# load training weight
state_dict = torch.load('logs/ep149-loss0.112-val_loss0.283.pth')
model.load_state_dict(state_dict)
# convert weight
repvgg_model_convert(model,save_path='deploy_weights/efficient_retinanet.pth')
