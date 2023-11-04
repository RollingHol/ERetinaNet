#       train
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import os

from nets.efficient_reatinanet import ERetinaNet
from nets.retinanet_change_backbone import retinanet

from nets.retinanet_training import FocalLoss
from utils.callbacks import LossHistory
from utils.dataloader import RetinanetDataset, retinanet_dataset_collate
from utils.dataset_format import datasetFormat
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    model_path = ''
    # Whether to train ERetinaNet.
    train_ERetinaNet = True
    input_shape = [600, 600]
    # Whether to load pre-training weights.
    # Note: The pre-training weights for RepVGG and FRepVGG need to be stored under model_data
    pretrained = True
    if train_ERetinaNet:
        # deploy is set to False during training
        deploy = False
    else:
        # chose backbone, Optional
        # resnet-18, resnet-34, resnet-50, resnet-101, resnet-152
        # repvgg-train, frepvgg-train
        # vgg-16, inception-v3, mobilenet-v3
        backbone = "mobilenet-v3"   # If train_ERetinaNet is True, backbone is not available

    Init_Epoch = 0
    End_Epoch = 200
    Batch_Size = 4  # 8
    Init_Lr = 1e-4

    # Weights and log file save folders
    save_dir = 'logs'
    # Random data augmentation during training, Optional
    train_random = True  # False
    # save_period   Save weights every save_period epoch
    save_period = 5

    num_workers = 4
    #   Adjusting the dataset format
    # dataset root
    data_dir = "VOCdevkit"
    # Path to the category file
    classes_path = 'model_data/breast_classes.txt'
    train_txt, val_txt, _ = datasetFormat(data_dir, classes_path)
    #   training set and validation set
    train_annotation_path = train_txt
    val_annotation_path = val_txt

    class_names, num_classes = get_classes(classes_path)
    if train_ERetinaNet:
        model = ERetinaNet(num_classes, pretrained, deploy)
    else:
        model = retinanet(num_classes, backbone, pretrained)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    focal_loss = FocalLoss()
    # log loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if True:
        batch_size = Batch_Size
        lr = Init_Lr
        start_epoch = Init_Epoch
        end_epoch = End_Epoch

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small for training, please expand the dataset.")
        # start time
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = RetinanetDataset(train_lines, input_shape, num_classes, train=train_random)
        val_dataset = RetinanetDataset(val_lines, input_shape, num_classes, train=False)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=retinanet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=retinanet_dataset_collate)

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, focal_loss, loss_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, save_period, save_dir)
            lr_scheduler.step()

        # end time
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


