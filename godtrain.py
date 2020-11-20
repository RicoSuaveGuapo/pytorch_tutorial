import os
import cv2
import time
import argparse
import easydict
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from goddataset import *
from godmodel import *

# ====================
# || Hyperparameter ||
# ====================
args = easydict.EasyDict({
        # Basic
        'image_size': 256,
        'val_split':0.3,
        'test_split':0.1,
        'seed':42,
        # FC
        'hidden_dim':128,
        'dropout':0.5,
        # Additional Hyperparameter
        'classes':2,
        'model':'pretrained',
        # Training Related
        'optim':'Adam',
        'momentum':0.9,
        'weight_decay':0.01,
        'lr':0.0001,
        'lr_name':'ReduceLROnPlateau',
        # Loop control
        'epoch':100,
        'batch_size': 5
        })

# ====================
# ||   Helper Funs  ||
# ====================
# use it to control overall hyper-parameters
def check_args(args):
    assert args.optim in ['Adam','SGD'], 'chose optim not implement'
    assert args.lr_name in ['ReduceLROnPlateau'], 'I am lazy, choose ReduceLROnPlateau instead :>'
    assert args.model in ['pretrained','self-define'], 'keywords: pretrained, self-define'

    print('\n---- Training parameters ----')
    print(f'image size: {args.image_size}')
    print(f'Optimizer : {args.optim}')
    print(f'Hidden dim: {args.hidden_dim}')
    print(f'Randomseed: {args.seed}')
    print(f'Initial lr: {args.lr}')
    print(f'lr name   : {args.lr_name}')
    print(f'Output cls: {args.classes}')
    print(f'Epoch     : {args.epoch}')
    print(f'Batch size: {args.batch_size}')


# ====================
# ||    The train   ||
# ====================
def train(args):
    check_args(args)
    # use GPU to do the training
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if not os.path.exists(os.path.join(os.getcwd(), 'runs')):
        os.mkdir(os.path.join(os.getcwd(), 'runs'))
    trial_num = os.listdir(os.path.join(os.getcwd(), 'runs'))
    if trial_num == []:
        exp_num = 0
    else:
        exp_num = [int(num.replace('trial_', '')) for num in trial_num]
        exp_num = max(exp_num) +1 
    print(f'\n== Trial {exp_num} begins ==\n')

    print('\n-------- Data Preparing --------\n')
    train_loader, val_loader, test_loader = datasets(args)
    print('-------- Data Preparing Done! --------')

    print('-------- Preparing Model --------')
    if args.model == 'pretrained':
        model = GodModelPretrained(hidden_dim=args.hidden_dim, dropout=args.dropout, classes=args.classes)
    else:
        model = GodModelSelf(args=args, hidden_dim=args.hidden_dim, dropout=args.dropout, classes=args.classes)
    # move model to GPU
    model = model.to(device)
    # loss
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'Adam':
        # before acc 80 %
        optimizer = optim.Adam(model.parameters())
    elif args.optim == 'SGD':
        # after acc 80 %
        optimizer = optim.SGD(model.parameters(), momentum=args.momentum, lr=args.lr, nesterov=True, weight_decay=args.weight_decay)
    # lr scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=6)
    print('-------- Preparing Model Done! --------')

    print('\n-------- Starting Training --------\n')
    # use Tensorbroad to record
    writer = SummaryWriter(f'runs/trial_{exp_num}')

    # comparsion of accuracy, only save the best weight
    accuracies = [0.]
    k = 0
    for epoch in range(args.epoch):
        start_time = time.time()
        train_running_loss = 0.0
        print(f'--- The {epoch+1}/{args.epoch} epoch ---')
        #  --------------------------- TRAINING LOOP ---------------------------
        print('\n--- Training Loop begins ---')
        print('[Epoch, Batch] : Loss')
        # set the optim to zero gradient
        optimizer.zero_grad()
        # set the model into training mode
        model.train()
        for i, data in enumerate(train_loader, start=0):
            # move data to GPU
            input, target = data[0].to(device), data[1].to(device)
            # foward
            output = model(input)
            loss = criterion(output, target)
            # backward
            loss.backward()
            train_running_loss += loss.item()
            # using the forward and backward info to update the parameters
            optimizer.step()
            # set the optim to zero gradient, in order to calculate new gradient in the next run
            # else will be accumlate (which also is the virture of Pytorch)
            optimizer.zero_grad()
            # record loss evey 50 iterations
            if (i+1)%50 == 0:
                    k += 1
                    writer.add_scalar('Batch-Averaged loss', train_running_loss, k)
                    print( f"[{epoch+1}, {i+1}]: %.3f" % train_running_loss)
                    train_running_loss = 0.0
        
        # record the lr use in the epoch
        lr = [group['lr'] for group in optimizer.param_groups]
        print('Epoch:', f'{epoch+1}/{args.epoch}',' LR:', lr[0])
        writer.add_scalar('Learning Rate', lr[0], epoch)
        print('--- Training Loop ends ---\n')
        print(f'--- Training spend time: %.1f sec ---' % (time.time() - start_time))

        #  --------------------------- VALIDATION LOOP ---------------------------
        # which ensure all the parameters are not calulcating gradients
        with torch.no_grad():
            # set the model into evalution mode, basically what torch.no_grad() does
            model.eval()
            val_run_loss = 0.0
            print('\n--- Validaion Loop begins ---')
            start_time = time.time()
            batch_count = 0
            total_count = 0
            correct_count = 0
            for data in tqdm(val_loader, desc='Validation'):
                input, target = data[0].to(device), data[1].to(device)
                output = model(input)
                _, predicted = torch.max(output, 1)
                loss = criterion(output, target)
                val_run_loss  += loss.item()
                correct_count += (predicted == target).sum().item()
                batch_count += 1
                total_count += target.size(0)
            accuracy = (100 * correct_count/total_count)
            val_run_loss = val_run_loss/batch_count

            if not os.path.exists(os.path.join(os.getcwd(), 'model_save')):
                os.mkdir(os.path.join(os.getcwd(), 'model_save'))
            if max(accuracies) < accuracy:
                savepath = os.path.join(os.getcwd(),'model_save',f'{exp_num}_best.pth')
                torch.save(model.state_dict(), savepath)
                print('\n-------- Saveing the best weight --------')
            else:
                print('\n-------- Accuracy is not improving --------')
            accuracies.append(accuracy)

            scheduler.step(val_run_loss)
            writer.add_scalar('Validation accuracy', accuracy, epoch)
            writer.add_scalar('Validation loss', val_run_loss, epoch)

            print(f"Loss of {epoch+1} epoch is {val_run_loss:.3f}")
            print(f"Accuracy is {accuracy:.2f} % \n")
            print('--- Validaion Loop ends ---\n')
            print(f'--- Validaion spend time: %.1f sec ---' % (time.time() - start_time))
    writer.close()
    print('\n-------- End Training --------\n')
    print(f'\n--- Best accuracy: {max(accuracies):.2f} % ---')
    print(f'\n== Trial {exp_num} finished ==\n')


if __name__ == "__main__":
    start_time = time.time()
    train(args)
    print('--- Total Execution time ---')
    exe_time = (time.time() - start_time)
    hr = int(exe_time // 3600)
    min = int(((exe_time / 3600) - hr) * 60)
    sec = ((((exe_time / 3600) - hr) * 60) - min)*60
    print(f'--- {hr}:{min}:{sec:.1f} (hr:min:sec)---')