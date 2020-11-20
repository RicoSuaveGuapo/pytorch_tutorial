import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from goddataset import *
from godmodel import *

classes = ['God','Goddess']

def confusionMatrix(model_path, mode):
    model = GodModelPretrained(hidden_dim=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.cuda()
    with torch.no_grad():
        dataset   = GodDataset(mode=mode)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, 
                                pin_memory=True, num_workers=2*os.cpu_count())
        targets = torch.Tensor().type(torch.long)
        predicts = torch.Tensor().type(torch.long).cuda()
        for data in tqdm(dataloader, desc='Validation'):
            input, target = data[0].cuda(), data[1]
            targets = torch.cat((targets, target))
            output = model(input) 
            _, predicted = torch.max(output, 1)
            predicts = torch.cat((predicts, predicted))
        
        targets  = targets.numpy()
        predicts = predicts.cpu().numpy()
        c_matrix = confusion_matrix(targets, predicts, normalize='true',
                                    labels=[i for i in range(len(classes))])    
        return c_matrix
    
def format_func(value, tick_number):
    if value >= 0 and value < 2:
        return classes[value.astype(np.int)]



if __name__ == "__main__":
    start_time = time.time()
    mode = 'test'
    path = 'model_save/0_best.pth'

    c_matrix = confusionMatrix(model_path=path, mode=mode)
    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.matshow(c_matrix)
    axes.set_title(f'Confusion Matrix: {mode} set')
    axes.set(xlabel = 'Predicted',ylabel = 'Truth')
    axes.set_xticks(np.arange(0, len(classes)))
    axes.set_yticks(np.arange(0, len(classes)))
    caxes = axes.matshow(c_matrix, interpolation ='nearest') 
    figure.colorbar(caxes)
    axes.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    axes.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    for row_i, row in enumerate(c_matrix):
        for col_i, col in enumerate(row):
            axes.text(col_i,row_i,f'{col:.2f}',color='red')
    plt.show()