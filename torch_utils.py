from matplotlib.lines import Line2D
from torchvision.transforms import ToTensor
import PIL.Image
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import python_utils
import sklearn.metrics
import subprocess
import torch

def save_params(params, savePth):
    # save params in savePth
    if not os.path.exists(savePth):
        os.makedirs(savePth)

    paramsFile = python_utils.get_new_name(os.path.join(savePth, 'parameters'),'.txt')
    with open(paramsFile, 'w+')  as f:
        for key in params.keys():
            value = params[key]
            if (isinstance(value, str) or isinstance(value, int) or
                isinstance(value, float) or isinstance(value, bool)):
                f.write('%s: %s\n' % (key, str(value)))
            elif (isinstance(value, list)):
                f.write('%s:\n' % key)
                for v in value:
                    f.write(v + '\n')
            elif isinstance(value, dict):
                f.write('%s:\n' %key)
                for key2 in value.keys():
                    value2 = value[key2]
                    f.write('\t%s: %s\n' %(key2, str(value2)))
            else:
                raise ValueError('For params dict [key: %s, value: %s], value instance type not defined' %(key,value))

    f.close()

def selectDevice(tcuda):
    deviceID = 0
    total = tcuda.get_device_properties(deviceID).total_memory/(1024.0*1024.0)  # Total GPU memory in MB
    occupied = get_gpu_memory_map()
    if occupied < total*0.3:
        return 'cuda'
    else:
        return 'cpu'


def get_gpu_memory_map():
    """Get the current gpu usage."""
    cmd = 'nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader'
    process = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    return float(process.communicate()[0].strip())


def get_git_head_hash(path):
    cmd = 'git  --git-dir ' + os.path.join(path,'.git') + ' rev-parse HEAD'
    process = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    return process.communicate()[0].strip()


def get_git_info(path):
    cmd = 'git --git-dir ' + os.path.join(path,'.git') + ' log --oneline -1'
    process = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    return process.communicate()[0].strip()

def matlplot_to_image(plt):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    plt.clf()
    return image

def plot_grad_flow(named_parameters, epoch, iteration, cdata, fig=None):
    '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for id, namedParam in enumerate(named_parameters):
        n, p = namedParam
        if (p.requires_grad) and ("bias" not in n):
            if cdata is not None:
                if p._cdata in cdata:
                    id = cdata[p._cdata]-1
                else:
                    id = 999
            layers.append("%s-%d" %(n.replace('.weight',''),id+1))
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.barh(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.barh(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.vlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.yticks(range(0, len(ave_grads), 1), layers, fontsize=8)
    plt.xticks(np.arange(0, 0.02, 0.005))
    plt.ylim(bottom=-0.001, top=len(ave_grads))
    plt.xlim(left=0, right=0.02)  # zoom in on the lower gradient regions
    plt.ylabel("Layers")
    plt.xlabel("average gradient")
    plt.title("Gradient flow for epoch %d, iteration %d" %(epoch, iteration))
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.tight_layout()

    # resiye y-axis to accomodate yticks
    plt.gca().margins(y=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_yticklabels()
    maxsize = max([t.get_window_extent().height for t in tl])
    m = 0.25  # inch margin
    s = maxsize / plt.gcf().dpi * len(layers) + 2 * m
    margin = m / plt.gcf().get_size_inches()[1]
    plt.gcf().subplots_adjust(bottom=margin, top=1. - margin)
    plt.gcf().set_size_inches(plt.gcf().get_size_inches()[0],s)

    if fig:
        fig.canvas.draw()
    return plt

def image_grad_flow(named_parameters, epoch, iteration, cdata=None):
    plt = plot_grad_flow(named_parameters, epoch, iteration, cdata)
    return matlplot_to_image(plt)

def tanh2sigmoid(tanh):
    # scale from tanh [-1 1] to sigmoid [0 1]
    sigmoid = (tanh + torch.ones_like(tanh)) / 2. * torch.ones_like(tanh)
    return sigmoid

def image_conf_matrix(y_true, y_pred, labels=None):
    conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred, labels, normalize='true')
    fig = plt.figure()
    im = plt.imshow(conf_mat)
    fig.colorbar(im)
    plt.xticks(range(0, len(labels), 1), labels, fontsize=8, rotation=-90)
    plt.yticks(range(0, len(labels), 1), labels, fontsize=8)

    diag = np.diagonal(conf_mat)
    return matlplot_to_image(plt), sum(diag)/len(diag)