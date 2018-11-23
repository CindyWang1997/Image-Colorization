import os
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from myimgfolder import TrainImageFolderBaseline
from colornet_baseline import ColorNet

original_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

have_cuda = torch.cuda.is_available()
epochs = 3

data_dir = "./places365_standard/train/"
train_set = TrainImageFolderBaseline(data_dir, original_transform)
train_set_size = len(train_set)
train_set_classes = train_set.classes
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
color_model = ColorNet()
if os.path.exists('./pretrained/colornet_params_baseline.pkl'):
    color_model.load_state_dict(torch.load('./pretrained/colornet_params_baseline.pkl'))
if have_cuda:
    color_model.cuda()
optimizer = optim.Adadelta(color_model.parameters())

if have_cuda: 
    print ("Have cuda")
else:
    print ("No cuda")

def train(epoch):
    color_model.train()

    try:
        for batch_idx, (data, classes) in enumerate(train_loader):
            print ('batch_idx: %d' %batch_idx)
            messagefile = open('./message.txt', 'a')
            original_img = data[0].unsqueeze(1).float()
            img_ab = data[1].float()
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()
            class_output, output = color_model(original_img, original_img)
            if have_cuda:
                ems_loss = torch.pow((img_ab - output), 2).sum() / Variable(torch.from_numpy(np.array(list(output.size()))).cuda()).prod().float()
            else:
                ems_loss = torch.pow((img_ab - output), 2).sum() / Variable(torch.from_numpy(np.array(list(output.size())))).prod().float()
            cross_entropy_loss = 1/300 * F.cross_entropy(class_output, classes)
            loss = ems_loss + cross_entropy_loss
            lossmsg = 'loss: %.9f\n' % (loss.item())
            messagefile.write(lossmsg)
            ems_loss.backward(retain_graph=True)
            cross_entropy_loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * 32, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())
                messagefile.write(message)
                torch.save(color_model.state_dict(), 'colornet_params_baseline_%d.pkl' %epoch)
            messagefile.close()
                # print('Train Epoch: {}[{}/{}({:.0f}%)]\tLoss: {:.9f}\n'.format(
                #     epoch, batch_idx * len(data), len(train_loader.dataset),
                #     100. * batch_idx / len(train_loader), loss.data[0]))
    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'colornet_params_baseline.pkl')


for epoch in range(1, epochs + 1):
    train(epoch)
