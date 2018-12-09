import os
import traceback

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

from myimgfolder import TrainImageFolder
from colornet import ColorNet
from loss import CE_loss

original_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    #transforms.ToTensor()
])

have_cuda = torch.cuda.is_available()
start_epoch = 2
epochs = 3

data_dir = "./places365_standard/train/"
train_set = TrainImageFolder(data_dir, original_transform)
train_set_size = len(train_set)
train_set_classes = train_set.classes
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8)
color_model = torch.nn.DataParallel(ColorNet())
if os.path.exists('./pretrained/colornet_params.pkl'):
    color_model.load_state_dict(torch.load('./pretrained/colornet_params.pkl'))
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
            # original_img = data[0].unsqueeze(1).float()
            original_img = data[0].float()
            original_img = (original_img - 50) * 0.02

            img_ab = data[1].float()
            if have_cuda:
                original_img = original_img.cuda()
                img_ab = img_ab.cuda()
                classes = classes.cuda()
            original_img = Variable(original_img)
            img_ab = Variable(img_ab)
            classes = Variable(classes)
            optimizer.zero_grad()

            class_output, output, target = color_model(original_img, original_img, img_ab)

            criterion = CE_loss()
            output_loss = criterion(output, target)
            class_cross_entropy_loss = 0.2* F.cross_entropy(class_output, classes)
            loss = output_loss + class_cross_entropy_loss
            # print ("output_loss: %.9f" %output_loss.item())
            # print ("class_cross_entropy_loss: %.9f" %class_cross_entropy_loss.item())
            lossmsg = 'loss: %.9f\n' % (loss.item())
            messagefile.write(lossmsg)
            output_loss.backward(retain_graph=True)
            class_cross_entropy_loss.backward()
            optimizer.step()
            if batch_idx % 500 == 0:
                message = 'Train Epoch:%d\tPercent:[%d/%d (%.0f%%)]\tLoss:%.9f\n' % (
                    epoch, batch_idx * 32, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())
                messagefile.write(message)
                torch.save(color_model.state_dict(), 'colornet_params_%d.pkl' %epoch)
            messagefile.close()

    except Exception:
        logfile = open('log.txt', 'w')
        logfile.write(traceback.format_exc())
        logfile.close()
    finally:
        torch.save(color_model.state_dict(), 'colornet_params.pkl')


for epoch in range(start_epoch, epochs + 1):
    train(epoch)
