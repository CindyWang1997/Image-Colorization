import os

import torch
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from skimage.color import lab2rgb
from skimage import io
from colornet import ColorNet
from myimgfolder import ValImageFolder
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


data_dir = "places365_standard/val"
# data_dir = "custom_test"
gamut = np.load('models/custom_layers/pts_in_hull.npy')
have_cuda = torch.cuda.is_available()

val_set = ValImageFolder(data_dir)
val_set_size = len(val_set)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

color_model = torch.nn.DataParallel(ColorNet())
if have_cuda:
    color_model.load_state_dict(torch.load('./pretrained/colornet_params.pkl', ))
    color_model.cuda()
else:
    color_model.load_state_dict(torch.load('./pretrained/colornet_params.pkl', map_location='cpu'))


def save_imgs(filename, im):
    img_rgb_out = (255*np.clip(lab2rgb(im),0,1)).astype('uint8')
    plt.imsave(filename , img_rgb_out, vmin=0, vmax=255 )


def val():
    color_model.eval()
    softmax_op = torch.nn.Softmax()
    
    i = 0
    for data, _ in val_loader:
        original_img = data[0].float()
        original_copy = original_img
        original_img = (original_img  - 50) * 0.02
        gray_name = './gray/' + str(i) + '.jpg'
        for img in original_img:
            pic = img.squeeze().numpy()
            pic = pic.astype(np.float64)
            plt.imsave(gray_name, pic, cmap='gray')
        scale_img = data[1].float()
        scale_img = (scale_img - 50) * 0.02
        img_ab = data[2].float()
        # img_gray = data[3].float()
        if have_cuda:
            original_img, scale_img, img_ab = original_img.cuda(), scale_img.cuda(), img_ab.cuda()
        # print (scale_img.shape)
        # print (img_ab.shape)
        # print (img_ab)
        original_img, scale_img = Variable(original_img, volatile=True), Variable(scale_img)
        output_img, output, target = color_model(original_img, scale_img, img_ab)

        output_img *= 2.606
        output_img = softmax_op(output_img).cpu().data.numpy()
        fac_a = gamut[:,0][np.newaxis,:,np.newaxis,np.newaxis] * 1.6
        fac_b = gamut[:,1][np.newaxis,:,np.newaxis,np.newaxis] * 1.6
        # print (original_img)
        img_l = (original_copy).cpu().data.numpy().transpose(0,2,3,1)
        frs_pred_ab = np.concatenate((np.sum(output_img * fac_a, axis=1, keepdims=True), np.sum(output_img * fac_b, axis=1, keepdims=True)), axis=1).transpose(0,2,3,1)
        # print (frs_pred_ab)

        frs_predic_imgs = np.concatenate((img_l, frs_pred_ab ), axis = 3)
        for img in frs_predic_imgs:
            print ("img: %d" %i)
            color_name = './colorimg/' + str(i) + '.jpg'
            save_imgs(color_name, img)
            i += 1

val()
