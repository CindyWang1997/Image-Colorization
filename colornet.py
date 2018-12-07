import torch.nn as nn
import torch.nn.functional as F
import torch
from models.custom_layers.trainable_layers import *
from torch.autograd import Variable

have_cuda = torch.cuda.is_available()


class LowLevelFeatNet(nn.Module):
    def __init__(self):
        super(LowLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))
        x1 = F.relu(self.bn5(self.conv5(x1)))
        x1 = F.relu(self.bn6(self.conv6(x1)))
        if self.training:
            x2 = x1.clone()
        else:
            x2 = F.relu(self.bn1(self.conv1(x2)))
            x2 = F.relu(self.bn2(self.conv2(x2)))
            x2 = F.relu(self.bn3(self.conv3(x2)))
            x2 = F.relu(self.bn4(self.conv4(x2)))
            x2 = F.relu(self.bn5(self.conv5(x2)))
            x2 = F.relu(self.bn6(self.conv6(x2)))
        return x1, x2


class MidLevelFeatNet(nn.Module):
    def __init__(self):
        super(MidLevelFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class GlobalFeatNet(nn.Module):
    def __init__(self):
        super(GlobalFeatNet, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(25088, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, 25088)
        x = F.relu(self.bn5(self.fc1(x)))
        output_512 = F.relu(self.bn6(self.fc2(x)))
        output_256 = F.relu(self.bn7(self.fc3(output_512)))
        return output_512, output_256


class ClassificationNet(nn.Module):
    def __init__(self):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 365)
        self.bn2 = nn.BatchNorm1d(365)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.log_softmax(self.bn2(self.fc2(x)))
        return x


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(256)
        #self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, mid_input, global_input):
        w = mid_input.size()[2]
        h = mid_input.size()[3]
        global_input = global_input.unsqueeze(2).unsqueeze(2).expand_as(mid_input)
        fusion_layer = torch.cat((mid_input, global_input), 1)
        fusion_layer = fusion_layer.permute(2, 3, 0, 1).contiguous()
        fusion_layer = fusion_layer.view(-1, 512)
        fusion_layer = self.bn1(self.fc1(fusion_layer))
        fusion_layer = fusion_layer.view(w, h, -1, 256)

        x = fusion_layer.permute(2, 3, 0, 1).contiguous()
        x = F.relu(self.bn2(self.conv1(x)))
        # x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        #x = F.relu(self.bn4(self.conv3(x)))
        # x = F.relu(self.bn5(self.conv4(x)))
        x = self.upsample(x)
        # x = F.sigmoid(self.bn5(self.conv4(x)))
        # x = self.upsample(self.conv5(x))
        return x

def conv( in_c, out_c,blocks, strides,  kernel_size=3,batchNorm=True, bias=True):

    model = []
    assert len(strides) == blocks

    for i in range(blocks):
        model += [nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=strides[i], padding=(kernel_size-1)//2, bias=bias),
            nn.ReLU()]
        in_c = out_c

    if batchNorm:
        model += [nn.BatchNorm2d(out_c)]
        return nn.Sequential(*model)

    else:
        return nn.Sequential(*model)


class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.low_lv_feat_net = LowLevelFeatNet()
        self.mid_lv_feat_net = MidLevelFeatNet()
        self.global_feat_net = GlobalFeatNet()
        self.class_net = ClassificationNet()
        self.upsample_col_net = FusionNet()
        self.nnecnclayer = NNEncLayer()
        self.priorboostlayer = PriorBoostLayer()
        self.nongraymasklayer = NonGrayMaskLayer()
        self.rebalancelayer = Rebalance_Op.apply
        self.pool = nn.AvgPool2d(4,4)
        self.upsample = nn.Upsample(scale_factor=4)
        self.conv_8 = conv(256,256,2,[1,1], batchNorm=False)
        self.conv313 = nn.Conv2d(256,313,1,1)


    def forward(self, x_1, x_2, img_ab):
        x1, x2 = self.low_lv_feat_net(x_1, x_2)
        #print('after low_lv, mid_input is:{}, global_input is:{}'.format(x1.size(), x2.size()))
        x1 = self.mid_lv_feat_net(x1)
        #print('after mid_lv, mid2fusion_input is:{}'.format(x1.size()))
        class_input, x2 = self.global_feat_net(x2)
        #print('after global_lv, class_input is:{}, global2fusion_input is:{}'.format(class_input.size(), x2.size()))
        class_output = self.class_net(class_input)
        #print('after class_lv, class_output is:{}'.format(class_output.size()))
        fusion = self.upsample_col_net(x1, x2)
        #print('after upsample_lv, output is:{}'.format(output.size()))
        x = self.conv_8(fusion)
        gen = self.conv313(x)

        # ************ probability distribution step **************
        gt_img_ab = self.pool(img_ab).cpu().data.numpy()
        enc = self.nnecnclayer(gt_img_ab)
        ngm = self.nongraymasklayer(gt_img_ab)
        pb = self.priorboostlayer(enc)
        boost_factor = (pb * ngm).astype('float32')
        if have_cuda:
            boost_factor = Variable(torch.from_numpy(boost_factor).cuda())
        else:
            boost_factor = Variable(torch.from_numpy(boost_factor))

        wei_output = self.rebalancelayer(gen, boost_factor)
        if self.training:
            if have_cuda:
                return class_output, wei_output, Variable(torch.from_numpy(enc).cuda())
            else:
                return class_output, wei_output, Variable(torch.from_numpy(enc))
        else:
            if have_cuda:
                return self.upsample(gen), wei_output, Variable(torch.from_numpy(enc).cuda())
            else:
                return self.upsample(gen), wei_output, Variable(torch.from_numpy(enc))
