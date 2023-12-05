from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss,TripletLoss,CrossEntropyLoss_Neg,CrossEntropyLoss_PerImg
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers,read_image
from torchreid.models.GD import L_norm
from torchreid import metrics
import random
from torchreid.optim import build_optimizer,build_lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import FloatTensor
import scipy.io as io

class ComplementEntropy(nn.Module):

    def __init__(self):
        super(ComplementEntropy, self).__init__()
        
    # here we implemented step by step for corresponding to our formula
    # described in the paper
    def forward(self, yHat, y):
        self.batch_size = len(y)
        self.classes = 702
        yHat = F.softmax(yHat, dim=1)
        Yg = torch.gather(yHat, 1, torch.unsqueeze(y, 1))
        Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
        Px = yHat / Yg_.view(len(yHat), 1)
        Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
        y_zerohot = torch.ones(self.batch_size, self.classes).scatter_(
            1, y.view(self.batch_size, 1).data.cpu(), 0)
        output = Px * Px_log * y_zerohot.cuda()
        loss = torch.sum(output)
        loss /= float(self.batch_size)
        loss /= float(self.classes)
        return loss
import math
from torchvision import transforms
import os.path as osp
def tensor_to_PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    loader = transforms.Compose([transforms.ToTensor()])  
    image = image.convert("L") 
    image = loader(image).unsqueeze(0)
    image = image.cuda()

    return image
import numpy as np
import cv2
from PIL import Image
class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(self, datamanager, model, G, optimizer_m, optimizer_G, scheduler_m=None,scheduler_G=None, use_gpu=True,
                 label_smooth=True):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, G, optimizer_m, optimizer_G, scheduler_m, scheduler_G, use_gpu)
        
        self.complemntloss=ComplementEntropy()
        self.fakediscrimin=nn.Sequential(
            nn.Linear(4096,2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048,1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128,1),
            nn.Sigmoid(),
            ).cuda()
        self.cameradiscrimin=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,2,bias=False))


        self.criterion1 = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion2 = CrossEntropyLoss_Neg(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion3 = CrossEntropyLoss_PerImg(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.adversarial_loss = torch.nn.BCELoss()
        self.optimizer_D=build_optimizer(
            self.fakediscrimin, optim='adam', lr=0.0003
        )
        self.scheduler_D = build_lr_scheduler(
            self.optimizer_D,
            lr_scheduler='single_step',
            stepsize=20
        )


        self.numid=0
        #self.criterion_m = torch.nn.BCELoss(reduction='mean')
        self.criterion_t = TripletLoss(margin=0.3)
    def cal_newimages(self,imgs,maskimages,pids):
        imagesnum=9
        index_random=random.sample(range(0,imgs.shape[0]),imagesnum)
        fakeimages=torch.zeros((imagesnum,imgs.shape[1],imgs.shape[2],imgs.shape[3]))
        fakepids=torch.zeros(imagesnum)

        fillimage=torch.zeros((imgs.shape[1],imgs.shape[2],imgs.shape[3]))
        fillimage[0,:,:]=0.4300
        fillimage[1,:,:]=0.4236
        fillimage[2,:,:]=0.4393

        fakepids=fakepids.int()
        original_images=imgs
        maskimages_person=maskimages
        maskimages_back=torch.abs(maskimages-1)
        for i in range(imagesnum):

            fillmask=maskimages_person[i]+maskimages_back[index_random[i]]
            fillmask[fillmask>0]=1
            fillmask=torch.abs(fillmask-1)
            imgs_tmp=imgs[i]*maskimages_person[i]+imgs[index_random[i]]*maskimages_back[index_random[i]]+fillimage*fillmask#torch.zeros(size=(imgs.shape[1],imgs.shape[2],imgs.shape[3]))
            
            fakeimages[i]=imgs_tmp
            fakepids[i]=pids[i]
        #fakepids=torch.Tensor(fakepids)
        #fakeimages=torch.Tensor(fakeimages)
        return fakepids,fakeimages


    def cal_colorimage(self,imgs,maskimages,pids):
        imagesnum=12
        index_random=random.sample(range(0,imgs.shape[0]),imagesnum)
        fakeimages=torch.zeros((imagesnum,imgs.shape[1],imgs.shape[2],imgs.shape[3]))
        fakepids=torch.zeros(imagesnum)

        fakepids=fakepids.int()
        maskimages_person=maskimages
        maskimages_back=torch.abs(maskimages-1)
        for j in range(imagesnum):
            i=index_random[j]
            augmentimages=(imgs[i,0,:,:]+random.uniform(0.05,0.2))
            augmentimages[augmentimages>1]=0.9
            dissimages=(imgs[i,0,:,:]-random.uniform(0.1,0.15))
            dissimages[dissimages<0]=0.02              
            imgs_tmp=augmentimages*maskimages_person[i]+dissimages*maskimages_back[i]
            fakeimages[j]=imgs_tmp
            fakepids[j]=pids[i]
        return fakepids,fakeimages
    def kl_loss(self,mu,logvar):
        kl_loss = -(1 + logvar - mu.pow(2) - logvar.exp()) / 2
        kl_loss = kl_loss.sum(dim=1).mean()
        loss    = 0.0003 * kl_loss
        return loss
    def color_erase(self,img,erase_x,erase_y):
        sl=0.3
        sh=0.6
        r1=0.5
        new=img.clone()
        for i in range(img.size(0)):
            new[i,:,:,:]=tensor_to_PIL(img[i,:,:,:])

        for i in range(img.size(0)):
            for attempt in range(10000000000):
                area = img.size()[2] * img.size()[3]
                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1 / r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img.size()[3] and h < img.size()[2]:
                    x1 = erase_x[i]
                    y1 = erase_y[i]
                    if x1+h>img.size()[2]:
                        x1=img.size()[2]-h
                    if y1+w>img.size()[3]:
                        y1=img.size()[3]-w
                    if img.size()[1] == 3:
                        img[i, 0, x1:x1 + h, y1:y1 + w] = new[i, 0, x1:x1 + h, y1:y1 + w]
                        img[i, 1, x1:x1 + h, y1:y1 + w] = new[i, 0, x1:x1 + h, y1:y1 + w]
                        img[i, 2, x1:x1 + h, y1:y1 + w] = new[i, 0, x1:x1 + h, y1:y1 + w]
       
                        break
        return img

    def perturb(self, imgs, features=None, train_or_test='test', epoch = 60, epsilon = 0.1):
        b = imgs.size()[0]
        imgs_copy = copy.deepcopy(imgs)
        mask = torch.ones([b, 1, imgs.size()[2], imgs.size()[3]])
        num_size = torch.zeros(2)
        if train_or_test == 'test':
            imgs_copy = torch.cat([imgs_copy, imgs_copy], dim = 0)
            k = 2
            pos = torch.empty([b, 2])
            output_pos = self.G(imgs,features)
            pos_probs = F.softmax(output_pos, dim=-1)
            output_pos_small = pos_probs[:,::2]
            output_pos_large = pos_probs[:,1::2]
            output_position = output_pos_small + output_pos_large

            randnum = random.random()
            if randnum <= 0.8:
                output_pos_max = output_position.multinomial(num_samples=2).cpu().numpy()
                for i in range(b):
                    np.random.shuffle(output_pos_max[i])
                output_pos_max = torch.from_numpy(output_pos_max).cuda()
            else:
                output_pos_max = torch.from_numpy(np.random.randint(0,8,size=(36, 2))).cuda()
        else:
            imgs_copy = torch.cat([imgs_copy, imgs_copy], dim = 0)
            k = 2
            pos = torch.empty([b, 2])
            output_pos = self.G(imgs,features)
            pos_probs = F.softmax(output_pos, dim=-1)
            log_pos_probs = F.log_softmax(output_pos, dim=-1)
            pos_entropy = -(log_pos_probs * pos_probs).mean()
            # output_pos_max = pos_probs.multinomial(num_samples=2)
            output_pos_max = torch.from_numpy(np.random.randint(0,8,size=(36, 2))).cuda()

        for j in range(k):
            row_x = torch.floor(output_pos_max[:,j] / 2)
            row_y = output_pos_max[:,j] % 2
            pos_x = (row_x + 1) * 77 - 1
            pos_y = (row_y + 1) * 43 - 1
            area = imgs.size()[2] * imgs.size()[3]

            for i in range(imgs.size()[0]):
                choice_pos = torch.zeros(2)
                choice_pos[0] = pos_probs[i, 2*output_pos_max[i, j]]
                choice_pos[1] = pos_probs[i, 2*output_pos_max[i, j] + 1]
                output_range_max = choice_pos.multinomial(num_samples=1).cuda()
                num_size[output_range_max] = num_size[output_range_max] + 1
                pos[i, j] = pos_probs[i][2*output_pos_max[i, j] + output_range_max]
                for attempt in range(10000000000):
                    target_area = (output_range_max + 2) * area / 12
                    # target_area = 2 * area / 12
                    aspect_ratio = random.uniform(0.8, 1.25)
                    length_x = int(round(math.sqrt(target_area * aspect_ratio)))
                    length_y = int(round(math.sqrt(target_area / aspect_ratio)))
            
                    if length_y < imgs.size()[3] and length_x < imgs.size()[2]:
                        height = pos_x[i]
                        width = pos_y[i]
                        lenx = length_x
                        leny = length_y

                        half_lenx = int(lenx / 2)
                        half_leny = int(leny / 2)

                        if(height - half_lenx < 0):
                            staj = 0
                        else:
                            staj = int(height - half_lenx)

                        if(height + half_lenx > imgs.size()[2]):
                            finj = imgs.size()[2]
                        else:
                            finj = int(height + half_lenx)


                        if(width - half_leny < 0):
                            stak = 0
                        else:
                            stak = int(width - half_leny)

                        if(width + half_leny > imgs.size()[3]):
                            fink = imgs.size()[3]
                        else:
                            fink = int(width + half_leny)
 
                        mask[i, 0, staj:finj, stak:fink] = 0
                        imgs_copy[i + j * b, 0, staj:finj, stak:fink] = random.uniform(0,1)
                        imgs_copy[i + j * b, 1, staj:finj, stak:fink] = random.uniform(0,1)
                        imgs_copy[i + j * b, 2, staj:finj, stak:fink] = random.uniform(0,1)

                        break
        if(epoch >= 0 and train_or_test == 'test'):
            np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/pos/" + 'pos.txt', output_pos_max.cpu().numpy())
            np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/rang_pro/" +'final.txt', num_size.cpu().numpy())
            np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/pos_pro/" +'pos_pro.txt', pos_probs.detach().cpu().numpy())
            # np.savetxt("/mnt/data/qzf/Paritalreid/rang/" +'rang.txt', pos_entropy.detach().cpu().numpy())
        if train_or_test == 'train':
            return imgs_copy, pos.cuda(), pos_entropy.cuda()

        else:
            return imgs_copy

    def set_requires_grad(self,net: nn.Module, mode=True):
        for p in net.parameters():
            p.requires_grad_(mode)



    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):

        
        losses = AverageMeter()
        losses_x1 = AverageMeter()
        losses_x2_fake = AverageMeter()
        losses_x2_true = AverageMeter()
        losses_x3 = AverageMeter()
        losses_x4 = AverageMeter()
        losses_mask = AverageMeter()
        losses_cont = AverageMeter()
        # losses_piddis = AverageMeter()
        # losses_complement = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        # func=nn.Sigmoid()
        loss_new=torch.nn.MSELoss()
        # pos = torch.zeros([4, 50, 2])
        # loss_distance = torch.zeros([4, 50, 2], requires_grad = False)
        # pos_entropy = torch.zeros([4])
        num = 0
        # adversarial_loss = torch.nn.BCELoss()
        self.model.train()
        self.G.train()
        if (epoch+1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            # open_specified_layers(self.model, open_layers)
            open_all_layers(self.model)
            # open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        

        loss_l1=torch.nn.L1Loss(reduction='mean')
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)
            imgs, pids,img_paths,maskimages = self._parse_data_for_train(data)
            b = imgs.size()[0]
            maskimages=torch.sum(maskimages,dim=1).cuda()
            maskimages[maskimages<1]=0
            maskimages[maskimages>2]=0
            maskimages[maskimages>0]=1

            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            if batch_idx % 220 == 0:
                imgs_fix = imgs
                pids_fix = pids
                features = self.model(imgs_fix, return_featuremaps=True)
                img_GAN, pos, pos_entropy = self.perturb(imgs_fix, features, 'train', epoch = epoch)
                output1, output2 = self.model(img_GAN[0:b], img_GAN[b:2*b], state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
                loss_controller_1 = self._compute_loss(self.criterion3, output1, pids_fix)
                loss_controller_2 = self._compute_loss(self.criterion3, output2, pids_fix)
                loss_cont = torch.stack([loss_controller_1, loss_controller_2], dim = 1).detach()
                loss_average = torch.zeros(b, 2).cuda()
                loss_average = loss_average * 0.95 + loss_cont * 0.05
                continue

            #img_GAN, mask_2 = self.perturb(imgs, 'train')
            # newimage= self.model(imgs, epoch=epoch,segmentmask=maskimages,returnflag=True)
            # if batch_idx == 0:
            #     imgs_fix = imgs
            #     pids_fix = pids

            # newimages=newimage[0].unsqueeze(0)
            # for i in range(len(newimage)-1):
            #     newimages=torch.cat([newimages,newimage[i+1].unsqueeze(0)],dim=0)

            # newimage=newimages.float()
            features = self.model(imgs, return_featuremaps=True)
            img_GAN = self.perturb(imgs, features, 'test', epoch = epoch)
            # img_GAN_dis, _, _, _, _ = self.perturb1(img_GAN.detach(), 'train_1')
            output1, output2, output3 = self.model(img_GAN[0:b], img_GAN[b:2*b], epoch=epoch,segmentmask=maskimages.cuda())
            #pids=torch.cat([pids,pids[:int(pids.shape[0]/2)]],0)
            #print(epoch)
            # if epoch >= 0:
            #     for i in range(imgs.shape[0]):
            #         image = Image.open(img_paths[i]) 
            #         image=image.resize((128,384))
            #         image = np.array(image, dtype=np.float32)
            #         # m1 = mask_1[i].reshape((384,128)).cpu().detach().numpy()
            #         mask1=mask[i].reshape((384,128,1))
            #         mask1=torch.cat([mask1,mask1,mask1],dim=2).detach()
            #         # m1 = output_pos_max
            #         # m2 = output_range_max
            #         # m3 = output_pos
            #         # m4 = output_range
            #         # mask2=mask_2[i].reshape((384,128,1))
            #         # mask2=torch.cat([mask2,mask2,mask2],dim=2).detach()
            #         maskimage1=mask1.cpu().numpy()*image
            #         # maskimage1=maskimage1.transpose((1,2,0))
            #         # maskimage2=mask2.cpu().numpy()*image
            #         # #maskimage2=maskimage2.transpose((1,2,0))
            #         # #grid_img = 255 * np.ones((384, 128, 3), dtype=np.uint8)
            #         # #grid_img[:, :, :] = maskimage1
            #         # #grid_img1 = 255 * np.ones((384, 128, 3), dtype=np.uint8)
            #         # #grid_img1[:, :, :] = maskimage2
            #         # #with open("/media/tongji/data/typ/Partialreid/maskimage/"+str(self.numid)+'.txt', 'w') as outfile:
            #         # #    for slice_2d in maskimage1:
            #         # #        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')

            #         # #with open("/media/tongji/data/typ/Partialreid/originalimage/"+str(self.numid)+'.txt', 'w') as outfile:
            #         # #    for slice_2d in maskimage2:
            #         # #        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')
            #         # print(maskimage1.shape)
            #         im1 = Image.fromarray(np.uint8(maskimage1))
            #         # im2 = Image.fromarray(np.uint8(maskimage2))

            #         im1.save("/mnt/data/qzf/Paritalreid/maskimage/"+str(self.numid)+'.jpg')
            #         # im2.save("/media/tongji/data/qzf/Paritalreid-1/originalimage/"+str(self.numid)+'.jpg')
            #         # np.savetxt("/media/tongji/data/qzf/Paritalreid-1/pos/" + str(i)+'.txt', m1.cpu().numpy())
            #         # np.savetxt("/media/tongji/data/qzf/Paritalreid-1/rang/" + str(i)+'.txt', m2.cpu().numpy())
            #         # np.savetxt("/media/tongji/data/qzf/Paritalreid-1/pos_pro/" + str(i)+'.txt', m3.detach().cpu().numpy())
            #         # np.savetxt("/media/tongji/data/qzf/Paritalreid-1/rang_pro/" + str(i)+'.txt', m4.detach().cpu().numpy())

            #         #print(True)
            #         #cv2.imwrite(osp.join("/media/tongji/data/typ/Partialreid/maskimage/", str(self.numid)+'.jpg'), grid_img)
            #         #cv2.imwrite(osp.join("/media/tongji/data/typ/Partialreid/originalimage/", str(self.numid)+'.jpg'), grid_img1)

            #         self.numid+=1
            #         if self.numid==800:
            #             self.numid=0

            # imgs_color=imgs.clone()
            # img_GAN_clone = img_GAN.clone()
            # imgs_color=self.color_erase(imgs_color,erase_x,erase_y)
            # imgs_color_GAN=self.color_erase(img_GAN_clone,erase_x,erase_y)
            # output1_color, output2_color, output3_color,output4_color,_,_,_= self.model(imgs_color,imgs_color_GAN,epoch=epoch,segmentmask=maskimages)

            

            #loss_l1new=loss_l1(imgs,newimages)




            loss_x1 = self._compute_loss(self.criterion1, output1, pids)

            loss_x2_true = self._compute_loss(self.criterion1, output2, pids)
            loss_x3 = self._compute_loss(self.criterion1, output3, pids)
            # loss_x4 = self._compute_loss(self.criterion1, output4, pids)
            # loss_x5 = self._compute_loss(self.criterion1, output5, pids)
            # mask_1 = mask_1 * 1.0
            loss_distill=(loss_kd(output1,pids,output2.detach()) + loss_kd(output2,pids,output1.detach()))
            # loss_self=(loss_kd(output1_color,pids,output1.detach())+loss_kd(output2_color,pids,output2.detach())+loss_kd(output3_color,pids,output3.detach())+loss_kd(output4_color,pids,output4.detach()))
            loss_x = torch.stack([loss_x1, loss_x2_true, loss_x3], dim=0)
            loss_w = F.softmax(loss_x, dim=-1)
            loss = torch.sum(loss_w * loss_x) + loss_distill
            
            self.optimizer_m.zero_grad()
            loss.backward()
            self.optimizer_m.step()

            features = self.model(imgs_fix, return_featuremaps=True)
            img_GAN, pos, pos_entropy = self.perturb(imgs_fix, features, 'train', epoch = epoch)
            output1, output2 = self.model(img_GAN[0:b], img_GAN[b:2*b], state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
            loss_controller_1 = self._compute_loss(self.criterion3, output1, pids_fix)
            loss_controller_2 = self._compute_loss(self.criterion3, output2, pids_fix) 

            loss_cont = torch.stack([loss_controller_1, loss_controller_2], dim = 1).detach()
            
            loss_distance = loss_cont - loss_average
            loss_distance = (loss_distance - loss_distance.mean()) / (torch.std(loss_distance) + 1e-4)
            # loss_controller[batch_idx % 3] = (loss_distance.detach() * pos).mean() - 1e-4 * pos_entropy
            loss_average = loss_average * 0.95 + loss_cont * 0.05

            loss_controller = (loss_distance.detach() * pos).mean() - 0.1 * pos_entropy
            self.optimizer_G.zero_grad()
            loss_controller.backward()
            self.optimizer_G.step()

            losses_cont.update(loss_controller.item(),pids_fix.size(0))
                # pos = torch.zeros([4, 50, 2])
                # loss_distance = torch.zeros([4, 50, 2], requires_grad = False)
                # pos_entropy = torch.zeros([4])

            # print(self.G.state_dict()['module.layer3.2.bn1.weight'])
            # print(self.G.state_dict()['module.linear_pos.weight'])
            # loss_x2_fake = self._compute_loss(self.criterion2, output2, pids)
            # # loss_x2_fake = self._compute_loss(self.criterion2, output2.detach(), pids)
            # loss_mask = loss_new(mask_1.detach().cuda(),mask_2.cuda()) * 10
            # loss_bran2 = loss_mask + loss_x2_fake
            
            # self.optimizer_G.zero_grad()
            # loss_bran2.backward()
            # self.optimizer_G.step()
            
            
            # losses_mask.update(loss_mask.item(),pids.size(0))
            batch_time.update(time.time() - end)
            losses.update(loss.item(), pids.size(0))
            losses_x1.update(loss_x1.item(), pids.size(0))
            # losses_x1.update(loss_x2_true.item().item(), pids.size(0))
            # losses_x2_fake.update(loss_x2_fake.item(), pids.size(0))
            losses_x2_true.update(loss_x2_true.item(), pids.size(0))
            losses_x3.update(loss_x3.item(), pids.size(0))
            # losses_x4.update(loss_x4.item(), pids.size(0))

            # losses_self.update(loss_self.item(), pids.size(0))



            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'loss_cont {loss_cont.val:.4f} ({loss_cont.avg:.4f})\t'
                      'Loss_x1 {loss_x1.val:.4f} ({loss_x1.avg:.4f})\t'
                    #   'Loss_x2_fake {loss_x2_fake.val:.4f} ({loss_x2_fake.avg:.4f})\t'
                      'Loss_x2_true {loss_x2_true.val:.4f} ({loss_x2_true.avg:.4f})\t'
                      'Loss_x3 {loss_x3.val:.4f} ({loss_x3.avg:.4f})\t'  
                    #   'Loss_x4 {loss_x4.val:.4f} ({loss_x4.avg:.4f})\t'
                      'Loss_mask {loss_mask.val:.4f} ({loss_mask.avg:.4f})\t'
                    #   'Loss_self {loss_self.val:.4f} ({loss_self.avg:.4f})\t'  

                      #'Loss_l1 {loss_l1:.4f} ({loss_l1:.4f})\t'  

                    #   'Loss_piddis {loss_piddis.val:.4f} ({loss_piddis.avg:.4f})\t'  
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      loss=losses,
                      loss_cont=losses_cont,
                      loss_x1=losses_x1,
                    #   loss_x2_fake=losses_x2_fake,
                      loss_x2_true=losses_x2_true,
                      loss_x3=losses_x3,
                    #   loss_x4=losses_x4,
                      loss_mask = losses_mask,
                    #   loss_self=losses_self,

                      #loss_l1=loss_l1new,
                    #   loss_piddis=losses_piddis,
                      lr=self.optimizer_m.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', losses.avg, n_iter)
                self.writer.add_scalar('Train/Loss_bran2', losses_cont.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x1', losses_x1.avg, n_iter)
                # self.writer.add_scalar('Train/Loss_x2_fake', losses_x2_fake.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x2_true', losses_x2_true.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x3', losses_x3.avg, n_iter)
                self.writer.add_scalar('Train/Loss_mask', losses_mask.avg, n_iter)
                
                #self.writer.add_scalar('Train/Loss_x4', losses_x4.avg, n_iter)

                #self.writer.add_scalar('Train/Loss_fake', losses_fake.avg, n_iter)
                #self.writer.add_scalar('Train/Loss_m', losses_m.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer_m.param_groups[0]['lr'], n_iter)

                # self.writer.add_scalars('Train',{'loss':losses.avg,'Loss_x1':losses_x1.avg, 'Loss_x2':losses_x2.avg},n_iter)
            
            end = time.time()
        if self.scheduler_m is not None:
            self.scheduler_m.step()
            # self.scheduler_G.step()
def loss_kd(outputs, labels, teacher_outputs):
    """
    loss function for Knowledge Distillation (KD)
    """
    alpha = 0.95
    T = 6
    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha)*loss_CE + alpha*D_KL

    return KD_loss

