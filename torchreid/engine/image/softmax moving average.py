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
import pandas as pd
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

    def perturb(self, imgs, features, train_or_test='test', camid = 0, epoch = 60, epsilon = 0.1):
        b = imgs.size()[0]
        imgs_copy = copy.deepcopy(imgs)
        mask = torch.ones([b, 1, imgs.size()[2], imgs.size()[3]])
        num_size = torch.zeros(2)
        if train_or_test == 'train':
            imgs_copy = torch.cat([imgs_copy, imgs_copy, imgs_copy], dim = 0)
            k = 3
            pos = torch.empty([b, k])
            output_pos = self.G(features)
            # output_pos = np.random.rand(50,30)
            # output_pos = torch.from_numpy(output_pos)
            pos_probs = F.softmax(output_pos, dim=-1)
            log_pos_probs = F.log_softmax(output_pos, dim=-1)
            pos_entropy = -(log_pos_probs * pos_probs).mean()
            output_pos_small = pos_probs[:,::2]
            output_pos_large = pos_probs[:,1::2]
            output_position = output_pos_small + output_pos_large

            randnum = random.random()
            # if randnum <= 0.9:
            output_pos_max = output_position.multinomial(num_samples=k).cpu().numpy()
            for i in range(b):
                np.random.shuffle(output_pos_max[i])
            output_pos_max = torch.from_numpy(output_pos_max).cuda()
            # else:
            #     output_pos_max = torch.from_numpy(np.random.randint(0,10,size=(b, k))).cuda()
        # else:
        #     imgs_copy = torch.cat([imgs_copy, imgs_copy, imgs_copy], dim = 0)
        #     k = 3
        #     pos = torch.empty([b, k])
        #     output_pos = self.G(imgs, features)
        #     pos_probs = F.softmax(output_pos, dim=-1)
        #     log_pos_probs = F.log_softmax(output_pos, dim=-1)
        #     pos_entropy = -(log_pos_probs * pos_probs).mean()
        #     # output_pos_max = pos_probs.multinomial(num_samples=2)
        #     output_pos_max = torch.from_numpy(np.random.randint(0,8,size=(b, k))).cuda()

        for j in range(k):
            row_x = torch.floor(output_pos_max[:,j] / 3)
            row_y = output_pos_max[:,j] % 3
            pos_x = (row_x + 1) * 64 - 1
            pos_y = (row_y + 1) * 32 - 1
            area = imgs.size()[2] * imgs.size()[3]

            for i in range(imgs.size()[0]):
                if train_or_test == 'train':
                    choice_pos = torch.zeros(2)
                    choice_pos[0] = pos_probs[i, 2*output_pos_max[i, j]]
                    choice_pos[1] = pos_probs[i, 2*output_pos_max[i, j] + 1]
                    output_range_max = choice_pos.multinomial(num_samples=1).cuda()
                else:
                    output_range_max = random.randint(0, 1)

                num_size[output_range_max] = num_size[output_range_max] + 1
                pos[i, j] = log_pos_probs[i][2*output_pos_max[i, j] + output_range_max]
                output_pos_max[i, j] = 2*output_pos_max[i, j] + output_range_max
                for attempt in range(10000000000):
                    target_area = (output_range_max + 1) * area / 8
                    # target_area = 2 * area / 12
                    aspect_ratio = random.uniform(0.8, 1.25)
                    length_x = int(round(math.sqrt(target_area) * aspect_ratio))
                    length_y = int(round(math.sqrt(target_area) / aspect_ratio))
            
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
        if(epoch >= 0 and train_or_test == 'train'):
            np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/pos/" + 'pos.txt', output_pos_max.cpu().numpy())
            np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/pos_pro/" +'pos_pro.txt', pos_probs.detach().cpu().numpy())
            # np.savetxt("/mnt/data/qzf/Paritalreid/rang/" +'rang.txt', pos_entropy.detach().cpu().numpy())
        if train_or_test == 'train':
            return imgs_copy, pos.cuda(), pos_entropy.cuda(), output_pos_max.cuda()

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
        # pos = torch.zeros([5, 50, 3])
        # loss_distance = torch.zeros([5, 50, 3], requires_grad = False)
        # loss_valid = torch.zeros([5,3], requires_grad = False)
        # pos_entropy = torch.zeros([5])
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
            imgs, pids, camids, img_paths,maskimages = self._parse_data_for_train(data)
            b = imgs.size()[0]
            maskimages=torch.sum(maskimages,dim=1).cuda()
            maskimages[maskimages<1]=0
            maskimages[maskimages>2]=0
            maskimages[maskimages>0]=1

            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            if batch_idx == 0:
                imgs_fix = imgs
                pids_fix = pids
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
            img_GAN, pos, pos_entropy, output_pos_max = self.perturb(imgs, features, 'train', epoch = epoch)

            # img_GAN_dis, _, _, _, _ = self.perturb1(img_GAN.detach(), 'train_1')
            output1, output2, output3, output4 = self.model(img_GAN[0:b], img_GAN[b:2*b], img_GAN[2*b:3*b], epoch=epoch,segmentmask=maskimages.cuda())

            loss_x1 = self._compute_loss(self.criterion1, output1, pids)
            loss_x2_true = self._compute_loss(self.criterion1, output2, pids)
            loss_x3 = self._compute_loss(self.criterion1, output3, pids)
            loss_x4 = self._compute_loss(self.criterion1, output4, pids)
            loss_distill=(loss_kd(output1,pids,output2.detach()) + loss_kd(output2,pids,output1.detach()) + loss_kd(output1,pids,output3.detach()) + loss_kd(output3,pids,output1.detach()) + loss_kd(output2,pids,output3.detach()) + loss_kd(output3,pids,output2.detach()))
            # mask_1 = mask_1 * 1.0
            # loss_self=(loss_kd(output1_color,pids,output1.detach())+loss_kd(output2_color,pids,output2.detach())+loss_kd(output3_color,pids,output3.detach())+loss_kd(output4_color,pids,output4.detach()))
            loss_x = torch.stack([loss_x1, loss_x2_true, loss_x3, loss_x4], dim=0)
            loss_w = F.softmax(loss_x, dim=-1)
            loss = torch.sum(loss_w * loss_x) + loss_distill
            
            
            # output1_train_pre, output2_train_pre = self.model(imgs_fix, state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
            # loss_controller_train_1_pre = self._compute_loss(self.criterion1, output1_train_pre, pids_fix)
            # loss_controller_train_2_pre = self._compute_loss(self.criterion1, output2_train_pre, pids_fix)
               
            self.optimizer_m.zero_grad()
            loss.backward()
            self.optimizer_m.step()
            
            output1_train, output2_train, output3_train = self.model(imgs, state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
            loss_controller_train_1 = self._compute_loss(self.criterion3, output1_train, pids).detach()
            loss_controller_train_2 = self._compute_loss(self.criterion3, output2_train, pids).detach()
            loss_controller_train_3 = self._compute_loss(self.criterion3, output3_train, pids).detach()
            
            loss_cont = torch.stack([loss_controller_train_1, loss_controller_train_2, loss_controller_train_3], dim = 1)
            # loss_cont_min, _ = torch.min(loss_cont, dim=1, keepdim=True)
            loss_cont = (loss_cont - loss_cont.mean(dim=1,keepdim = True))
            loss_cont = F.normalize(loss_cont, p=2, dim=1)
            
            output1_valid, output2_valid, output3_valid = self.model(imgs_fix, state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
            loss_controller_valid_1 = self._compute_loss(self.criterion3, output1_valid, pids_fix)
            loss_controller_valid_2 = self._compute_loss(self.criterion3, output2_valid, pids_fix)
            loss_controller_valid_3 = self._compute_loss(self.criterion3, output3_valid, pids_fix)
            loss_val = torch.stack([loss_controller_valid_1, loss_controller_valid_2, loss_controller_valid_3], dim = 1)
            loss_val = (loss_val - loss_val.mean(dim=1,keepdim = True))
            loss_val = F.normalize(loss_val, p=2, dim=1)
            self.replayer.store(features, output_pos_max.detach(), pos.detach(), loss_cont.detach(), loss_val.detach())
            
            # for i in range(len(loss_cont)):
            #     for j in range(loss_cont.size()[1]):
            #         if loss_cont[i,j] > 0:
            #             loss_cont[i,j] = loss_cont[i,j] * 1.2
            # np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/rang_pro/" +'final.txt', loss_cont.detach().cpu().numpy())
            # loss_controller_1 = (loss_controller_1 - loss_controller_1.mean()) / (torch.std(loss_controller_1) + 1e-4)
            # loss_controller_2 = (loss_controller_2 - loss_controller_2.mean()) / (torch.std(loss_controller_2) + 1e-4)
        
            # output1_valid, output2_valid = self.model(imgs_fix, state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
            # loss_x1_valid = self._compute_loss(self.criterion1, output1_valid, pids_fix).detach()
            # loss_x2_valid = self._compute_loss(self.criterion1, output2_valid, pids_fix).detach()
    
            # loss_valid = torch.Tensor([loss_x1_valid - loss_x1_valid_average, loss_x2_valid - loss_x2_valid_average]).cuda().detach()
            # loss_x1_valid_average = 0.95 * loss_x1_valid_average + 0.05 * loss_x1_valid
            # loss_x2_valid_average = 0.95 * loss_x2_valid_average + 0.05 * loss_x2_valid
            # loss_distance = loss_cont - loss_average
            # loss_distance = (loss_distance - loss_distance.mean()) / (torch.std(loss_distance) + 1e-4)
            # loss_average = loss_average * 0.95 + loss_cont * 0.05
            # loss_controller = (loss_cont.detach() * pos).mean() + (loss_valid.detach() * pos).mean() - 0.1 * pos_entropy
            if batch_idx % 20 == 19:
                for step in range(4):
                    features, actions, prob, rewards, valid = self.replayer.sample(300)
                    prob_new = torch.zeros(features.size()[0], 3).cuda()
                    features = features.cuda()
                    actions = actions.cuda()
                    prob = prob.cuda().detach()
                    rewards = rewards.cuda().detach()
                    valid = valid.cuda().detach()
                    output_pos = self.G(features)
                    pos_probs = F.softmax(output_pos, dim=-1)
                    log_pos_probs = F.log_softmax(output_pos, dim=-1)
                    pos_entropy = -(log_pos_probs * pos_probs).mean()
                    for i in range(features.size()[0]):
                        for j in range(3):
                            prob_new[i,j] = log_pos_probs[i][int(actions[i,j])]
                    ratio = torch.exp(prob_new - prob)
                    alpha = torch.mean(rewards.detach()) / torch.mean(valid.detach())
                    L1 = (ratio * (rewards * 20 + valid * alpha)).mean()
                    L2 = (torch.clamp(ratio, 0.8, 1.2) * (rewards * 20 + valid * alpha)).mean()
                    loss_controller = torch.max(L1,L2) - 1e-3 * pos_entropy
                    self.optimizer_G.zero_grad()
                    loss_controller.backward()
                    self.optimizer_G.step()

                self.replayer.clear()

            losses.update(loss.item(), pids.size(0))
            losses_x1.update(loss_x1.item(), pids.size(0))
            # losses_x1.update(loss_x2_true.item().item(), pids.size(0))
            # losses_x2_fake.update(loss_x2_fake.item(), pids.size(0))
            losses_x2_true.update(loss_x2_true.item(), pids.size(0))
            losses_x3.update(loss_x3.item(), pids.size(0))
            # losses_x4.update(loss_x4.item(), pids.size(0))

            # losses_self.update(loss_self.item(), pids.size(0))

            # print(self.G.state_dict()['module.linear_pos.linear1.weight'])
            # print(self.G.state_dict()['module.linear_pos.weight'])

            batch_time.update(time.time() - end)



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
