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
import torchvision.transforms as T
import torch
from torch import FloatTensor
import scipy.io as io
import os
import cv2
import _pickle as cPickle
class RandomErasing_Background(object):
    def __init__(self, EPSILON = 0.5, root='/home/v-zefanqu/v-zefanqu/CAAO/Paritalreid/reid-data/P-DukeMTMC/crop_backgrounds/crop_backgrounds'):
        self.EPSILON = EPSILON
        self.root = root
        self.occ_imgs = os.listdir(self.root)

        for img in self.occ_imgs:
            if not img.endswith('.jpg'):
                self.occ_imgs.remove(img)

        self.len = len(self.occ_imgs)
       
    def __call__(self, img):            
        index = random.randint(0, self.len-1)
        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')

        h, w = img.size()[1], img.size()[2] 
        h_, w_ = occ_img.height, occ_img.width

        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
        normalize = T.Normalize(mean=norm_mean, std=norm_std)
        ratio = h_ / w_
        if ratio > 2:
            re_size = (h, random.randint(w//8, w//4))
            # re_size = (h, random.randint(w//4, w//2))
            function = T.Compose([
                # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize,
            ])
            occ_img = function(occ_img)
        else:
            re_size = (random.randint(h//4, h//2), random.randint(w//2, w))
            # re_size = (random.randint(h//4, h//2), w)
            function = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                normalize,
            ])
            occ_img = function(occ_img)

        h_, w_ = re_size[0], re_size[1]

        index_ = random.sample(range(4), 3)
        # points = [(0, 0), (0, w), (h, 0), (h, w)]
        imgs_copy = copy.deepcopy(img).unsqueeze(0)
        imgs_copy = torch.cat([imgs_copy, imgs_copy, imgs_copy], dim = 0)
        for m in range(3):
            if index_[m]==0:
                imgs_copy[m, :, 0:h_, 0:w_] = occ_img
            elif index_[m]==1:
                imgs_copy[m, :, 0:h_, w-w_:w] = occ_img
            elif index_[m]==2:
                imgs_copy[m, :, h-h_:h, 0:w_] = occ_img
            else:
                imgs_copy[m, :, h-h_:h, w-w_:w] = occ_img

        return imgs_copy


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

def get_masks(im_names, all_masks=None, prob_diff=None):
    """Get a batch of masks for the input batch.
    Returns:
        masks: numpy array with shape [len(im_names), im_h, im_w]
    """
    masks = []
    m_key = 'mirrored'
    for name in im_names:
        if np.random.uniform() < 0.75:
            p_d = prob_diff[m_key][name]
            p_d = p_d.flatten()
            ind1 = np.argsort(- p_d)[0]
            ind2 = np.argsort(- p_d)[1]
            ind3 = np.argsort(- p_d)[2]
            mask1 = torch.from_numpy(all_masks[ind1]).cuda().unsqueeze(0)
            mask2 = torch.from_numpy(all_masks[ind2]).cuda().unsqueeze(0)
            mask3 = torch.from_numpy(all_masks[ind3]).cuda().unsqueeze(0)
            mask = torch.cat([mask1, mask2, mask3], dim = 0)
        else:
            mask = torch.ones(3, 384, 128).cuda()
            # mask_new.append(mask)
        # mask = torch.cat(mask_new, dim = 0)
        masks.append(mask)
    masks = torch.stack(masks)
    return masks

def blur_prob_diff(prob_diff):
    """Blur each prob_diff map (with shape [num_h_pos, num_w_pos]) using a 3x3 kernel.
    Whether it's effective is not analysed -- it's just intuitively applied."""
    return {m_key: {im_name: cv2.blur(p_d_, (3, 3))
                    for im_name, p_d_ in p_d.items()}
            for m_key, p_d in prob_diff.items()}

def load_pickle(path):
    """Check and load pickle object.
    According to this post: https://stackoverflow.com/a/41733927, cPickle and
    disabling garbage collector helps with loading speed."""
    assert osp.exists(path), "File not exists: {}".format(path)
    # gc.disable()
    with open(path, 'rb') as f:
        ret = cPickle.load(f)
    # gc.enable()
    print('Loaded pickle file {}'.format(path))
    return ret

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

        self.FED_erase = RandomErasing_Background()
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

    def perturb(self, imgs, mask = None, train_or_test='test', epoch = 60, epsilon = 0.1):
        b = imgs.size()[0]
        imgs_copy = copy.deepcopy(imgs)
        flip = T.RandomHorizontalFlip(p=1.0)
        if train_or_test == 'train':
            imgs_copy = torch.cat([imgs_copy, imgs_copy, imgs_copy], dim = 0)
            k = 3
            # pos = torch.empty([b, k])
            # output_pos = self.G(imgs, features)
            # # output_pos = np.random.rand(50,30)
            # # output_pos = torch.from_numpy(output_pos)
            # pos_probs = F.softmax(output_pos, dim=-1)
            # log_pos_probs = F.log_softmax(output_pos, dim=-1)
            # pos_entropy = -(log_pos_probs * pos_probs).mean()
            # output_pos_small = pos_probs[:,::2]
            # output_pos_large = pos_probs[:,1::2]
            # output_position = output_pos_small + output_pos_large

            # randnum = random.random()
            # # if randnum <= 0.9:
            # output_pos_max = output_position.multinomial(num_samples=k).cpu().numpy()
            # for i in range(b):
            #     np.random.shuffle(output_pos_max[i])
            # # np.random.shuffle(output_pos_max)
            # output_pos_max = torch.from_numpy(output_pos_max).cuda()
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
        for i in range(imgs.size()[0]):
            # row_x = torch.floor(output_pos_max[:,j] / 3)
            # row_y = output_pos_max[:,j] % 3
            # pos_x = (row_x + 1) * 55 - 1
            # pos_y = (row_y + 1) * 32 - 1
            # area = imgs.size()[2] * imgs.size()[3]
            # enhanced_image = self.FED_erase(imgs[i])
            flip_prob = random.uniform(0, 1)
            for j in range(k):
                imgs_copy[i + j * b] = imgs_copy[i + j * b] * mask[i, j]
                if flip_prob >= 0.5:
                    imgs_copy[i + j * b] = flip(imgs_copy[i + j * b])
                # if train_or_test == 'train':
                #     choice_pos = torch.zeros(2)
                #     choice_pos[0] = pos_probs[i, 2*output_pos_max[i, j]]
                #     choice_pos[1] = pos_probs[i, 2*output_pos_max[i, j] + 1]
                #     output_range_max = choice_pos.multinomial(num_samples=1).cuda()
                # else:
                #     output_range_max = random.randint(0, 1)

                # num_size[output_range_max] = num_size[output_range_max] + 1
                # pos[i, j] = log_pos_probs[i][2*output_pos_max[i, j] + output_range_max]
                # for attempt in range(10000):
                #     target_area = (output_range_max + 1) * area / 6
                #     # target_area = 2 * area / 12
                #     aspect_ratio = random.uniform(0.8, 1.25)
                #     length_x = int(round(math.sqrt(target_area) * aspect_ratio))
                #     length_y = int(round(math.sqrt(target_area) / aspect_ratio))
            
                #     if length_y < imgs.size()[3] and length_x < imgs.size()[2]:
                #         height = pos_x[i]
                #         width = pos_y[i]
                #         lenx = length_x
                #         leny = length_y

                #         half_lenx = int(lenx / 2)
                #         half_leny = int(leny / 2)

                #         if(height - half_lenx < 0):
                #             staj = 0
                #         else:
                #             staj = int(height - half_lenx)

                #         if(height + half_lenx > imgs.size()[2]):
                #             finj = imgs.size()[2]
                #         else:
                #             finj = int(height + half_lenx)


                #         if(width - half_leny < 0):
                #             stak = 0
                #         else:
                #             stak = int(width - half_leny)

                #         if(width + half_leny > imgs.size()[3]):
                #             fink = imgs.size()[3]
                #         else:
                #             fink = int(width + half_leny)
                        
                #         mask[i, 0, staj:finj, stak:fink] = 0
                #         imgs_copy[i + j * b, 0, staj:finj, stak:fink] = random.uniform(0,1)
                #         imgs_copy[i + j * b, 1, staj:finj, stak:fink] = random.uniform(0,1)
                #         imgs_copy[i + j * b, 2, staj:finj, stak:fink] = random.uniform(0,1)

                        # break
        # if(epoch >= 0 and train_or_test == 'train'):
        #     np.savetxt("./pos/pos.txt", output_pos_max.cpu().numpy())
        #     np.savetxt("./pos_pro/pos_pro.txt", pos_probs.detach().cpu().numpy())
        #     # np.savetxt("/mnt/data/qzf/Paritalreid/rang/" +'rang.txt', pos_entropy.detach().cpu().numpy())
        if train_or_test == 'train':
            return imgs_copy

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
        pos = torch.zeros([5, 40, 3])
        loss_distance = torch.zeros([5, 40, 3], requires_grad = False)
        loss_valid = torch.zeros([5, 40, 3], requires_grad = False)
        loss_train_alpha = torch.zeros([5, 40, 3], requires_grad = False)
        loss_valid_alpha = torch.zeros([5, 40, 3], requires_grad = False)
        # imgs_fix = torch.zeros([100, 3, 384, 128])
        # pids_fix = torch.zeros([100], dtype=torch.int64)
        pos_entropy = torch.zeros([5])
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

        prob_diff, all_masks = None, None
        self.prob_diff = load_pickle('/home/v-zefanqu/v-zefanqu/CAAO/Paritalreid/log/Pduke_sw_occlusion_0.1range/prob_diff.pkl')
        self.prob_diff = blur_prob_diff(self.prob_diff)
        print(len(self.prob_diff['mirrored'].keys()))
        self.all_masks = load_pickle('/home/v-zefanqu/v-zefanqu/CAAO/Paritalreid/log/Pduke_sw_occlusion_0.1range/all_masks.pkl')
        
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

            # if batch_idx == 0:
            #     imgs_fix = imgs
            #     pids_fix = pids
            #     continue
            
            #img_GAN, mask_2 = self.perturb(imgs, 'train')
            # newimage= self.model(imgs, epoch=epoch,segmentmask=maskimages,returnflag=True)
            # if batch_idx == 0:
            #     imgs_fix = imgs
            #     pids_fix = pids

            # newimages=newimage[0].unsqueeze(0)
            # for i in range(len(newimage)-1):
            #     newimages=torch.cat([newimages,newimage[i+1].unsqueeze(0)],dim=0)

            # newimage=newimages.float()
            # features = self.model(imgs, return_featuremaps=True)
            # masks = get_masks(img_paths, self.all_masks, self.prob_diff)
            # img_GAN = self.perturb(imgs=imgs, mask = masks, train_or_test='train', epoch = epoch)
            # pos[batch_idx % 5] = pos1
            # pos_entropy[batch_idx % 5] = pos_entropy1
            output1, output2, output3, output4 = self.model(imgs, imgs, imgs, epoch=epoch,segmentmask=maskimages.cuda())
            # output1, output2, output3, output4 = self.model(imgs, imgs, imgs, epoch=epoch,segmentmask=maskimages.cuda())
            loss_x1 = self._compute_loss(self.criterion1, output1, pids)
            loss_x2_true = self._compute_loss(self.criterion1, output2, pids)
            loss_x3 = self._compute_loss(self.criterion1, output3, pids)
            loss_x4 = self._compute_loss(self.criterion1, output4, pids)

            # mask_1 = mask_1 * 1.0
            loss_distill=(loss_kd(output1,pids,output2.detach()) + loss_kd(output2,pids,output1.detach()) + loss_kd(output1,pids,output3.detach()) + loss_kd(output3,pids,output1.detach()) + loss_kd(output2,pids,output3.detach()) + loss_kd(output3,pids,output2.detach()))
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
            
            # output1_train, output2_train, output3_train = self.model(imgs, state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
            # loss_controller_train_1 = self._compute_loss(self.criterion3, output1_train, pids).detach()
            # loss_controller_train_2 = self._compute_loss(self.criterion3, output2_train, pids).detach()
            # loss_controller_train_3 = self._compute_loss(self.criterion3, output3_train, pids).detach()
            
            # loss_cont = torch.stack([loss_controller_train_1, loss_controller_train_2, loss_controller_train_3], dim = 1)
            # # loss_cont_min, _ = torch.min(loss_cont, dim=1, keepdim=True)
            # loss_train_alpha[batch_idx % 5] = loss_cont
            # loss_cont = (loss_cont - loss_cont.mean(dim=1,keepdim = True))
            # loss_cont = F.normalize(loss_cont, p=2, dim=1)
            # loss_distance[batch_idx % 5] = loss_cont
            
            # # for i in range(len(loss_cont)):
            # #     for j in range(loss_cont.size()[1]):
            # #         if loss_cont[i,j] > 0:
            # #             loss_cont[i,j] = loss_cont[i,j] * 1.2
            # # np.savetxt("/media/tongji/data/qzf/Paritalreid-Check/rang_pro/" +'final.txt', loss_cont.detach().cpu().numpy())
            # # loss_controller_1 = (loss_controller_1 - loss_controller_1.mean()) / (torch.std(loss_controller_1) + 1e-4)
            # # loss_controller_2 = (loss_controller_2 - loss_controller_2.mean()) / (torch.std(loss_controller_2) + 1e-4)
        
            # output1_valid, output2_valid, output3_valid = self.model(imgs_fix, state = 'train', epoch=epoch,segmentmask=maskimages.cuda())
            # loss_controller_valid_1 = self._compute_loss(self.criterion3, output1_valid, pids_fix).detach()
            # loss_controller_valid_2 = self._compute_loss(self.criterion3, output2_valid, pids_fix).detach()
            # loss_controller_valid_3 = self._compute_loss(self.criterion3, output3_valid, pids_fix).detach()
            
            # loss_val = torch.stack([loss_controller_valid_1, loss_controller_valid_2, loss_controller_valid_3], dim = 1)
            # loss_valid_alpha[batch_idx % 5] = loss_val
            # # loss_cont_min, _ = torch.min(loss_cont, dim=1, keepdim=True)
            # loss_val = (loss_val - loss_val.mean(dim=1,keepdim = True))
            # loss_val = F.normalize(loss_val, p=2, dim=1)
            # loss_valid[batch_idx % 5] = loss_val
            
            # # self.model(imgs, update_param=True)
            
            # if batch_idx % 5 == 4:
            #     # pos_double = torch.cat([pos,pos], dim = 1)
            #     # alpha = torch.mean(loss_valid.detach()) / torch.mean(loss_distance.detach())
            #     alpha = torch.mean(loss_valid.detach()) / torch.mean(loss_distance.detach())
            #     loss_controller = (loss_distance.detach() * pos).mean() + (loss_valid * alpha * pos).mean() - 1e-4 * pos_entropy.mean()
            #     # loss_controller = (loss_distance.detach() * pos * 5).mean() + (loss_valid * alpha * pos).mean() - 1e-4 * pos_entropy.mean()
            #     self.optimizer_G.zero_grad()
            #     loss_controller.backward()
            #     self.optimizer_G.step()

            #     losses_cont.update(loss_controller.item(),pids.size(0))
            #     pos = torch.zeros([5, 40, 3])
            #     loss_distance = torch.zeros([5, 40, 3], requires_grad = False)
            #     loss_valid = torch.zeros([5, 40, 3], requires_grad = False)
            #     loss_train_alpha = torch.zeros([5, 40, 3], requires_grad = False)
            #     loss_valid_alpha = torch.zeros([5, 40, 3], requires_grad = False)
            #     pos_entropy = torch.zeros([5])

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
    T = 12
    loss_CE = F.cross_entropy(outputs, labels)
    D_KL = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (T * T)
    KD_loss =  (1. - alpha)*loss_CE + alpha*D_KL

    return KD_loss
