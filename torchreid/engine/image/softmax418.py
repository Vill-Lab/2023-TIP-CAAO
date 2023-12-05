from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time
import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchreid.engine import Engine
from torchreid.losses import CrossEntropyLoss,TripletLoss
from torchreid.utils import AverageMeter, open_specified_layers, open_all_layers,read_image
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

    def __init__(self, datamanager, model, optimizer, scheduler=None, use_gpu=True,
                 label_smooth=True):
        super(ImageSoftmaxEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)
        

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


        self.criterion = CrossEntropyLoss(
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
        loss    = 0.0001 * kl_loss
        return loss

    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):

        
        losses = AverageMeter()
        losses_x1 = AverageMeter()
        losses_x2 = AverageMeter()
        losses_x3 = AverageMeter()
        losses_piddis = AverageMeter()
        losses_x4 = AverageMeter()
        losses_x5 = AverageMeter()
        losses_x6 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        func=nn.Sigmoid()
        loss_new=torch.nn.MSELoss()
        self.model.train()
        if (epoch+1) <= fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()


        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time() - end)
            imgs, pids,maskimages = self._parse_data_for_train(data)
            maskimages=torch.sum(maskimages,dim=1)
            maskimages[maskimages<1]=0
            maskimages[maskimages>2]=0
            maskimages[maskimages>0]=1

            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            
            self.optimizer.zero_grad()
            output1, output2, output3,output4,output1_new, output2_new, output3_new,output4_new= self.model(imgs,epoch=epoch,segmentmask=maskimages)



            


            loss_x1 = self._compute_loss(self.criterion, output1, pids)
            loss_x2 = self._compute_loss(self.criterion, output2, pids)
            loss_x3 = self._compute_loss(self.criterion, output3, pids)
            
            loss_distill=(loss_kd(output1_new,pids,output1.detach())+loss_kd(output2_new,pids,output2.detach())+loss_kd(output3_new,pids,output3.detach())+loss_kd(output4_new,pids,output4.detach()))#+loss_kd(output4_new,pids,output4.detach()))
            
            loss_x4 = self._compute_loss(self.criterion, output4, pids)#(self.kl_loss(mu1,logvar1)+self.kl_loss(mu2,logvar2)+self.kl_loss(mu1_new,logvar1_new)+self.kl_loss(mu2_new,logvar2_new))

            loss_x = torch.stack([loss_x1, loss_x2, loss_x4,loss_x3], dim=0)
            loss_w = F.softmax(loss_x, dim=-1)

            loss = torch.sum(loss_w * loss_x)+loss_distill
            loss.backward()

            self.optimizer.step()



            batch_time.update(time.time() - end)
            losses.update(loss.item(), pids.size(0))
            losses_x1.update(loss_x1.item(), pids.size(0))
            losses_x2.update(loss_x2.item(), pids.size(0))
            losses_x3.update(loss_x3.item(), pids.size(0))
            losses_x4.update(loss_x4.item(), pids.size(0))
            losses_piddis.update(loss_distill.item(), pids.size(0))




            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_x1 {loss_x1.val:.4f} ({loss_x1.avg:.4f})\t'
                      'Loss_x2 {loss_x2.val:.4f} ({loss_x2.avg:.4f})\t'
                      'Loss_x3 {loss_x3.val:.4f} ({loss_x3.avg:.4f})\t'  
                      'Loss_x4 {loss_x4.val:.4f} ({loss_x4.avg:.4f})\t'  

                      'Loss_piddis {loss_piddis.val:.4f} ({loss_piddis.avg:.4f})\t'  
                      #'Loss_featdis {loss_featdis.val:.4f} ({loss_featdis.avg:.4f})\t'    

                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                      epoch+1, max_epoch, batch_idx+1, num_batches,
                      loss=losses,
                      loss_x1=losses_x1,
                      loss_x2=losses_x2,
                      loss_x3=losses_x3,
                      loss_x4=losses_x4,
                      loss_piddis=losses_piddis,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                    )
                )

            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Loss', losses.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x1', losses_x1.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x2', losses_x2.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x3', losses_x3.avg, n_iter)
                self.writer.add_scalar('Train/Loss_x4', losses_x4.avg, n_iter)

                #self.writer.add_scalar('Train/Loss_fake', losses_fake.avg, n_iter)
                #self.writer.add_scalar('Train/Loss_m', losses_m.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)

                # self.writer.add_scalars('Train',{'loss':losses.avg,'Loss_x1':losses_x1.avg, 'Loss_x2':losses_x2.avg},n_iter)
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
            #self.scheduler_fake.step()
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