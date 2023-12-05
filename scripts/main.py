import sys
import os
import os.path as osp
import time
import argparse
# sys.path.append("/mnt/Data/qzf/Paritalreid/")
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
from default_config import (
    get_default_config, imagedata_kwargs, videodata_kwargs,
    optimizer_kwargs, lr_scheduler_kwargs, engine_run_kwargs
)
import torchreid
from torchreid.utils import (
    Logger, set_random_seed, check_isfile, resume_from_checkpoint,
    load_pretrained_weights, compute_model_complexity, collect_env_info
)

def build_datamanager(cfg):
    if cfg.data.type == 'image':
        return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, G, optimizer_m, scheduler_m, optimizer_G, scheduler_G = None):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                G,
                optimizer_m,
                optimizer_G,
                scheduler_m=scheduler_m,
                scheduler_G=scheduler_G,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )
    
    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )
        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-file', type=str, default='configs/bfe.yaml', help='path to config file')
    parser.add_argument('-s', '--sources', type=str, nargs='+', help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, nargs='+', help='target datasets (delimited by space)')
    parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation')
    parser.add_argument('--root', type=str, default='reid-data', help='path to data root')
    parser.add_argument('--seed', type=int, default=0,
                        help="manual seed")
    parser.add_argument('--gpu-devices', type=str, default='0')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help="use available gpus instead of specified devices (useful when using managed clusters)")
    parser.add_argument('--normalization', type=str, default='bn', help='bn or in')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='Modify config options using the command-line')
    parser.add_argument('--num_ker', type=int, default=32, help='generator filters in first conv layer')
    args = parser.parse_args()

    cfg = get_default_config()
    #cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    reset_config(cfg, args)
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)

    torch.manual_seed(args.seed)
    if not args.use_avai_gpus: os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    cfg.use_gpu = torch.cuda.is_available()

    if cfg.use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)

    #if cfg.use_gpu and args.gpu_devices:
        # if gpu_devices is not specified, all available gpus will be use
    #    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    log_name = 'test.log' if cfg.test.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))
    print("====================================")
    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    
    #if cfg.use_gpu:
    #    torch.backends.cudnn.benchmark = True
    

    
    datamanager = build_datamanager(cfg)
    
    print('Building model: {}'.format(cfg.model.name))
    G, model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    print(f"Model Total number of parameters: {total_params}")
    
    total_params = count_parameters(G)
    print(f"G Total number of parameters: {total_params}")
    #num_params, flops = compute_model_complexity(model, (1, 3, cfg.data.height, cfg.data.width))
    #print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
        # load_pretrained_weights(G, '/mnt/data/qzf/Paritalreid/log/brd_10_duke/迭代训练G.tar-60')
    
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
        G = nn.DataParallel(G).cuda()

    optimizer_m = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    optimizer_G = torchreid.optim.build_optimizer(G, lr=0.0005)
    scheduler_m = torchreid.optim.build_lr_scheduler(optimizer_m, lr_scheduler='multi_step', max_epoch = 270, stepsize=[25,45,65,85], gamma=0.1)
    # scheduler_G = torchreid.optim.build_lr_scheduler(optimizer_G, lr_scheduler='multi_step', max_epoch = 120, stepsize=[15,60], gamma=0.2)
    # scheduler_G = torchreid.optim.build_lr_scheduler(optimizer_G, lr_scheduler='multi_step', max_epoch = 120,stepsize=[20,40])
    # scheduler_G = torchreid.optim.build_lr_scheduler(optimizer_G, lr_scheduler='multi_step', max_epoch = 240,stepsize=[40,80])

    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(cfg.model.resume, model, optimizer=optimizer_m)

    print('Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type))
    engine = build_engine(cfg, datamanager, model, G, optimizer_m, scheduler_m, optimizer_G)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
