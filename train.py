import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, 
                    default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/Lora_checkopints',
                    help='Trained LORA Checkpoints')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument(
    '--phase',
    choices=['AP', 'VP'],
    default='AP',
    help='Contrast-enhanced CT phase defining the label schema and vessel selection rule',
)
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network') 
parser.add_argument('--max_iterations', type=int,
                    default=21000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=5, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=5, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument(
    '--negative_per_positive',
    type=int,
    default=2,
    help='Number of tumour-absent slices sampled per tumour-present slice',
)
parser.add_argument(
    '--train_data_type',
    choices=['inhouse_3d', 'pants_2d'],
    default='inhouse_3d',
    help='Dataset layout used for training',
)
parser.add_argument(
    '--train_list',
    type=str,
    default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/lists/lists_Pancreas/train_list.txt',
    help='Training manifest containing case directories or 2D label paths',
)
parser.add_argument(
    '--val_list',
    type=str,
    required=True,
    help='Fixed case-level validation manifest used for checkpoint selection',
)
parser.add_argument(
    '--rotation_k',
    type=int,
    choices=[0, 1, 2, 3],
    default=3,
    help='Fixed in-plane 90-degree counterclockwise rotations for in-house data',
)
parser.add_argument(
    '--enable_random_orientation',
    action='store_true',
    help='Enable random 90-degree rotations and flips after the fixed orientation',
)
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.00005,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, 
                    default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--save_iteration', type=int, default=1000, help='How many iterations are needed to save lora checkpoints')
parser.add_argument('--ce_weight', type=float, default=0.1, help='The weight of normal CrossEntropyLoss')
parser.add_argument('--dice_weight', type=float, default=0.8, help='The weight of Weighted Diceloss')
parser.add_argument('--focal_weight', type=float, default=0.1, help='The weight of Focal loss')

args = parser.parse_args()

if args.phase == 'VP' and args.num_classes != 3:
    parser.error('VP requires --num_classes 3 for tumour, PV, and SMV.')

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'num_classes': args.num_classes,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_path = snapshot_path + '_vp' if args.phase == 'VP' else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                pixel_std=[1, 1, 1])

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    # net = LoRA_Sam(sam, args.rank).cuda()
    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {'Synapse': trainer_synapse}
    trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res)
