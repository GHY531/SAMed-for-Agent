import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.test_dataset import TestDataset


class_to_name = {0: 'background',
                 1: 'tumour',
                 2: 'aorta',
                 3: 'superior mesenteric artery',
                 4: 'celiac axis',
                 }


def inference(
    args, 
    multimask_output, 
    z_spacing,
    model, 
    test_save_path=None
):
    db_test = TestDataset(list_path=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)    
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0

    valid_case_count = 0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0]

        # ------------------- Optimized Slice Filtering (Pure PyTorch) -------------------
        # label shape: [1, N, H, W] -> label_sq shape: [N, H, W]
        label_sq = label.squeeze(0)
        
        # Check non-zero slices and slices containing class 2 (Aorta)
        has_nonzero = (label_sq != 0).flatten(1).any(dim=1)
        has_aorta = (label_sq == 2).flatten(1).any(dim=1)
        
        # Combine conditions and extract valid slice indices
        valid_mask = has_nonzero & has_aorta
        valid_indices = torch.where(valid_mask)[0]

        # Skip evaluation if no slices meet the criteria
        if len(valid_indices) == 0:
            logging.warning(f"Case {case_name} has no valid slices (all background or missing class 2), skipped.")
            continue

        # Keep only valid slices along the N dimension (dim=1)
        image = image[:, valid_indices, :, :]
        label = label[:, valid_indices, :, :]
        # --------------------------------------------------------------------------------

        metric_i = test_single_volume(
            image, label, model, classes=args.num_classes, multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
            test_save_path=test_save_path, case=case_name, z_spacing=z_spacing
        )
        
        metric_list += np.array(metric_i)
        valid_case_count += 1
        
        case_mean = np.mean(metric_i, axis=0)
        logging.info('idx %d case %s mean_dice %f mean_iou %f' % (
            i_batch, case_name, case_mean[0], case_mean[1]))
        
        
        for j in range(1, args.num_classes + 1):
            logging.info('name %s dice %f iou %f' % 
                (
                class_to_name.get(j, f'class_{j}'), 
                metric_i[j - 1][0], 
                metric_i[j - 1][1]
                )
            )

    if valid_case_count == 0:
        logging.error("No valid cases were evaluated!")
        return 0

    # Divide by actual evaluated cases count to avoid skewing average metrics
    metric_list = metric_list / valid_case_count
    for i in range(1, args.num_classes + 1):
        logging.info('Mean class %d name %s mean_dice %f mean_iou %f' % (
            i, class_to_name[i], metric_list[i - 1][0], metric_list[i - 1][1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_iou = np.mean(metric_list, axis=0)[1]
    
    logging.info('Testing performance in best val model: mean_dice : %f mean_iou : %f' % (
        performance, mean_iou))

    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--list_dir', type=str, default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/lists/lists_Pancreas/test_list.txt', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/test_result')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--lora_ckpt', type=str, default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/Lora_checkopints/Synapse_512_pretrain_vit_b_21k_epo50_bs32_lr0.0004/best.pth', 
                        help='The checkpoint from LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

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
    z_spacing = 1
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0],
                                                                    pixel_std=[1, 1, 1])
    
    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, z_spacing, net, test_save_path)