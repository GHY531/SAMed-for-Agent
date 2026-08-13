import argparse
import csv
import logging
import os
import random
import sys
from importlib import import_module

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.test_2d_dataset import TestDataset
from segment_anything import sam_model_registry
from utils import test_single_slice


class_to_name = {
    0: 'background',
    1: 'tumour',
    2: 'aorta',
    3: 'superior mesenteric artery',
    4: 'celiac axis',
}

test_list = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/test'


def save_tumour_area_statistics(records, output_dir):
    """Save per-slice tumour areas and a distribution figure."""
    csv_path = os.path.join(output_dir, 'tumour_area_per_slice.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                'case_name',
                'tumour_pixels',
                'tumour_area_fraction',
                'tumour_dice',
            ),
        )
        writer.writeheader()
        writer.writerows(records)

    # Use a non-interactive backend so plotting works on headless servers.
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tumour_areas = np.asarray(
        [record['tumour_pixels'] for record in records], dtype=np.int64
    )
    positive_areas = tumour_areas[tumour_areas > 0]
    area_groups = np.asarray(
        [
            np.count_nonzero(tumour_areas == 0),
            np.count_nonzero((tumour_areas >= 1) & (tumour_areas <= 10)),
            np.count_nonzero((tumour_areas >= 11) & (tumour_areas <= 50)),
            np.count_nonzero((tumour_areas >= 51) & (tumour_areas <= 200)),
            np.count_nonzero(tumour_areas > 200),
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    if positive_areas.size > 0:
        max_area = int(positive_areas.max())
        if max_area == 1:
            histogram_bins = np.asarray([0.5, 1.5])
        else:
            histogram_bins = np.unique(
                np.geomspace(1, max_area + 1, num=min(40, max_area + 1))
            )
        axes[0].hist(positive_areas, bins=histogram_bins, edgecolor='black')
        axes[0].set_xscale('log')
    else:
        axes[0].text(
            0.5,
            0.5,
            'No tumour-positive slices',
            ha='center',
            va='center',
            transform=axes[0].transAxes,
        )
    axes[0].set_title('Tumour Area Distribution (Positive Slices)')
    axes[0].set_xlabel('GT tumour area (pixels, log scale)')
    axes[0].set_ylabel('Number of slices')
    axes[0].grid(axis='y', alpha=0.25)

    group_labels = ('0', '1-10', '11-50', '51-200', '>200')
    bars = axes[1].bar(group_labels, area_groups, edgecolor='black')
    axes[1].bar_label(bars, padding=3)
    axes[1].set_title('Slices Grouped by GT Tumour Area')
    axes[1].set_xlabel('GT tumour area (pixels)')
    axes[1].set_ylabel('Number of slices')
    axes[1].grid(axis='y', alpha=0.25)

    figure.suptitle(f'Tumour Area Statistics for {len(records)} Test Slices')
    figure.tight_layout()
    histogram_path = os.path.join(output_dir, 'tumour_area_histogram.png')
    figure.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close(figure)

    logging.info('Saved tumour area statistics CSV to %s', csv_path)
    logging.info('Saved tumour area histogram to %s', histogram_path)
    logging.info(
        'Tumour area groups [0, 1-10, 11-50, 51-200, >200]: %s',
        area_groups.tolist(),
    )


def inference(args, multimask_output, model, test_save_path=None):
    # Require NIfTI output because the low-Dice list points to saved triplets.
    if test_save_path is None:
        raise ValueError(
            '--is_savenii is required because low_dice_sample.txt must point '
            'to saved NIfTI triplets.'
        )

    db_test = TestDataset(test_dir=test_list)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f'{len(testloader)} test 2D slices in total')

    model.eval()
    metric_list = 0.0
    valid_case_count = 0
    low_dice_samples = []
    tumour_area_records = []

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, case_name = (
            sampled_batch['image'],
            sampled_batch['label'],
            sampled_batch['case_name'][0],
        )

        # Run direct 2D inference and evaluate all foreground classes.
        metric_i = test_single_slice(
            image,
            label,
            model,
            classes=args.num_classes,
            multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
        )

        metric_list += np.array(metric_i)
        valid_case_count += 1

        # The tumour is class 1, so its metric is stored at index 0.
        tumour_dice = float(metric_i[0][0])
        label_array = label.squeeze().cpu().numpy()
        tumour_pixels = int(np.count_nonzero(label_array == 1))
        tumour_area_records.append(
            {
                'case_name': case_name,
                'tumour_pixels': tumour_pixels,
                'tumour_area_fraction': tumour_pixels / label_array.size,
                'tumour_dice': tumour_dice,
            }
        )
        if tumour_dice < args.low_dice_threshold:
            low_dice_samples.append(
                os.path.abspath(
                    os.path.join(test_save_path, f'{case_name}_gt.nii.gz')
                )
            )

        # Calculate average metrics for the current 2D slice.
        case_mean = np.mean(metric_i, axis=0)
        logging.info(
            'idx %d slice %s mean_dice %f mean_iou %f'
            % (i_batch, case_name, case_mean[0], case_mean[1])
        )

        # Log metrics for each individual class in the current slice.
        for j in range(1, args.num_classes + 1):
            logging.info(
                'name %s dice %f iou %f'
                % (
                    class_to_name.get(j, f'class_{j}'),
                    metric_i[j - 1][0],
                    metric_i[j - 1][1],
                )
            )

    low_dice_file = os.path.join(args.output_dir, 'low_dice_sample.txt')
    with open(low_dice_file, 'w', encoding='utf-8') as file:
        for image_path in low_dice_samples:
            file.write(f'{image_path}\n')
    logging.info(
        'Saved %d tumour slices with Dice < %.3f to %s',
        len(low_dice_samples),
        args.low_dice_threshold,
        low_dice_file,
    )
    save_tumour_area_statistics(tumour_area_records, args.output_dir)

    if valid_case_count == 0:
        logging.error('No valid 2D slices were evaluated!')
        return 0

    # Calculate overall average metrics across all evaluated 2D slices.
    metric_list = metric_list / valid_case_count
    for i in range(1, args.num_classes + 1):
        logging.info(
            'Mean class %d name %s mean_dice %f mean_iou %f'
            % (
                i,
                class_to_name.get(i, f'class_{i}'),
                metric_list[i - 1][0],
                metric_list[i - 1][1],
            )
        )

    performance = np.mean(metric_list, axis=0)[0]
    mean_iou = np.mean(metric_list, axis=0)[1]

    logging.info(
        'Overall 2D Testing Performance: mean_dice: %f | mean_iou: %f'
        % (performance, mean_iou)
    )
    logging.info('Testing Finished!')
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as file:
        items = file.readlines()
    for item in items:
        key, value = item.strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='The config file provided by the trained model',
    )
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument(
        '--low_dice_threshold',
        type=float,
        default=0.5,
        help='Tumour Dice threshold used to collect low-quality slices',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/test_result',
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=512,
        help='Input image size of the network',
    )
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument(
        '--is_savenii',
        action='store_true',
        help='Whether to save results during inference',
    )
    parser.add_argument(
        '--deterministic',
        type=int,
        default=1,
        help='whether use deterministic training',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/checkpoints/sam_vit_b_01ec64.pth',
        help='Pretrained checkpoint',
    )
    parser.add_argument(
        '--lora_ckpt',
        type=str,
        default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/Lora_checkopints/Synapse_512_pretrain_vit_b_21k_epo50_bs32_lr0.0004/best.pth',
        help='The checkpoint from LoRA',
    )
    parser.add_argument(
        '--vit_name', type=str, default='vit_b', help='Select one vit model'
    )
    parser.add_argument(
        '--rank', type=int, default=4, help='Rank for LoRA adaptation'
    )
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')

    args = parser.parse_args()

    if args.config is not None:
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Register the segmentation model.
    sam, img_embedding_size = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )

    pkg = import_module(args.module)
    net = pkg.LoRA_Sam(sam, args.rank).cuda()

    assert args.lora_ckpt is not None
    net.load_lora_parameters(args.lora_ckpt)

    multimask_output = args.num_classes > 1

    # Initialize file and console logging.
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, multimask_output, net, test_save_path)
