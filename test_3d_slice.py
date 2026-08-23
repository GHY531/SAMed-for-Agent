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

from datasets.test_dataset import TestDataset
from segment_anything import sam_model_registry
from test_3d import (
    config_to_dict,
    get_class_to_name,
    get_reference_class_ids,
    resolve_phase_configuration,
)
from utils import calculate_slice_metrics, test_single_volume


TUMOUR_SLICE_FIELDNAMES = (
    'case_name',
    'label_path',
    'slice_index',
    'tumour_gt_pixels',
    'tumour_pred_pixels',
    'tumour_dice',
    'tumour_iou',
    'tumour_hd95',
    'tumour_hd95_status',
    'below_low_dice_threshold',
)


def inference(args, multimask_output, z_spacing, model, test_save_path=None):
    """Run 2D slice inference and aggregate global Dice across all valid slices."""
    # Keep the dataset construction and list-file loading identical to test_3d.py.
    reference_class_ids = get_reference_class_ids(args.phase)
    db_test = TestDataset(
        list_path=args.list_dir,
        required_label_ids=reference_class_ids,
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info('%d test iterations per epoch', len(testloader))
    model.eval()
    class_to_name = get_class_to_name(args.phase)
    reference_vessel_names = ', '.join(
        class_to_name[class_id] for class_id in reference_class_ids
    )

    global_true_positive = np.zeros(args.num_classes, dtype=np.int64)
    global_false_positive = np.zeros(args.num_classes, dtype=np.int64)
    global_false_negative = np.zeros(args.num_classes, dtype=np.int64)
    hd95_values = [[] for _ in range(args.num_classes)]
    hd95_status_counts = [dict() for _ in range(args.num_classes)]
    tumour_slice_records = []
    low_dice_samples = []
    evaluated_slice_count = 0
    low_dice_threshold = float(args.low_dice_threshold)
    rotation_k = int(args.rotation_k)
    if rotation_k not in (0, 1, 2, 3):
        raise ValueError('rotation_k must be one of 0, 1, 2, or 3.')

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch['case_name'][0]
        label_path = sampled_batch['label_path'][0]

        image = torch.rot90(image, k=rotation_k, dims=(-2, -1))
        label = torch.rot90(label, k=rotation_k, dims=(-2, -1))

        # Use the same phase-specific valid-slice selection rule as test_3d.py.
        label_sq = label.squeeze(0)
        has_nonzero = (label_sq != 0).flatten(1).any(dim=1)
        # AP requires aorta; VP requires portal vein or superior mesenteric vein.
        has_reference_vessel = torch.stack(
            [
                (label_sq == class_id).flatten(1).any(dim=1)
                for class_id in reference_class_ids
            ]
        ).any(dim=0)
        valid_indices = torch.where(has_nonzero & has_reference_vessel)[0]
        if len(valid_indices) == 0:
            logging.warning(
                'Case %s has no valid slices (all background or missing %s), skipped.',
                case_name,
                reference_vessel_names,
            )
            continue

        _, prediction, volume_label, evaluated_indices = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=z_spacing,
            valid_slice_indices=valid_indices,
            return_prediction=True,
        )

        for slice_index in evaluated_indices:
            slice_index = int(slice_index)
            evaluated_slice_count += 1
            slice_prediction = prediction[slice_index]
            slice_label = volume_label[slice_index]
            for class_index in range(args.num_classes):
                class_id = class_index + 1
                prediction_mask = slice_prediction == class_id
                label_mask = slice_label == class_id
                global_true_positive[class_index] += np.logical_and(
                    prediction_mask, label_mask
                ).sum()
                global_false_positive[class_index] += np.logical_and(
                    prediction_mask, ~label_mask
                ).sum()
                global_false_negative[class_index] += np.logical_and(
                    ~prediction_mask, label_mask
                ).sum()

                dice, iou, hd95, hd95_status = calculate_slice_metrics(
                    prediction_mask, label_mask
                )
                hd95_status_counts[class_index][hd95_status] = (
                    hd95_status_counts[class_index].get(hd95_status, 0) + 1
                )
                if hd95_status == 'valid':
                    hd95_values[class_index].append(hd95)

                if class_id == 1:
                    record = {
                        'case_name': case_name,
                        'label_path': os.path.abspath(label_path),
                        'slice_index': slice_index,
                        'tumour_gt_pixels': int(label_mask.sum()),
                        'tumour_pred_pixels': int(prediction_mask.sum()),
                        'tumour_dice': dice,
                        'tumour_iou': iou,
                        'tumour_hd95': hd95,
                        'tumour_hd95_status': hd95_status,
                        'below_low_dice_threshold': (
                            bool(label_mask.any()) and dice < low_dice_threshold
                        ),
                    }
                    tumour_slice_records.append(record)
                    if record['below_low_dice_threshold']:
                        low_dice_samples.append(record)

        logging.info(
            'idx %d case %s evaluated_slices %d',
            i_batch,
            case_name,
            len(evaluated_indices),
        )

    if evaluated_slice_count == 0:
        logging.error('No valid slices were evaluated!')
        return 0

    low_dice_file = os.path.join(args.output_dir, 'low_dice_sample.txt')
    with open(low_dice_file, 'w', encoding='utf-8') as file:
        for record in low_dice_samples:
            file.write(
                f"{record['case_name']}\t{record['label_path']}\t"
                f"{record['slice_index']}\t{record['tumour_dice']:.6f}\n"
            )
    tumour_metrics_file = os.path.join(args.output_dir, 'tumour_slice_metrics.csv')
    with open(tumour_metrics_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=TUMOUR_SLICE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(tumour_slice_records)
    logging.info(
        'Saved %d GT-positive tumour slices with Dice < %.3f to %s',
        len(low_dice_samples),
        low_dice_threshold,
        low_dice_file,
    )
    logging.info('Saved tumour slice audit records to %s', tumour_metrics_file)

    global_dice_values = []
    global_iou_values = []
    for class_index in range(args.num_classes):
        true_positive = global_true_positive[class_index]
        false_positive = global_false_positive[class_index]
        false_negative = global_false_negative[class_index]
        denominator = 2 * true_positive + false_positive + false_negative
        # Assign a perfect score when both prediction and ground truth are empty.
        global_dice = 1.0 if denominator == 0 else 2 * true_positive / denominator
        iou_denominator = true_positive + false_positive + false_negative
        global_iou = 1.0 if iou_denominator == 0 else true_positive / iou_denominator
        global_dice_values.append(global_dice)
        global_iou_values.append(global_iou)

        class_hd95_values = hd95_values[class_index]
        mean_hd95 = np.mean(class_hd95_values) if class_hd95_values else np.nan
        median_hd95 = np.median(class_hd95_values) if class_hd95_values else np.nan
        logging.info(
            'Global class %d name %s dice %f; iou %f; valid_slice_hd95_mean %f; '
            'valid_slice_hd95_median %f; hd95_valid_count %d; '
            'hd95_status_counts %s',
            class_index + 1,
            class_to_name.get(class_index + 1, f'class_{class_index + 1}'),
            global_dice,
            global_iou,
            mean_hd95,
            median_hd95,
            len(class_hd95_values),
            hd95_status_counts[class_index],
        )

    micro_true_positive = int(global_true_positive.sum())
    micro_false_positive = int(global_false_positive.sum())
    micro_false_negative = int(global_false_negative.sum())
    micro_iou_denominator = (
        micro_true_positive + micro_false_positive + micro_false_negative
    )
    # Compute one foreground-class IoU after aggregating every class's errors.
    micro_iou = (
        1.0
        if micro_iou_denominator == 0
        else micro_true_positive / micro_iou_denominator
    )
    logging.info(
        'Testing performance: mean_global_dice %f; mean_global_iou %f; '
        'micro_iou %f over %d valid slices',
        np.mean(global_dice_values),
        np.mean(global_iou_values),
        micro_iou,
        evaluated_slice_count,
    )
    logging.info('Testing Finished!')
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--phase', type=str, choices=['AP', 'VP'], default='AP')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--rotation_k', type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--low_dice_threshold', type=float, default=0.5)
    parser.add_argument(
        '--list_dir',
        type=str,
        default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/lists/lists_Pancreas/test_list.txt',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/test_result',
    )
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--is_savenii', action='store_true')
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument(
        '--ckpt',
        type=str,
        default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/checkpoints/sam_vit_b_01ec64.pth',
    )
    parser.add_argument(
        '--lora_ckpt',
        type=str,
        default='/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/SAMed/Lora_checkopints/self_dataset_ckpt/epoch_118_iter_12000.pth',
    )
    parser.add_argument('--vit_name', type=str, default='vit_b')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
    args = parser.parse_args()

    if args.config is not None:
        config_dict = config_to_dict(args.config)
        for key, value in config_dict.items():
            setattr(args, key, value)
    resolve_phase_configuration(args)

    cudnn.benchmark = not args.deterministic
    cudnn.deterministic = bool(args.deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1],
    )
    net = import_module(args.module).LoRA_Sam(sam, args.rank).cuda()
    net.load_lora_parameters(args.lora_ckpt)
    multimask_output = args.num_classes > 1

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

    test_save_path = None
    if args.is_savenii:
        test_save_path = os.path.abspath(os.path.join(args.output_dir, 'predictions'))
        os.makedirs(test_save_path, exist_ok=True)
    inference(args, multimask_output, 1, net, test_save_path)
