import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, Focal_loss, WeightedDiceLoss, test_single_volume
from torchvision import transforms
from datasets.direct_dataset import (
    DirectDataset,
    RandomGenerator as DirectRandomGenerator,
    TumourBalancedBatchSampler,
)
from datasets.merged_dataset import (
    MergedDataset,
    RandomGenerator as MergedRandomGenerator,
)
from datasets.test_dataset import TestDataset

dice_weights_list = [0.2, 3, 0.5, 1.5, 1.5]
focal_weights_list = [0.05, 10, 0.5, 1.5, 1.5]


def validate_case_tumour_dice(
    model,
    val_loader,
    multimask_output,
    patch_size,
    rotation_k,
):
    """Return mean case-level tumour Dice on tumour-bearing evaluation subsets."""
    model.eval()
    case_dice_scores = []
    skipped_cases = 0

    for sampled_batch in val_loader:
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch['case_name'][0]
        label_slices = label.squeeze(0)
        valid_mask = (label_slices == 2).flatten(1).any(dim=1)
        valid_indices = torch.where(valid_mask)[0]

        if len(valid_indices) == 0:
            logging.warning(
                'Validation case %s has no aorta-visible slice and was skipped',
                case_name,
            )
            skipped_cases += 1
            continue
        if not (label_slices[valid_indices] == 1).any():
            logging.warning(
                'Validation case %s has no tumour GT in the evaluation subset '
                'and was excluded from tumour-Dice model selection',
                case_name,
            )
            skipped_cases += 1
            continue

        image = torch.rot90(image, k=rotation_k, dims=(-2, -1))
        label = torch.rot90(label, k=rotation_k, dims=(-2, -1))
        metric_list = test_single_volume(
            image=image,
            label=label,
            net=model,
            classes=1,
            multimask_output=multimask_output,
            patch_size=[patch_size, patch_size],
            valid_slice_indices=valid_indices,
        )
        tumour_dice = float(metric_list[0][0])
        case_dice_scores.append(tumour_dice)
        logging.info(
            'Validation case %s tumour Dice: %.6f',
            case_name,
            tumour_dice,
        )

    model.train()
    if not case_dice_scores:
        raise RuntimeError(
            'No validation case contains tumour GT in the evaluation subset.'
        )
    mean_tumour_dice = float(np.mean(case_dice_scores))
    logging.info(
        'Validation mean case-level tumour Dice: %.6f over %d cases; skipped %d',
        mean_tumour_dice,
        len(case_dice_scores),
        skipped_cases,
    )
    return mean_tumour_dice

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, focal_loss,
            ce_weight:float=0.1,
            dice_weight:float=0.45,
            focal_weight:float=0.45
            ):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss_focal = focal_loss(low_res_logits, low_res_label_batch)
    loss = ce_weight * loss_ce + dice_weight * loss_dice + focal_weight * loss_focal
    return loss, loss_ce, loss_dice, loss_focal


def trainer_synapse(args, model, snapshot_path, multimask_output, low_res):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    if args.train_data_type == 'inhouse_3d':
        db_train = MergedDataset(
            list_path=args.train_list,
            rotation_k=args.rotation_k,
            num_classes=args.num_classes,
            transform=transforms.Compose([
                MergedRandomGenerator(
                    output_size=[args.img_size, args.img_size],
                    low_res=[low_res, low_res],
                    enable_random_orientation=args.enable_random_orientation,
                )
            ]),
        )
    else:
        db_train = DirectDataset(
            label_path=args.train_list,
            transform=transforms.Compose([
                DirectRandomGenerator(
                    output_size=[args.img_size, args.img_size],
                    low_res=[low_res, low_res],
                )
            ]),
        )
    logging.info(
        'Training data type: %s; list: %s; fixed rotation_k: %d',
        args.train_data_type,
        args.train_list,
        args.rotation_k if args.train_data_type == 'inhouse_3d' else 0,
    )
    print("The length of train set is: {}".format(len(db_train)))
    batch_sampler = TumourBalancedBatchSampler(
        positive_indices=db_train.positive_indices,
        negative_indices=db_train.negative_indices,
        batch_size=batch_size,
        negative_per_positive=args.negative_per_positive,
        seed=args.seed,
    )
    logging.info(
        'Positive-case training slices: %d tumour-present and %d tumour-absent',
        len(db_train.positive_indices),
        len(db_train.negative_indices),
    )
    logging.info(
        'Global batch size: %d (%d per GPU across %d GPUs)',
        batch_size,
        args.batch_size,
        args.n_gpu,
    )
    logging.info(
        'Tumour-present slices per batch: %d-%d; epoch ratio: %.4f',
        min(batch_sampler.positive_counts_per_batch),
        max(batch_sampler.positive_counts_per_batch),
        batch_sampler.positive_count_per_epoch
        / (len(batch_sampler) * batch_size),
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    val_dataset = TestDataset(list_path=args.val_list)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    if len(val_dataset) == 0:
        raise ValueError(f'No validation cases found in {args.val_list}.')
    logging.info(
        'Validation cases: %d; list: %s; selection metric: case tumour Dice',
        len(val_dataset),
        args.val_list,
    )
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    dice_class_weights = torch.tensor(dice_weights_list, dtype=torch.float32).cuda()
    ce_loss = CrossEntropyLoss()
    dice_loss = WeightedDiceLoss(n_classes=num_classes + 1, weight=dice_class_weights)
    focal_loss = Focal_loss(num_classes=num_classes + 1, alpha=focal_weights_list)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_tumour_dice = float('-inf')
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        batch_sampler.set_epoch(epoch_num)
        epoch_dice_sum = 0.0
        epoch_focal_sum = 0.0
        epoch_batch_count = 0
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w]
            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            assert image_batch.max() <= 3, f'image_batch max: {image_batch.max()}'
            outputs = model(image_batch, multimask_output, args.img_size)
            loss, loss_ce, loss_dice, loss_focal = calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, focal_loss, 
            ce_weight = args.ce_weight, 
            dice_weight = args.dice_weight, 
            focal_weight = args.focal_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('learning_rate', lr_, iter_num)
            writer.add_scalar('Total_loss', loss, iter_num)
            writer.add_scalar('Loss_ce', loss_ce, iter_num)
            writer.add_scalar('Loss_dice', loss_dice, iter_num)
            writer.add_scalar('Loss_focal', loss_focal, iter_num)

            print('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_focal: %f' % (
                iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_focal.item()))
            
            if iter_num % 50 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_focal: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_focal.item()))
            
            epoch_dice_sum += loss_dice.item()
            epoch_focal_sum += loss_focal.item()
            epoch_batch_count += 1

            if iter_num % args.save_iteration == 0:
                iter_save_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_iter_{iter_num}.pth')
                try:
                    model.save_lora_parameters(iter_save_path)
                except AttributeError:
                    model.module.save_lora_parameters(iter_save_path)
                logging.info(f"Saved LoRA weights at iteration {iter_num} to {iter_save_path}")

        # Select the best checkpoint using case-level tumour Dice.
        epoch_avg_dice = epoch_dice_sum / max(epoch_batch_count, 1)
        epoch_avg_focal = epoch_focal_sum / max(epoch_batch_count, 1)

        logging.info("epoch %d : avg loss_dice: %f, ave loss_focal: %f" % (epoch_num, epoch_avg_dice, epoch_avg_focal))
        validation_tumour_dice = validate_case_tumour_dice(
            model=model,
            val_loader=val_loader,
            multimask_output=multimask_output,
            patch_size=args.img_size,
            rotation_k=args.rotation_k,
        )
        writer.add_scalar(
            'Validation/case_tumour_dice',
            validation_tumour_dice,
            epoch_num + 1,
        )
        if validation_tumour_dice > best_tumour_dice:
            best_tumour_dice = validation_tumour_dice
            best_save_path = os.path.join(snapshot_path, 'best.pth')
            try:
                model.save_lora_parameters(best_save_path)
            except AttributeError:
                model.module.save_lora_parameters(best_save_path)
            logging.info(
                'New best epoch %d (validation case tumour Dice %.6f), '
                'saved model to %s',
                epoch_num + 1,
                best_tumour_dice,
                best_save_path,
            )

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            last_save_path = os.path.join(snapshot_path, 'last.pth')
            try:
                model.save_lora_parameters(last_save_path)
            except AttributeError:
                model.module.save_lora_parameters(last_save_path)
            logging.info("save last epoch model to {}".format(last_save_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
