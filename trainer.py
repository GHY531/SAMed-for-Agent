import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss, Focal_loss, WeightedDiceLoss
from torchvision import transforms
from datasets.direct_dataset import (
    DirectDataset,
    RandomGenerator,
    TumourBalancedBatchSampler,
)

dice_weights_list = [0.2, 3, 0.5, 1.5, 1.5]
focal_weights_list = [0.05, 10, 0.5, 1.5, 1.5]

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
    train_list_path = '/home/bml/storage/mnt/v-3f30eb9261b04a32/org/HY/GHY/Dataset_AP/positive_sample/train.txt'
    #train_list_path could be your own path to your train_list
    db_train = DirectDataset(label_path = train_list_path,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))
    batch_sampler = TumourBalancedBatchSampler(
        positive_indices=db_train.positive_indices,
        negative_indices=db_train.negative_indices,
        batch_size=batch_size,
        negative_per_positive=args.negative_per_positive,
        seed=args.seed,
    )
    logging.info(
        'Training slices: %d tumour-positive and %d tumour-negative',
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
        'Tumour-positive slices per batch: %d-%d; epoch ratio: %.4f',
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
    best_performance = float('inf')
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

        #save the best and last epoch
        epoch_avg_dice = epoch_dice_sum / max(epoch_batch_count, 1)
        epoch_avg_focal = epoch_focal_sum / max(epoch_batch_count, 1)
        epoch_avg_loss = 0.7 * epoch_avg_dice + 0.3 * epoch_avg_focal

        logging.info("epoch %d : avg loss_dice: %f, ave loss_focal: %f" % (epoch_num, epoch_avg_dice, epoch_avg_focal))
        if epoch_avg_loss < best_performance:
            best_performance = epoch_avg_loss
            best_save_path = os.path.join(snapshot_path, 'best.pth')
            try:
                model.save_lora_parameters(best_save_path)
            except:
                model.module.save_lora_parameters(best_save_path)
            logging.info("new best epoch %d (avg loss_dice %f), save model to %s"
                         % (epoch_num, best_performance, best_save_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            last_save_path = os.path.join(snapshot_path, 'last.pth')
            try:
                model.save_lora_parameters(last_save_path)
            except:
                model.module.save_lora_parameters(last_save_path)
            logging.info("save last epoch model to {}".format(last_save_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
