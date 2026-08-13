import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
from einops import repeat


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, n_classes, weight=None):
        super(WeightedDiceLoss, self).__init__()
        self.n_classes = n_classes
        # Convert weight to buffer if provided at init
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def _one_hot_encoder(self, input_tensor):
        # Squeeze channel dim if target shape is (B, 1, H, W)
        if input_tensor.dim() == 4 and input_tensor.size(1) == 1:
            input_tensor = input_tensor.squeeze(1)

        # F.one_hot expects long dtype: (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
        one_hot = F.one_hot(input_tensor.long(), num_classes=self.n_classes)
        return one_hot.permute(0, 3, 1, 2).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1.0 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target)

        assert (
            inputs.size() == target.size()
        ), f"Predict {inputs.size()} & target {target.size()} shape do not match"

        # Determine which weight array to use
        if weight is None:
            if self.weight is not None:
                weight = self.weight
            else:
                weight = torch.ones(
                    self.n_classes, device=inputs.device, dtype=torch.float32
                )
        else:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(
                    weight, device=inputs.device, dtype=torch.float32
                )
            else:
                weight = weight.to(inputs.device)

        loss = 0.0
        for i in range(self.n_classes):
            dice_loss = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice_loss * weight[i]

        # Normalize by the sum of weights instead of n_classes
        return loss / torch.sum(weight)

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    
    pred_has_pixel = pred.sum() > 0
    gt_has_pixel = gt.sum() > 0

    if pred_has_pixel and gt_has_pixel:
        dice = metric.binary.dc(pred, gt)
        
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = intersection / (union + 1e-8)
        return dice, iou

    elif not pred_has_pixel and not gt_has_pixel:
        return 1.0, 1.0

    else:
        return 0.0, 0.0

def test_single_volume(
    image,
    label,
    net,
    classes,
    multimask_output,
    patch_size=[256, 256],
    test_save_path=None,
    case=None,
    z_spacing=1,
    valid_slice_indices=None,
):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if image.ndim != 3 or label.ndim != 3:
        raise ValueError(
            'test_single_volume expects image and label volumes with shape '
            f'[N, H, W], but received {image.shape} and {label.shape}.'
        )
    if image.shape != label.shape:
        raise ValueError(
            'Image and label volumes must have identical shapes, but received '
            f'{image.shape} and {label.shape}.'
        )

    if valid_slice_indices is None:
        valid_slice_indices = np.arange(image.shape[0], dtype=np.int64)
    elif isinstance(valid_slice_indices, torch.Tensor):
        valid_slice_indices = valid_slice_indices.cpu().numpy().astype(np.int64)
    else:
        valid_slice_indices = np.asarray(valid_slice_indices, dtype=np.int64)

    if valid_slice_indices.ndim != 1:
        raise ValueError('valid_slice_indices must be a one-dimensional sequence.')
    if valid_slice_indices.size == 0:
        raise ValueError('At least one valid slice index is required.')
    if np.any(valid_slice_indices < 0) or np.any(valid_slice_indices >= image.shape[0]):
        raise IndexError('valid_slice_indices contains an out-of-range slice index.')

    prediction = np.zeros_like(label)
    for ind in valid_slice_indices:
        slice = image[ind, :, :]
        x, y = slice.shape

        # Resize directly from the original resolution to the model input resolution.
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(
                slice,
                (patch_size[0] / x, patch_size[1] / y),
                order=3,
            )
        inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            pred = out.cpu().detach().numpy()
            out_h, out_w = pred.shape
            if x != out_h or y != out_w:
                pred = zoom(pred, (x / out_h, y / out_w), order=0)
            prediction[ind] = pred
    metric_list = []
    evaluated_prediction = prediction[valid_slice_indices]
    evaluated_label = label[valid_slice_indices]
    for i in range(1, classes + 1):
        metric_list.append(
            calculate_metric_percase(
                evaluated_prediction == i,
                evaluated_label == i,
            )
        )

    if test_save_path is not None:
        if not case:
            raise ValueError('A non-empty case name is required when saving NIfTI files.')
        case_save_path = os.path.join(test_save_path, case)
        os.makedirs(case_save_path, exist_ok=True)
        evaluation_mask = np.zeros_like(label, dtype=np.uint8)
        evaluation_mask[valid_slice_indices] = 1
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
        lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
        mask_itk = sitk.GetImageFromArray(evaluation_mask)
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        mask_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, os.path.join(case_save_path, 'prediction.nii.gz'))
        sitk.WriteImage(img_itk, os.path.join(case_save_path, 'image.nii.gz'))
        sitk.WriteImage(lab_itk, os.path.join(case_save_path, 'ground_truth.nii.gz'))
        sitk.WriteImage(mask_itk, os.path.join(case_save_path, 'evaluation_mask.nii.gz'))
    return metric_list

def test_single_slice(image, label, net, classes, multimask_output, patch_size=[256, 256],
                      test_save_path=None, case=None):
    # Squeeze batch and extra dimensions to ensure 2D shape (H, W)
    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().detach().numpy()
    else:
        image = np.squeeze(image)
        
    if isinstance(label, torch.Tensor):
        label = label.squeeze().cpu().detach().numpy()
    else:
        label = np.squeeze(label)

    # Record original spatial dimensions
    x, y = image.shape[0], image.shape[1]

    # Resize directly from the original resolution to the model input resolution.
    slice_img = image
    if x != patch_size[0] or y != patch_size[1]:
        slice_img = zoom(
            slice_img,
            (patch_size[0] / x, patch_size[1] / y),
            order=3,
        )

    # Prepare network input tensor (1, 3, H, W)
    inputs = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()
    inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)

    # Model evaluation
    net.eval()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks = outputs['masks']
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        out_h, out_w = out.shape
        # Restore prediction to original image resolution using nearest neighbor interpolation
        if x != out_h or y != out_w:
            prediction = zoom(out, (x / out_h, y / out_w), order=0)
        else:
            prediction = out

    # Compute metric for each class
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # Save 2D slice files if path is provided
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        
        # Set 2D image spacing
        img_itk.SetSpacing((1.0, 1.0))
        prd_itk.SetSpacing((1.0, 1.0))
        lab_itk.SetSpacing((1.0, 1.0))
        
        sitk.WriteImage(prd_itk, os.path.join(test_save_path, f"{case}_pred.nii.gz"))
        sitk.WriteImage(img_itk, os.path.join(test_save_path, f"{case}_img.nii.gz"))
        sitk.WriteImage(lab_itk, os.path.join(test_save_path, f"{case}_gt.nii.gz"))

    return metric_list
