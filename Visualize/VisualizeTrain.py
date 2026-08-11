import numpy as np
from scipy.ndimage.interpolation import zoom
from einops import repeat
import torch

def rotateImage(arr, k=-1):
    """
    When training, I found that the images need to rotate.
    If your image needs to rotate k*90°, use this!
    """
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    return np.rot90(arr, k=k).copy()

def normalize(img):
    """
    Stretch raw CT images to grey-scale images (0-255 uint8)
    """
    if hasattr(img, 'detach'):
        img = img.detach().cpu().numpy()
        
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
    return img.astype(np.uint8)


def overlay_mask(ct_display, mask, color_map, alpha=0.5):
    """
    Colour the stretched picture to show different masks
    """
    if hasattr(ct_display, 'detach'):
        ct_display = ct_display.detach().cpu().numpy()
    if hasattr(mask, 'detach'):
        mask = mask.detach().cpu().numpy()

    ct_display = np.squeeze(ct_display)
    mask = np.squeeze(mask)

    if ct_display.shape != mask.shape:
        raise ValueError(f"Shape Mismatch! ct_display: {ct_display.shape}, mask: {mask.shape}. "
                         f"Please check whether the pred_slice/gt_slice's dimension is correct!")

    base = np.stack([ct_display] * 3, axis=-1).astype(np.float32)
    overlay = base.copy()

    for cls_id, rgb in color_map.items():
        region = (mask == cls_id)
        if not np.any(region):
            continue
        
        rgb_arr = np.array(rgb, dtype=np.float32)
        
        overlay[region] = base[region] * (1 - alpha) + rgb_arr * alpha

    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_comparison(ct_display, gt_overlay, pred_overlay, gap=10):
    """
    Stitch the three pictures together
    """
    if hasattr(ct_display, 'detach'):
        ct_display = ct_display.detach().cpu().numpy()
        
    ct_display = np.squeeze(ct_display)
    
    if ct_display.ndim == 2:
        ct_rgb = np.stack([ct_display] * 3, axis=-1)
    else:
        ct_rgb = ct_display

    h = ct_rgb.shape[0]
    gap_col = np.zeros((h, gap, 3), dtype=np.uint8)

    panel = np.concatenate([ct_rgb, gap_col, gt_overlay, gap_col, pred_overlay], axis=1)
    return panel.astype(np.uint8)

def build_panel(ct_slice, gt_slice, pred_slice, color_map, rot_k=-1, alpha=0.5, gap=10):
    """
    If needs,you can use this to rotate the input image and stitch them together
    """
    ct_slice = rotateImage(ct_slice, k=rot_k)
    gt_slice = rotateImage(gt_slice, k=rot_k)
    pred_slice = rotateImage(pred_slice, k=rot_k)

    ct_display = normalize(ct_slice)
    gt_overlay = overlay_mask(ct_display, gt_slice, color_map, alpha=alpha)
    pred_overplay = overlay_mask(ct_display, pred_slice, color_map, alpha=alpha)

    return make_comparison(ct_display, gt_overlay, pred_overplay, gap=gap)

def build_multi_panel(vis_images, vis_gt_slices, pred_slices, color_map, gap=10):
    panels = []
    for vis_image, gt_slice, pred_slice in zip(vis_images, vis_gt_slices, pred_slices):
        ct_slice = vis_image[0, 0, :, :].cpu().numpy()
        panels.append(build_panel(ct_slice, gt_slice, pred_slice, color_map))

    w = panels[0].shape[1]
    gap_row = np.zeros((gap, w, 3), dtype=np.uint8)

    stacked = panels[0]
    for p in panels[1:]:
        stacked = np.concatenate([stacked, gap_row, p], axis=0)
    
    return stacked

def get_prediction(model, vis_images, image_size):
    """
    Perform a forward propagation to get predicted image
    """
    model.eval()
    preds = []
    with torch.no_grad():
       for vis_image in vis_images:
           outputs = model(batched_input=vis_image, multimask_output=True, image_size=image_size)
           pred_mask = torch.argmax(outputs['masks'], dim=1)
           preds.append(pred_mask.squeeze(0).cpu().numpy())
    model.train()
    return preds

class VisualizeTransform(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        if x!=self.output_size[0] or y!=self.output_size[1]:
            image = zoom(image, (self.output_size[0]/x, self.output_size[1]/y), order=3)
            label = zoom(label, (self.output_size[0]/x, self.output_size[1]/y), order=0)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
            image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}

        return sample