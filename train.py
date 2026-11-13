"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.landslide import landslide_fewshot
from dataloaders.transforms import (RandomMirror, RandomFlip, RandomRotation,
                                    ColorJitter, Resize, ToTensorNormalize)
from util.utils import set_seed, CLASS_LABELS
from config import ex


try:  # torch >= 2.0 mixed-precision API
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    def _make_autocast(enabled, dtype=torch.float16):
        return _autocast(device_type='cuda', dtype=dtype, enabled=enabled)

    def _make_grad_scaler(enabled):
        return _GradScaler(device='cuda', enabled=enabled)
except ImportError:  # older torch
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler

    def _make_autocast(enabled, dtype=torch.float16):
        return _autocast(enabled=enabled, dtype=dtype)

    def _make_grad_scaler(enabled):
        return _GradScaler(enabled=enabled)


def resolve_amp(amp):
    """
    Resolve the AMP setting into an autocast dtype.

    Args:
        amp:
            'auto'/True   -> bf16 if the GPU supports it, else fp16
            'bf16'        -> bf16 if supported, else fp16
            'fp16'        -> fp16
            False/None/'' -> AMP disabled
    Returns:
        torch.dtype to autocast to, or None when AMP is disabled / no GPU.
    """
    if not amp or not torch.cuda.is_available():
        return None
    requested = str(amp).lower()
    if requested in ('false', 'none', 'off', '0'):
        return None
    bf16_ok = torch.cuda.is_bf16_supported()
    if requested == 'fp16':
        return torch.float16
    # 'auto', 'true', 'bf16' or anything else: prefer bf16 when available
    return torch.bfloat16 if bf16_ok else torch.float16


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    model.train()


    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    elif data_name == 'landslide':
        make_data = landslide_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    # Landslide patches (128x128) are upsampled to a per-dataset size; the
    # Resize transform uses bilinear for the multispectral image and nearest
    # neighbour for the label.
    if data_name == 'landslide':
        input_size = _config['path']['landslide']['size']
    else:
        input_size = _config['input_size']
    transforms = [Resize(size=input_size)]
    if data_name == 'landslide':
        # Random flip + rotation (geometric) and colour perturbation on the
        # multispectral image only (see config['aug']).
        aug = _config['aug']
        if aug['flip']:
            transforms.append(RandomFlip())
        if aug['rotation'] > 0:
            transforms.append(RandomRotation(degrees=aug['rotation']))
        transforms.append(ColorJitter(**aug['color_jitter']))
    else:
        transforms.append(RandomMirror())
    transforms = Compose(transforms)
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=_config['num_workers'],
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])

    # Automatic mixed precision: bf16 when the GPU supports it, else fp16
    amp_dtype = resolve_amp(_config['amp'])
    use_amp = amp_dtype is not None
    # Gradient scaling is only needed for fp16 (bf16 has fp32's exponent range)
    scaler = _make_grad_scaler(use_amp and amp_dtype == torch.float16)
    _log.info(f'###### AMP: {amp_dtype if use_amp else "disabled"} ######')

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        with _make_autocast(use_amp, amp_dtype):
            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask,
                                           query_images)
            query_loss = criterion(query_pred, query_labels)
            loss = query_loss + align_loss * _config['align_loss_scaler']
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss


        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
