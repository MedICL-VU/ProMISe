import logging
import os
import time
import torch
import shutil
import numpy as np
import nibabel as nib
import pandas

def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)

    log_time = get_timestamp()
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, log_time))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg, log_time


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def get_points(args, seg, num_positive=10, num_negative=20):
    l = len(torch.where(seg == 1)[0])
    points_torch = None
    if l > 0:
        sample = np.random.choice(np.arange(l), num_positive, replace=True)
        x = torch.where(seg == 1)[1][sample].unsqueeze(1)
        y = torch.where(seg == 1)[3][sample].unsqueeze(1)
        z = torch.where(seg == 1)[2][sample].unsqueeze(1)
        points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
        points_torch = points.to(args.device)
        points_torch = points_torch.transpose(0, 1)
    l = len(torch.where(seg < 10)[0])
    sample = np.random.choice(np.arange(l), num_negative, replace=True)
    x = torch.where(seg < 10)[1][sample].unsqueeze(1)
    y = torch.where(seg < 10)[3][sample].unsqueeze(1)
    z = torch.where(seg < 10)[2][sample].unsqueeze(1)
    points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
    points_torch_negative = points.to(args.device)
    points_torch_negative = points_torch_negative.transpose(0, 1)
    if points_torch is not None:
        points_torch = torch.cat([points_torch, points_torch_negative], dim=1)
    else:
        points_torch = points_torch_negative

    return points_torch




def save_predict(args, logger,
        final_pred, seg, pred,
        points_dict,
        idx, test_data, image_data,
        patient_name, save_predict_dir):

    device = args.device
    save_predict_dir = os.path.join(save_predict_dir, 'predictions')
    if not os.path.exists(save_predict_dir):
        os.makedirs(save_predict_dir)

    x, y, z = points_dict['x_location'], points_dict['y_location'], points_dict['z_location']
    x_dimension, y_dimension, z_dimension = points_dict['x_dimension'], points_dict['y_dimension'], points_dict['z_dimension']
    # order see save_name and apply permute --> permute(test_data.dataset.spatial_index)
    new_x = torch.round(x * seg.shape[3] / x_dimension).long()
    new_y = torch.round(y * seg.shape[4] / y_dimension).long()
    new_z = torch.round(z * seg.shape[2] / z_dimension).long()

    # save_image_aug_path = os.path.join(save_predict_dir, patient_name.replace('.nii.gz', '_image_seed_' + str(seed) + '.nii.gz'))
    # nib.save(nib.Nifti1Image(img_orig_space[0, 0, :].permute(test_data.dataset.spatial_index).cpu().numpy(),
    #                          image_data.affine, image_data.header), save_image_aug_path)

    save_prediction_path = os.path.join(save_predict_dir, patient_name.replace('.nii.gz', '_prediction_seed_' + '.nii.gz'))
    nib.save(nib.Nifti1Image(pred[0, 0, :].permute(test_data.dataset.spatial_index).cpu().numpy(),
                             image_data.affine, image_data.header), save_prediction_path)

    seg_points = torch.zeros_like(seg).to(device)
    seg_points[0, 0, new_z, new_x, new_y] = 1
    save_point_path = os.path.join(save_predict_dir, patient_name.replace('.nii.gz', '_point_seed_' + '.nii.gz'))
    nib.save(nib.Nifti1Image(seg_points[0, 0, :].permute(test_data.dataset.spatial_index).cpu().numpy(),
                        image_data.affine, image_data.header), save_point_path)

    save_probability_path = os.path.join(save_predict_dir, patient_name.replace('.nii.gz', '_probability_seed_' + '.nii.gz'))
    nib.save(nib.Nifti1Image(final_pred[0, 0, :].permute(test_data.dataset.spatial_index).cpu().numpy(),
                        image_data.affine, image_data.header), save_probability_path)

    logger.info(
        "- Case {} - x {} | y {} | z{} | prediction saved".format(test_data.dataset.img_dict[idx],
                                                                  new_x.cpu().numpy()[0][0],
                                                                  new_y.cpu().numpy()[0][0],
                                                                  new_z.cpu().numpy()[0][0]
                                                                  ))

