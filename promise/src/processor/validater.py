import torch
import torch.nn.functional as F
from src.utils.util import get_points
import numpy as np


def validater(args, val_data, logger, epoch_num,
          img_encoder, prompt_encoder_list, mask_decoder, loss_validation):
    patch_size = args.rand_crop_size[0]
    device = args.device
    with torch.no_grad():
        loss_summary = []
        for idx, (img, seg, spacing) in enumerate(val_data):
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1)
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)

            points_torch = get_points(args, seg, num_positive=10, num_negative=10, split='validation')

            new_feature = []
            for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
                if i == 3:
                    new_feature.append(
                        prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                    )
                else:
                    new_feature.append(feature)
            img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device),
                                       scale_factor=64 / patch_size,
                                       mode='trilinear')
            new_feature.append(img_resize)
            masks = mask_decoder(new_feature, 2, patch_size // 64)
            masks = masks.permute(0, 1, 4, 2, 3)
            seg = seg.to(device)
            seg = seg.unsqueeze(1)
            loss = loss_validation(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))
    return loss_summary