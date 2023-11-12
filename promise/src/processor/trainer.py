import torch
import torch.nn.functional as F
from src.utils.util import get_points
def trainer(args, logger, epoch_num, train_data, img_encoder, prompt_encoder_list, mask_decoder,
             pooling_layer, encoder_opt, prompt_opt, decoder_opt,
                                loss_summary, loss_boundary, loss_segmentation
             ):

    patch_size = args.rand_crop_size[0]
    device = args.device
    for idx, (img, seg, spacing) in enumerate(train_data):
        out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
        input_batch = out.to(device)
        if args.batch_size == 1:
            input_batch = input_batch[0].transpose(0, 1)
        else:
            input_batch = input_batch.transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)

        points_torch = get_points(args, seg)
        new_feature = []
        for i, (feature, prompt_encoder) in enumerate(zip(feature_list, prompt_encoder_list)):
            if i == 3:
                new_feature.append(
                    prompt_encoder(feature, points_torch.clone(), [patch_size, patch_size, patch_size])
                )
            else:
                new_feature.append(feature)

        img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(device), scale_factor=64 / patch_size,
                                   mode='trilinear')
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, patch_size // 64)
        masks = masks.permute(0, 1, 4, 2, 3)

        seg = seg.to(device)
        seg = seg.unsqueeze(1)
        loss_dice = loss_segmentation(masks, seg)

        if seg.sum() > 0:
            seg_edge = abs(seg - pooling_layer(seg))
            mask_probs = F.softmax(masks, dim=1)
            mask_probs.requires_grad_(True)

            _, mask_binary = torch.max(mask_probs.data, 1)
            mask_binary = mask_binary.unsqueeze(1).float().requires_grad_(True)
            mask_edge = abs(mask_binary - pooling_layer(mask_binary))
            loss_distance = loss_boundary(mask_edge, seg_edge) * 10
        else:
            loss_distance = torch.tensor(0)
        loss = loss_dice + loss_distance
        loss_summary.append(loss_dice.detach().cpu().numpy())

        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        prompt_opt.zero_grad()

        loss.backward()
        logger.info(
            'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data))
            + ": loss:" + str(loss_summary[-1].flatten()[0])
            + ": loss_dice:" + str(round(loss_dice.item(), 4))
            + ": loss_distance:" + str(round(loss_distance.item(), 4))
        )

        logger.info(
            'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
                loss_summary[-1].flatten()[0]))
        torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(prompt_encoder_list[-1].parameters(), 1.0)

        encoder_opt.step()
        decoder_opt.step()
        prompt_opt.step()


    opt_sche_dict['image_encoder']['optimizer'] = encoder_opt
    opt_sche_dict['prompt_encoder']['optimizer'] = prompt_opt
    opt_sche_dict['decoder']['optimizer'] = decoder_opt
    return opt_sche_dict
