from src.dataset.datasets import load_data_volume
from src.models.image_encoder import Promise
from src.models.prompt_encoder import PromptEncoder, TwoWayTransformer
from src.models.mask_decoder import VIT_MLAHead
import os
import torch
from functools import partial
import torch.nn as nn
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.optim import AdamW
from monai.losses import DiceCELoss, DiceLoss
# def load_data_set(args, split=''):
#     if split == 'train':
#         data = load_data_volume(
#             data=args.data,
#             data_dir=args.data_dir,
#             batch_size=args.batch_size,
#             augmentation=True,
#             split=split,
#             rand_crop_spatial_size=args.rand_crop_size,
#             num_worker=args.num_worker
#         )
#     elif split == 'val':
#         data = load_data_volume(
#             data=args.data,
#             data_dir=args.data_dir,
#             batch_size=1,
#             augmentation=False,
#             split=split,
#             deterministic=True,
#             rand_crop_spatial_size=args.rand_crop_size,
#             num_worker=args.num_worker
#         )
#     else:
#         data = load_data_volume(
#         data=args.data,
#         batch_size=1,
#         data_dir=args.data_dir,
#         augmentation=False,
#         split=split,
#         rand_crop_spatial_size=args.rand_crop_size,
#         convert_to_sam=False,
#         do_test_crop=False,
#         deterministic=True,
#         num_worker=args.num_worker
#     )
#
#     return data


def load_data_set(args, split=''):

    if split == 'train':
        augmentation = True
        deterministic = False
    else:
        augmentation = False
        deterministic = True
        args.batch_size = 1

    data = load_data_volume(

        data=args.data,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augmentation=augmentation,
        split=split,
        deterministic=deterministic,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=args.num_worker,
    )

    return data


def load_model(args, logger):
    if args.split == 'test':
        if args.use_pretrain:
            file_path = args.pretrain_path
            logger.info("- using pretrained model: {}".format(args.pretrain_path))
        else:
            if args.checkpoint == "last":
                file = "last.pth.tar"
            else:
                file = "best.pth.tar"
            file_path = os.path.join(args.save_dir, file)
            logger.info("- using pretrained model: {}".format(file_path))
        pretrained_model = torch.load(file_path, map_location='cpu')
    else:
        # please download pretrained SAM model (vit_b), and put it in the "/src/ckpl"
        sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint_sam)
        mask_generator = SamAutomaticMaskGenerator(sam)

    # image encoder
    img_encoder = Promise(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            cubic_window_size=8,
            out_chans=256,
            num_slice=16)
    if args.split == 'test':
        img_encoder.load_state_dict(pretrained_model["encoder_dict"], strict=True)
        img_encoder.to(args.device)
    else:
        img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
        del sam
        img_encoder.to(args.device)
        for p in img_encoder.parameters():
            p.requires_grad = False
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.adapter.parameters():
                p.requires_grad = True
            for p in i.adapter_back.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for i in img_encoder.neck_3d:
            for p in i.parameters():
                p.requires_grad = True

    # prompt encoder
    parameter_list = []
    prompt_encoder_list = []
    for i in range(4):
        prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8))
        if args.split == 'test':
            prompt_encoder.load_state_dict(pretrained_model["feature_dict"][i], strict=True)


        prompt_encoder.to(args.device)
        parameter_list.extend([i for i in prompt_encoder.parameters() if i.requires_grad == True])
        prompt_encoder_list.append(prompt_encoder)

    # mask decoder
    mask_decoder = VIT_MLAHead(img_size=96, num_classes=2).to(args.device)
    if args.split == 'test':
        if 'pretrain_promise' in args.pretrain_path:
            print('using pretrained ProMISe')
            pretrained_model['decoder_dict']['head.0.weight'] = pretrained_model['decoder_dict']['cls_hao.0.weight']
            del pretrained_model['decoder_dict']['cls_hao.0.weight']
            pretrained_model['decoder_dict']['head.3.weight'] = pretrained_model['decoder_dict']['cls_hao.3.weight']
            del pretrained_model['decoder_dict']['cls_hao.3.weight']

        mask_decoder.load_state_dict(pretrained_model["decoder_dict"], strict=True)
    mask_decoder.to(args.device)


    model_dict = {'img_encoder': img_encoder, 'prompt_encoder_list': prompt_encoder_list, 'mask_decoder': mask_decoder}

    if args.split == 'test':
        img_encoder.eval()
        for i in prompt_encoder_list:
            i.eval()
        mask_decoder.eval()
        return model_dict
    else:
        return model_dict, parameter_list


