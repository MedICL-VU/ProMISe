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
def load_data_set(args, split=''):
    test_data = load_data_volume(
                        data=args.data,
                        batch_size=1,
                        path_prefix=args.data_dir,
                        augmentation=False,
                        split=split,
                        rand_crop_spatial_size=args.rand_crop_size,
                        convert_to_sam=False,
                        do_test_crop=False,
                        deterministic=True,
                        num_worker=0,
                    )
    return test_data



def load_model(args):
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
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
        img_encoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["encoder_dict"], strict=True)
        img_encoder.to(args.device)
    else:
        # please download pretrained SAM model (vit_b), and put it in the "/src/ckpl"
        sam = sam_model_registry["vit_b"](checkpoint="src/ckpt/sam_vit_b_01ec64.pth")
        mask_generator = SamAutomaticMaskGenerator(sam)
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
            prompt_encoder.load_state_dict(
                torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["feature_dict"][i], strict=True)


        prompt_encoder.to(args.device)
        parameter_list.extend([i for i in prompt_encoder.parameters() if i.requires_grad == True])
        prompt_encoder_list.append(prompt_encoder)

    # mask decoder
    mask_decoder = VIT_MLAHead(img_size=96).to(args.device)
    mask_decoder.load_state_dict(torch.load(os.path.join(args.snapshot_path, file), map_location='cpu')["decoder_dict"],
                          strict=True)
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


def load_optimizer_scheduler_loss(args, model_dict, parameter_list):
    img_encoder, prompt_encoder_list, mask_decoder = model_dict['img_encoder'], model_dict['prompt_encoder_list'], \
        model_dict['mask_decoder']

    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=500)
    feature_opt = AdamW(parameter_list, lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=500)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=500)

    opt_sche_dict = {'image_encoder': {'optimizer': encoder_opt, 'scheduler': encoder_scheduler},
                     'prompt_encoder': {'optimizer': feature_opt, 'scheduler': feature_scheduler},
                     'decoder': {'optimizer': encoder_opt, 'scheduler': decoder_scheduler}
                     }
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)

    loss_dict = {'dice': dice_loss, 'dice_ce': loss_cal}
    return opt_sche_dict, loss_dict