import numpy as np
import logging
import torch.nn as nn
from utils.util import setup_logger, save_checkpoint
from config.config_args import *
from config.config_setup import load_data_set, load_model
from processor.trainer import trainer
from processor.validater import validater
import torch
from torch.optim import AdamW, lr_scheduler
from monai.losses import DiceCELoss, DiceLoss


def main():
    args = parser.parse_args()
    device, file = check_and_setup_parser(args)

    log_name = args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    train_data, val_data = load_data_set(args, split='train'), load_data_set(args, split='val')
    model_dict, parameter_list = load_model(args)
    img_encoder, prompt_encoder_list, mask_decoder = model_dict['img_encoder'], model_dict['prompt_encoder_list'], \
        model_dict['mask_decoder']

    encoder_opt = AdamW([i for i in img_encoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    encoder_scheduler = lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)

    prompt_opt = AdamW(parameter_list, lr=args.lr, weight_decay=0)
    prompt_scheduler = lr_scheduler.LinearLR(prompt_opt, start_factor=1.0, end_factor=0.01, total_iters=500)

    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)

    loss_boundary = nn.MSELoss()
    loss_segmentation = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    loss_validation = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")

    best_loss = np.inf
    pooling_layer = nn.AvgPool3d((args.boundary_kernel_size,args.boundary_kernel_size,1), stride=1,
                                 padding=(int((args.boundary_kernel_size-1)/2),int((args.boundary_kernel_size-1)/2),0)).cuda()


    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.train()
        for module in prompt_encoder_list:
            module.train()
        mask_decoder.train()

        # rewrite this as class later on
        trainer(args, logger, epoch_num, train_data, img_encoder, prompt_encoder_list, mask_decoder,
                                pooling_layer, encoder_opt, prompt_opt, decoder_opt,
                                loss_summary, loss_boundary, loss_segmentation
                                )

        encoder_scheduler.step()
        prompt_scheduler.step()
        decoder_scheduler.step()


        # validation
        img_encoder.eval()
        for module in prompt_encoder_list:
            module.eval()
        mask_decoder.eval()
        # rewrite this as class later on
        loss_summary_vali = validater(args, val_data, logger, epoch_num,
              img_encoder, prompt_encoder_list, mask_decoder, loss_validation)


        is_best = False
        if np.mean(loss_summary_vali) < best_loss:
            best_loss = np.mean(loss_summary_vali)
            is_best = True

        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": [i.state_dict() for i in prompt_encoder_list],
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": prompt_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.save_dir)
        logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()

