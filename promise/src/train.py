import numpy as np
import logging
import torch.nn as nn
from utils.util import setup_logger, save_checkpoint
from config.config_args import *
from config.config_setup import load_data_set, load_model, load_optimizer_scheduler_loss
from processor.trainer import trainer
from processor.valid import valid
def main():
    args = parser.parse_args()
    device, file = check_and_setup_parser(args)

    log_name = args.save_name
    setup_logger(logger_name=log_name, root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    train_data, val_data = load_data_set(args, split='train'), load_data_set(args, split='val')
    model_dict, parameter_list = load_model(args)
    img_encoder, prompt_encoder_list, mask_decoder = model_dict['img_encoder'], model_dict['prompt_encoder_list'], \
        model_dict['mask_decoder']
    opt_sche_dict, loss_dict = load_optimizer_scheduler_loss(args, model_dict, parameter_list)
    loss_dict['mse_loss'] = nn.MSELoss()
    best_loss = np.inf
    pooling_layer = nn.AvgPool3d((5,5,1), stride=1, padding=(2,2,0)).cuda()
    for epoch_num in range(args.max_epoch):
        loss_summary = []
        img_encoder.train()
        for module in prompt_encoder_list:
            module.train()
        mask_decoder.train()

        # rewrite this as class later on
        trainer(args, logger, epoch_num, train_data, img_encoder, prompt_encoder_list, mask_decoder,
                 pooling_layer, opt_sche_dict, loss_summary, loss_dict
                 )

        opt_sche_dict['image_encoder']['scheduler'].step()
        opt_sche_dict['prompt_encoder']['scheduler'].step()
        opt_sche_dict['decoder']['scheduler'].step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        img_encoder.eval()
        for module in prompt_encoder_list:
            module.eval()
        mask_decoder.eval()
        # rewrite this as class later on
        valid(args, val_data, logger, epoch_num,
              img_encoder, prompt_encoder_list, mask_decoder, loss_dict)

        logger.info("- Val metrics: " + str(np.mean(loss_summary)))


        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": [i.state_dict() for i in prompt_encoder_list],
                         "encoder_opt": opt_sche_dict['image_encoder']['optimizer'].state_dict(),
                         "feature_opt": opt_sche_dict['prompt_encoder']['optimizer'].state_dict(),
                         "decoder_opt": opt_sche_dict['decoder']['optimizer'].state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))


if __name__ == "__main__":
    main()

