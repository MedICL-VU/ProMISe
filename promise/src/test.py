from processor.tester import tester
import logging
from monai.losses import DiceLoss
import torch
from utils.util import setup_logger
from config.config_args import *
from config.config_setup import load_data_set, load_model
def main():
    args = parser.parse_args()
    check_and_setup_parser(args)

    log_name = 'test_' + args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    test_data = load_data_set(args, split='test')
    model_dict = load_model(args, logger)

    dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")

    loss_summary, loss_nsd = [], []
    with torch.no_grad():
        tester(args, logger, model_dict, test_data, loss_summary, loss_nsd, dice_loss)

    logger.info("- Test done")
if __name__ == "__main__":
    main()

