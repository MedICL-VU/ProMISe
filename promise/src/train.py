import logging
from utils.util import setup_logger
from config.config_args import *
from processor.trainer import Trainer

def main():
    args = parser.parse_args()
    check_and_setup_parser(args)

    log_name = 'train_' + args.save_name
    setup_logger(logger_name=log_name, root=args.save_dir, screen=True, tofile=True)
    logger = logging.getLogger(log_name)
    logger.info(str(args))

    Trainer(args, logger).run()


if __name__ == "__main__":
    main()

