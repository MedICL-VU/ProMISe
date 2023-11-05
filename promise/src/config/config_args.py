import argparse
import os

parser = argparse.ArgumentParser()

# data
parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
parser.add_argument("--snapshot_path", default="", type=str, )
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--split", default="", type=str)
parser.add_argument("--num_worker", default=1, type=int)


# network
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--network_config", default="", type=str)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--rand_crop_size", default=0, nargs='+', type=int)
parser.add_argument("--checkpoint", default="best", type=str)
parser.add_argument("--num_prompts", default=1, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--tolerance", default=5, type=int)



# saving
parser.add_argument("--save_predictions", action="store_true")
parser.add_argument("--save_dir", default='', type=str, )
parser.add_argument("--save_name", default='', type=str, )
parser.add_argument("--save_csv", action="store_true")





def check_and_setup_parser(args):
    assert args.save_name != '', "[save_name] should has a real name"
    assert args.split != '', "[split] should be train/val/test"
    if args.checkpoint == "last":
        file = "last.pth.tar"
    else:
        file = "best.pth.tar"
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    if args.network_config == "":
        args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    else:
        args.snapshot_path = os.path.join(args.snapshot_path, args.data + '_' + args.network_config)

    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    return device, file
