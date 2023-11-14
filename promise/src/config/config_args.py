import argparse
import os
import warnings

parser = argparse.ArgumentParser()





# data
parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
parser.add_argument("--save_dir", default="", type=str)
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--num_worker", default=6, type=int)
parser.add_argument("--split", default="train", type=str)



# network
parser.add_argument("--lr", default=4e-4, type=float)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--max_epoch", default=200, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--rand_crop_size", default=0, nargs='+', type=int)
parser.add_argument("--checkpoint", default="best", type=str)
parser.add_argument("--checkpoint_sam", default="./checkpoint_sam/sam_vit_b_01ec64.pth", type=str,
                    help='path of pretrained SAM')
parser.add_argument("--num_prompts", default=1, type=int)
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--tolerance", default=5, type=int)
parser.add_argument("--boundary_kernel_size", default=5, type=int,
                    help='an integer for kernel size of avepooling layer for boundary generation')
parser.add_argument("--use_pretrain", action="store_true")
parser.add_argument("--pretrain_path", default="", type=str)

# saving
parser.add_argument("--save_predictions", action="store_true")
parser.add_argument("--save_csv", action="store_true")
parser.add_argument("--save_base_dir", default='', type=str)
parser.add_argument("--save_name", default='testing_only', type=str)






def check_and_setup_parser(args):
    if args.save_name == 'testing_only':
        warnings.warn("[save_name] (--save_name) should be a real name, currently is for testing purpose (--save_name=testing_only)")

    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["colon", "pancreas", "lits", "kits"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)

    args.save_dir = os.path.join(args.save_dir, args.data, args.save_name)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    return device
