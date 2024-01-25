import torch
import torch.nn.functional as F
from src.utils.util import get_points
import numpy as np
from torch.optim import AdamW, lr_scheduler
from src.config.config_setup import load_data_set, load_model
from monai.losses import DiceCELoss, DiceLoss
import torch.nn as nn
from src.processor.validater import validater
from src.utils.util import save_checkpoint
import time

class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        self.train_data, self.val_data = load_data_set(args, split='train'), load_data_set(args, split='val')
        a = time.time()
        print('loading models and setting up')
        self.model_dict, self.parameter_list = load_model(args, logger)
        self.img_encoder = self.model_dict['img_encoder']
        self.prompt_encoder_list = self.model_dict['prompt_encoder_list']
        self.mask_decoder = self.model_dict['mask_decoder']

        self.best_loss, self.best_epoch = np.inf, 0
        self.pooling_layer = nn.AvgPool3d((self.args.boundary_kernel_size, self.args.boundary_kernel_size, 1), stride=1,
                                     padding=(int((self.args.boundary_kernel_size - 1) / 2),
                                              int((self.args.boundary_kernel_size - 1) / 2),
                                              0)).cuda()

        self.setup()
        print('models are loaded and others are set, spent {}'.format(round(time.time() - a, 4)))

    def run(self):

        for epoch_num in range(self.args.max_epoch):
            for module in self.prompt_encoder_list:
                module.train()
            self.mask_decoder.train()
            self.img_encoder.train()


            self.train(epoch_num)

            current_loss = self.validate(epoch_num)

            self.save_model(current_loss, epoch_num)
    def validate(self, epoch_num):
        self.img_encoder.eval()
        for module in self.prompt_encoder_list:
            module.eval()
        self.mask_decoder.eval()

        loss = validater(self.args, self.val_data, self.logger, epoch_num,
                         self.img_encoder, self.prompt_encoder_list, self.mask_decoder, self.loss_validation)


        return loss

    def train(self, epoch_num):

        loss_summary = []
        patch_size = self.args.rand_crop_size[0]
        device = self.args.device

        for idx, (img, seg, spacing) in enumerate(self.train_data):
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            input_batch = out.to(device)
            if self.args.batch_size == 1:
                input_batch = input_batch[0].transpose(0, 1)
            else:
                input_batch = input_batch.transpose(0, 1)
            batch_features, feature_list = self.img_encoder(input_batch) #batch_feature size (b,c,x,z,y) x,y,z is the image size after dataloader, ignore the transpose!
            feature_list.append(batch_features)

            points_torch = get_points(self.args, seg, split='train')



            new_feature = []
            for i, (feature, prompt_encoder) in enumerate(zip(feature_list, self.prompt_encoder_list)):
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
            masks = self.mask_decoder(new_feature, 2, patch_size // 64)
            masks = masks.permute(0, 1, 4, 2, 3)

            seg = seg.to(device)
            seg = seg.unsqueeze(1)

            loss_dice = self.loss_segmentation(masks, seg)

            if seg.sum() > 0:
                seg_edge = abs(seg - self.pooling_layer(seg))
                mask_probs = torch.softmax(masks, dim=1)
                mask_edge = abs(mask_probs - self.pooling_layer(mask_probs))
                loss_distance = self.loss_boundary(mask_edge, seg_edge) * 10

                # mask_relu = nn.ReLU()
                # mask_probs_relu = mask_relu(mask_probs-0.5)
                # mask_loss = torch.ones_like(mask_probs_relu)
                # mask_loss[mask_probs_relu != 0] = mask_probs_relu[mask_probs_relu != 0] / mask_probs_relu[mask_probs_relu != 0]
                # mask_loss = mask_loss[0, 1, :].unsqueeze(0).unsqueeze(0)
                # mask_edge = abs(mask_loss - self.pooling_layer(mask_loss))
                # loss_distance = self.loss_boundary(mask_probs, seg_edge) * 10
            else:
                loss_distance = torch.tensor(0)
            loss = loss_dice + loss_distance
            loss_summary.append(loss.detach().cpu().numpy())

            self.encoder_opt.zero_grad()
            self.prompt_opt.zero_grad()
            self.decoder_opt.zero_grad()

            loss.backward()
            self.logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.train_data))
                + ": loss:" + str(round(loss_summary[-1].flatten()[0], 4))
                + ": loss_dice:" + str(round(loss_dice.item(), 4))
                + ": loss_distance:" + str(round(loss_distance.item(), 4))
            )

            torch.nn.utils.clip_grad_norm_(self.img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.prompt_encoder_list[-1].parameters(), 1.0)

            self.encoder_opt.step()
            self.prompt_opt.step()
            self.decoder_opt.step()

        self.encoder_scheduler.step()
        self.prompt_scheduler.step()
        self.decoder_scheduler.step()
        self.logger.info("- Train metrics: " + str(np.mean(loss_summary)))


    def save_model(self, current_loss, epoch_num):
        is_best = False
        if np.mean(current_loss) < self.best_loss:
            self.best_loss = np.mean(current_loss)
            self.best_epoch = epoch_num
            is_best = True

        save_checkpoint({"epoch": epoch_num + 1,
                         "best_val_loss": self.best_loss,
                         "encoder_dict": self.img_encoder.state_dict(),
                         "decoder_dict": self.mask_decoder.state_dict(),
                         "feature_dict": [i.state_dict() for i in self.prompt_encoder_list],
                         "encoder_opt": self.encoder_opt.state_dict(),
                         "feature_opt": self.prompt_opt.state_dict(),
                         "decoder_opt": self.decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=self.args.save_dir)
        self.logger.info("- Val metrics best: {} at epoch {} " .format(self.best_loss, self.best_epoch))



    def setup(self):
        self.setup_loss()
        self.setup_optimizier()
        self.setup_scheduler()
    def setup_loss(self):
        self.loss_boundary = nn.MSELoss()
        self.loss_segmentation = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5,
                                       lambda_ce=0.5)
        self.loss_validation = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    def setup_optimizier(self):
        self.encoder_opt = AdamW([i for i in self.img_encoder.parameters() if i.requires_grad == True], lr=self.args.lr,
                            weight_decay=0)
        self.prompt_opt = AdamW(self.parameter_list, lr=self.args.lr, weight_decay=0)
        self.decoder_opt = AdamW([i for i in self.mask_decoder.parameters() if i.requires_grad == True], lr=self.args.lr,
                            weight_decay=0)
    def setup_scheduler(self):
        self.encoder_scheduler = lr_scheduler.LinearLR(self.encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
        self.prompt_scheduler = lr_scheduler.LinearLR(self.prompt_opt, start_factor=1.0, end_factor=0.01, total_iters=500)
        self.decoder_scheduler = lr_scheduler.LinearLR(self.decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)

