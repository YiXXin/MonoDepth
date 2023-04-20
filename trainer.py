from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json
import os

from utils import *
from kitti_utils import *
from layers import *
# from networks import *


from torch import nn

import datasets
import networks
from networks import bi_encoder
from networks import net_utils


class Trainer:
    def __init__(self, options, _log):
        self.opt = options
        # self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self._log = _log
        _log.info("=> fetching img pairs.")
        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)  # 39810
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)  # 39810
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)  # 6635
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)  # 4424
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True) # 737
        self.val_iter = iter(self.val_loader)

        _log.info('{} samples found, {} train samples and {} test samples '.format(
        len(train_dataset) + len(val_dataset),
        len(train_dataset),
        len(val_dataset)))

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # from ipdb import set_trace 
        # set_trace()

        if self.opt.use_flow:
            _log.info("=> Training depth and optical flow jointly.")
        else:
            _log.info("=> Only training depth.")
        _log.info("Depth encoder is {}.".format(self.opt.depth_encoder))

        norm_cfg = dict(type='BN', requires_grad=True)

        if not self.opt.use_flow:
            if self.opt.depth_encoder == 'unet':
                self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")

                self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
            
            elif self.opt.depth_encoder == 'swin':
                self.models["encoder"] = networks.H_Transformer(window_size=4, embed_dim=64, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
                ckpt = torch.load('./checkpoints/imagenet/swin_checkpoint.pth', map_location='cpu')
                self.models["encoder"].load_state_dict(ckpt['encoder'])

                self.models["depth"] = networks.DCMNet(in_channels=[64, 128, 256, 512], in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6),
                                channels=128,
                                dropout_ratio=0.1,
                                num_classes=1,
                                norm_cfg=norm_cfg,
                                align_corners=False)
            
            elif self.opt.depth_encoder == 'biformer':
                self.models["encoder"] = bi_encoder.biformer_tiny()
                ckpt = torch.load('./checkpoints/imagenet/biformer_tiny_best.pth', map_location='cpu')
                i=0
                weight={}
                for k,v in ckpt['model'].items():
                    i=i+1
                    if i>6:
                        weight.update({k:v})
                # from ipdb import set_trace 
                # set_trace()
                self.models["encoder"].load_state_dict(weight, strict=False)

                self.models["depth"] = networks.DCMNet(in_channels=[64, 128, 256, 512], in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6),
                                                channels=128,
                                                dropout_ratio=0.1,
                                                num_classes=1,
                                                norm_cfg=norm_cfg,
                                                align_corners=False)
        else:
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")

            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)

            # self.models["flow"] = 


        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        # print(self.device)
        self.models["depth"].to(self.device)
        # print(self.models["depth"].parameters())
        self.parameters_to_train += list(self.models["depth"].parameters())


        self.models["pose_encoder"] = networks.ResnetEncoder(
            18,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)

        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)


        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        for model_name, model in self.models.items():
            self.models[model_name] = nn.DataParallel(model)

        if self.opt.predictive_mask:   # false
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        _log.info("Training is using {} with split {}.".format(self.device, self.opt.split))
        # print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.opt.save_root, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        # print("Using split:\n  ", self.opt.split)
        # print("There are {:d} training items and {:d} validation items\n".format(
        #     len(train_dataset), len(val_dataset)))

        # self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        from ipdb import set_trace 
        set_trace()

        self._log.info("=> Start training.")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:  # [6, 1, 375, 1242]
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)  # what's this
                self.val()

            self.step = self.step + 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        from ipdb import set_trace 
        set_trace()
        if not self.opt.use_flow:
        # inputs["color_aug", 0, 0].shape: [6, 3, 192, 640]
            features = self.models["encoder"](inputs["color_aug", 0, 0])
        # features[0]: [6, 64, 96, 320], features[1]: [6, 128, 48, 160], features[2]: [6, 256, 24, 80], features[3]: [6, 512, 12, 40]
            outputs = self.models["depth"](features)
        # outputs[('disp', 3)]: [6, 1, 24, 80], outputs[('disp', 2)]: [6, 1, 48, 160], outputs[('disp', 1)]:[6, 1, 96, 320], outputs[('disp', 0)]: [6, 1, 192, 640]

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        from ipdb import set_trace 
        set_trace()
        # In this setting, we compute the pose to each source frame via a
        # separate forward pass through the pose network.
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}  # keys: 0,-1,1

        for f_i in self.opt.frame_ids[1:]:  # [-1,1]
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            axisangle, translation = self.models["pose"](pose_inputs) # [4,2,1,3]
            outputs["pose", f_i] = torch.cat((torch.squeeze(axisangle),torch.squeeze(translation)),2)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def build_rigid_warp_flow(self):
        global n_iter
        # NOTE: this should be a python list,
        # since the sizes of different level of the pyramid are not same
        """
        Uses self.poses and self.depth, computed through build_posenet() and build_dispnet(), respectively
        """
        args = self.opt
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []

        #print(self.depth[0].shape)
        for scale in range(self.num_scales):    #num_scales is 4

            for src in range(args.num_source):  #num_source is 2
                # self.depth: (4, 12, _, _)
                # self.poses: (4, 2, 6)
                # self.multi_scale_intrinsices: (4, 4, 3, 3)
                                
                # (4, h, w, 2) for each particular scale
                fwd_rigid_flow = net_utils.compute_rigid_flow( # Checks out
                    self.poses[:, src, :],
                    self.depth[scale][:args.batch_size, :, :], #the first disparity
                    self.multi_scale_intrinsices[:, scale, :, :], False)
        
                # (4, h, w, 2)
                bwd_rigid_flow = net_utils.compute_rigid_flow(
                    self.poses[:, src, :],
                    self.depth[scale][args.batch_size * (
                        src + 1):args.batch_size * (src + 2), :, :],
                    self.multi_scale_intrinsices[:, scale, :, :], True)
                
                if not src:
                    fwd_rigid_flow_cat = fwd_rigid_flow
                    bwd_rigid_flow_cat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_cat = torch.cat(
                        (fwd_rigid_flow_cat, fwd_rigid_flow), dim=0)
                    bwd_rigid_flow_cat = torch.cat(
                        (bwd_rigid_flow_cat, bwd_rigid_flow), dim=0)
            
            # After the inner loop runs: fwd_rigid_flow_cat - (b*src_imgs, h, w, 2)
            
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_cat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_cat)

        #After the outer loop runs: fwd_rigid_flow_pyramid: (scales, b*src_imgs, h, w, 2) like (4, 8, h, w, 2)
        
        self.fwd_rigid_warp_pyramid = [
            net_utils.flow_warp(self.src_views_pyramid[scale],
                      self.fwd_rigid_flow_pyramid[scale])
            for scale in range(args.num_scales)
        ]
                
#         print(self.fwd_rigid_warp_pyramid[0].shape, self.fwd_rigid_warp_pyramid) - different
#         print(self.tmp_pyramid[0].shape, self.tmp_pyramid)
        
        self.bwd_rigid_warp_pyramid = [
            net_utils.flow_warp(self.tgt_view_tile_pyramid[scale],
                      self.bwd_rigid_flow_pyramid[scale])
            for scale in range(args.num_scales)
        ]

        #print(len(self.fwd_rigid_warp_pyramid), " ", self.fwd_rigid_warp_pyramid[0].size())
        #fwd_rigid_warp_pyramid: (8,128,416,3), (8,64,208,3), (8,32,104,3), (8,16,52,3)
        
        if n_iter % 10000 == 0:
            for j in range(len(self.fwd_rigid_warp_pyramid)):
                x = self.fwd_rigid_warp_pyramid[j].permute(0, 3, 1, 2)
                x = (x - torch.min(x))/(torch.max(x)-torch.min(x))
                self.tensorboard_writer.add_images('fwd_rigid_warp_scale' + str(j), x, n_iter)
 
            for j in range(len(self.bwd_rigid_warp_pyramid)):
                x = self.fwd_rigid_warp_pyramid[j].permute(0, 3, 1, 2)
                x = (x - torch.min(x))/(torch.max(x)-torch.min(x))
                self.tensorboard_writer.add_images('bwd_rigid_warp_scale' + str(j), x, n_iter)

        self.fwd_rigid_error_pyramid = [
            self.image_similarity(args.simi_alpha,
                             self.tgt_view_tile_pyramid[scale],
                             self.fwd_rigid_warp_pyramid[scale])
            for scale in range(args.num_scales)
        ]
        self.bwd_rigid_error_pyramid = [
            self.image_similarity(args.simi_alpha, self.src_views_pyramid[scale],
                             self.bwd_rigid_warp_pyramid[scale])
            for scale in range(args.num_scales)
        ]
        
        if n_iter % 10000 == 0:
            self.fwd_rigid_error_scale=[]
            self.bwd_rigid_error_scale=[]
            #fwd_rigid_error_pyramid[0]: (8, 3, 128, 416)

            for j in range(len(self.fwd_rigid_error_pyramid)):
                tmp=torch.mean(self.fwd_rigid_error_pyramid[j].permute(0, 3, 1, 2), dim=1, keepdim=True)
                #tmp: (8, 1, 128, 416) in 1st iteration
                self.tensorboard_writer.add_images('fwd_rigid_error_scale' + str(j), tmp, n_iter)
                self.fwd_rigid_error_scale.append(tmp)

            for j in range(len(self.bwd_rigid_error_pyramid)):
                tmp=torch.mean(self.bwd_rigid_error_pyramid[j].permute(0, 3, 1, 2), dim=1, keepdim=True)
                self.tensorboard_writer.add_images('bwd_rigid_error_scale' + str(j), tmp, n_iter)
                self.bwd_rigid_error_scale.append(tmp)

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """

        for scale in self.opt.scales:

            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]


                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss = loss + weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                        idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss = loss + to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss = loss + self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            loss = loss*10**(-scale)

            total_loss = total_loss + loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= (1 + 1e-1 + 1e-2 + 1e-3)


        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def image_similarity(alpha,x,y):
    # print('alpha*DSSIM(x,y): {:.16f}\n torch.abs(x-y): {:.16f}'.format(torch.mean(alpha*DSSIM(x,y)),torch.mean((1-alpha)*torch.abs(x-y))))
        return alpha * net_utils.DSSIM(x,y) + (1-alpha) * torch.abs(x - y)

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "bi_models_0414", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            #to_save = model.state_dict()
            to_save = model.module.state_dict()

            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
