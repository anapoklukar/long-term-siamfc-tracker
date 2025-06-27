from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)

        ######################################################################
        # cumulative responses and frames
        self.past_qt = 0
        self.n_frames = 0

        # number of samples for sampling strategy
        self.n_samples = 30

        # threshold for tracking uncertainty
        self.treshold = 3
        ######################################################################

        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 16,  # 32
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        ######################################################################
        self.img_size = img.shape[:2]
        ######################################################################
        
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)

    ######################################################################
    # Sampling strategies

    # Uniform sampling: samples uniformly around the center
    def uniform_sampling(self):
        sf = self.upscale_sz // 2
        x_samples = np.random.randint(1 + sf, self.img_size[1] - sf, self.n_samples)
        y_samples = np.random.randint(1 + sf, self.img_size[0] - sf, self.n_samples)
        samples = np.stack([y_samples, x_samples], axis=1)
        return samples

    # Gaussian sampling: samples from a Gaussian distribution centered at the target
    def gaussian_sampling(self):
        max_sf = self.upscale_sz / 2
        sf = min(self.upscale_sz * 1 / 2, max_sf)
        x_samples = np.random.normal(loc=self.center[1], scale=sf, size=self.n_samples)
        y_samples = np.random.normal(loc=self.center[0], scale=sf, size=self.n_samples)
        x_samples = np.clip(x_samples, 1 + sf, self.img_size[1] - sf)
        y_samples = np.clip(y_samples, 1 + sf, self.img_size[0] - sf)
        samples = np.stack([y_samples, x_samples], axis=1)
        return samples
    ######################################################################


    @torch.no_grad()
    def update(self, img):
        # Set the model to evaluation mode
        self.net.eval()

        def find_peak(response):
            response -= response.min()
            response /= (response.sum() + 1e-16)  # Normalize to sum to 1
            response = ((1 - self.cfg.window_influence) * response +
                        self.cfg.window_influence * self.hann_window)
            peak_loc = np.unravel_index(np.argmax(response), response.shape)
            return response, peak_loc

        # Crop search region at multiple scales around the current target center
        scaled_patches = [
            ops.crop_and_resize(
                img, self.center, self.x_sz * scale,
                out_size=self.cfg.instance_sz,
                border_value=self.avg_color
            )
            for scale in self.scale_factors
        ]

        # Convert to tensor and prepare for network input
        x = torch.from_numpy(np.stack(scaled_patches)).to(self.device).permute(0, 3, 1, 2).float()

        # Extract features using the backbone network
        features = self.net.backbone(x)

        # Compute response maps using the correlation head
        responses = self.net.head(self.kernel, features).squeeze(1).cpu().numpy()

        # Upsample response maps to a finer resolution
        responses = np.stack([
            cv2.resize(r, (self.upscale_sz, self.upscale_sz), interpolation=cv2.INTER_CUBIC)
            for r in responses
        ])

        # Penalize large scale changes to encourage stability
        num_scales = self.cfg.scale_num
        responses[:num_scales // 2] *= self.cfg.scale_penalty
        responses[num_scales // 2 + 1:] *= self.cfg.scale_penalty

        # Find the scale index with the strongest response
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        response = responses[scale_id]
        max_resp = max(0, response.max())  # Get peak response value

        # Compute PSR (Peak-to-Sidelobe Ratio) to estimate confidence
        mask = response < max_resp
        mean = np.mean(response[mask])
        std = np.std(response[mask])
        psr = (max_resp - mean) / (std + 1e-6)
        qt = max_resp * psr

        # Update average response to track confidence over time
        avg_response = (self.past_qt + qt) / (self.n_frames + 1)
        tracking_uncertainty = avg_response / qt

        if tracking_uncertainty >= self.treshold:
            # Tracking is uncertain: resample using uniform distribution
            samples = self.uniform_sampling()

            # ================== Visualization for debugging ==================
            # Uncomment these lines to visualize uniform samples and target area
            # img_draw = img.copy()
            # half_sz = self.upscale_sz // 2
            #
            # for y, x in samples:
            #     top_left = (int(x - half_sz), int(y - half_sz))
            #     bottom_right = (int(x + half_sz), int(y + half_sz))
            #     cv2.rectangle(img_draw, top_left, bottom_right, color=(0, 0, 255), thickness=1)
            #
            # top_left = (int(self.center[1] - half_sz), int(self.center[0] - half_sz))
            # bottom_right = (int(self.center[1] + half_sz), int(self.center[0] + half_sz))
            # cv2.rectangle(img_draw, top_left, bottom_right, color=(0, 255, 0), thickness=2)
            #
            # prediction = np.array([
            #     self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            #     self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            #     self.target_sz[1], self.target_sz[0]
            # ])
            #
            # tl_ = (int(round(prediction[0])), int(round(prediction[1])))
            # br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
            # cv2.rectangle(img_draw, tl_, br_, color=(0, 255, 0), thickness=1)
            #
            # cv2.imshow('Uniform Samples', img_draw)
            # cv2.imwrite(f'./uniform_samples_frame_{self.i}.png', img_draw)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ==================================================================

            # Evaluate all uniformly sampled patches
            patches = [
                ops.crop_and_resize(
                    img, center, self.x_sz,
                    out_size=self.cfg.instance_sz,
                    border_value=self.avg_color
                )
                for center in samples
            ]
            patches = torch.from_numpy(np.stack(patches)).to(self.device).permute(0, 3, 1, 2).float()
            features = self.net.backbone(patches)
            responses = self.net.head(self.kernel, features).squeeze(1).cpu().numpy()

            responses = np.stack([
                cv2.resize(r, (self.upscale_sz, self.upscale_sz), interpolation=cv2.INTER_CUBIC)
                for r in responses
            ])

            # Choose the best sample based on response peak
            scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
            response = responses[scale_id]
            _, peak_loc = find_peak(response)
            self.center = samples[scale_id].astype(np.float64)

        else:
            # Tracking is confident — update target center and scale
            self.past_qt += qt
            self.n_frames += 1
            response, peak_loc = find_peak(response)

            # Compute displacement from response peak
            disp = np.array(peak_loc) - (self.upscale_sz - 1) / 2
            disp *= self.cfg.total_stride / self.cfg.response_up
            disp *= self.x_sz * self.scale_factors[scale_id] / self.cfg.instance_sz
            self.center += disp

            # Update scale using learning rate
            scale = (1 - self.cfg.scale_lr) + self.cfg.scale_lr * self.scale_factors[scale_id]
            self.target_sz *= scale
            self.z_sz *= scale
            self.x_sz *= scale

        # Return updated bounding box in [x, y, width, height] format (1-indexed)
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1],
            self.target_sz[0]
        ])
        return box, max_resp

    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels