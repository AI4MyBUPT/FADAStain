# #将正负样本计算分开，加位置约束/不加位置约束
import numpy as np
import torch

import itertools
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from .gauss_pyramid import Gauss_Pyramid_Conv
import util.util as util

import PIL.Image as Image
import os
from torch.nn import init
import torch.nn as nn
from util import losses
from collections import defaultdict
from util.losses import MS_SSIM_Loss
from .PALS import MLPA_LOSS
from .PCLS import UNet_pro, CTPC_LOSS
from focal_frequency_loss import FocalFrequencyLoss as FFL
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class FADAStain(BaseModel):
    """ Contrastive Paired Translation (CPT).
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=0.1, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.set_defaults(pool_size=0)  # no image pooling

        # FDL:
        parser.add_argument('--lambda_gp', type=float, default=1.0, help='weight for Gaussian Pyramid reconstruction loss')
        parser.add_argument('--gp_weights', type=str, default='uniform', help='weights for reconstruction pyramids.')
        
        
        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=False,
                n_epochs=20, n_epochs_decay=10
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'ssim']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        
        self.feature_layers_pos=[0,4,8,12,16]
        self.feature_layers_neg=[0,4,8,12,16]


        self.batch_size = opt.batch_size
        self.image_order = {}
        self.my_dict = defaultdict(list)
        

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
    
        self.netF_pos = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_neg = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        if self.isTrain:
            self.netSeg = UNet_pro(in_chns=3,class_num=2)
            self.netSeg.load_state_dict(torch.load(f'pretrain/{opt.unet_seg}.pth'))
            self.netSeg = self.netSeg.to(self.device)

            self.train_dataset_size = opt.train_dataset_size

#-------------------------------------------#
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterion_ssim = MS_SSIM_Loss(data_range=1.0, size_average=True, channel=3)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionMLPA = MLPA_LOSS().to(self.device)
            

            self.ce_loss = torch.nn.modules.loss.CrossEntropyLoss().to(self.device)
            
            self.criterionCTPC = CTPC_LOSS#(weight,input_logits, target_logits)
            
           
            self.optimizer_seg = torch.optim.Adam(self.netSeg.parameters(), lr=self.opt.lr , betas=(self.opt.beta1, self.opt.beta2))
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr/20, betas=(opt.beta1, opt.beta2))
            
            self.optimizers.append(self.optimizer_seg)
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            

            if self.opt.lambda_gp > 0:
                self.P = Gauss_Pyramid_Conv(num_high=5)
                self.criterionGP = torch.nn.L1Loss().to(self.device)
                if self.opt.gp_weights == 'uniform':
                    self.gp_weights = [1.0] * 6
                else:
                    self.gp_weights = eval(self.opt.gp_weights)
                self.loss_names += ['GP']
                # self.loss_names += ['focal_frequency']
                self.loss_names += ['feature_all']
                

            


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data,data_init=1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_F_pos = torch.optim.Adam(self.netF_pos.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_F_neg = torch.optim.Adam(self.netF_neg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)
            self.optimizers.append(self.optimizer_F_pos)
            self.optimizers.append(self.optimizer_F_neg)          

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad([self.netG,self.netSeg], False)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD,self.netSeg], False)
        self.set_requires_grad([self.netG] ,True)
        self.optimizer_G.zero_grad()
        
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
            self.optimizer_F_pos.zero_grad()
            self.optimizer_F_neg.zero_grad()
        self.loss_G = self.compute_G_loss()
       
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
            self.optimizer_F_pos.step()
            self.optimizer_F_neg.step()

    def set_input(self, input, data_init = 0):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_name = []
        
        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']


    def forward(self):
        # self.netG.print()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D
    
    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
  
        fake = self.fake_B
        fake_B = self.netG(self.real_B)

        self.loss_G_cyc = self.criterionIdt(fake_B,self.real_B)
        self.loss_ssim = self.criterion_ssim(self.fake_B, self.real_B)
        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)

        with torch.no_grad():
            feat_real_B_pos = self.netG(self.real_B, self.feature_layers_pos, encode_only=True)
            feat_fake_B_pos = self.netG(self.fake_B, self.feature_layers_pos, encode_only=True)
            feat_real_B_neg = self.netG(self.real_B, self.feature_layers_neg, encode_only=True)
            feat_fake_B_neg = self.netG(self.fake_B, self.feature_layers_neg, encode_only=True)
        
        # MLPA: Multi-Level Protein Awareness Loss
        self.loss_MLPA, self.mask_A, self.mask_B = self.criterionMLPA(self.fake_B,self.real_B)

        if self.opt.nce_idt:
            feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0
        loss_NCE_all = self.loss_NCE

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF, self.nce_layers)
        else:
            self.loss_NCE_Y = 0.0
        loss_NCE_all += self.loss_NCE_Y

        self.loss_feature_all = 0     

        loss_feature_Y = self.calculate_feature_loss1(feat_fake_B_pos, feat_real_B_pos, feat_fake_B_neg, feat_real_B_neg, self.netF_pos, self.netF_neg, self.feature_layers_neg,self.feature_layers_pos)
        self.loss_feature_all += loss_feature_Y


        # FDL: compute loss on Gaussian pyramids
        if self.opt.lambda_gp > 0:
            p_fake_B = self.P(self.fake_B)
            p_real_B = self.P(self.real_B)
            loss_pyramid = [self.criterionGP(pf, pr) for pf, pr in zip(p_fake_B, p_real_B)]
            weights = self.gp_weights
            loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
            self.loss_GP = torch.mean(torch.stack(loss_pyramid)) * self.opt.lambda_gp
        else:
            self.loss_GP = 0
        
        self.loss_G = self.loss_G_GAN + loss_NCE_all + self.loss_GP  + 0.05* self.loss_ssim  + self.loss_feature_all

        return self.loss_G
    
    def calculate_NCE_loss(self, feat_src, feat_tgt, netF, nce_layers):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            
            loss = self.criterionNCE(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def calculate_feature_loss1(self, feat_src_pos, feat_tgt_pos, feat_src_neg, feat_tgt_neg, netF_pos, netF_neg, nce_layers_neg, nce_layers_pos):
        n_layers_pos = len(feat_src_pos)
        n_layers_neg = len(feat_src_neg)
        feat_q_neg = feat_tgt_neg
        feat_q_pos = feat_tgt_pos
        
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q_pos = [torch.flip(fq, [3]) for fq in feat_q_pos]
            feat_q_neg = [torch.flip(fq, [3]) for fq in feat_q_neg]
        
        feat_k_pos = feat_src_pos
        feat_k_neg = feat_src_neg

        patch_size_neg = 8
        patch_size_pos = 8
        num_patches_neg = 256
        num_patches_pos = 256
        num_positive_patches = 32
        num_negative_patches = 224
        stride_neg = 4
        stride_pos = 4
        
        feat_k_pool_neg, sample_ids_k_neg = netF_neg.forward2(feat_k_neg, num_patches_neg, patch_size_neg, None)
        feat_k_pool_pos, sample_ids_k_pos = netF_pos.forward2(feat_k_pos, num_patches_pos, patch_size_pos, None)
        feat_q_pool_neg, sample_ids_q_neg = netF_neg.forward2(feat_q_neg, num_patches_neg, patch_size_neg, None)
        feat_q_pool_pos, sample_ids_q_pos, all_num_poses, all_num_negs = netF_pos.forward3(
            feats=feat_q_pos,
            num_positive_patches=num_positive_patches,
            num_negative_patches=num_negative_patches,
            patch_size=patch_size_pos,
            patch_ids=None,
            stride=stride_pos,
            mask=self.mask_B
        )
        
        f_q_pos = torch.stack(feat_q_pool_pos, dim=0)
        f_q_neg = torch.stack(feat_q_pool_neg, dim=0)
        f_k_pos = torch.stack(feat_k_pool_pos, dim=0)
        f_k_neg = torch.stack(feat_k_pool_neg, dim=0)
        id_q_pos = torch.stack(sample_ids_q_pos, dim=0)
        id_q_neg = torch.stack(sample_ids_q_neg, dim=0)
        id_k_pos = torch.stack(sample_ids_k_pos, dim=0)
        id_k_neg = torch.stack(sample_ids_k_neg, dim=0)
        
        f_size = (feat_k_neg[0].shape[-1], feat_k_neg[0].shape[-1])

        visualize_this_step = self.should_visualize()
        if visualize_this_step:
            loss_pos, vis_data = self.calculate_l1_loss_pos(
                f_k_pos, f_q_pos, id_k_pos, id_q_pos, all_num_poses, all_num_negs, f_size, 
                pos_weight=0.1, threshold=0.95, visualize=True, 
                epoch=getattr(self, 'current_epoch', 0), 
                batch_idx=getattr(self, 'current_iter', 0)
            )
            
            if hasattr(self, 'real_B') and hasattr(self, 'fake_B'):
                self.visualize_matched_patches(
                    vis_data, self.real_B, self.fake_B,
                    epoch=getattr(self, 'current_epoch', 0), 
                    batch_idx=getattr(self, 'current_iter', 0)
                )
        else:
            loss_pos = self.calculate_l1_loss_pos(
                f_k_pos, f_q_pos, id_k_pos, id_q_pos, all_num_poses, all_num_negs, f_size, 
                pos_weight=0.1, threshold=0.95, visualize=False
            )
        
        loss_neg = self.calculate_l1_loss_neg(f_k_neg, f_q_neg, id_k_neg, id_q_neg, f_size, pos_weight=0.3) / n_layers_neg
        loss_pos = loss_pos / n_layers_pos

        decay_rate = 0.9
        decay_steps = 80
        base_weight = 1.0
        growth_rate = 1.1
        decay_steps_pos = 80
        
        decayed_weight = (decay_rate ** (self.current_epoch / decay_steps))
        current_weight = base_weight * (growth_rate ** (self.current_epoch / decay_steps_pos) - 1)  
        
        loss = decayed_weight * loss_neg + current_weight * loss_pos
        total_feature_loss = loss.mean()

        return total_feature_loss

    def calculate_l1_loss_pos(self, feat_k_pool, feat_q_pool, patch_ids_k, patch_ids_q, 
                     all_num_poses_q, all_num_negs_q, f_size, pos_weight=0.1, threshold=0.5,
                     visualize=False, epoch=None, batch_idx=None):
        """
        添加可视化功能的损失计算
        """
        total_loss = 0.0
        n_layers = feat_k_pool.shape[0]
        num_patches = 256
        
        visualization_data = {
            'matched_pairs': [],
            'similarity_scores': [],
            'layer_info': []
        }
        
        temperature = 0.07
        
        for layer_id in range(n_layers):
            layer_feat_k = feat_k_pool[layer_id].view(len(all_num_poses_q[layer_id]), -1, feat_k_pool[layer_id].shape[-1])
            layer_feat_q = feat_q_pool[layer_id].view(len(all_num_poses_q[layer_id]), -1, feat_q_pool[layer_id].shape[-1])
            
            layer_loss = 0.0
            layer_visualization = []
            
            for batch_id in range(len(all_num_poses_q[layer_id])):
                actual_pos = min(all_num_poses_q[layer_id][batch_id], 32)
                actual_neg = all_num_poses_q[layer_id][batch_id] + all_num_negs_q[layer_id][batch_id] - actual_pos
                
                sample_feat_q = layer_feat_q[batch_id]
                sample_feat_k = layer_feat_k[batch_id]
                
                sim_matrix = F.cosine_similarity(
                    sample_feat_q.unsqueeze(1),
                    sample_feat_k.unsqueeze(0),
                    dim=-1
                ) / temperature

                pos_sim = sim_matrix[:actual_pos]
                batch_visualization = {}
                
                if pos_sim.size(0) > 0:
                    pos_max_values, pos_max_indices = torch.max(pos_sim, dim=1)
                    mask = pos_max_values > threshold

                    valid_q_indices = torch.nonzero(mask).squeeze(1)
                    valid_k_indices = pos_max_indices[mask]

                    if valid_q_indices.numel() > 0:
                        if visualize:
                            batch_visualization = {
                                'layer_id': layer_id,
                                'batch_id': batch_id,
                                'q_indices': valid_q_indices.detach().cpu().numpy(),  
                                'k_indices': valid_k_indices.detach().cpu().numpy(),  
                                'similarities': pos_max_values[mask].detach().cpu().numpy(),  
                                'q_coords': patch_ids_q[layer_id][batch_id*num_patches:(batch_id+1)*num_patches][:actual_pos][mask].detach().cpu().numpy(),  # 添加 detach()
                                'k_coords': patch_ids_k[layer_id][batch_id*num_patches:(batch_id+1)*num_patches][pos_max_indices][mask].detach().cpu().numpy(),  # 添加 detach()
                                'threshold': threshold,
                                'num_matches': len(valid_q_indices)
                            }
                            layer_visualization.append(batch_visualization)
                        
                        valid_queries = sample_feat_q[:actual_pos][mask]
                        valid_targets = pos_max_indices[mask]
                        pos_logits = sim_matrix[valid_q_indices]
                        
                        contrastive_loss = F.cross_entropy(
                            pos_logits,
                            valid_targets,
                            reduction='sum'
                        )
                        
                        pos_coords_q = patch_ids_q[layer_id][batch_id*num_patches:(batch_id+1)*num_patches][:actual_pos][mask] / f_size[0]
                        pos_coords_k = patch_ids_k[layer_id][batch_id*num_patches:(batch_id+1)*num_patches][pos_max_indices][mask] / f_size[0]
                        pos_coord_loss = F.mse_loss(pos_coords_k, pos_coords_q, reduction='sum') if pos_coords_q.size(0) > 0 else 0.0
                    else:
                        contrastive_loss = 0.0
                        pos_coord_loss = 0.0
                else:
                    contrastive_loss = 0.0
                    pos_coord_loss = 0.0
                
                sample_loss = contrastive_loss + pos_weight * pos_coord_loss
                layer_loss += sample_loss
            
            if visualize and layer_visualization:
                visualization_data['layer_info'].extend(layer_visualization)
            
            total_loss += layer_loss / len(all_num_poses_q[layer_id])
        
        final_loss = total_loss / n_layers
        
        if visualize:
            return final_loss, visualization_data
        else:
            return final_loss


    def calculate_l1_loss_neg(self, feat_k_pool, feat_q_pool, patch_ids_k, patch_ids_q, f_size, pos_weight=0.1):
        assert feat_k_pool.shape == feat_q_pool.shape, "feat_k_pool 和 feat_q_pool 的形状必须相同"
        batch_size, num_patches, feature_dim = feat_k_pool.shape

        f_k_batch = feat_k_pool.unsqueeze(2)  
        f_q_batch = feat_q_pool.unsqueeze(1)  
        cos_sim = F.cosine_similarity(f_k_batch, f_q_batch, dim=-1) 
        min_sim_indices = torch.argmin(cos_sim, dim=2)
        min_sim_features = torch.gather(feat_q_pool, 1, min_sim_indices.unsqueeze(-1).expand(-1, -1, feature_dim))
        l1_loss = F.l1_loss(feat_k_pool, min_sim_features, reduction='sum')  

        f_size_norm = f_size[0]
        pos_k = patch_ids_k / f_size_norm  
        pos_q = torch.gather(patch_ids_q, 1, min_sim_indices.unsqueeze(-1).expand(-1, -1, 2)) / f_size_norm  
        pos_loss = F.mse_loss(pos_k, pos_q, reduction='sum')  
        total_loss = l1_loss + pos_weight * pos_loss
        return total_loss / batch_size


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>