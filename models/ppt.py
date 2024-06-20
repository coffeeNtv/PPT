import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import numpy as np

from .focalnce import FocalNCELoss
from .frequency_loss import Gauss_Pyramid_Conv
from .patch_alignment_loss import PatchAlignmentLoss
from .content_loss import VGGLoss

class PPT_model(BaseModel):

    def name(self):
        return 'PPT_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.nce_layers = '0,4,8,12,16'
        self.dropout = False
        self.init_gain=0.02
        self.no_antialias=False
        self.no_antialias_up=False
        self.num_patches = 256
        self.num_D = opt.num_D
        self.n_layers_D = opt.n_layers_D

        self.isTrain = opt.isTrain
        self.nce_layers = [int(i) for i in self.nce_layers.split(',')]

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.model_type, opt.norm, self.dropout, opt.init_type, self.init_gain, self.no_antialias, self.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.norm, self.dropout, opt.init_type, self.init_gain, self.no_antialias, self.gpu_ids, opt)
        self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.init_type, self.init_gain, self.no_antialias, self.gpu_ids, opt)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan

            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionNCE = FocalNCELoss(opt)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionVGG = VGGLoss(self.gpu_ids)
            self.P = Gauss_Pyramid_Conv(num_high=5)
            self.gp_weights = [1.0] * 6
            self.criterionMisalignment = PatchAlignmentLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.backward_D()                  # calculate gradients for D
            self.backward_G()                  # calculate graidents for G
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
            self.optimizers.append(self.optimizer_F)

    def set_input(self, input):
        if self.isTrain:
            self.real_A = input['A']
            self.real_B = input['B']
            if len(self.gpu_ids) > 0:
                self.real_A = input['A'].cuda(self.gpu_ids[0], non_blocking = True)
                self.real_B = input['B'].cuda(self.gpu_ids[0], non_blocking = True)
            self.image_paths = input['A_paths']
            if 'current_epoch' in input:
                self.current_epoch = input['current_epoch']
            if 'current_iter' in input:
                self.current_iter = input['current_iter']
        else:
            self.real_A = input['A']
            if len(self.gpu_ids) > 0:
                self.real_A = input['A'].cuda(self.gpu_ids[0], non_blocking = True)
            self.image_paths = input['A_paths']

    def forward(self):
        self.real = torch.cat((self.real_A, self.real_B), dim=0)
        self.fake = self.netG(self.real, layers=[])
        self.fake_B = self.fake[:self.real_A.size(0)]
        self.idt_B = self.fake[self.real_A.size(0):]

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
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
        return self.loss_D.backward()
   
    def backward_G(self):

        """Calculate GAN and NCE loss for the generator"""
        feat_real_A = self.netG(self.real_A, self.nce_layers, encode_only=True)
        feat_fake_B = self.netG(self.fake_B, self.nce_layers, encode_only=True)
        feat_real_B = self.netG(self.real_B, self.nce_layers, encode_only=True)
        feat_idt_B = self.netG(self.idt_B, self.nce_layers, encode_only=True)

        pred_fake = self.netD(self.fake_B)
        pred_real = self.netD(self.real_B)

        # advsarial loss
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean()

        # contrastive loss
        self.loss_NCE = self.calculate_NCE_loss(feat_real_A, feat_fake_B, self.netF)
        self.loss_NCE_Y = self.calculate_NCE_loss(feat_real_B, feat_idt_B, self.netF)
        self.loss_NCE_Y_hat = self.calculate_NCE_loss(feat_idt_B, feat_real_B, self.netF)
        self.loss_contrast = self.loss_NCE + self.loss_NCE_Y + self.loss_NCE_Y_hat
    
        # patch alignment loss
        self.loss_patch_alignment = self.criterionMisalignment(self.real_B,self.fake_B)
        
        # content loss
        self.loss_content = self.criterionVGG(self.fake_B, self.real_B)

        # freqency loss
        p_fake_B = self.P(self.fake_B)
        p_real_B = self.P(self.real_B)
        loss_pyramid = [self.criterionL1(pf, pr) for pf, pr in zip(p_fake_B, p_real_B)]
        weights = self.gp_weights
        loss_pyramid = [l * w for l, w in zip(loss_pyramid, weights)]
        self.loss_freq = torch.mean(torch.stack(loss_pyramid))
     
        # total loss
        self.loss_G = self.loss_G_GAN + self.loss_freq + self.loss_contrast + self.loss_patch_alignment + self.loss_content
        return self.loss_G.backward()
        
    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.fake_B = self.netG(self.real_A)

    def optimize_parameters(self):
        self.forward()

        # update D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update F
        self.optimizer_F.step()

    def get_current_errors(self):
        return OrderedDict([('loss_G_GAN', self.loss_G_GAN.item()),
                           ('loss_freq', self.loss_freq.item()),
                           ('loss_D', self.loss_D.item()),
                           ('loss_contrast', self.loss_contrast.item()),
                           ('loss_content', self.loss_content.item()),
                           ('loss_patch_alignment', self.loss_patch_alignment.item()),
                            ])
        
    def get_current_visuals(self):
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('fake_B',fake_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)


    def calculate_NCE_loss(self, feat_src, feat_tgt, netF):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.criterionNCE(f_q, f_k)
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
