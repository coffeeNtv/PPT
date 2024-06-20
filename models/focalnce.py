from packaging import version
import torch
from torch import nn

import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
 
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)
 
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
       
        probs = (P*class_mask).sum(1).view(-1,1)
 
        log_p = probs.log()
 
        batch_loss = -self.alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

#reference: https://github.com/taesungp/contrastive-unpaired-translation/tree/master

class FocalNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.opt.nce_T = 0.07
        self.focal_loss = FocalLoss() 
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.opt.choose_patch = 100
        
    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        # neg logit
        batch_dim_for_bmm = self.opt.batchSize

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

       # l_neg, _ = l_neg.sort(dim=0, descending=True)
        #print("score",l_neg[:50, :])
       # l_neg = l_neg[:50, :]

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.focal_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
