# https://github.com/wdd614/pytorch-yolo/blob/master/FocalLoss.py
import torch
import torch.nn as nn
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
            reduce(bool): reduce(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field reduce is set to False, the losses are
                                instead summed for each minibatch.

    """
    def __init__(self, num_classes, alpha=None, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduce = reduce

    def forward(self, input, target):
        N = input.size(0) # batch_size
        C = input.size(1)
        P = nn.Softmax(dim=1)(input)

        class_mask = input.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = target.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        
        if input.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.reduce:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss