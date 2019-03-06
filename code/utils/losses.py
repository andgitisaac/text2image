import torch
from torch import nn

from utils.utils import roll

def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def compute_generator_loss(netD, fake_imgs, real_labels, conditions):
    criterion = nn.BCELoss()

    conditions = conditions.detach()

    fake_logits = netD(fake_imgs, conditions)
    errG_fake = criterion(fake_logits, real_labels)

    errG = errG_fake
    return errG
    

def compute_discriminator_loss(netD, imgs, labels, conditions):
    criterion = nn.BCELoss()

    real_imgs, fake_imgs = imgs
    real_labels, fake_labels = labels
    conditions, fake_imgs = conditions.detach(), fake_imgs.detach()

    # Real Pairs
    real_logits = netD(real_imgs, conditions)
    errD_real = criterion(real_logits, real_labels)

    # Wrong Pairs
    rolled_conditions = roll(conditions, shift=1, axis=0)
    wrong_logits = netD(real_imgs, rolled_conditions)
    errD_wrong = criterion(wrong_logits, fake_labels)

    # Fake Pairs
    fake_logits = netD(fake_imgs, conditions)
    errD_fake = criterion(fake_logits, fake_labels)

    errD = errD_real + (errD_wrong + errD_fake) / 2
    return errD, errD_real, errD_wrong, errD_fake

