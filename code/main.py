import os
import pickle
import logging

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

from utils.logger import LogWriter
from utils.losses import compute_generator_loss, compute_discriminator_loss, KL_loss
from utils.utils import weights_init, adjust_learning_rate, visualize_caption
from utils.vocab import Vocabulary
from fuel.dataset import ImgCapDataset
from fuel.dataset import train_transform, collate_fn
from fuel.model import TextEncoder, Global_G, Global_D

vocab_path = '../data/coco/captions/vocab.pickle'
root_path = '../data/coco'

max_epoch = 100
batch_size = 4
crop_size = 64

generator_lr = 0.0002
discriminator_lr = 0.0002
lr_decay_epoch = 20

summary_step = 5


vocab_size = 27297
word_dim = 300
embed_size = 256


logwriter = LogWriter(level=logging.INFO)
model_logger = logwriter.create_logger('model')
trainer_logger = logwriter.create_logger('trainer')


cudnn.benchmark = True
device = torch.device('cpu')
# device = torch.device('cuda')

log_dir = 'logs/'
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
summary_writer = SummaryWriter(log_dir=log_dir)

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)


dataset = ImgCapDataset(root_path, transform=train_transform(crop_size), split='train')
dataloader = iter(DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn))


text_encoder = TextEncoder(vocab_size, word_dim, embed_size)
text_encoder.load_state_dict(torch.load('/home/andgitisaac/text2image/models/text_encoder100.pth', map_location='cpu'))
text_encoder.eval()
text_encoder.to(device)

global_g = Global_G()
global_g.apply(weights_init)
global_g.train()
global_g.to(device)

global_d = Global_D()
global_d.apply(weights_init)
global_d.train()
global_d.to(device)


optimizer_G = optim.Adam(global_g.parameters(), lr=generator_lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(global_d.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))


for epoch in range(1, max_epoch+1):
    for i in tqdm(range(1, len(dataloader)+1)):
        global_step = (epoch - 1) * len(dataloader) + i

        if epoch % lr_decay_epoch == 0:
            adjust_learning_rate(optimizer_G, generator_lr, lr_decay_epoch)
            adjust_learning_rate(optimizer_D, discriminator_lr, lr_decay_epoch)


        noise = torch.FloatTensor(batch_size, 128).normal_(0, 1)
        real_labels = torch.FloatTensor(batch_size).fill_(1)
        fake_labels = torch.FloatTensor(batch_size).fill_(0)

        noise = noise.to(device)
        real_labels = real_labels.to(device)
        fake_labels = fake_labels.to(device)


        real_imgs, caps, cap_lens = next(dataloader)

        real_imgs = real_imgs.to(device)
        caps = caps.to(device)

        sent_caps, word_caps, cap_lens = text_encoder(caps, cap_lens, batch_size)
        sent_caps, word_caps = sent_caps.detach(), word_caps.detach()
        
        # trainer_logger.debug(
        #     'Require Grads => sent_caps:{}, word_caps:{}, imgs:{}'.format(
        #         sent_caps.requires_grad, word_caps.requires_grad, real_imgs.requires_grad
        #     )
        # )


        ### Generate Fake Image ###
        fake_imgs, mu, logvar = global_g(sent_caps, noise)


        ### Update D Network ###
        global_d.zero_grad()

        errD, errD_real, errD_wrong, errD_fake = \
            compute_discriminator_loss(
                                global_d, (real_imgs, fake_imgs),
                                (real_labels, fake_labels), mu
                            )

        errD.backward()
        optimizer_D.step()
                            

        ### Update G Network ###
        global_g.zero_grad()

        errG_fake = compute_generator_loss(global_d, fake_imgs, real_labels, mu)
        kl_loss = KL_loss(mu, logvar)

        kl_weight = 1
        errG = errG_fake + kl_loss * kl_weight

        errG.backward()
        optimizer_G.step()

        if (i+1) % summary_step == 0:
            
            captions = list()
            for (cap_len, cap_idx) in zip(cap_lens, caps.cpu().numpy()):
                cap = visualize_caption(vocab, cap_idx, cap_len)
                captions.append(' '.join(cap))
            captions = '\n'.join(captions)

            vutils.save_image(real_imgs[0], '{}.jpg'.format(global_step), normalize=True)

            real_imgs = vutils.make_grid(real_imgs[:4], nrow=4, normalize=True)
            fake_imgs = vutils.make_grid(fake_imgs[:4], nrow=4)            


            summary_writer.add_text('0_Captions', captions, global_step=global_step)

            summary_writer.add_image('1_Real Images', real_imgs, global_step=global_step)
            summary_writer.add_image('2_Fake Images', fake_imgs, global_step=global_step)

            summary_writer.add_scalars(
                '3_Discriminator Losses',
                {
                    'loss_D': errD.item(),
                    'loss_D_real': errD_real.item(),
                    'loss_D_wrong': errD_wrong.item(),
                    'loss_D_fake': errD_fake.item()
                },
                global_step=global_step
            )

            summary_writer.add_scalars(
                '4_Generator Losses',
                {
                    'loss_G': errG.item(),
                    'loss_G_fake': errG_fake.item(),
                    'loss_kl': kl_loss
                },
                global_step=global_step
            )




    
