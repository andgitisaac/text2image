import logging

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def conv(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes,
                    kernel_size=3, stride=stride,
                    padding=1, bias=False)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )
    return block


def deconv(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 
                    kernel_size=4, stride=2,
                    padding=1, bias=False)


def downBlock(in_planes, out_planes):
    block = nn.Sequential(        
        deconv(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block




class TextEncoder(nn.Module):
    def __init__(self, vocab_size, word_dim, embed_size,
                num_layers=1, batch_first=True,
                drop_prob=0.5, bidirectional=True):
        super(TextEncoder, self).__init__()
        self.logger = logging.getLogger('model')

        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.num_layers = num_layers
        self.batch_first = batch_first
        self.drop_prob = drop_prob
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # number of features in the hidden state
        self.embed_size = embed_size // self.num_directions

        self.define_module()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.logger.info('Initializing embedding layer with range {}~{}'.format(-initrange, initrange))
        self.encoder.weight.data.uniform_(-initrange, initrange)
    
    def define_module(self):
        # Word Embedding
        self.encoder = nn.Embedding(self.vocab_size, self.word_dim)

        # Dropout Layer
        self.drop = nn.Dropout(self.drop_prob)

        # Caption Embedding    
        self.rnn = nn.LSTM(self.word_dim, self.embed_size,
                        num_layers=self.num_layers, batch_first=self.batch_first,
                        dropout=self.drop_prob, bidirectional=self.bidirectional)

    def forward(self, captions, lengths, batch_size):
        x = self.drop(self.encoder(captions)) # batch_size x seq_len --> batch_size x seq_len x word_dim

        packed = pack_padded_sequence(x, lengths, batch_first=self.batch_first)        
        output, hidden = self.rnn(packed) # output: (batch_size, seq_len, hidden_size * num_directions)

        output = pad_packed_sequence(output, batch_first=self.batch_first)[0]

        word_caps = output.transpose(1, 2)
        sentence_caps = hidden[0].transpose(0, 1).contiguous()
        sentence_caps = sentence_caps.view(-1, self.embed_size * self.num_directions)

        self.logger.debug('Sentence Embedding Size: {}'.format(sentence_caps.size()))
        self.logger.debug('Word Embedding Size: {}'.format(word_caps.size()))

        return sentence_caps, word_caps, lengths


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 256 # text embedding dim
        self.c_dim = 128 # generator condition dim
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class Global_G(nn.Module):
    def __init__(self):
        super(Global_G, self).__init__()

        self.logger = logging.getLogger('model')

        self.z_dim = 128
        self.c_dim = 128
        self.input_dim = self.z_dim + self.c_dim
        self.fc_dim = 1024

        self.define_module()
    
    def define_module(self):
        self.ca_net = CA_NET()

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_dim * 4 * 4, bias=False),
            nn.BatchNorm1d(self.fc_dim * 4 * 4),
            nn.ReLU(inplace=True)
        )

        self.upsample1 = upBlock(self.fc_dim, self.fc_dim // 2)
        self.upsample2 = upBlock(self.fc_dim // 2, self.fc_dim // 4)
        self.upsample3 = upBlock(self.fc_dim // 4, self.fc_dim // 8)
        self.upsample4 = upBlock(self.fc_dim // 8, self.fc_dim // 16)

        # To-RGB insteadc
        self.img = nn.Sequential(
            conv(self.fc_dim // 16, 3),
            nn.Tanh()
        )
    
    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)        
        self.logger.debug('Condition & noise size: {}'.format(z_c_code.size()))

        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.fc_dim, 4, 4)
        self.logger.debug('FC reshape size: {}'.format(h_code.size()))
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        self.logger.debug('Fake image size: {}'.format(fake_img.size()))
        return fake_img, mu, logvar
    


class Global_D(nn.Module):
    def __init__(self):
        super(Global_D, self).__init__()
        self.df_dim = 64
        self.c_dim = 128

        self.define_module()

    def define_module(self):
        self.downsample1 = downBlock(3, self.df_dim)
        self.downsample2 = downBlock(self.df_dim, self.df_dim * 2)
        self.downsample3 = downBlock(self.df_dim * 2, self.df_dim * 4)
        self.downsample4 = downBlock(self.df_dim * 4, self.df_dim * 8)
        # self.downsample5 = downBlock(self.df_dim * 8, self.df_dim * 16)

        # self.get_cond_logits = D_GET_LOGITS(self.df_dim, self.c_dim)
        # self.get_uncond_logits = None
        self.logit_layer = nn.Sequential(
            conv(self.df_dim * 8 + self.c_dim, self.df_dim * 8),
            nn.BatchNorm2d(self.df_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.df_dim * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

    def forward(self, image, c_code):
        h_code = self.downsample1(image)
        h_code = self.downsample2(h_code)
        h_code = self.downsample3(h_code)
        h_code = self.downsample4(h_code)
        # h_code = self.downsample5(h_code)
        
        H, W = h_code.size()[2:]
        c_code = c_code.view(-1, self.c_dim, 1, 1)
        c_code = c_code.repeat(1, 1, H, W)

        h_c_code = torch.cat((h_code, c_code), 1)

        output = self.logit_layer(h_c_code)
        output = output.view(-1)

        return output