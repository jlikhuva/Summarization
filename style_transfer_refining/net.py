import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce

def tensor_mean_std(t, dim, *args, **kwargs):
    num_elements = reduce(operator.mul, (t.shape[d] for d in dim), 1)
    t_mean = torch.mean(t, dim=dim, *args, **kwargs)
    return t_mean, torch.sum((t - t_mean) ** 2, dim=dim, *args, **kwargs) / num_elements

class StyleTransfer(nn.Module):
    '''
    Constructor
    Parameters:
    @param conv_layers - list of tuples (filter_size, stride, num_filters, adain) or ('MaxPool', stride, size, adain)
    '''
    def __init__(self, vocab_size, embed_size=100, hidden_dim=512, conv_layers=None, batch_norm=False):
        super(StyleTransfer, self).__init__()

        if conv_layers is None:
            vgg1 = [(3, 1, 64, True)]
            vgg2 = [(3, 1, 128, True)]
            vgg3 = [(3, 1, 256, True)]
            vgg4 = [(3, 1, 512, True)]
            vgg5 = [(3, 1, 512, True)]
            maxpool = [('MaxPool', 1, 2, True)]
            last_conv = [(3, 1, 1, True)]
            #conv_layers = vgg1 * 2 + maxpool + vgg2 * 2 + maxpool + vgg3 * 4 + maxpool + vgg4 * 4 + maxpool + vgg5 * 4 + last_conv
            conv_layers = vgg1 * 2 + maxpool + vgg2 * 2 + maxpool + last_conv
        self.conv_layers = conv_layers
        self.conv_weights = []
        self.adain_booleans = []

        prev_channels = 1
        for filter_size, stride, num_filters, adain in conv_layers:
            if filter_size == 'MaxPool':
                layer = nn.MaxPool2d(num_filters, stride=stride)
            else:
                layer = nn.Conv2d(prev_channels, num_filters, filter_size, stride=stride)
                prev_channels = num_filters
            self.conv_weights.append(layer)
            self.adain_booleans.append(adain)

            if batch_norm:
                self.conv_weights.append(nn.BatchNorm2d(prev_channels))
                self.adain_booleans.append(False)

            self.conv_weights.append(nn.RReLU())
            self.adain_booleans.append(False)
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.content_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim)
        self.style_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim)

    def forward(self, content, style):
        content, _ = self.content_lstm(self.embeddings(content))
        style, _ = self.style_lstm(self.embeddings(style))
        style = style.unsqueeze(1); content = content.unsqueeze(1)
        style_outputs = []
        prev_style_output = style
        for layer in self.conv_weights:
            new_style_output = layer(prev_style_output)
            prev_style_output = new_style_output
            style_outputs.append(new_style_output)

        prev_content_output = content
        for adain, layer, style_output in zip(self.adain_booleans, self.conv_weights, style_outputs):
            new_content_output = layer(prev_content_output)

            if adain:
                style_mean, style_std = tensor_mean_std(style_output, [-1, -2])
                content_mean, content_std = tensor_mean_std(new_content_output, [-1, -2])

                new_content_output = style_std * ((new_content_output - content_mean) / content_std) + style_mean
            prev_content_output = new_content_output

        return prev_content_output


class TwoDimensionalAttention(nn.Module):
    def __init__(self, hidden_size, nchannels, height, width):
        super(TwoDimensionalAttention, self).__init__()
        self.attention_weights_linear = nn.Linear(hidden_size+nchannels, height * width)
        self.attention_projection_linear = nn.Linear(hidden_size+nchannels, hidden_size)

        self.image_shape = (nchannels, height, width)

    def forward(self, image, hidden_state):
        assert(image.shape[1:] == self.image_shape)

        permuted_image = image.permute((0, 2, 3, 1))

        expanded_hidden = hidden_state.view(hidden_state.shape[0], 1, 1, hidden_state.shape[1])
        expanded_hidden = expanded_hidden.expand(-1, image.shape[-2], image.shape[-1], -1)

        concatenated_image = torch.cat([permuted_image, expanded_hidden], dim=-1)

        attention_weights = self.attention_weights_linear(concatenated_image)
        attention_weights /= torch.sum(attention_weights, dim=[-3, -2])

        attention_vectors = torch.sum(torch.mul(attention_weights, permuted_image), dim=[-3, -2])

        hidden_attention_concat = torch.cat([hidden_state, attention_vectors], dim=-1)

        new_hidden_vectors = self.attention_projection_linear(hidden_attention_concat)

        return new_hidden_vectors
