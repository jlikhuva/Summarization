import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, conv_layers=None, batch_norm=False):
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
            
       
    def forward(self, content, style):
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
            