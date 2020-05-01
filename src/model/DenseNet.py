import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from inplace_abn import InPlaceABN

class BottleNeck(nn.Sequential) :
    def __init__(self, num_input_features, bottle_neck_size, growth_rate, bias,) :
        super(BottleNeck, self).__init__()
        #self.add_module('norm', nn.BatchNorm2d(num_input_features), )
        #self.add_module('relu', nn.ReLU(inplace=True), )
        self.add_module('abn', InPlaceABN(num_input_features), )
        self.add_module('conv', nn.Conv2d(num_input_features, bottle_neck_size * growth_rate, 
                                          kernel_size=1, stride=1, bias=bias,), )

class _DenseLayer(nn.Module) :
    def __init__(self, num_input_features, growth_rate, bottle_neck_size, 
                 dropout_rate, bias, memory_efficient ) :
        super(_DenseLayer, self).__init__()
        if bottle_neck_size > 0 :
            self.bottle_neck = True
            self.add_module('bottle_neck_layer', BottleNeck(num_input_features, bottle_neck_size, 
                                                      growth_rate, bias, ) )
            self.add_module('layer', nn.Sequential( 
                #nn.BatchNorm2d(bottle_neck_size * growth_rate),
                #nn.ReLU(inplace=True),
                InPlaceABN(bottle_neck_size * growth_rate),
                nn.Conv2d(bottle_neck_size * growth_rate, growth_rate,
                          kernel_size=3, stride=1, padding=1, bias=bias,),
                nn.Dropout(dropout_rate, inplace = True),
            ) )
        else :
            self.bottle_neck = False
            self.add_module('layer', nn.Sequential( 
                #nn.BatchNorm2d(num_input_features),
                #nn.ReLU(inplace=True),
                InPlaceABN(num_input_features),
                nn.Conv2d(num_input_features, growth_rate,
                          kernel_size=3, stride=1, padding=1, bias=bias,),
                nn.Dropout(dropout_rate, inplace = True),
            ) )
        self.memory_efficient = memory_efficient
    def concat_features(self, inputs ) :
        return torch.cat( inputs, dim = 1 )
    def memory_checkpoint(self, func) :
        def closure(inputs) :
            concat_features = self.concat_features(inputs)
            return func(concat_features)
        return closure
                            
    def forward(self, prev_features) :
        if isinstance(prev_features, Tensor) :
            prev_features = [prev_features]
        
        if self.memory_efficient and any([tensor.requires_grad for tensor in prev_features]) :
            if self.bottle_neck :
                bottle_neck_features = cp.checkpoint(self.memory_checkpoint(self.bottle_neck_layer), prev_features)
                output = self.layer(bottle_neck_features)
            else :
                output = memory_checkpoint(self.layer, prev_features)
        else :
            if self.bottle_neck :
                bottle_neck_features = self.bottle_neck_layer(self.concat_features(prev_features))
                output = self.layer(bottle_neck_features)
            else :
                output = self.layer(self.concat_features(prev_features))
        return output
    
class DenseBlock(nn.ModuleDict) :
    def __init__(self,num_input_features, num_layer, growth_rate, bottle_neck_size,
                 dropout_rate, bias, memory_efficient) :
        super(DenseBlock, self).__init__()
        for n_th_layer in range(num_layer) :
            layer = _DenseLayer(
                num_input_features + n_th_layer * growth_rate, 
                growth_rate, 
                bottle_neck_size, 
                dropout_rate, 
                bias, 
                memory_efficient )
            self.add_module('DenseLayer_%d' % n_th_layer, layer)
    def forward(self, inputs) :
        features = [inputs]
        for name, layer in self.items() :
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, dim = 1)

class Transition(nn.Sequential) :
    def __init__(self, num_input_features, num_output_features, bias):
        super(Transition, self).__init__()
        #self.add_module('norm', nn.BatchNorm2d(num_input_features))
        #self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('abn', InPlaceABN(num_input_features), )
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=bias,))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
    
class DenseNet(nn.Module) :
    def __init__(self, growth_rate=12, DenseBlock_layer_num=(40,40,40), 
                 bottle_neck_size=4, dropout_rate=0.2, compression_rate=0.5, num_init_features = 16,
                 num_input_features=3, num_classes=10, bias=False, memory_efficient=False) :
        super(DenseNet, self).__init__()
        self.features_layers = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_features,
                                num_init_features, 
                                kernel_size=5, 
                                stride=1,
                                padding=2,
                                bias=bias, )),
            #('norm0', nn.BatchNorm2d(num_init_features)),
            #('relu0', nn.ReLU(inplace=True)),
            ('abn0', InPlaceABN(num_init_features), ),
            #('pool0', nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        num_features = num_init_features
        for idx, num_layers in enumerate( DenseBlock_layer_num ) :
            layer = DenseBlock( num_features, 
                               num_layers, 
                               growth_rate, 
                               bottle_neck_size,
                               dropout_rate, 
                               bias, 
                               memory_efficient)
            self.features_layers.add_module('DenseBlock%d' % idx, layer)
            num_features += growth_rate * num_layers
            if idx < len(DenseBlock_layer_num) - 1 :
                num_output_features=int((1-compression_rate) * num_features)
                Transition_layer = Transition(num_features, num_output_features, bias)
                self.features_layers.add_module('Transition%d' % idx, Transition_layer)
                num_features = num_output_features
        idx+=1
        #self.features_layers.add_module('norm%d' % idx, nn.BatchNorm2d(num_features))
        #self.features_layers.add_module('relu%d' % idx, nn.ReLU(inplace=True))
        self.add_module('abn%d' % idx, InPlaceABN(num_features), )
        self.features_layers.add_module('GlobalAvgPool%d' % idx, nn.AdaptiveAvgPool2d(1))
        
        self.classifier = nn.Linear(num_features, num_classes, bias=True,)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        features = self.features_layers(x)
        out = torch.flatten(features, 1)
        out = self.classifier(out)
        return out