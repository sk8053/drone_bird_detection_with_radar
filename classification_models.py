# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:20:34 2023

@author: seongjoon kang
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class MyDataSet(Dataset):
  def __init__(self, image, target):
    self.image = image
    self.target = target
  def __len__(self):
    return len(self.target)
  def __getitem__(self, idx):
    return self.image[idx], self.target[idx]

class View(nn.Module):
    def __init__(self, shape):
      super().__init__()
      self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)
    


class Res_Convnet(nn.Module):
    def __init__(self, input_channels, num_classes):
        """
        Parameters
        ----------
        input_channels : Number of input channels
        num_classes : Number of classes for the final prediction 
        """
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
          
            )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=2, padding=1,stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            )
        self.resid_connection = nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=2, padding =1, stride =2)
        
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding=1,stride = 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        
            nn.Flatten(),
            #nn.Linear(17*17*32, num_classes),
            nn.Linear(11*16*64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
            )


    def forward(self, x):
        """
        Parameters
        ----------
        x
        Returns
        -------
        output : Result after running through the model
        """

        output = self.block_1(x)
        output = self.block_2(output) + self.resid_connection(output)
        output = self.block_3(output)
        return output

# the convolutional lstm code is referenced from the following link:
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        __padding_map={2:1, 3:1, 4:2} # padding values to keep the image size the same
        __dilation_map = {2:2, 3:1, 4:1} # the corresponding dilation valuees to keep the same size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = [__padding_map[kernel_size[0]], __padding_map[kernel_size[1]]]
        self.dilation = [__dilation_map[kernel_size[0]], __dilation_map[kernel_size[1]]]
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              dilation = self.dilation,
                              bias=self.bias)
                             
                    
        self.actv = nn.PReLU()
        self.norm = nn.BatchNorm2d(4 * self.hidden_dim)
        
    def forward(self, input_tensor, cur_state):
        H_t, C_t = cur_state
        combined = torch.cat([input_tensor, H_t], dim=1)  # concatenate along channel axis
        combined_conv = self.norm(self.actv(self.conv(combined)))
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
      
        c_next = f * C_t + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, 
                 return_all_layers=False, img_size = np.array([20,30])):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = [kernel_size]*num_layers
        hidden_dim = [hidden_dim]*num_layers
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
       
        lstm_cell_list = []
        cur_input_dim = self.input_dim
        for i in range(0, self.num_layers):
             
            lstm_cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
            cur_input_dim = self.hidden_dim[i]
            

        self.lstm_cell_list = nn.ModuleList(lstm_cell_list)
        
        self.linear_fc = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(hidden_dim[0]* img_size[0]*img_size[1], 100),    
            nn.PReLU(),
            nn.Linear(100, 2),     
        )
    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output, class_labels
        """
        if not self.batch_first: # always, it should be batch first 
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
      
        
        for layer_idx in range(self.num_layers):
            # for each LSTM layer, use the initial hidden state values
            # which is set as zeros
            h, c = hidden_state[layer_idx]
            
            output_inner = []
            for t in range(seq_len):
                h, c = self.lstm_cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
        
        # take the last hidden and cell states
        last_states = last_state_list[-1] 
        last_h,last_c = last_states
        
        
        # input vector to FCC is last hidden state, H_t
        input_fc = last_h
        class_labels = self.linear_fc(input_fc)

        return layer_output_list, last_state_list, class_labels

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.lstm_cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')






'''
#class VGG16(nn.Module):
class ResidualConvnet(nn.Module):
        
    def __init__(self, input_channels=1, num_classes=2):
        super(ResidualConvnet, self).__init__()
        ## first layer
        self.first_layer = nn.Sequential(
                  nn.Conv2d(in_channels = input_channels, out_channels = 16, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16))
        
        ## first block
        self.first_block = nn.Sequential(
                  nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),
                  
                  nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(16),

                  nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size=3, padding=1),
                  nn.ReLU()
        )
        self.first_BatchNorm_MaxPool = nn.Sequential(         ##
                  nn.BatchNorm2d(16),
                  nn.MaxPool2d(kernel_size=2, stride=2)  # it halves the resolution of the image             
        ) 
         
        # second block
        self.second_block = nn.Sequential(
          nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(32),
         
          nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, padding=1),
          nn.ReLU(),
         nn.BatchNorm2d(32),

          nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, padding=1),
          nn.ReLU()
          )
        self.second_BatchNorm_MaxPool = nn.Sequential(
          nn.BatchNorm2d(32),
          nn.MaxPool2d(kernel_size=2, stride=2)  # it halves the resolution of the image             
            )
          # change the number of input channel as that of output chanel
        self.conv_64_128 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=3, padding=1)

        ## thrid block
        self.third_block = nn.Sequential(
          nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding=1),
          nn.ReLU(),
          nn.BatchNorm2d(64),

          nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1),
          nn.ReLU(), 
          nn.BatchNorm2d(64),

          nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1),
          nn.ReLU()
          )
        self.thrid_BatchNorm_MaxPool = nn.Sequential(
          nn.BatchNorm2d(64),
          nn.MaxPool2d(kernel_size=8)  # it makes the resolution 1*1 size         
            )
          # change the number of input channel as that of output chanel
        self.conv_128_256 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding=1)

        # projection layer
        self.projection = nn.Sequential(
            View((-1,64)),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # TODO 

        # first conv layer 
        x_first_layer = self.first_layer(x)

        # let's make residual net here
        x_first_b = self.first_block(x_first_layer) + x_first_layer
        # after then do batch normalization
        x_first_b = self.first_BatchNorm_MaxPool(x_first_b) # channel number is 64

        #  make residual net but we need to make the number of chanel same
        x_second_b = self.second_block(x_first_b) + self.conv_64_128(x_first_b) # chanel number is 128
        x_second_b = self.second_BatchNorm_MaxPool(x_second_b) # bathc normalization

        # in the same way
        x_third_b = self.third_block (x_second_b) + self.conv_128_256(x_second_b)
        x_third_b = self.thrid_BatchNorm_MaxPool(x_third_b)

        #  projection 
        output = self.projection(x_third_b)
        
        return output

'''