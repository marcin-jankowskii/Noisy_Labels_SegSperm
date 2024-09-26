import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class QuadUNetPlusPlus(nn.Module):
    def __init__(self, in_chan=3, num_classes=4):
        super(QuadUNetPlusPlus, self).__init__()


        self.unet1 = smp.UnetPlusPlus(encoder_name='resnet18', in_channels=in_chan, classes=num_classes, encoder_weights=None)
        self.unet2 = smp.UnetPlusPlus(encoder_name='resnet18', in_channels=num_classes + in_chan, classes=num_classes, encoder_weights=None)
        self.unet3 = smp.UnetPlusPlus(encoder_name='resnet18', in_channels=num_classes + in_chan, classes=num_classes, encoder_weights=None)
        self.unet4 = smp.UnetPlusPlus(encoder_name='resnet18', in_channels=num_classes + in_chan, classes=num_classes, encoder_weights=None)

    def forward(self, x):

        output1 = self.unet1(x)
        combined_input = torch.cat([x, output1], dim=1)
        output2 = self.unet2(combined_input)
        output3 = self.unet3(combined_input)
        output4 = self.unet4(combined_input)

        return output1, output2, output3, output4
