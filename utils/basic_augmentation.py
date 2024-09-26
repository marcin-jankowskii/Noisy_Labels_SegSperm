import torch
import torch.nn as nn
import kornia as K
import kornia.augmentation as Ka


class BasicAugmentation(nn.Module):
    def __init__(self):
        super(BasicAugmentation, self).__init__()
        self.k1 = Ka.ColorJitter(brightness=0.2, contrast=0.3, p=0.5)
        self.k2 = Ka.RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False)
        self.k3 = Ka.RandomVerticalFlip(p=0.5, p_batch=1.0, same_on_batch=False, keepdim=False)
        self.k6 = Ka.RandomRotation(45.0, same_on_batch=False, align_corners=True, p=0.5, keepdim=False, resample='nearest')
        self.k5 = Ka.RandomGaussianBlur((3, 9), (0.1, 2.0), p=0.5)
        self.k4 = Ka.RandomResizedCrop((512, 512), scale=(0.67, 0.67), ratio=(0.75, 1.333), same_on_batch=False,resample='bilinear',p=0.5, align_corners= True)
        self.resize = Ka.Resize((512, 512))

    def forward(self, img: torch.Tensor, mask: torch.Tensor):
        img_out, mask_out = img, mask


        img_out = self.k1(img_out)
        img_out = self.k2(img_out)
        img_out = self.k3(img_out)
        img_out = self.k4(img_out)
        img_out = self.k5(img_out)
        img_out = self.k6(img_out)
        img_out = self.resize(img_out)
        mask_out = self.k2(mask_out, self.k2._params)
        mask_out = self.k3(mask_out, self.k3._params)
        mask_out = self.k4(mask_out, self.k4._params)
        mask_out = self.k6(mask_out, self.k6._params)
        mask_out = self.resize(mask_out)


        return img_out, mask_out

