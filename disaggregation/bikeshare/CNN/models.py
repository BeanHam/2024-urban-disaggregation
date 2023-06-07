import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## seed
seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic=True

# ---------------------
# weights initialization
# ---------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
        torch.nn.init.constant_(m.bias, 0.01)
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.01)
        torch.nn.init.zeros_(m.bias)

        
# ---------------------
# unet block
# ---------------------
class unet_block(nn.Module):
   
    '''
    Single block for UNet
    '''
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        
        h = self.conv(input)
        return h

# ---------------------
# Disaggregation Model
# ---------------------
class DisAgg(nn.Module):
    
    """
    Super-Resolution of graphs
    
    """
    
    def __init__(self, ndf=32):
        super().__init__()
        
        ## encoders   
        self.enc_1 = unet_block(1,     ndf*1)
        self.enc_2 = unet_block(ndf*1, ndf*2)
        self.enc_3 = unet_block(ndf*2, ndf*4)
        self.enc_4 = unet_block(ndf*4, ndf*8)
        self.enc_5 = unet_block(ndf*8, ndf*16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # decoders
        self.upconv_4 = nn.ConvTranspose2d(ndf*16, ndf*8, kernel_size=2, stride=2)
        self.upconv_3 = nn.ConvTranspose2d(ndf*8,  ndf*4, kernel_size=2, stride=2)
        self.upconv_2 = nn.ConvTranspose2d(ndf*4,  ndf*2, kernel_size=2, stride=2)
        self.upconv_1 = nn.ConvTranspose2d(ndf*2,  ndf*1, kernel_size=2, stride=2)
        self.dec_4 = unet_block(ndf*16, ndf*8)
        self.dec_3 = unet_block(ndf*8, ndf*4)
        self.dec_2 = unet_block(ndf*4, ndf*2)
        self.dec_1 = unet_block(ndf*2, ndf*1)
        self.dec_0 = nn.Conv2d(ndf*1, 1, kernel_size=1)
        
    def forward(self, input):
        
        ## encoder
        enc1 = self.enc_1(input)
        enc2 = self.enc_2(self.pool(enc1))
        enc3 = self.enc_3(self.pool(enc2))
        enc4 = self.enc_4(self.pool(enc3))
        bottleneck = self.enc_5(self.pool(enc4))

        ## decoder
        dec4 = self.upconv_4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec_4(dec4)
        dec3 = self.upconv_3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec_3(dec3)
        dec2 = self.upconv_2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec_2(dec2)
        dec1 = self.upconv_1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_1(dec1)
        output = self.dec_0(dec1)
        
        return output