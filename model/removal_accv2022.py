import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from model.efficient import get_efficientnet
from model.unet import UNetModule
from model.convgru import ConvGRUCell

model_infos = {
    'b5':[24,40,64,176,2048], 'b4':[24,32,56,160,1792], 'b3':[24,32,48,136,1536], 
    'b2':[16,24,48,120,1408], 'b1':[16,24,40,112,1280], 'b0':[16,24,40,112,1280],
}

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(inplace=True))

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class EfficientEncoder(nn.Module):
    def __init__(self, backbone):
        super(EfficientEncoder, self).__init__()
        self.original_model = get_efficientnet(backbone)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features

class Decoder(nn.Module):
    def __init__(self, n_channel_features):
        super(Decoder, self).__init__()
        features = n_channel_features[-1]

        self.conv2 = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + n_channel_features[-2], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + n_channel_features[-3], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + n_channel_features[-4], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + n_channel_features[-5], output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return torch.tanh(out)

class lightnessAdjustModule(nn.Module):
    def __init__(self, n_class, ch_in, ch_base, ch_hidden, n_loop=2):
        super(lightnessAdjustModule, self).__init__()
        self.n_loop = n_loop
        # features extractor
        self.encoder = UNetModule(n_class, ch_in, ch_base, False)
        # convgru
        self.gru_cell = ConvGRUCell(ch_base, hidden_size=ch_hidden, kernel_size=3)

        # output
        self.output = nn.Sequential(
            nn.Conv2d(ch_hidden, n_class, 1),
            nn.Tanh()
        )

    def forward(self, x, hidden=None):
        fea = self.encoder(x)
        for _ in range(self.n_loop):
            hidden = self.gru_cell(fea, hidden)

        output = self.output(hidden) # scale factor for v (hsv)
        return output, hidden

class ShadowRemoval(nn.Module):
    def __init__(self, n_cells=4):
        super(ShadowRemoval, self).__init__()
        self.n_layers = n_cells
        self.convgru_cells = nn.ModuleList([lightnessAdjustModule(3, 4) for _ in range(n_cells)])

    def forward(self, rgb, mask):
        pred_rgbs = []
        hidden_state = None

        cur_pred = rgb
        for i in range(self.n_layers):
            input = torch.cat((cur_pred, mask), axis=1)
            convgru_cell = self.convgru_cells[i] 
            output, new_hidden_state = convgru_cell(input, hidden_state)
            hidden_state = new_hidden_state
            
            cur_pred = cur_pred + mask*output

            pred_rgbs.append(cur_pred)

        return pred_rgbs

class SRNet(nn.Module):
    def __init__(self, args):
        super(SRNet, self).__init__()
        self.backbone = args.backbone
        # encoder
        n_channel_features = model_infos[self.backbone]
        self.efficient_encoder = EfficientEncoder(self.backbone)

        # decoder
        features = n_channel_features[-1]

        self.conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + n_channel_features[-2], output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + n_channel_features[-3], output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + n_channel_features[-4], output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + n_channel_features[-5], output_features=features // 16)

        n_dim_align, ch_base = 32, 64
        self.align1 = nn.Sequential(
            nn.Conv2d(features // 2, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align2 = nn.Sequential(
            nn.Conv2d(features // 4, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align3 = nn.Sequential(
            nn.Conv2d(features // 8, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        self.align4 = nn.Sequential(
            nn.Conv2d(features // 16, n_dim_align, 1, bias=False), nn.BatchNorm2d(n_dim_align), nn.LeakyReLU(inplace=True)
        )
        
        self.pred_rgb = lightnessAdjustModule(3, n_dim_align+1, ch_base, n_dim_align)
        self.predict_mask = nn.Sequential(
            nn.Conv2d(n_dim_align, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 1, 1), nn.Sigmoid()
        )
        
    def forward(self, rgb):
        features = self.efficient_encoder(rgb)
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        ##############################################    
        # upsampling
        ##############################################

        x_d0 = self.conv(x_block4)
        # layer01
        x_d1 = self.up1(x_d0, x_block3)

        # layer02
        x_d2 = self.up2(x_d1, x_block2)
        x_align2 = self.align2(x_d2)
        
        # layer03
        x_d3 = self.up3(x_d2, x_block1)
        x_align3 = self.align3(x_d3)
        pred_mask3 = self.predict_mask(x_align3)
        hidden_state = F.interpolate(x_align2, x_block1.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb3, hidden_state = self.pred_rgb(torch.cat((x_align3, pred_mask3), dim=1), hidden_state)
        
        # layer04
        x_d4 = self.up4(x_d3, x_block0)
        x_align4 = self.align4(x_d4)
        pred_mask4 = self.predict_mask(x_align4)
        hidden_state = F.interpolate(hidden_state, x_block0.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb4, _ = self.pred_rgb(torch.cat((x_align4, pred_mask4), dim=1), hidden_state)

        pred_mask3 = F.interpolate(pred_mask3, rgb.shape[-2:], mode='bilinear', align_corners=True)
        pred_mask4 = F.interpolate(pred_mask3, rgb.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb3 = F.interpolate(pred_rgb3, rgb.shape[-2:], mode='bilinear', align_corners=True)
        pred_rgb4 = F.interpolate(pred_rgb4, rgb.shape[-2:], mode='bilinear', align_corners=True)

        pred_rgb3 = torch.clamp(rgb+pred_rgb3*pred_mask3, 0.0, 1.0)
        pred_rgb4 = torch.clamp(rgb+pred_rgb4*pred_mask4, 0.0, 1.0)
        
        return [pred_mask3, pred_mask4], [pred_rgb3, pred_rgb4]

    def name(self):
        return 'accv2022'