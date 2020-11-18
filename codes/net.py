from pytorch_lightning import LightningModule
import torch
from torchvision.models.resnet import resnet50, resnet34
import torch.nn as nn
from convlstm.convlstm import ConvLSTM

class LyftNet(LightningModule):
    def __init__(
        self, 
        history_num_frames, 
        future_num_frames,
        pretrained=True,
        num_modes=3,
        **kwargs
    ):
        super().__init__()
        num_history_channels = (history_num_frames+1) * 2
        num_in_channels = 3 + num_history_channels
        num_targets = 2 * future_num_frames
        self.future_len = future_num_frames

        self.ConvLSTM = ConvLSTM(
            2, [
                2,2,2,2,2,2,2,2,2,2,2
            ], (3,3), 11)
        self.backbone = resnet34(pretrained=pretrained)
        #for name, param in resnet.named_parameters():
        #    if param.requires_grad:
        #        print(name + " " * (30 - len(name)) + str(param.shape))
        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )
        backbone_out_features = 512
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )
        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes
        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        # print(x.shape)
        x[:,:22,...] = torch.stack(self.ConvLSTM(
            torch.stack([
                torch.stack((x[:,i,...], x[:,i+11,...]), dim=1)
                for i in range(11)
            ]))[0], 
        dim=1).squeeze(1).view(-1, 22, 224, 224)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
