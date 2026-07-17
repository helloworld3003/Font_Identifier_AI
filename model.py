import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HE_Block(nn.Module):
    def __init__(self, drop_ratio=0.15):
        super(HE_Block, self).__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if not self.training:
            return x
            
        batch_size, channels = x.size()
        k = max(1, int(channels * self.drop_ratio))
        _, topk_indices = torch.topk(x, k, dim=1)
        
        mask = torch.ones_like(x)
        mask.scatter_(1, topk_indices, 0.0)
        return x * mask

class ConvNeXtFontEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ConvNeXtFontEncoder, self).__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.he_block = HE_Block(drop_ratio=0.15)
        self.fc = nn.Linear(self.backbone.num_features, embedding_dim)

    def forward(self, x):
        features = self.backbone(x)
        features = self.he_block(features)
        embeddings = self.fc(features)
        return F.normalize(embeddings.float(), p=2, dim=1)
