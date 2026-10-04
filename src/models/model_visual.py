"""
Visual Model, with AGLFF and UWBMoE.
Used for visual classification, and future late-fusion strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # initial conv
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # global avg pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        # initial conv and pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50_backbone(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        from torchvision.models import resnet50, ResNet50_Weights
        weights = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
        model.load_state_dict({key: value for key, value in weights.items() if not key.startswith('fc.')}, strict=True)

    return model

def resnet50_backbone_6ch(pretrained=False, **kwargs):
    adapter = nn.Sequential(
        nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True)
    )

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    class ResNetWithAdapter(nn.Module):
        def __init__(self, adapter, resnet):
            super().__init__()
            self.adapter = adapter
            self.resnet = resnet
        def forward(self, x):
            return self.resnet(self.adapter(x))

    return ResNetWithAdapter(adapter, model)

def resnet50_backbone_12ch(pretrained=False, **kwargs):
    adapter = nn.Sequential(
        nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU(inplace=True)
    )

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    class ResNetWithAdapter(nn.Module):
        def __init__(self, adapter, resnet):
            super().__init__()
            self.adapter = adapter
            self.resnet = resnet
        def forward(self, x):
            return self.resnet(self.adapter(x))

    return ResNetWithAdapter(adapter, model)


class ExpertMLP(nn.Module):

    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class ConditionalTopKMoE(nn.Module):

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        renorm_topk: bool = True,
        cond_mode: str = "mean",
    ):
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.renorm_topk = renorm_topk
        self.cond_mode = cond_mode

        self.router_x = nn.Linear(d_model, num_experts)
        self.router_c = nn.Linear(d_model, num_experts)

        self.experts = nn.ModuleList(
            [ExpertMLP(d_model, hidden_mult=hidden_mult, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, F_main, F_cond):

        B, N, D = F_main.shape
        assert D == self.d_model
        assert F_cond.dim() == 3 and F_cond.size(-1) == D

        logits = self.router_x(F_main) + self.router_c(F_cond)
        WG = F.softmax(logits, dim=-1)

        topk_w, topk_idx = WG.topk(self.top_k, dim=-1)

        if self.renorm_topk:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)

        Y = torch.stack([exp(F_main) for exp in self.experts], dim=2)

        idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        Y_topk = torch.gather(Y, dim=2, index=idx)

        X_moe = (topk_w.unsqueeze(-1) * Y_topk).sum(dim=2)

        F_refined = F_main + X_moe

        return X_moe, F_refined, WG


class BidirectionalMoE(nn.Module):

    def __init__(
        self,
        d_model: int = 2048,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        renorm_topk: bool = True,
    ):
        super().__init__()

        # syndrome -> organ MoE
        self.moe_syndrome_to_organ = ConditionalTopKMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            hidden_mult=hidden_mult,
            dropout=dropout,
            renorm_topk=renorm_topk,
        )

        # organ -> syndrome MoE
        self.moe_organ_to_syndrome = ConditionalTopKMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            hidden_mult=hidden_mult,
            dropout=dropout,
            renorm_topk=renorm_topk,
        )

    def forward(self, F_organ, F_syndrome):
        X_moe_s2o, F_organ_refined, WG_s2o = self.moe_syndrome_to_organ(
            F_main=F_organ,
            F_cond=F_syndrome
        )

        X_moe_o2s, F_syndrome_refined, WG_o2s = self.moe_organ_to_syndrome(
            F_main=F_syndrome,
            F_cond=F_organ
        )

        return (
            X_moe_s2o, F_organ_refined, WG_s2o,
            X_moe_o2s, F_syndrome_refined, WG_o2s
        )


def uncertainty_confidence(first, second, eps=1e-8):
    uncertainty = (1.0 - F.cosine_similarity(first, second, dim=-1, eps=eps).unsqueeze(-1)) / 2.0
    minimum = uncertainty.amin(dim=1, keepdim=True)
    maximum = uncertainty.amax(dim=1, keepdim=True)
    return 1.0 - (uncertainty - minimum) / (maximum - minimum + eps)


# CycleTCM network
class CycleTCM(nn.Module):
    def __init__(self, num_classes1=8, num_classes2=5, pretrained=False,
                 aglff=True, uwbmoe=True, mllm=False, **kwargs):
        super(CycleTCM, self).__init__()
        self.backbone_whole = resnet50_backbone(pretrained=pretrained, **kwargs)
        self.backbone_syndrome = resnet50_backbone_6ch(pretrained=pretrained, **kwargs)
        self.backbone_organ = resnet50_backbone_12ch(pretrained=pretrained, **kwargs)

        self.aglff, self.uwbmoe, self.mllm = aglff, uwbmoe, mllm

        # global-local fusion (AGLFF)
        self.avgpool_whole = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool_syndrome = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool_organ = nn.AdaptiveAvgPool2d((1,1))

        embedding_dim = 2048
        if aglff:
            # patch syndrome
            self.conv_syndrome_1_1 = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1,stride=1,padding=0)
            self.conv_syndrome_1_2 = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=3,stride=1,padding=1)

            # patch organ
            self.conv_organ_1_1 = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1,stride=1,padding=0)
            self.conv_organ_1_2 = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=3,stride=1,padding=1)

            # patch whole
            self.conv_whole_1_1 = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=1,stride=1,padding=0)
            self.conv_whole_1_2 = nn.Conv2d(in_channels=2048,out_channels=2048,kernel_size=3,stride=1,padding=1)

            self.avgpool_syndrome_augmented_1 = nn.AdaptiveAvgPool2d((1,1))
            self.avgpool_organ_augmented_1 = nn.AdaptiveAvgPool2d((1,1))

            self.avgpool_syndrome_augmented_2 = nn.AdaptiveAvgPool2d((1,1))
            self.avgpool_organ_augmented_2 = nn.AdaptiveAvgPool2d((1,1))

            # gate
            self.gate_syndrome = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=1)
            self.gate_organ = nn.Conv2d(in_channels=2048,out_channels=1,kernel_size=1)

            # class token
            self.cross_attention1 = nn.MultiheadAttention(embed_dim=2048, num_heads=8, dropout=0.1, batch_first=True)
            self.cross_attention2 = nn.MultiheadAttention(embed_dim=2048, num_heads=8, dropout=0.1, batch_first=True)

            self.norm1 = nn.LayerNorm(2048)
            self.norm2 = nn.LayerNorm(2048)

            embedding_dim = 2048
            patch_nums = 49

            self.avgpool_whole = nn.AdaptiveAvgPool2d((1,1))

            self.position_whole = nn.Parameter(torch.zeros(1,patch_nums,embedding_dim))
            self.position_syndrome = nn.Parameter(torch.zeros(1,patch_nums,embedding_dim))
            self.position_organ = nn.Parameter(torch.zeros(1,patch_nums,embedding_dim))

        if uwbmoe:
            self.bidirectional_moe = BidirectionalMoE(
                d_model=2048,
                num_experts=4,
                top_k=2,
                hidden_mult=4,
                dropout=0.1
            )

        self.avgpool_syndrome_after = nn.AdaptiveAvgPool1d(1)
        self.avgpool_organ_after = nn.AdaptiveAvgPool1d(1)

        if mllm:
            from models.model_mllm import MLLM_Adapter
            self.mllm_adapter = MLLM_Adapter()
        width = embedding_dim * (3 if mllm else 2)

        self.bn_syndrome = nn.BatchNorm1d(width)
        self.dropout_syndrome = nn.Dropout(0.3)
        self.bn_organ = nn.BatchNorm1d(width)
        self.dropout_organ = nn.Dropout(0.3)

        self.classifier1 = nn.Linear(width , num_classes1)

        self.classifier2 = nn.Linear(width , num_classes2)


    def forward(self, x_whole, x_edge, x_body, x_heart_lung, x_spleen, x_liver, x_kidney, mllm_feature=None):
        # syndrome part
        x_syndrome = torch.cat([x_edge, x_body], dim=1)
        patch_syndrome = self.backbone_syndrome(x_syndrome)
        cls_syndrome = self.avgpool_syndrome(patch_syndrome)
        cls_syndrome = torch.flatten(cls_syndrome, 1)
        cls_syndrome = cls_syndrome.unsqueeze(1)

        # whole part
        patch_whole = self.backbone_whole(x_whole)
        cls_whole = self.avgpool_whole(patch_whole)
        cls_whole = torch.flatten(cls_whole , 1)
        cls_whole = cls_whole.unsqueeze(1)

        # organ part
        x_organ = torch.cat([x_heart_lung, x_spleen, x_liver, x_kidney], dim=1)
        patch_organ = self.backbone_organ(x_organ)
        cls_organ = self.avgpool_organ(patch_organ)
        cls_organ = torch.flatten(cls_organ, 1)
        cls_organ = cls_organ.unsqueeze(1)

        if self.aglff:
            # patch token augment
            patch_syndrome = self.conv_syndrome_1_2(F.relu(self.conv_syndrome_1_1(patch_syndrome)))
            patch_organ = self.conv_organ_1_2(F.relu(self.conv_organ_1_1(patch_organ)))

            patch_whole = self.conv_whole_1_2(F.relu(self.conv_whole_1_1(patch_whole)))

            syndrome_avg = self.avgpool_syndrome_augmented_1(patch_syndrome)
            organ_avg = self.avgpool_organ_augmented_1(patch_organ)

            patch_syndrome = patch_syndrome + syndrome_avg
            patch_organ = patch_organ + organ_avg

            gate_syndrome = torch.sigmoid(self.gate_syndrome(patch_syndrome))
            gate_organ = torch.sigmoid(self.gate_organ(patch_organ))

            fused_feature_syndrome = gate_syndrome * patch_syndrome + (1.0 - gate_syndrome) * patch_whole
            fused_feature_organ = gate_organ * patch_organ + (1.0 - gate_organ) * patch_whole

            # class token augment
            attn_syndrome , _ = self.cross_attention1(cls_whole, cls_syndrome, cls_syndrome)
            attn_organ , _ = self.cross_attention2(cls_whole, cls_organ, cls_organ)
            attn_syndrome = attn_syndrome.squeeze(1)
            attn_organ = attn_organ.squeeze(1)

            cls_whole = cls_whole.squeeze(1)
            cls_syndrome = cls_syndrome.squeeze(1)
            cls_organ = cls_organ.squeeze(1)

            cls_feat_whole_1 = self.norm1(cls_whole + attn_syndrome)
            cls_feat_whole_2 = self.norm2(cls_whole + attn_organ)

            cls_whole_augmented = (cls_feat_whole_1 + cls_feat_whole_2) / 2.0

        else:
            cls_whole_augmented = cls_whole.squeeze(1)
            fused_feature_syndrome = patch_syndrome
            fused_feature_organ = patch_organ

        # add positional embeddings
        patch_syndrome_embedding = fused_feature_syndrome.flatten(2).transpose(1,2)
        patch_organ_embedding = fused_feature_organ.flatten(2).transpose(1,2)

        if self.aglff:
            patch_syndrome_embedding = patch_syndrome_embedding + self.position_syndrome
            patch_organ_embedding = patch_organ_embedding + self.position_organ

        if self.uwbmoe:
            U_hat_syndrome = uncertainty_confidence(patch_syndrome_embedding, patch_organ_embedding)
            U_hat_organ = uncertainty_confidence(patch_organ_embedding, patch_syndrome_embedding)

            F_fused_syndrome = U_hat_syndrome * patch_syndrome_embedding + (1.0 - U_hat_syndrome) * patch_organ_embedding
            F_fused_organ = U_hat_organ * patch_organ_embedding + (1.0 - U_hat_organ) * patch_syndrome_embedding

            # bidirectional MoE between syndrome and organ
            X_moe_s2o, F_organ_refined, WG_s2o,X_moe_o2s, F_syndrome_refined, WG_o2s = self.bidirectional_moe(F_fused_organ, F_fused_syndrome)

        else:
            F_syndrome_refined = patch_syndrome_embedding
            F_organ_refined = patch_organ_embedding

        # classifier
        feat_syndrome = self.avgpool_syndrome_after(F_syndrome_refined.transpose(1,2)).squeeze(-1)
        feat_organ = self.avgpool_organ_after(F_organ_refined.transpose(1,2)).squeeze(-1)

        x_syndrome_all = torch.cat([cls_whole_augmented, feat_syndrome], dim=1)
        x_organ_all = torch.cat([cls_whole_augmented, feat_organ], dim=1)

        if self.mllm:
            semantic = self.mllm_adapter(mllm_feature)
            x_syndrome_all = torch.cat([x_syndrome_all, semantic], dim=1)
            x_organ_all = torch.cat([x_organ_all, semantic], dim=1)

        # BatchNorm and Dropout before classifiers
        x_syndrome_all = self.bn_syndrome(x_syndrome_all)
        x_syndrome_all = self.dropout_syndrome(x_syndrome_all)
        x_organ_all = self.bn_organ(x_organ_all)
        x_organ_all = self.dropout_organ(x_organ_all)


        syndrome_pred = self.classifier1(x_syndrome_all)
        organ_pred = self.classifier2(x_organ_all)

        return syndrome_pred, organ_pred


# count model parameters
def count_parameters(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return total_params, trainable_params, non_trainable_params

# print model parameter summary
def print_model_info(model):
    total_params, trainable_params, non_trainable_params = count_parameters(model)

    print("Model parameter summary")
    print(f"Total:        {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable:    {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Fixed:        {non_trainable_params:,} ({non_trainable_params/1e6:.2f}M)")


def _smoke_test() -> None:
    """Random-input forward pass; no ImageNet pretrained weights."""
    torch.manual_seed(0)
    model = CycleTCM(num_classes1=8, num_classes2=5, pretrained=False)
    model.eval()
    b, h, w = 2, 224, 224
    x_whole = torch.randn(b, 3, h, w)
    x_edge = torch.randn(b, 3, h, w)
    x_body = torch.randn(b, 3, h, w)
    x_heart_lung = torch.randn(b, 3, h, w)
    x_spleen = torch.randn(b, 3, h, w)
    x_liver = torch.randn(b, 3, h, w)
    x_kidney = torch.randn(b, 3, h, w)
    with torch.no_grad():
        syndrome_pred, organ_pred = model(
            x_whole, x_edge, x_body, x_heart_lung, x_spleen, x_liver, x_kidney
        )
    print("CycleTCM (visual) smoke_test ok")
    print("  syndrome_pred:", tuple(syndrome_pred.shape))
    print("  organ_pred:   ", tuple(organ_pred.shape))
    print_model_info(model)


if __name__ == "__main__":
    _smoke_test()
