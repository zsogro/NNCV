import torch
from torchvision.models import vit_b_16

BACKBONE_WEIGHTS = "Final assignment/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" 
model = torch.hub.load("dinov3", "dinov3_vitb16", source="local", pretrained=False)

state_dict = torch.load(BACKBONE_WEIGHTS, map_location="cpu")
model.load_state_dict(state_dict)