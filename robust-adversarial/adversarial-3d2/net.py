import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image

class AdversarialNet(nn.Module):
    def __init__(self, texture_np, device, cfg, inception):

        super(AdversarialNet, self).__init__()
        self.cfg = cfg
        self.device = device
        self.inception = inception
        # torchvision Inception은 ImageNet 정규화를 필요로 함
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
        self.ce_loss = nn.CrossEntropyLoss()
        
        # texture_np: [H, W, 3] -> tensor: [1, 3, H, W]
        texture_np = texture_np.transpose(2, 0, 1)  # (3, H, W)
        texture_tensor = torch.from_numpy(texture_np).unsqueeze(0).to(device).float()
        
        # 원본 텍스처 (업데이트하지 않으므로 buffer로 등록)
        self.register_buffer('std_texture', texture_tensor)
        # 적대적 텍스처 (학습 가능한 파라미터)
        self.adv_texture = nn.Parameter(texture_tensor.clone())
        
    def forward(self, uv_mapping):
        """
        Args:
            uv_mapping: tensor, shape [B, H, W, 2] (픽셀 좌표, 예: [0, width-1], [0, height-1])
                        여기서 B = cfg.batch_size
        Returns:
            dict with:
                - 'loss': 총 손실 (cross entropy + l2_loss)
                - 'predictions': top-5 예측 클래스 indices (numpy array)
                - 'diff': adv_texture - std_texture (학습된 차이)
        """
        B = uv_mapping.shape[0]
        # 배치 크기에 맞게 원본과 적대적 텍스처 확장: [B, 3, H_tex, W_tex]
        std_textures = self.std_texture.expand(B, -1, -1, -1)
        adv_textures = self.adv_texture.expand(B, -1, -1, -1)
        
        # [print_error] 각 배치마다 랜덤 multiplier와 addend 적용 (채널별 변형)
        if self.cfg.print_error:
            multiplier = torch.empty(B, 3, 1, 1, device=self.device).uniform_(self.cfg.channel_mult_min, self.cfg.channel_mult_max)
            addend = torch.empty(B, 3, 1, 1, device=self.device).uniform_(self.cfg.channel_add_min, self.cfg.channel_add_max)
            std_textures = self.transform(std_textures, multiplier, addend)
            adv_textures = self.transform(adv_textures, multiplier, addend)
        
        # uv_mapping: [B, H, W, 2] (픽셀 좌표)를 [-1,1] 범위로 정규화 (grid_sample 사용)
        _, _, H_tex, W_tex = std_textures.shape
        norm_uv = uv_mapping.clone().float()  # 복사본
        norm_uv[..., 0] = (uv_mapping[..., 0] / (W_tex - 1)) * 2 - 1
        norm_uv[..., 1] = (uv_mapping[..., 1] / (H_tex - 1)) * 2 - 1
        
        # grid_sample을 통해 텍스처에서 이미지 샘플링: 결과 [B, 3, H, W]
        std_images = F.grid_sample(std_textures, norm_uv, mode='bilinear', align_corners=True)
        adv_images = F.grid_sample(adv_textures, norm_uv, mode='bilinear', align_corners=True)
        
        # 배경 처리: uv_mapping이 모두 0인 부분은 배경 색상 적용
        # mask: [B, H, W, 1] → (B, 1, H, W)
        mask = (uv_mapping != 0).all(dim=3, keepdim=True).float().permute(0, 3, 1, 2)
        bg_color = torch.zeros(B, 3, 1, 1, device=self.device)
        std_images = self.set_background(std_images, mask, bg_color)
        adv_images = self.set_background(adv_images, mask, bg_color)
        
        # 광원 관련 변형 및 노이즈 추가 EOT
        if self.cfg.photo_error:
            multiplier = torch.empty(B, 1, 1, 1, device=self.device).uniform_(self.cfg.light_mult_min, self.cfg.light_mult_max)
            addend = torch.empty(B, 1, 1, 1, device=self.device).uniform_(self.cfg.light_add_min, self.cfg.light_add_max)
            std_images = self.transform(std_images, multiplier, addend)
            adv_images = self.transform(adv_images, multiplier, addend)
            
            noise_std = torch.empty(1, device=self.device).uniform_(0, self.cfg.stddev)
            gaussian_noise = torch.randn_like(std_images) * noise_std
            std_images = std_images + gaussian_noise
            adv_images = adv_images + gaussian_noise
        
        std_images, adv_images = self.normalize(std_images, adv_images)
        
        # config에 save_adv가 True일 경우 지정된 경로에 이미지를 저장
        if hasattr(self.cfg, 'save_adv') and self.cfg.save_adv:
            save_dir = self.cfg.save_adv_path if hasattr(self.cfg, 'save_adv_path') else './adv_images'
            os.makedirs(save_dir, exist_ok=True)
            for i in range(B):
                file_path = os.path.join(save_dir, f'adv_image_{i}.png')
                # adv_images는 [0,1] 범위이므로 save_image 사용 가능
                save_image(adv_images[i], file_path)
        
        # InceptionV3 입력에 맞게 스케일 조저ㅇ
        normalized_adv = adv_images.clone()
        for i in range(B):
            normalized_adv[i] = self.normalize_transform(normalized_adv[i])
        
        # InceptionV3에 입력
        logits = self.inception(normalized_adv)
        
        #손실값 계산
        # 타겟 클래스 (cfg.target)를 모든 배치에 대해 생성
        target_labels = torch.full((B,), self.cfg.target, dtype=torch.long, device=self.device)
        ce_loss = self.ce_loss(logits, target_labels)
        l2_loss = F.mse_loss(adv_images, std_images, reduction='mean')
        total_loss = ce_loss + self.cfg.l2_weight * l2_loss
        #토탈 로스는 클래스 loss + l2 loss (패치가 너무 변형되지 않도록 함)
        
        # Top-5 예측 추출
        _, top_indices = torch.topk(logits, 5, dim=1)
        diff = self.adv_texture - self.std_texture
        
        return {
            'loss': total_loss,
            'predictions': top_indices.detach().cpu().numpy(),
            'diff': diff
        }
    
    @staticmethod
    def transform(x, a, b):
        """Apply element-wise transform: a * x + b"""
        return a * x + b
    
    @staticmethod
    def set_background(x, mask, color):
        return mask * x + (1 - mask) * color
    
    @staticmethod
    def normalize(x, y):
        B = x.shape[0]
        x_flat = x.view(B, -1)
        y_flat = y.view(B, -1)
        min_val = torch.min(torch.min(x_flat, dim=1, keepdim=True)[0],
                            torch.min(y_flat, dim=1, keepdim=True)[0])
        max_val = torch.max(torch.max(x_flat, dim=1, keepdim=True)[0],
                            torch.max(y_flat, dim=1, keepdim=True)[0])
        min_val = torch.min(min_val, torch.zeros_like(min_val))
        max_val = torch.max(max_val, torch.ones_like(max_val))
        min_val = min_val.view(B, 1, 1, 1)
        max_val = max_val.view(B, 1, 1, 1)
        norm_x = (x - min_val) / (max_val - min_val + 1e-8)
        norm_y = (y - min_val) / (max_val - min_val + 1e-8)
        return norm_x, norm_y
