#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch import optim
from torchvision import models
from PIL import Image

from renderer import Renderer
from net import AdversarialNet
from config import cfg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 텍스처 이미지 로드 
    texture_img = Image.open(cfg.texture).convert('RGB')
    tex_width, tex_height = texture_img.size
    texture_np = np.array(texture_img).astype(np.float32) / 255.0  # shape: (H, W, 3)

    # 2. Renderer 초기화
    renderer = Renderer((299, 299))
    renderer.load_obj(cfg.obj)
    renderer.set_parameters(
        camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
        x_translation=(cfg.x_translation_min, cfg.x_translation_max),
        y_translation=(cfg.y_translation_min, cfg.y_translation_max)
    )

    # 3. 사전학습된 InceptionV3 모델 불러오기 
    inception = models.inception_v3(pretrained=True, aux_logits=True)
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception.to(device)

    # 4. 적대적 네트워크 모델 초기화
    model = AdversarialNet(texture_np, device, cfg, inception)
    #texture_np로 이미지에 대한 픽셀값 전달
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # TensorBoard SummaryWriter (옵션)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(cfg.logdir)

    # 5. 학습 루프
    for i in range(cfg.iterations):
        # 렌더러를 통해 UV 좌표 생성 (넘파이 배열, shape: [B, H, W, 2])
        uv = renderer.render(cfg.batch_size) * np.array([tex_width - 1, tex_height - 1], dtype=np.float32)
        uv_tensor = torch.from_numpy(uv).to(device)  # [B, H, W, 2]

        optimizer.zero_grad()
        outputs = model(uv_tensor)  # forward 시 손실, 예측, diff 등 반환
        loss = outputs['loss']
        loss.backward()
        optimizer.step() #파라미터 업데이트

        # adv_texture를 [0,1] 범위로 클리핑
        with torch.no_grad():
            model.adv_texture.clamp_(0.0, 1.0)

        print(f"Iteration {i} - Loss: {loss.item():.4f} | Diff Sum: {outputs['diff'].sum().item():.4f}")
        print("Top Predictions:", outputs['predictions'])
        writer.add_scalar('Loss/train', loss.item(), i)

        # 10 iteration마다 적대적 텍스처를 이미지로 저장
        if i % 10 == 0:
            adv_tex = model.adv_texture.detach().cpu().squeeze(0)  # shape: [3, H, W]
            # 채널 순서를 변경하여 [H, W, 3]로 변환한 후 0-255 범위로 스케일링
            adv_tex_img = adv_tex.permute(1, 2, 0).numpy()
            adv_tex_img = np.rint(adv_tex_img * 255).astype(np.uint8)
            adv_image = Image.fromarray(adv_tex_img)
            adv_image.save(os.path.join(cfg.image_dir, f"adv_{i}.jpg"))

if __name__ == '__main__':
    main()
