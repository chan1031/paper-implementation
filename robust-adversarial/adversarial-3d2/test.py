#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

from renderer import Renderer
from config import cfg

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #####################################################
    # 1. 적대적 텍스처 불러오기 (adv_500.jpg)
    #####################################################
    adv_texture_path = os.path.join(cfg.image_dir, "adv_990.jpg")
    if not os.path.isfile(adv_texture_path):
        raise FileNotFoundError(f"적대적 텍스처 {adv_texture_path}가 존재하지 않습니다.")

    # PIL -> NumPy -> Tensor
    adv_texture_img = Image.open(adv_texture_path).convert("RGB")
    adv_texture_np = np.array(adv_texture_img).astype(np.float32) / 255.0  # (H, W, 3)
    # PyTorch 텐서로 변환 (1, 3, H, W)
    adv_texture_tensor = torch.from_numpy(adv_texture_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    #####################################################
    # 2. 3D 모델 렌더러 초기화 및 OBJ 로드
    #####################################################
    renderer = Renderer((299, 299))  # 299x299 해상도로 렌더링
    renderer.load_obj(cfg.obj)       # 예: 3d_model/barrel.obj
    renderer.set_parameters(
        camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
        x_translation=(cfg.x_translation_min, cfg.x_translation_max),
        y_translation=(cfg.y_translation_min, cfg.y_translation_max)
    )

    #####################################################
    # 3. 렌더링 -> UV 맵 얻기, grid_sample로 최종 이미지 생성
    #####################################################
    # 배치 크기 1 (1장만 렌더링)
    uv = renderer.render(batch_size=1)  # shape: [1, 299, 299, 2]
    uv_tensor = torch.from_numpy(uv).float().to(device)

    # adv_texture_tensor shape: [1, 3, H_tex, W_tex]
    B, C, H_tex, W_tex = adv_texture_tensor.shape

    # UV 좌표를 [-1,1] 범위로 정규화
    norm_uv = uv_tensor.clone()
    norm_uv[..., 0] = (norm_uv[..., 0] / (W_tex - 1)) * 2.0 - 1.0
    norm_uv[..., 1] = (norm_uv[..., 1] / (H_tex - 1)) * 2.0 - 1.0

    # grid_sample -> 렌더링된 이미지 (1, 3, 299, 299)
    rendered_image = F.grid_sample(
        adv_texture_tensor, norm_uv, mode='bilinear', align_corners=True
    )

    #####################################################
    # 4. 렌더링된 이미지를 PIL로 변환 후 파일 저장
    #####################################################
    rendered_np = rendered_image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    rendered_np = np.clip(rendered_np * 255.0, 0, 255).astype(np.uint8)
    rendered_pil = Image.fromarray(rendered_np)
    
    save_rendered_path = os.path.join(cfg.image_dir, "rendered_adv.jpg")
    rendered_pil.save(save_rendered_path)
    print(f"렌더링된 이미지가 '{save_rendered_path}'에 저장되었습니다.")

    #####################################################
    # 5. InceptionV3 로 분류 (클래스 이름 출력)
    #####################################################
    # 5-1) 사전학습된 InceptionV3 로드 (ImageNet)
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights).to(device)
    model.eval()

    # 5-2) ImageNet 정규화/전처리
    #      weights.transforms()는 Resize(299), CenterCrop(299), ToTensor, Normalize 포함
    transform = weights.transforms()
    input_tensor = transform(rendered_pil).unsqueeze(0).to(device)  # (1, 3, 299, 299)

    # 5-3) 추론
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    # 5-4) Top-5 클래스
    top5 = torch.topk(probs, 5, dim=1)
    indices = top5.indices[0].cpu().numpy()
    scores = top5.values[0].cpu().numpy()

    # ImageNet 클래스 이름 (weights.meta["categories"])
    categories = weights.meta["categories"]

    print("=== 분류 결과 (Top-5) ===")
    for rank in range(5):
        cls_idx = indices[rank]
        score = scores[rank]
        cls_name = categories[cls_idx]
        print(f"{rank+1}) {cls_name} ({cls_idx}): {score:.4f}")

if __name__ == "__main__":
    main()
