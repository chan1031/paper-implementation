import torch
import torch.nn as nn
import torchvision.models as models

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, pretrained=True):
        """
        InceptionV3 모델을 생성합니다.
        
        Args:
            num_classes (int): 최종 출력 클래스 수. (기본: 1000)
            aux_logits (bool): 보조 로짓(auxiliary logits)을 사용할지 여부.
            transform_input (bool): 입력 변환을 적용할지 여부.
            pretrained (bool): 사전학습된 가중치를 사용할지 여부.
        """
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input

        # torchvision의 InceptionV3 모델을 로드합니다.
        self.inception = models.inception_v3(pretrained=pretrained, aux_logits=aux_logits, transform_input=transform_input)
        
        if num_classes != 1000:
            # 최종 fully-connected 레이어 교체 (필요시)
            self.inception.fc = nn.Linear(self.inception.fc.in_features, num_classes)
            if aux_logits and self.inception.AuxLogits is not None:
                self.inception.AuxLogits.fc = nn.Linear(self.inception.AuxLogits.fc.in_features, num_classes)

    def forward(self, x):
        """
        모델의 forward pass.
        학습 시 aux_logits를 사용하는 경우 두 개의 출력을 반환합니다.
        """
        if self.aux_logits and self.training:
            # 학습 모드에서는 (주 출력, 보조 출력)을 반환합니다.
            x, aux = self.inception(x)
            return x, aux
        else:
            x = self.inception(x)
            return x

def inception_v3(num_classes=1000, aux_logits=True, transform_input=False, pretrained=True):
    """
    TensorFlow 버전의 인터페이스와 유사하게 InceptionV3 모델을 생성합니다.
    
    Returns:
        InceptionV3 모듈.
    """
    return InceptionV3(num_classes=num_classes, aux_logits=aux_logits,
                       transform_input=transform_input, pretrained=pretrained)
