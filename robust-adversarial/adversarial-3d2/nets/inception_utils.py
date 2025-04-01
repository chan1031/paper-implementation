"""
inception_utils.py

PyTorch에서는 tf_slim의 arg_scope와 같은 개념이 필요하지 않습니다.
이 파일은 기존 코드와의 호환성을 위해 arg_scope와 관련된 함수를 제공하는 플레이스홀더입니다.
"""

def inception_arg_scope():
    # PyTorch에서는 네트워크 설정(초기화, 정규화 등)을 모듈 내에서 직접 처리하므로
    # 이 함수는 더미(dummy) 함수로 사용됩니다.
    return None
