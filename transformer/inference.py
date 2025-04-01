import torch
from config import get_config, get_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from dataset import causal_mask

# 1. 설정 및 디바이스 선택
config = get_config()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 2. 토크나이저 로드
tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))

# 3. 모델 생성 및 가중치 로드
model = build_transformer(
    src_vocab_size=tokenizer_src.get_vocab_size(),
    tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
    src_seq_len=config["seq_len"],
    tgt_seq_len=config["seq_len"],
    d_model=config["d_model"]
).to(device)

# 가중치 파일 불러오기 (가장 마지막 에폭 또는 원하는 에폭 번호로 수정)
model_path = get_weights_file_path(config, "16")  # 예: 19번째 에폭
state = torch.load(model_path, map_location=device)

# DataParallel로 학습된 가중치를 단일 GPU/CPU 모델에 로드하기 위해 'module.' 접두사 제거
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state["model_state_dict"].items():
    name = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

# 4. 입력 문장을 토큰화하고 텐서로 변환
def prepare_input(sentence, tokenizer, seq_len):
    tokens = tokenizer.encode(sentence).ids
    tokens = [tokenizer.token_to_id("[SOS]")] + tokens + [tokenizer.token_to_id("[EOS]")]
    num_padding = seq_len - len(tokens)
    if num_padding < 0:
        raise ValueError("Sentence too long")
    tokens += [tokenizer.token_to_id("[PAD]")] * num_padding
    input_tensor = torch.tensor(tokens).unsqueeze(0).to(device)  # (1, seq_len)
    mask = (input_tensor != tokenizer.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).int()  # (1,1,1,seq_len)
    return input_tensor, mask

# 5. greedy decoding 함수로 추론 실행
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.tensor([[sos_idx]], device=device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        logits = model.project(decoder_output[:, -1])
        next_token = torch.argmax(logits, dim=-1).item()
        decoder_input = torch.cat(
            [decoder_input, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)

# 6. 번역 함수
def translate(sentence):
    encoder_input, encoder_mask = prepare_input(sentence, tokenizer_src, config['seq_len'])
    output_tokens = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, config["seq_len"])
    translated_text = tokenizer_tgt.decode(output_tokens.tolist(), skip_special_tokens=True)
    return translated_text

# 7. 예시 실행
if __name__ == '__main__':
    while True:
        print()
        sentence = input("영어 문장 입력 (종료하려면 'exit'): ")
        if sentence.strip().lower() == "exit":
            break
        try:
            result = translate(sentence)
            print(f"번역 결과: {result}")
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
