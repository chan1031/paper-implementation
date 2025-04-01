import torch
import torch.nn as nn
from torch.utils.data import Dataset #파이토치 데이터셋을 만들기 위한 클래스

'''
파이토치에서 커스텀한 데이터셋을 로드
Dataset 클래스를 상속받으면 아래 3가지 클래스르 반드시 구현해야함
__init__, __len__, __getitem__
'''
class BilingualDataset(Dataset): #Dataset을 상속받아 Pytorch 데이터셋을 커스텀 하는 클래스

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len #최대 문장 길이
        self.ds = ds #허깅 페이스 데이터 셋
        self.tokenizer_src = tokenizer_src #원본 언어 토크나이저
        self.tokenizer_tgt = tokenizer_tgt #번역 언어 토크나이저
        self.src_lang = src_lang #원본 언어 코드
        self.tgt_lang = tgt_lang #번역언어 코드

        #특별 토큰 설정
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    #데이터셋 크기 반환
    def __len__(self):
        return len(self.ds)

    #데이터셋에서 특정 샘플 가져오기
    def __getitem__(self, idx):
        #선택한 데이텃세에서 해당 idx의 원본 언어(src_text)와 번역 언어(tgt_text) 문장을 가져옴
        src_target_pair = self.ds[idx] #전체 데이터셋중에 idx번째 데이터 가져오기
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # 문장을 토큰화하여 정수리스트로 반환
        '''
        예를 들어 '나는 학생입니다'일 경우
        [나는, 학생, 입니다] 를 [92,32, 563]으로 단어 사전에 맞는 숫자로 변환함
        '''
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids #src_text에 문장이 들어가면 이를 토큰화
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        '''
        sos, eos, pad 토큰을 추가하기 위한 계산
        encoder는 eos가 있지만,
        decoder는 eos가 없다. 왜냐하면 예측을 해야하기 때문이다.
        ex)<sos> 나는 학생입니다 에서 끝
        '''
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # 인코더는 eos,sos 둘다 들어가므로 -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 #디코더는 eos가 없으므로 -1만 빼줌

        # 입력 문장이 seq_len보다 길면 오류 발생
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        '''
        인코더 입력 생성
        [SOS] + 정수 토큰 + [EOS] + 패딩 추가
        '''
        encoder_input = torch.cat(
            [
                self.sos_token, #sos 토큰
                torch.tensor(enc_input_tokens, dtype=torch.int64), #토큰화 및 정수로 변환한 문장
                self.eos_token, #eos 토큰
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64), # 패딩 토큰 (패딩 토큰은 최대 문장 길이까지 연장하기위해 추가하는 코드)
            ],
            dim=0,
        )

        '''
        디코더 입력 추가
        [SOS] + 정수 토큰 + 패딩 추가
        [EOS]는 label에서만 추가 됨
        '''
        decoder_input = torch.cat(
            [
                self.sos_token, #sos 토큰
                torch.tensor(dec_input_tokens, dtype=torch.int64), #디코더 변환
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        '''
        정답 라벨을 제작
        여기서 label은 SOS 토큰이 없는데 그 이유는
        트랜스포머 모델은 다음 단어를 예측하면서 학습을 진행하기 때문이다.
        예를 들어서 '나는 학생이다'라는 단어를 학습한다고 하면
        [SOS] 부터 시작하므로 이 다음 단어를 예측해야한다.
        그렇기에 label 부분은 [SOS]의 다음 단어 부분에 대한 정답만 가지고 있다.
        하지만. EOS 토큰은 가지고 있다.
        '''
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            #인코더 마스크는 패딩을 마스킹 처리함
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            #디코더 마스크는 패딩도 보지말고, 미래 단어도 보지 못하도록 False 처리함
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

#디코더에서 미래 단어를 볼 수 없도록 설정하는 마스크
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0