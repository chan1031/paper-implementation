import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split #PyTorch 데이터셋과 미니배치 학습을 위한 유틸리티

from datasets import load_dataset #Hugging Face datasets 라이브러리에서 데이터셋을 로드
from tokenizers import Tokenizer #단어를 토큰 단위로 분리할 수 있는 클래스로 Tokenizer는 토큰화 방식을 결정해야함 (WordLevel, BPE, Unigram등 여러가지 토큰화 방법이 존재함)
#여기서 아래 코드에서 WordLevel을 불러온 것을 통해 Tokenizer의 토큰화 방법은 WordLevel임
from tokenizers.models import WordLevel #단어기반 토크나이저 모델
from tokenizers.trainers import WordLevelTrainer # WordLevel 토크나이저를 학습시키는 클래스
from tokenizers.pre_tokenizers import Whitespace # 공백 단위로 텍스트를 분리하는 Pre-tokenizer

from pathlib import Path

from tqdm import tqdm

from config import get_config, get_weights_file_path, latest_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer

import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    #특수 토큰 가져오기
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # 인코더로 소스 문장 인코딩
    encoder_output = model.module.encode(source, source_mask) if isinstance(model, nn.DataParallel) else model.encode(source, source_mask)
    #디코더 입력을 [SOS] 토큰으로 초기화
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    #반복문으로 토큰 생성
    '''
    이 경우는 inference 즉, 추론 과정에서 사용되는 코드이다.
    그러므로 처음에는 <SOS> 토큰 뒤에 나오게 될 값을 예측한다.
    이후 반복해서 <SOS> + 나는 뒤에 나올 토큰을 예측하고
    <EOS>가 나올때까지 반복한다.
    '''
    while True:
        if decoder_input.size(1) == max_len:
            break

        #현재까지의 출력에 대한 마스크 생성
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # 디코더로 다음 토큰 예측
        out = model.module.decode(encoder_output, source_mask, decoder_input, decoder_mask) if isinstance(model, nn.DataParallel) else model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        '''
        이 경우 처음에 decoder_input은 <SOS>만 들어가게 된다.
        이후 디코더 출력값이 나오면 밑의 project를 통과해야만 그 다음 예측값이 나오는 것이다.
        
        훈련단계와의 차이점은
        out에 문장 전체가 들어가서
        나는: [0.3, 0.4, 0.5]
        학생: [0.1, 0.2, 0.3]
        이다: [0.2, 0.3, 0.4]
        이렇게 전체 텐서의 형태가 [batch_size, seq_len, d_model]이 되지만
        
        out은 반복문으로 하나씩 토큰을 예측하므로
        처음에는 <SOS>를 입력하여 나온 결과: [0.3, 0.4, 0.5]
        <SOS> + 나는을 입력하여 나온 결과: [0.1, 0.2, 0.3]
        이렇게 하나씩 스코어 값을 뽑고 
        이후 proj로 다음 단어를 예측한다.
        '''
        
        # 다음 토큰의 확률 계산
        '''
        prob도 train에서의 proj_output과 동일한 형태를 보인다.
        하지만 반복문을 돌기때문에
        처음에는 <SOS>를 입력하여 나온 결과: [0.3, 0.4, 0.5]
        <SOS> + 나는을 입력하여 나온 결과: [0.1, 0.2, 0.3]
        이렇게 하나씩 스코어 값을 뽑고 
        이후 proj로 다음 단어를 예측한다.
        
        최종형태는 결국에 똑같아진다. (1, seq_len, vocab_size) 형태를 가지며 각 행은 다음 단어의 예측 확률을 의미한다.
        '''
        prob = model.module.project(out[:, -1]) if isinstance(model, nn.DataParallel) else model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        # [EOS] 토큰이 생성되면 종료
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)
        writer.flush()


def get_all_sentences(ds,lang): #ds: 번역 데이터셋, lang: 가져올 언어
    for item in ds:
        yield item['translation'][lang] #yield는 함수가 종료되지 않고 실행 상태를 유지하면서 값을 하나씩 반환함
        #즉, 데이터셋에서 해당 언어에 맞는 raw를 하나씩 반환

#WordLevel 토크나이저를 불러오거나 없으면 생성하는 함수
'''
토크나이저 학습은 딥러닝 학습과는 다르게 
가중치를 반복하면서 업데이트 하는 것이 아닌
단순히, 단어를 수집하는 것을 의미한다.
단어 출현 빈도를 기반으로 단어 리스트 즉, 단어 사전을 만든다.
'''
def get_or_build_tokenizer(config,ds,lang): #config 파일, 데이터셋, 언어
    
    # config['tokenizer_file'] = 'path'
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) #config에 저장된 tokenizer_file 경로에 설정된 언어에 맞는 tokenizer 설정파일을 저장한다.
    #tokenizer 경로가 없으면 새로 학습 즉, 토크나이저 파일이 없다면 새로 WordLevel 토크나이저를 만들고 학습함
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) #사전에 없는 새로운 단어는 UNK 토큰으로 지정
        tokenizer.pre_tokenizer = Whitespace() #띄워쓰기 단위로
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2) #스페셜 토큰을 지정하고, 등장 횟수가 2회 이상되는 단어만 학습하도록 설정
        #train_from_iterateor를 통해 데이터셋(opus_book)을 활용하여 토크나이저를 학습함 즉, 단어사전을 구축
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer) #데이터셋을 이용해 토크나이저 학습
        '''
        train_from_iterator는 허깅페이스의 tokenizer 라이브러리에서 제공하는 함수로
        tokenizer.train_from_iterator(iterator, trainer)로 구성되어있다.
        iterator는 문장들을 하나씩 반환하는 리스트나 제너레이터를 의미한다.
        즉, 데이터셋에서 한 문장씩 가져와서 정의된 훈련 규칙에 따라 
        단어사전을 구축한다.
        '''
        tokenizer.save(str(tokenizer_path)) #토크나이저 저장
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path)) #존재한다면 불러옴
    return tokenizer

#허깅 페이스의 datasets 라이브러리를 이용하여 데이터를 불러오고, 토크나이저를 적용, 훈련/검증 데이터셋으로 분류함
def get_ds(config):
    #load_dataset은 허깅페이스 dataset라이브러의 함수로 데이터셋을 불러오는 역할을 함
    ds_raw = load_dataset('opus100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='train') #opus-books데이터셋을 사용하고, 원본언어>번역영어를 결정하며 훈련 데이터셋만 로드함
    '''
    ds_raw의 예시 구조
    {
    id:12345
    "translation": {
        "en": "Hello, how are you?",
        "fr": "Bonjour, comment ça va?"
        }
    }
    '''
    #각 언어에 맞는 토크나이저를 불러오거나 생성함
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src']) #원본 언어 토크나이저
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt']) #번역 언어 토크나이저
    
    #길이가 긴 문장이 있을때 처리 (연구용이므로 이렇게 함)
    seq_len = config["seq_len"]

    # 길이가 너무 긴 문장 제거하도록 함
    def is_valid_sample(item):
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        return len(src_ids) + 2 <= seq_len and len(tgt_ids) + 1 <= seq_len

    ds_filtered = list(filter(is_valid_sample, ds_raw))

    print(f"Original dataset size: {len(ds_raw)}")
    print(f"Filtered dataset size: {len(ds_filtered)}")
    
    
    #Keep 90% for training and 10% for validation
    train_ds_size = int(0.9 * len(ds_filtered))
    val_ds_size = len(ds_filtered) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_filtered, [train_ds_size, val_ds_size])
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    
    #BilingualDataset에서 정의한 커스텀 데이터셋 클래스를 통해 DataLoader로 데이터를 실제로 배치 사이즈 만큼 불러옴
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True) # 지정한 배치 사이즈 만큼 데이터를 가져옴
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    '''
    파라미터 설명)
    vocab_src_len: 원본 문장 토큰 사전 길이
    vocat_tgt_len: 번역 문장 토큰 사전 길이
    config["seq_len"]: 최대길이
    d_model: 임베딩 차원
    '''
    return model

# 모델 훈련에 대한 함수
def train_model(config):
    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    # 가중치가 저장된 폴더의 위치
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    #데이터 로드
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    #모델 로드
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    
    # Multi-GPU 설정
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Tensorboard를 활용하여 훈련 변화 시각화
    writer = SummaryWriter(config['experiment_name'])

    #Optimizer는 Adam을 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    #즁간에 훈련이 중단되면 끊긴 부분부터 불러올 수 있도록 함
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        # Preloading된 모델을 불러오도록 함
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        
        # 단일 GPU에서 저장된 모델을 DataParallel 구조로 로드할 때 처리
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['model_state_dict'].items():
            if k.startswith('module.'):
                new_state_dict[k] = v
            else:
                new_state_dict[f'module.{k}'] = v
            
        model.load_state_dict(new_state_dict)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    #손실함수 정의
    #손실함수는 CrossEntropy 함수를 사용함 무시하는 토큰은 PAD 토큰이며, 스무딩을 0.1로 설정함
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)
    '''
    label_smoothing은 원-핫 인코딩 처럼 [0,0,1,0] 이렇게 정답을 분류하지 않고
    [0.25,0.25,0.95...] 이렇게 유연하게 분류하는 것을 의미함
    이를 통해 텍스트 학습시 너무 오버피팅 되지 않도록 함
    '''

    #에포크 설정
    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train() # 모델을 훈련 상태로 둠
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.module.encode(encoder_input, encoder_mask) if isinstance(model, nn.DataParallel) else model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) if isinstance(model, nn.DataParallel) else model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            #단어를 예측한 결과
            proj_output = model.module.project(decoder_output) if isinstance(model, nn.DataParallel) else model.project(decoder_output) # (B, seq_len, vocab_size)
            '''
            proj_output의 형태는 [batch_size, seq_len, vocab_size] 형태를 가진다.
            여기서 decoder_output의 형태와[batch_size, seq_len, d_model]의 차이점은
            vocab_size는 전체 단어사전의 크기로 확률을 나타낼때 가장 높은 값의 인덱스가 예측 토큰이라는 뜻이며
            d_model은 임베딩 차원으로 토큰의 벡터 표현을 의미하기에 이는 예측이 아닌 "어텐션": 문장간의 관계 결과 라는 차이가 있다는 것이다.
            '''
            
            
            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # 정답 label과 출력값 proj_output을 비교함
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # 역전파
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
        
        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # 에포크 마다 가중치 파일 저장
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
    