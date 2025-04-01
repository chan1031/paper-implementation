import torch
import torch.nn as nn
import math

#임베딩 레이어 구현
'''
임베딩 레이어는 단어를 의미를 담고있는 벡터로 변환해준다.
예를 들어 '나는 고양이를 좋아해'를 입력으로 넣으면
토큰화와 단어사전을 통해 각 단어를 토큰으로 분리후 단어 사전에 맞는 숫자로 변환해준다.
이후 해당 숫자에 의미를 넣어주어야 하기 때문에 이때 이 값을 임베딩 레이어에 입력하면
각 숫자(토큰)에 512차원의 벡터로 의미를 부여해주게 된다.
ex) 고양이 = [0.01, 0.23, -0.12 .....] 이런식으로 단어에 의미를 부여해줌
'''
class InputEmbeddings(nn.Module):
    #vocab_size: 모델이 학습할 수 있는 전체 단어의 크기
    def __init__(self, d_model: int, vocab_size: int): #d_model: 임베딩 차원으로 각 단어가 몇 개의 벡터로 표햐현되는지를 나타냄, vocab_size: 단어 사전의 크기로 총 단어 개수를 의미함
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) #Embedding layer에 vocab_size, d_model 사이즈의 학습 가능한 가중치가 존재함
        #Embedding layer를 통과하여 출력되는 텐서의 크기는 (vocab_size, d_model)임
        
    #nn.Module을 상속받으면 반드시 forward를 구현해줘야 함
    #forward는 InputEmbedding을 호출하면 자동으로 순전파를 통과함
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #크기가 너무 작아지는 것을 방지하기 위해 d_model의 제곱근을 곱하여 return 해줌
        
'''
이제 임베딩 벡터에다가 포지션 임베딩을 추가해야한다
seq_len의 경우 "나는 고양이를 좋아해"라고 입력한다면 "나는", "고양이를", "좋아해", [PAD], [PAD] 총 5개의 개수로 나뉨
'''
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None: #d_model: 임베팅 벡터 크기, seq_len: 한문장에 있는 단어의 최대 개수
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model) #위치별 임베딩을 저장할 공간
        
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) #position은 [0~seq_len-1]까지의 배열을 생성함 이후 unsqueeze(1)를 통해 (seq_len,1)로 차원을 변경함
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0) / d_model)) #주기적인 포지션 정보를 조정하는 값
        
        #슬라이싱 start:stop:step을 적용하여 열 부분에 짝수,홀수 열을 구분하여 sin,cos 적용
        pe[:,0::2] = torch.sin(position * div_term) #짝수 인덱스는 sin 적용
        pe[:,1::2] = torch.cos(position * div_term) #홀수 인덱스는 cos 적용
        '''
        즉, position값은 처음에는 단순한 위치 인덱스 값을 가진다.
        이후 0으로 초기화 된 pe(포지션 임베딩 행렬)에
        position값과 sin,cos 그리고 포지션 정보 조정값을 짝수,홀수에 맞게 곱해주어서
        의미를 가지는 pe 값을 가지게 한다.
        '''
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe) #pe를 모델의 버퍼로 등록하여 학습하지는 않지만 저장할수있게 구성
            
    def forward(self, x): #입력데이터 x는 단어의 임베딩 결과값
        x = x + (self.pe[:,:x.shape[1], :]).requires_grad_(False) #포지션 임베딩을 추가
        return self.dropout(x) #과적합을 방지하기 위해 dropout을 적용함 예를 들어 x (임베딩 + 포지션)의 값이 [1.0, 2.0, 3.0 ...] 이면 정해둔 확률 만큼 요소들을 0으로 바꾸어줌 [1.0, 0 , 3.0...]

'''
Transformer는 배치 정규화 대신
층 정규화를 씀 신경망의 각 레이어에서 입력 값의 분포를 조정하여 학습을 안정화하는 기법

왜 배치 정규화를 안 쓰고 층 정규화를 쓸까?
자연어 처리에서 배치 정규화는 그 크기가 다양해서 정규화가 어려워 질 수 있음
그렇기에 각 토큰 별로 정규화를 하는 층 정규화를 사용
ex) '나는' = [1.0, 2.0, 3.0 ...] -> 정규화
'''
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps #수치적 안정성을 위해 분모에 작은 값을 더해줌
        
        #nn.Parameter는 학습 가능한 파라미터로 등록함 즉 밑의 alpha와 bias는 학습 가능한 파라미터로 등록함
        #requireds_grad=False로 설정하면 학습을 하지 않게 만들수도 있음
        self.alpha = nn.Parameter(torch.ones(1)) #Multiplier
        self.bias = nn.Parameter(torch.zeros(1)) #Added
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias

#인코더의 마지막에서 사용되는 FFN
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int, d_ff: int, dropout: float)-> None:
        super().__init__()
        self.linear_1=nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2=nn.Linear(d_ff, d_model)
        
    def forward(self,x):
        #(batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

#멀티헤드어텐션으로 어텐션 스코어 계산
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model // h #d_k는 d_model을 head의 개수만큼 나눈 것을 의미
        '''
        어텐션연산은 각 토큰이 다른 토큰과 어떠한 관계를 가지는지를 나타냄
        '나는 파리에 놀러갔다'에서 '파리'와 '나는'을 어텐션 연산을 한다고 하면
        Q(쿼리): 파리
        K(키): 나는
        score = Q x V.T 
        이후 score값을 softmax로 확률화 하고 V를 곱하면 된다.
        
        Q,K,V값은 각각 가중치가 존재하며 임베딩 값 x와 곱해져서 의미를 가지게 된다.
        ex) Q = Q_W @ X
        '''
        #q,k,v의 가중치 행렬 (d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        #어텐션 연산 이후 최종적으로 곱해지는 가중치
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) -> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) #어텐션 스코어 계산: Q @ K / root(d_k)
        '''
        마스킹
        마스킹은 디코더에서 사용되는 부분이다.
        디코더에서는 미래 단어를 예측하는 역할을 하는데 
        이때 마스킹 없이 전체 단어에 대한 attention score가 나오면 안된다.
        그렇기에 마스킹으로 현재 단어 이후의 단어들을 가려버린다.
        ex) '나는 오늘 프랑스 파리에 갔다' 에서 
        디코다가 오늘 이후의 단어를 예측한다고 하면 나는, 오늘 까지의 정보는 알 수 있지만 그 이후 프랑스 부터는 알 수가 없어야 함
        '''
        #마스크가 필요하다면 마스크를 적용하며 미래 단어를 볼수 없도록 함
        '''
        "나는 학생이다"의 마스크는
        mask = [
            [1, 0, 0 ],
            [1, 1, 0 ],
            [1, 1, 1 ]
        ]
        이렇게 되어 현재 단어 이후의 단어를 볼 수 없도록 막아둠
        '''
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) #masked_fill_은 attention_scores값에서 mask가 0으로 설정된 부분을 -1e9로 바꿔준다는 뜻이다. 그렇게 되면 softmax에서 0으로 분류되기 때문이다.
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len) 소프트 맥스를 적용
        #드롭아웃 적용
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        #attention_scores와 value값을 곱해줘야먄 실제 나타내는 값을 가짐
        return (attention_scores @ value), attention_scores #attention_scores는 시각화를 위해 넣어둠
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        
        #view를 통해 멀티헤드 어텐션으로 바꿔줌
        #view()는 텐서의 차원을 바꾸워주는 역할을 함
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch,h,seq_len,d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        #어텐션 스코어는 각 토큰이 다른 토큰과 어떠한 관계를 가지는지를 벡터로 나타내게 됨
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        #이제 각각의 헤드들을 합침
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k) #contignous는 텐서의 메모리 배치를 연속적인 형태로 변환
        '''
        transpose를 사용하면 실제 메모리 주소는 변하지 않고 인덱스만 바뀌게 됨
        view는 메모리가 연속적인 경우에서만 사용이 가능함
        그렇기에 transpose를 쓰면 연속적이지 못한 메모리 배열을 가지기 때문에 contignous를 통해 연속적인 형태로 바꿔줘야함
        '''
        
        #(batch, seq_len, d_model) -> (batch, seq_len, d_model)
        return self.w_o(x) #이후 합쳐진 헤드를 w_O와 곱한 후 반환

#잔차연결
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x))) # 이전 입력 + 이전입력이 레이어를 통과한 값 즉, 잔차연결을 구현
    
'''
이제 지금까지 구현한 것들을 합친 큰 블록인
인코더 블록을 만들어야함
인코드는 크게 FFN과 멀티헤드 어텐션으로 이루어짐
'''

class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
'''
이제 여러개의 인코더를 정의
'''
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
'''
디코더 구현
디코더는 잔차 연결이 3개있고,
attention은 self-attention 1개와 cross-attention 1개로 구성되어 있음
corss-attention의 경우 인코더에서 출력된 값을 query와 key로 사용함
'''

#DecoderBlock은 디코더 한개를 의미함
class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask): #(tgt, encoder_output, src_mask, tgt_mask)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) #마스크 셀프 어텐션
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) #크로스 어텐션
        x = self.residual_connections[2](x, self.feed_forward_block) #피드포워드 블록
        return x

#디코더 여러개를 합친 것
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask): #
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self,x):
        #(Batch, Seq_Len, d_model) --> (Batch,Seq_len, Vocab_size)
        return torch.log_softmax(self.proj(x), dim= -1)
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len) -> (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        #tgt = decoder_input
        # (batch, seq_len) -> (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt) # decoder_input을 임베딩
        tgt = self.tgt_pos(tgt) #포지션 임베딩
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer