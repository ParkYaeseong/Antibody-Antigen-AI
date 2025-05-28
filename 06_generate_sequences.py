# -*- coding: utf-8 -*-
# generate_sequences.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# VAE 모델 정의 필요 (train_vh_vae.py 등에서 복사)
# --- VAE 모델 정의 (시작) ---
class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_seq_len, pad_idx):
        super(VAE, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        # 인코더
        self.embedding_enc = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm_enc = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim) # 양방향이므로 *2
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # 디코더
        self.fc_dec_init = nn.Linear(latent_dim, hidden_dim * 2) # LSTM 초기 상태용
        self.embedding_dec = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # 디코더 LSTM은 단방향
        self.lstm_dec = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True) # 입력에 latent vector 추가
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding_enc(x) # (batch, seq_len, embed_dim)
        _, (hn, _) = self.lstm_enc(embedded) # hn shape: (2*num_layers, batch, hidden_dim)
        # 양방향 LSTM의 마지막 hidden state 결합
        hidden = torch.cat((hn[0], hn[1]), dim=1) # (batch, hidden_dim * 2)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_input): # Teacher forcing을 위해 입력 시퀀스(x_input)도 받음
        # 잠재 벡터 z로 LSTM 초기 상태 생성
        batch_size = z.size(0)
        hidden_init = self.fc_dec_init(z) # (batch, hidden_dim * 2)
        # LSTM은 (num_layers, batch, hidden_size) 형태의 hidden/cell state 필요
        h0 = hidden_init[:, :self.hidden_dim].unsqueeze(0).contiguous() # <<< .contiguous() 추가
        c0 = hidden_init[:, self.hidden_dim:].unsqueeze(0).contiguous() # <<< .contiguous() 추가
        # 디코더 입력 준비 (Teacher forcing: 이전 타임스텝의 실제 입력 사용)
        # 첫 입력은 <sos> 토큰, 이후는 실제 입력 시퀀스 (패딩 제외 마지막 토큰까지)
        # 실제 구현에서는 <sos> 토큰을 vocab에 추가하고 처리해야 함 (여기서는 단순화)
        decoder_input_embedded = self.embedding_dec(x_input) # (batch, seq_len, embed_dim)

        # 매 타임스텝마다 잠재 벡터 z를 입력에 추가
        z_repeated = z.unsqueeze(1).repeat(1, self.max_seq_len, 1) # (batch, seq_len, latent_dim)
        decoder_input_with_z = torch.cat((decoder_input_embedded, z_repeated), dim=2) # (batch, seq_len, embed_dim + latent_dim)

        # 디코더 LSTM 실행
        outputs, _ = self.lstm_dec(decoder_input_with_z, (h0, c0)) # outputs: (batch, seq_len, hidden_dim)

        # 최종 출력 계산
        logits = self.fc_out(outputs) # (batch, seq_len, vocab_size)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        # 디코딩 시 teacher forcing 사용 (자기 자신을 복원하도록)
        logits = self.decode(z, x)
        return logits, mu, logvar

    # --- 서열 생성을 위한 generate 메소드 추가 ---
    def generate(self, z, start_token_idx, max_len, device, sampling_strategy='greedy'):
        """ 잠재 벡터 z로부터 서열을 생성합니다. """
        batch_size = z.size(0)
        # LSTM 초기 상태 생성
        hidden_init = self.fc_dec_init(z)
        h = hidden_init[:, :self.hidden_dim].unsqueeze(0).contiguous().to(device)
        c = hidden_init[:, self.hidden_dim:].unsqueeze(0).contiguous().to(device)

        # 시작 토큰 (<sos> 또는 첫 아미노산) - 여기서는 PAD 토큰 인덱스를 시작으로 가정
        current_tokens = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=device)
        generated_indices = []

        self.eval() # 생성 시에는 평가 모드
        with torch.no_grad():
            z_unsqueeze = z.unsqueeze(1) # 루프 내 연산을 위해 차원 추가

            for _ in range(max_len):
                # 현재 토큰 임베딩
                embedded = self.embedding_dec(current_tokens) # (batch, 1, embed_dim)

                # 잠재 벡터와 결합
                decoder_input_with_z = torch.cat((embedded, z_unsqueeze), dim=2) # (batch, 1, embed_dim + latent_dim)

                # LSTM 실행
                output, (h, c) = self.lstm_dec(decoder_input_with_z, (h, c)) # output: (batch, 1, hidden_dim)

                # 다음 토큰 예측 (logits)
                logits = self.fc_out(output.squeeze(1)) # (batch, vocab_size)
                generated_indices.append(logits.argmax(dim=-1)) # 로짓 저장 대신 인덱스 저장

                # 다음 입력 토큰 결정
                if sampling_strategy == 'greedy':
                    next_tokens = logits.argmax(dim=-1) # 가장 확률 높은 토큰 선택
                elif sampling_strategy == 'multinomial':
                    # 확률 분포에서 샘플링 (더 다양한 서열 생성 가능)
                    probs = F.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    raise ValueError("Unknown sampling strategy")

                current_tokens = next_tokens.unsqueeze(1) # 다음 스텝 입력으로

                # (선택사항) <eos> 토큰 처리 로직 추가 가능

        # 생성된 인덱스들을 모아 반환 (batch, seq_len)
        return torch.stack(generated_indices, dim=1)
# --- VAE 모델 정의 (끝) ---

print("--- 필요한 라이브러리 및 VAE 모델 정의 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    model_save_dir = os.path.join(base_dir, '4_trained_models')
    vh_vae_model_path = os.path.join(model_save_dir, 'vh_vae_best.pth')
    vl_vae_model_path = os.path.join(model_save_dir, 'vl_vae_best.pth')
    if not os.path.exists(vh_vae_model_path) or not os.path.exists(vl_vae_model_path):
        raise FileNotFoundError("오류: 학습된 VH 또는 VL VAE 모델 파일(.pth)을 찾을 수 없습니다.")
except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- VAE 파라미터 (학습 시 사용했던 값과 동일하게 설정) ---
MAX_SEQ_LEN = 150
VOCAB = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
VOCAB_SIZE = len(VOCAB)
VAE_EMBEDDING_DIM = 64
VAE_HIDDEN_DIM = 256
VAE_LATENT_DIM = 64
PAD_IDX = VOCAB.index('-')
# --- 생성 관련 설정 ---
NUM_SAMPLES_TO_GENERATE = 10 # 생성할 항체 서열 쌍 개수
SAMPLING_STRATEGY = 'multinomial' # 'greedy' 또는 'multinomial'

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 정의
int_to_char = {i: char for i, char in enumerate(VOCAB)}

# --- 1. 학습된 VAE 모델 로드 ---
print("\n--- 1. 학습된 VAE 모델 로드 ---")
try:
    vae_vh = VAE(VOCAB_SIZE, VAE_EMBEDDING_DIM, VAE_HIDDEN_DIM, VAE_LATENT_DIM, MAX_SEQ_LEN, PAD_IDX).to(device)
    vae_vh.load_state_dict(torch.load(vh_vae_model_path, map_location=device))
    vae_vh.eval()
    print("VH VAE 모델 로드 완료.")

    vae_vl = VAE(VOCAB_SIZE, VAE_EMBEDDING_DIM, VAE_HIDDEN_DIM, VAE_LATENT_DIM, MAX_SEQ_LEN, PAD_IDX).to(device)
    vae_vl.load_state_dict(torch.load(vl_vae_model_path, map_location=device))
    vae_vl.eval()
    print("VL VAE 모델 로드 완료.")
except Exception as e:
    print(f"VAE 모델 로드 중 오류: {e}"); exit()

# --- 2. 잠재 공간에서 샘플링 및 서열 생성 ---
print(f"\n--- 2. 잠재 공간 샘플링 및 서열 생성 ({NUM_SAMPLES_TO_GENERATE}개) ---")

# 표준 정규 분포에서 잠재 벡터 샘플링
z_vh_samples = torch.randn(NUM_SAMPLES_TO_GENERATE, VAE_LATENT_DIM).to(device)
z_vl_samples = torch.randn(NUM_SAMPLES_TO_GENERATE, VAE_LATENT_DIM).to(device)

# VAE 디코더로 서열 생성 (generate 메소드 사용)
# 시작 토큰 인덱스를 PAD_IDX (0)으로 가정 (또는 <sos> 토큰 인덱스 사용)
start_token_idx = PAD_IDX
generated_vh_indices = vae_vh.generate(z_vh_samples, start_token_idx, MAX_SEQ_LEN, device, SAMPLING_STRATEGY).cpu().numpy()
generated_vl_indices = vae_vl.generate(z_vl_samples, start_token_idx, MAX_SEQ_LEN, device, SAMPLING_STRATEGY).cpu().numpy()

# 숫자 인덱스를 아미노산 서열로 변환
generated_vh_seqs = []
generated_vl_seqs = []

for indices in generated_vh_indices:
    seq = "".join([int_to_char.get(idx, '?') for idx in indices if idx != PAD_IDX])
    generated_vh_seqs.append(seq)

for indices in generated_vl_indices:
    seq = "".join([int_to_char.get(idx, '?') for idx in indices if idx != PAD_IDX])
    generated_vl_seqs.append(seq)

# --- 3. 생성된 서열 출력 ---
print("\n--- 3. 생성된 항체 서열 쌍 ---")
for i in range(NUM_SAMPLES_TO_GENERATE):
    print(f"--- 샘플 {i+1} ---")
    print(f"VH: {generated_vh_seqs[i]}")
    print(f"VL: {generated_vl_seqs[i]}\n")

print("--- 서열 생성 완료 ---")