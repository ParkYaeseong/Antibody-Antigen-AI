# -*- coding: utf-8 -*-
# train_vh_vae.py

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models')
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")
except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- VAE 및 학습 관련 하이퍼파라미터 ---
SEQ_COLUMN = 'vl_sequence' # 또는 'vl_sequence'
MAX_SEQ_LEN = 150  # VH/VL 최대 길이에 맞춰 설정 (이전 노트북 확인 필요)
VOCAB = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'] # X는 비표준 아미노산
VOCAB_SIZE = len(VOCAB)
EMBEDDING_DIM = 64   # 아미노산 임베딩 차원
HIDDEN_DIM = 256   # LSTM 등 RNN/CNN의 은닉 차원
LATENT_DIM = 64    # 잠재 공간 차원
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100   # 필요에 따라 조절
BETA = 1e-4 # KL 손실 가중치 (튜닝 필요)
VAL_SPLIT_RATIO = 0.1 # 내부 검증용 비율

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 데이터 로딩 및 전처리 ---
print("\n--- 1. 데이터 로딩 및 전처리 ---")
try:
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 메타데이터 로드 완료: {meta_train_path} ({len(metadata_train)} 항목)")

    # 서열 데이터 추출 및 결측치 처리 (만약 있다면)
    sequences = metadata_train[SEQ_COLUMN].dropna().tolist()
    print(f"사용할 {SEQ_COLUMN} 서열 수: {len(sequences)}")
    if len(sequences) == 0: raise ValueError("유효한 서열 데이터가 없습니다.")

    # 사전 및 인코딩 함수 정의
    char_to_int = {char: i for i, char in enumerate(VOCAB)}
    int_to_char = {i: char for i, char in enumerate(VOCAB)}
    PAD_IDX = char_to_int['-']

    def encode_sequence(seq):
        return [char_to_int.get(aa, char_to_int['X']) for aa in seq]

    def pad_encoded_sequence(encoded_seq, max_len):
        padded = encoded_seq[:max_len]
        padded.extend([PAD_IDX] * (max_len - len(padded)))
        return padded

    # 모든 서열 인코딩 및 패딩
    encoded_sequences = [encode_sequence(seq) for seq in sequences]
    padded_sequences = [pad_encoded_sequence(enc_seq, MAX_SEQ_LEN) for enc_seq in encoded_sequences]

    # Tensor로 변환
    data_tensor = torch.tensor(padded_sequences, dtype=torch.long)

    # Dataset 및 DataLoader 생성 (내부 검증 분할 포함)
    full_dataset = TensorDataset(data_tensor) # VAE는 입력만 필요 (자기 자신 복원)
    val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("DataLoader 생성 완료.")

except Exception as e:
    print(f"데이터 준비 중 오류: {e}"); exit()


# --- 2. VAE 모델 정의 (LSTM 예시) ---
print("\n--- 2. VAE 모델 정의 ---")
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

# 모델 인스턴스 생성
model = VAE(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM, MAX_SEQ_LEN, PAD_IDX).to(device)
print("VAE 모델 정의 완료.")
print(model)

# --- 3. 손실 함수 및 옵티마이저 정의 ---
print("\n--- 3. 손실 함수 및 옵티마이저 정의 ---")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def vae_loss_function(recon_x_logits, x, mu, logvar, beta, pad_idx):
    # 재구성 손실 (Reconstruction Loss) - CrossEntropyLoss 사용
    # recon_x_logits: (batch, seq_len, vocab_size)
    # x: (batch, seq_len) - 정수 인덱스 형태
    batch_size, seq_len, vocab_size = recon_x_logits.shape
    # CrossEntropyLoss는 (N, C) 형태 입력을 기대하므로 reshape 필요
    # 또한 padding index는 무시해야 함
    recon_loss = F.cross_entropy(recon_x_logits.view(-1, vocab_size), x.view(-1),
                                 ignore_index=pad_idx, reduction='sum') / batch_size

    # KL 발산 손실 (KL Divergence Loss)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    return recon_loss + beta * kld_loss, recon_loss, kld_loss

print("VAE 손실 함수 및 옵티마이저 정의 완료.")


# --- 4. 모델 학습 함수 --- (VAE용으로 수정)
print("\n--- 4. 모델 학습 함수 정의 ---")
def train_vae_model(model, train_loader, val_loader, optimizer, num_epochs, device, beta, pad_idx):
    train_losses, val_losses = [], []
    train_recon_losses, val_recon_losses = [], []
    train_kld_losses, val_kld_losses = [], []
    best_val_loss = float('inf')
    model_save_path = os.path.join(model_save_dir, 'vl_vae_best.pth') # 모델 파일명 변경

    print(f"\n--- VAE 모델 학습 시작 (Epochs: {num_epochs}) ---")
    start_time_train = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss, running_recon_loss, running_kld_loss = 0.0, 0.0, 0.0

        for i, data in enumerate(train_loader):
            inputs = data[0].to(device) # VAE는 입력만 사용
            optimizer.zero_grad()
            recon_batch_logits, mu, logvar = model(inputs)
            loss, recon_loss, kld_loss = vae_loss_function(recon_batch_logits, inputs, mu, logvar, beta, pad_idx)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_kld_loss += kld_loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_recon_loss = running_recon_loss / len(train_loader)
        epoch_train_kld_loss = running_kld_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        train_recon_losses.append(epoch_train_recon_loss)
        train_kld_losses.append(epoch_train_kld_loss)

        # 검증 단계
        model.eval()
        running_val_loss, running_val_recon_loss, running_val_kld_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                recon_batch_logits, mu, logvar = model(inputs)
                loss, recon_loss, kld_loss = vae_loss_function(recon_batch_logits, inputs, mu, logvar, beta, pad_idx)
                running_val_loss += loss.item()
                running_val_recon_loss += recon_loss.item()
                running_val_kld_loss += kld_loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_recon_loss = running_val_recon_loss / len(val_loader)
        epoch_val_kld_loss = running_val_kld_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_recon_losses.append(epoch_val_recon_loss)
        val_kld_losses.append(epoch_val_kld_loss)
        epoch_end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f} (Recon: {epoch_train_recon_loss:.4f}, KLD: {epoch_train_kld_loss:.4f}), "
              f"Val Loss: {epoch_val_loss:.4f} (Recon: {epoch_val_recon_loss:.4f}, KLD: {epoch_val_kld_loss:.4f}), "
              f"Time: {epoch_end_time - epoch_start_time:.2f} sec")

        # 최고 성능 모델 저장 (Val Loss 기준)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  => Best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    end_time_train = time.time()
    print(f"--- VAE 모델 학습 완료 (총 소요 시간: {end_time_train - start_time_train:.2f} 초) ---")
    return train_losses, val_losses # 필요시 다른 손실도 반환 가능

print("VAE 모델 학습 함수 정의 완료.")

# --- 6. 메인 실행 블록 ---
if __name__ == '__main__':
    # 모델 학습 실행
    train_losses, val_losses = train_vae_model(
        model, train_loader, val_loader, optimizer, NUM_EPOCHS, device, BETA, PAD_IDX
    )

    # (선택사항) 학습 곡선 시각화
    import matplotlib.pyplot as plt
    plt.plot(train_losses, label='Train Total Loss')
    plt.plot(val_losses, label='Validation Total Loss')
    plt.legend()
    plt.title('VAE Loss Curve')
    plt.show()

    # (선택사항) 재구성(Reconstruction) 예시 확인
    model.load_state_dict(torch.load(os.path.join(model_save_dir, 'vl_vae_best.pth')))
    model.eval()
    with torch.no_grad():
        # val_loader 대신 train_loader 사용 가능 (데이터 양이 많음)
        sample_input = next(iter(val_loader))[0][:5].to(device) # 검증 데이터 5개 샘플
        recon_logits, _, _ = model(sample_input)
        recon_probs = F.softmax(recon_logits, dim=-1)
        _, recon_indices = torch.max(recon_probs, dim=-1)
        print("\n--- 재구성 예시 (Original vs Reconstructed) ---")
        for i in range(5):
            original_seq = "".join([int_to_char.get(idx.item(), '?') for idx in sample_input[i] if idx != PAD_IDX])
            # 복원된 서열 길이를 원본 길이에 맞추거나, <eos> 토큰 기반으로 자르는 로직 추가 가능
            recon_seq = "".join([int_to_char.get(idx.item(), '?') for idx in recon_indices[i]]) # 패딩 포함 출력 후 비교
            # 패딩 제외 시: recon_seq = "".join([int_to_char.get(idx.item(), '?') for idx in recon_indices[i] if idx != PAD_IDX])
            print(f"Original:  {original_seq}")
            print(f"Reconstr: {recon_seq.replace('-', '')}\n") # 패딩 문자 제거 후 출력 예시

    print("\n--- VAE 모델 학습 완료 ---")