# -*- coding: utf-8 -*-
# evaluate_vae_reconstruction.py

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset # TensorDataset 추가
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split # 필요시 사용

# --- VAE 모델 정의 (train_vh_vae.py 에서 복사) ---
print("--- VAE 모델 클래스 정의 ---")
class VAE(nn.Module):
    # ... (VAE 클래스 정의 전체 복사) ...
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_seq_len, pad_idx):
        super(VAE, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.embedding_enc = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm_enc = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_dec_init = nn.Linear(latent_dim, hidden_dim * 2)
        self.embedding_dec = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm_dec = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding_enc(x)
        _, (hn, _) = self.lstm_enc(embedded)
        hidden = torch.cat((hn[0], hn[1]), dim=1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, decoder_input):
        batch_size = z.size(0)
        seq_len = decoder_input.size(1)
        hidden_init = self.fc_dec_init(z)
        h0 = hidden_init[:, :self.hidden_dim].unsqueeze(0).contiguous()
        c0 = hidden_init[:, self.hidden_dim:].unsqueeze(0).contiguous()
        decoder_input_embedded = self.embedding_dec(decoder_input)
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input_with_z = torch.cat((decoder_input_embedded, z_repeated), dim=2)
        outputs, _ = self.lstm_dec(decoder_input_with_z, (h0, c0))
        logits = self.fc_out(outputs)
        return logits

    def forward(self, encoder_input, decoder_input):
        mu, logvar = self.encode(encoder_input)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, decoder_input)
        return logits, mu, logvar
print("VAE 클래스 정의 완료.")


# --- 설정 ---
SEQ_COL_TO_EVAL = 'vl_sequence' # 또는 vh_sequence
MODEL_FILENAME = f"{SEQ_COL_TO_EVAL}_vae_sos_eos_anneal_best.pth" # 평가할 모델 파일

base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
split_dir = os.path.join(base_dir, '3_split_data')
model_save_dir = os.path.join(base_dir, '4_trained_models')
visualization_dir = os.path.join(base_dir, '5_visualizations')
model_path = os.path.join(model_save_dir, MODEL_FILENAME)

# --- VAE 파라미터 (학습 시 사용했던 값과 동일하게 설정) ---
MAX_SEQ_LEN = 150
PAD_TOKEN = '-'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = 'X'
VOCAB = [PAD_TOKEN] + sorted("ACDEFGHIKLMNPQRSTVWY") + [UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB.index(PAD_TOKEN)
SOS_IDX = VOCAB.index(SOS_TOKEN)
EOS_IDX = VOCAB.index(EOS_TOKEN)
UNK_IDX = VOCAB.index(UNK_TOKEN)
VAE_EMBEDDING_DIM = 64
VAE_HIDDEN_DIM = 256
VAE_LATENT_DIM = 64
BATCH_SIZE = 128 # 평가 시 배치 크기

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 및 인코딩 함수 정의
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}
def encode_sequence(seq):
    if not isinstance(seq, str): return []
    return [char_to_int.get(aa, UNK_IDX) for aa in seq]

# 데이터셋 클래스 (학습 시 사용했던 것과 동일)
class AntibodyDataset(Dataset):
    def __init__(self, dataframe, seq_column, max_len, sos_idx, eos_idx, pad_idx):
        self.sequences = dataframe[seq_column].dropna().tolist()
        self.max_len = max_len
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        encoded = encode_sequence(seq_str)
        enc_input = encoded[:self.max_len]
        enc_input_padded = enc_input + [self.pad_idx] * (self.max_len - len(enc_input))
        dec_input = [self.sos_idx] + encoded[:self.max_len-1]
        dec_input_padded = dec_input + [self.pad_idx] * (self.max_len - len(dec_input))
        dec_target = encoded[:self.max_len-1] + [self.eos_idx]
        dec_target_padded = dec_target + [self.pad_idx] * (self.max_len - len(dec_target))
        return {'encoder_input': torch.tensor(enc_input_padded, dtype=torch.long),
                'decoder_input': torch.tensor(dec_input_padded, dtype=torch.long),
                'decoder_target': torch.tensor(dec_target_padded, dtype=torch.long)}

# --- 모델 로드 ---
print("\n--- 모델 로드 ---")
if not os.path.exists(model_path):
    print(f"오류: 모델 파일을 찾을 수 없습니다 - {model_path}")
    exit()
try:
    vae_model = VAE(VOCAB_SIZE, VAE_EMBEDDING_DIM, VAE_HIDDEN_DIM, VAE_LATENT_DIM, MAX_SEQ_LEN, PAD_IDX).to(device)
    vae_model.load_state_dict(torch.load(model_path, map_location=device))
    vae_model.eval()
    print(f"VAE 모델 로드 완료: {model_path}")
except Exception as e:
    print(f"모델 로드 오류: {e}"); exit()

# --- 데이터 로드 (검증 또는 테스트 데이터 사용) ---
print("\n--- 평가 데이터 로드 ---")
try:
    # 검증/테스트 데이터 로드 (데이터 분할 방식에 따라 경로/파일명 확인)
    # 여기서는 훈련 데이터 중 일부를 다시 사용하거나, 별도 저장된 검증 데이터 사용
    meta_data_path = os.path.join(split_dir, 'metadata_train.parquet') # 예시: 훈련 데이터 사용
    meta_data = pd.read_parquet(meta_data_path)
    valid_indices = meta_data[SEQ_COL_TO_EVAL].notna()
    meta_filtered = meta_data[valid_indices]

    # 내부 검증 세트 인덱스를 사용하거나, 전체 데이터 사용
    # 여기서는 간단히 전체 필터링된 데이터를 평가용으로 사용
    eval_dataset = AntibodyDataset(meta_filtered, SEQ_COL_TO_EVAL, MAX_SEQ_LEN, SOS_IDX, EOS_IDX, PAD_IDX)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"평가 데이터 로더 생성 완료 (샘플 수: {len(eval_dataset)})")

except Exception as e:
    print(f"평가 데이터 로딩 오류: {e}"); exit()

# --- 정량적 재구성 평가 ---
print("\n--- 정량적 재구성 평가 시작 ---")
all_identities = []
vae_model.eval()
with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating Reconstruction"):
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)

        recon_logits, _, _ = vae_model(encoder_input, decoder_input)
        recon_indices = torch.argmax(recon_logits, dim=2).cpu().numpy()
        target_indices = decoder_target.cpu().numpy()

        for i in range(encoder_input.size(0)):
            # 원본 서열 (타겟 기준, EOS/PAD 제외)
            target_i = target_indices[i]
            eos_pos_orig = np.where(target_i == EOS_IDX)[0]
            pad_pos_orig = np.where(target_i == PAD_IDX)[0]
            end_pos_orig = min(eos_pos_orig[0] if len(eos_pos_orig) > 0 else len(target_i),
                               pad_pos_orig[0] if len(pad_pos_orig) > 0 else len(target_i))
            original_valid_indices = target_i[:end_pos_orig]

            # 재구성 서열 (동일 길이만큼 비교)
            reconstructed_indices_i = recon_indices[i][:end_pos_orig]

            # 일치도 계산
            if len(original_valid_indices) > 0:
                matches = (original_valid_indices == reconstructed_indices_i).sum()
                identity = matches / len(original_valid_indices)
                all_identities.append(identity)
            else:
                all_identities.append(0.0) # 원본 길이가 0인 경우

print("재구성 평가 완료.")

# --- 결과 분석 및 시각화 ---
if all_identities:
    all_identities = np.array(all_identities) * 100 # 백분율로 변환
    print("\n--- 재구성 정확도(%) 통계 ---")
    print(f"평균 (Mean): {np.mean(all_identities):.2f}%")
    print(f"중간값 (Median): {np.median(all_identities):.2f}%")
    print(f"표준편차 (Std Dev): {np.std(all_identities):.2f}%")
    print(f"최소 (Min): {np.min(all_identities):.2f}%")
    print(f"최대 (Max): {np.max(all_identities):.2f}%")
    print(f"95% 이상 정확도 비율: {np.mean(all_identities >= 95)*100:.2f}%")
    print(f"50% 미만 정확도 비율: {np.mean(all_identities < 50)*100:.2f}%")

    # 히스토그램 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(all_identities, bins=20, edgecolor='black')
    plt.title(f'{SEQ_COL_TO_EVAL} VAE Reconstruction Accuracy Distribution')
    plt.xlabel('Per-Sequence Reconstruction Accuracy (%)')
    plt.ylabel('Number of Sequences')
    plt.grid(axis='y', alpha=0.75)
    hist_save_path = os.path.join(visualization_dir, f'{SEQ_COL_TO_EVAL}_vae_recon_accuracy_hist.png')
    plt.savefig(hist_save_path)
    print(f"재구성 정확도 분포 히스토그램 저장 완료: {hist_save_path}")
    # plt.show()
else:
    print("계산된 재구성 정확도가 없습니다.")

print("\n--- 스크립트 완료 ---")