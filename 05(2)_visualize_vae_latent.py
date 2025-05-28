# -*- coding: utf-8 -*-
# visualize_vae_latent.py

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# --- VAE 모델 정의 필요 (train_vh_vae.py 등에서 복사) ---
print("--- VAE 모델 클래스 정의 ---")
class VAE(nn.Module):
    # (이전 스크립트의 VAE 클래스 정의 전체를 여기에 복사해 넣으세요)
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
        # 디코더 부분은 시각화에 필요 없지만 로드를 위해 정의는 유지
        self.fc_dec_init = nn.Linear(latent_dim, hidden_dim * 2)
        self.embedding_dec = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm_dec = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding_enc(x)
        _, (hn, _) = self.lstm_enc(embedded) # Use hn for final state
        hidden = torch.cat((hn[0], hn[1]), dim=1) # Correct way to concat bidirectional final hidden states
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar): # 사용 안 함
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_input): # 사용 안 함
        # ... (decode logic) ...
        pass

    def forward(self, x): # 사용 안 함
        # ... (forward logic) ...
        pass
print("VAE 클래스 정의 완료.")


# --- 시각화 및 필요 라이브러리 ---
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models')
    visualization_dir = os.path.join(base_dir, '5_visualizations')
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")
    print(f"시각화 저장 경로: {visualization_dir}")

    # --- 불러올 학습된 VAE 모델 경로 ---
    # 베타 낮춰서 재학습한 모델을 사용할지, 원래 모델을 사용할지 결정 필요
    # 여기서는 원래 모델 사용 가정 (파일명 확인 필요)
    vh_vae_model_path = os.path.join(model_save_dir, 'vh_vae_best.pth')
    vl_vae_model_path = os.path.join(model_save_dir, 'vl_vae_best.pth')
    if not os.path.exists(vh_vae_model_path) or not os.path.exists(vl_vae_model_path):
        raise FileNotFoundError("오류: 학습된 VH 또는 VL VAE 모델 파일(.pth)을 찾을 수 없습니다.")

except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- VAE 및 시각화 관련 설정 ---
TARGET_COLUMN = 'isotype_heavy'     # 색상 구분 기준
VH_SEQ_COL = 'vh_sequence'
VL_SEQ_COL = 'vl_sequence'

# VAE 파라미터 (학습 시 사용했던 값과 동일하게 설정)
MAX_SEQ_LEN = 150
VOCAB = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
VOCAB_SIZE = len(VOCAB)
VAE_EMBEDDING_DIM = 64
VAE_HIDDEN_DIM = 256
VAE_LATENT_DIM = 64
PAD_IDX = VOCAB.index('-')
BATCH_SIZE = 128 # 인코딩 시 배치 크기

# UMAP 파라미터
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine' # 또는 'euclidean'

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 및 인코딩 함수 정의
char_to_int = {char: i for i, char in enumerate(VOCAB)}

def encode_sequence(seq):
    if not isinstance(seq, str): return []
    return [char_to_int.get(aa, char_to_int['X']) for aa in seq]

def pad_encoded_sequence(encoded_seq, max_len):
    padded = encoded_seq[:max_len]
    padded.extend([PAD_IDX] * (max_len - len(padded)))
    return padded

# --- 1. VAE 모델 로드 ---
print("\n--- 1. VAE 모델 로드 ---")
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
    print(f"VAE 모델 로드 중 오류: {e}"); import traceback; traceback.print_exc(); exit()

# --- 2. 데이터 로딩, 전처리 및 VAE 잠재 벡터 생성 ---
print("\n--- 2. 데이터 로딩, 전처리 및 VAE 잠재 벡터 생성 ---")
try:
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 메타데이터 로드 완료: {meta_train_path} ({len(metadata_train)} 항목)")

    # 필요한 컬럼 및 결측치 처리
    required_cols = [TARGET_COLUMN, VH_SEQ_COL, VL_SEQ_COL]
    valid_indices = metadata_train[required_cols].notna().all(axis=1)
    metadata_filtered = metadata_train[valid_indices].copy().reset_index(drop=True)
    num_removed = len(metadata_train) - len(metadata_filtered)
    print(f"Isotype 또는 서열 결측치 포함 항목 {num_removed}개 제외됨.")
    print(f"시각화에 사용할 데이터 수: {len(metadata_filtered)}")

    if len(metadata_filtered) == 0: raise ValueError("시각화할 유효한 데이터가 없습니다.")

    # 실제 라벨 저장
    y_labels = metadata_filtered[TARGET_COLUMN].tolist()
    unique_labels = sorted(list(set(y_labels)))
    print(f"Isotype 클래스 종류: {unique_labels}")

    # 서열 리스트 준비
    vh_sequences = metadata_filtered[VH_SEQ_COL].tolist()
    vl_sequences = metadata_filtered[VL_SEQ_COL].tolist()

    # VAE 잠재 벡터 추출 함수 (mu 값 사용)
    def get_vae_latent_vectors(sequences, vae_model, max_len, batch_size):
        all_mu = []
        encoded = [encode_sequence(seq) for seq in sequences]
        padded = [pad_encoded_sequence(enc_seq, max_len) for enc_seq in encoded]
        seq_tensor = torch.tensor(padded, dtype=torch.long).to(device) # GPU로 바로 보냄

        print(f"총 {len(sequences)}개 서열 잠재 벡터 추출 시작 (배치 크기: {batch_size})...")
        vae_model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(seq_tensor), batch_size), desc="Encoding sequences"):
                batch = seq_tensor[i:i+batch_size]
                mu, _ = vae_model.encode(batch)
                all_mu.append(mu.cpu()) # 결과는 CPU로 모음
        return torch.cat(all_mu, dim=0)

    # VH, VL 잠재 벡터 생성
    vh_latent_mu = get_vae_latent_vectors(vh_sequences, vae_vh, MAX_SEQ_LEN, BATCH_SIZE)
    vl_latent_mu = get_vae_latent_vectors(vl_sequences, vae_vl, MAX_SEQ_LEN, BATCH_SIZE)
    print("VH/VL Latent Mu 벡터 추출 완료.")

    # VH, VL 잠재 벡터 결합
    combined_latent = torch.cat((vh_latent_mu, vl_latent_mu), dim=1).numpy()
    print(f"결합된 VAE 잠재 벡터 Shape: {combined_latent.shape}") # (N, 128)

except Exception as e:
    print(f"데이터 준비/잠재 벡터 생성 중 오류: {e}"); import traceback; traceback.print_exc(); exit()


# --- 3. UMAP 차원 축소 및 시각화 ---
print("\n--- 3. UMAP 차원 축소 및 시각화 ---")
try:
    print("UMAP 차원 축소 중...")
    start_time_umap = time.time()
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST,
                        n_components=2,
                        metric=UMAP_METRIC,
                        random_state=42)
    embedding_2d = reducer.fit_transform(combined_latent) # <<< VAE 잠재 벡터 사용
    print(f"UMAP 차원 축소 완료. 소요 시간: {time.time() - start_time_umap:.2f} 초")
    print(f"2D 임베딩 Shape: {embedding_2d.shape}")

    # 시각화
    print("시각화 진행 중...")
    plt.figure(figsize=(12, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        indices = [idx for idx, l in enumerate(y_labels) if l == label]
        if not indices: continue
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1],
                    color=colors[i], label=label, s=10, alpha=0.7)

    plt.title('VAE Latent Space Visualization (Colored by Isotype)') # <<< 제목 변경
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Isotype', markerscale=2, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 시각화 결과 저장 및 보기
    save_path = os.path.join(visualization_dir, 'vae_latent_space_isotype.png') # <<< 파일명 변경
    plt.savefig(save_path, dpi=300)
    print(f"시각화 결과 저장 완료: {save_path}")
    # plt.show()

except ImportError:
     print("오류: 시각화에 필요한 라이브러리(umap-learn, matplotlib)가 설치되지 않았습니다.")
     print("pip install umap-learn matplotlib")
except Exception as e:
    print(f"UMAP 시각화 중 오류 발생: {e}")

print("\n--- 스크립트 완료 ---")