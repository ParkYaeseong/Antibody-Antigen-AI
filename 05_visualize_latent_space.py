# -*- coding: utf-8 -*-
# visualize_latent_space.py

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# VAE 모델 정의 필요 (train_vh_vae.py 등에서 복사)
# --- VAE 모델 정의 (시작) ---
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
        self.fc_dec_init = nn.Linear(latent_dim, hidden_dim * 2)
        self.embedding_dec = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm_dec = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding_enc(x)
        outputs, (hn, cn) = self.lstm_enc(embedded)
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_input):
        batch_size = z.size(0)
        hidden_init = self.fc_dec_init(z)
        h0 = hidden_init[:, :self.hidden_dim].unsqueeze(0).contiguous()
        c0 = hidden_init[:, self.hidden_dim:].unsqueeze(0).contiguous()
        decoder_input_embedded = self.embedding_dec(x_input)
        z_repeated = z.unsqueeze(1).repeat(1, self.max_seq_len, 1)
        decoder_input_with_z = torch.cat((decoder_input_embedded, z_repeated), dim=2)
        outputs, _ = self.lstm_dec(decoder_input_with_z, (h0, c0))
        logits = self.fc_out(outputs)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x)
        return logits, mu, logvar
# --- VAE 모델 정의 (끝) ---

# 시각화 및 차원 축소 라이브러리
import umap # UMAP 설치 필요: pip install umap-learn
import matplotlib.pyplot as plt
import matplotlib.cm as cm # 컬러맵 사용

print("--- 필요한 라이브러리 및 VAE 모델 정의 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models')
    # 시각화 결과 저장 경로 (선택사항)
    visualization_dir = os.path.join(base_dir, '5_visualizations')
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")
    print(f"시각화 저장 경로: {visualization_dir}")

    # VAE 모델 경로
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
BATCH_SIZE = 128 # 인코딩 시 사용할 배치 크기 (메모리에 맞게 조절)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 및 인코딩 함수 정의
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}

def encode_sequence(seq):
    if not isinstance(seq, str): return []
    return [char_to_int.get(aa, char_to_int['X']) for aa in seq]

def pad_encoded_sequence(encoded_seq, max_len):
    padded = encoded_seq[:max_len]
    padded.extend([PAD_IDX] * (max_len - len(padded)))
    return padded

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

# --- 2. 데이터 로딩 및 잠재 벡터 생성 (전체 훈련 데이터 사용) ---
print("\n--- 2. 데이터 로딩 및 잠재 벡터 생성 ---")
try:
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 메타데이터 로드 완료: {meta_train_path} ({len(metadata_train)} 항목)")

    target_col = 'isotype_heavy'
    vh_seq_col = 'vh_sequence'
    vl_seq_col = 'vl_sequence'

    # 시각화를 위해 Isotype 정보가 있는 데이터만 사용
    valid_indices = metadata_train[[target_col, vh_seq_col, vl_seq_col]].notna().all(axis=1)
    metadata_filtered = metadata_train[valid_indices].copy()
    num_removed = len(metadata_train) - len(metadata_filtered)
    print(f"Isotype 또는 서열 결측치 포함 항목 {num_removed}개 제외됨.")
    print(f"시각화에 사용할 데이터 수: {len(metadata_filtered)}")

    if len(metadata_filtered) == 0: raise ValueError("시각화할 유효한 데이터가 없습니다.")

    # 실제 라벨 저장 (시각화용)
    y_labels = metadata_filtered[target_col].tolist()
    unique_labels = sorted(list(set(y_labels))) # 고유 라벨 (색상 매핑용)

    # 서열 인코딩 및 잠재 벡터 추출 함수 (이전과 동일)
    def get_latent_vectors(sequences, vae_model, max_len, batch_size):
        encoded = [encode_sequence(seq) for seq in sequences]
        padded = [pad_encoded_sequence(enc_seq, max_len) for enc_seq in encoded]
        seq_tensor = torch.tensor(padded, dtype=torch.long).to(device)
        latent_vectors = []
        vae_model.eval()
        with torch.no_grad():
            for i in range(0, len(seq_tensor), batch_size):
                batch = seq_tensor[i:i+batch_size]
                mu, _ = vae_model.encode(batch)
                latent_vectors.append(mu.cpu())
        return torch.cat(latent_vectors, dim=0)

    # VH, VL 잠재 벡터 생성
    print("VH 서열 잠재 벡터 생성 중...")
    vh_latent = get_latent_vectors(metadata_filtered[vh_seq_col].tolist(), vae_vh, MAX_SEQ_LEN, BATCH_SIZE)
    print("VL 서열 잠재 벡터 생성 중...")
    vl_latent = get_latent_vectors(metadata_filtered[vl_seq_col].tolist(), vae_vl, MAX_SEQ_LEN, BATCH_SIZE)
    print("VH/VL 잠재 벡터 생성 완료.")

    # VH, VL 잠재 벡터 결합
    latent_combined = torch.cat((vh_latent, vl_latent), dim=1).numpy() # Numpy 배열로 변환
    print(f"결합된 잠재 벡터 Shape: {latent_combined.shape}")

except Exception as e:
    print(f"데이터 준비/잠재 벡터 생성 중 오류: {e}"); exit()


# --- 3. UMAP 차원 축소 및 시각화 ---
print("\n--- 3. UMAP 차원 축소 및 시각화 ---")
try:
    print("UMAP 차원 축소 중... (시간이 소요될 수 있습니다)")
    start_time_umap = time.time()
    reducer = umap.UMAP(n_neighbors=15, # 주변 이웃 수 (조절 가능)
                        min_dist=0.1,   # 점들 사이 최소 거리 (조절 가능)
                        n_components=2, # 2차원으로 축소
                        metric='cosine', # 또는 'euclidean' 등 거리 측정 방식
                        random_state=42)
    embedding_2d = reducer.fit_transform(latent_combined)
    print(f"UMAP 차원 축소 완료. 소요 시간: {time.time() - start_time_umap:.2f} 초")
    print(f"2D 임베딩 Shape: {embedding_2d.shape}")

    # 시각화
    print("시각화 진행 중...")
    plt.figure(figsize=(12, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels))) # 클래스별 색상 자동 할당

    for i, label in enumerate(unique_labels):
        # 해당 라벨을 가진 데이터의 인덱스 찾기
        indices = [idx for idx, l in enumerate(y_labels) if l == label]
        # 2D 임베딩에서 해당 인덱스의 점들만 선택하여 그리기
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1],
                    color=colors[i], label=label, s=10) # s는 점 크기

    plt.title('VAE Latent Space Visualization (Colored by Isotype)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Isotype', markerscale=2) # 범례 표시
    plt.grid(True, linestyle='--', alpha=0.6)

    # 시각화 결과 저장 및 보기
    save_path = os.path.join(visualization_dir, 'vae_latent_space_isotype.png')
    plt.savefig(save_path, dpi=300)
    print(f"시각화 결과 저장 완료: {save_path}")
    # plt.show() # 주피터 노트북 등에서는 이 라인 주석 해제하여 바로 보기

except ImportError:
     print("오류: 시각화에 필요한 라이브러리(umap-learn, matplotlib)가 설치되지 않았습니다.")
     print("pip install umap-learn matplotlib")
except Exception as e:
    print(f"UMAP 시각화 중 오류 발생: {e}")

print("\n--- 스크립트 완료 ---")