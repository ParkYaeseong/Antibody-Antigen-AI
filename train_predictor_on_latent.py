# -*- coding: utf-8 -*-
# train_predictor_on_latent.py (Class Weighting Applied)

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

# --- 데이터 전처리, 모델 평가, VAE 모델 정의, 클래스 가중치 계산을 위한 라이브러리 ---
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight # <<< 클래스 가중치 계산용

# --- VAE 모델 정의 (시작) ---
# 이 부분은 VAE 학습 시 사용했던 것과 동일한 구조여야 모델 로딩이 가능합니다.
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

        # 디코더 (여기서는 인코더만 사용하지만, 로딩을 위해 정의는 필요)
        self.fc_dec_init = nn.Linear(latent_dim, hidden_dim * 2) # LSTM 초기 상태용
        self.embedding_dec = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm_dec = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True) # 입력에 latent vector 추가
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embedded = self.embedding_enc(x) # (batch, seq_len, embed_dim)
        outputs, (hn, cn) = self.lstm_enc(embedded) # hn shape: (2*num_layers, batch, hidden_dim)
        # 양방향 LSTM의 마지막 hidden state 결합
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1) # (batch, hidden_dim * 2)
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

print("--- 필요한 라이브러리 및 VAE 모델 정의 임포트 완료 ---")


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

    # --- 불러올 학습된 VAE 모델 경로 ---
    vh_vae_model_path = os.path.join(model_save_dir, 'vh_vae_best.pth')
    vl_vae_model_path = os.path.join(model_save_dir, 'vl_vae_best.pth')
    if not os.path.exists(vh_vae_model_path) or not os.path.exists(vl_vae_model_path):
        raise FileNotFoundError("오류: 학습된 VH 또는 VL VAE 모델 파일(.pth)을 찾을 수 없습니다.")

except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()


# --- VAE 및 예측 모델 관련 하이퍼파라미터 ---
# VAE 파라미터 (학습 시 사용했던 값과 동일하게 설정)
MAX_SEQ_LEN = 150
VOCAB = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
VOCAB_SIZE = len(VOCAB)
VAE_EMBEDDING_DIM = 64
VAE_HIDDEN_DIM = 256
VAE_LATENT_DIM = 64
PAD_IDX = VOCAB.index('-')

# 예측 MLP 모델 파라미터
PREDICTOR_INPUT_DIM = VAE_LATENT_DIM * 2 # z_vh(64) + z_vl(64)
PREDICTOR_HIDDEN_DIM_1 = 128
PREDICTOR_HIDDEN_DIM_2 = 64
# 출력 차원은 데이터 로드 후 결정됨 (num_classes)

# 학습 파라미터
LEARNING_RATE = 1e-4 # 이전과 동일하게 시작 (필요시 조정)
BATCH_SIZE = 64
NUM_EPOCHS = 50 # 이전과 동일하게 시작 (필요시 조정)
VAL_SPLIT_RATIO = 0.2

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 및 인코딩 함수 정의
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}

def encode_sequence(seq):
    # NaN이나 다른 타입이 들어올 경우 빈 리스트 반환하도록 예외 처리 추가
    if not isinstance(seq, str):
        return []
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


# --- 2. 데이터 로딩, 전처리 및 잠재 벡터 생성 ---
print("\n--- 2. 데이터 로딩, 전처리 및 잠재 벡터 생성 ---")
try:
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 메타데이터 로드 완료: {meta_train_path} ({len(metadata_train)} 항목)")

    target_col = 'isotype_heavy'
    vh_seq_col = 'vh_sequence'
    vl_seq_col = 'vl_sequence'

    if not all(col in metadata_train.columns for col in [target_col, vh_seq_col, vl_seq_col]):
        raise ValueError(f"오류: 메타데이터에 필요한 컬럼({target_col}, {vh_seq_col}, {vl_seq_col})이 없습니다.")

    valid_indices = metadata_train[[target_col, vh_seq_col, vl_seq_col]].notna().all(axis=1)
    metadata_filtered = metadata_train[valid_indices].copy()
    num_removed = len(metadata_train) - len(metadata_filtered)
    print(f"결측치 포함 항목 {num_removed}개 제거됨 (타겟 또는 서열 기준).")
    print(f"최종 사용할 훈련 데이터 수: {len(metadata_filtered)}")

    if len(metadata_filtered) == 0: raise ValueError("유효한 데이터가 없습니다.")

    y_labels = metadata_filtered[target_col].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    num_classes = len(label_encoder.classes_)
    print(f"'{target_col}' 라벨 인코딩 완료. 클래스 개수: {num_classes}")
    print(f"클래스 종류: {label_encoder.classes_}")

    # --- 클래스 가중치 계산 ---
    print("\n클래스 가중치 계산 중...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"계산된 클래스 가중치: {class_weights}")
    print(f"가중치 텐서: {class_weights_tensor}")
    # --- 클래스 가중치 계산 끝 ---

    # 서열 인코딩 및 잠재 벡터 추출 함수
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
    start_time_encode = time.time()
    vh_latent = get_latent_vectors(metadata_filtered[vh_seq_col].tolist(), vae_vh, MAX_SEQ_LEN, BATCH_SIZE)
    print(f"VL 서열 잠재 벡터 생성 중... (VH 생성 소요 시간: {time.time() - start_time_encode:.2f}초)")
    encode_start_time_vl = time.time() # VL 인코딩 시작 시간 기록
    vl_latent = get_latent_vectors(metadata_filtered[vl_seq_col].tolist(), vae_vl, MAX_SEQ_LEN, BATCH_SIZE)
    print(f"VH/VL 잠재 벡터 생성 완료. (VL 생성 소요 시간: {time.time() - encode_start_time_vl:.2f}초)") # VL 소요 시간 출력
    print(f"VH Latent shape: {vh_latent.shape}, VL Latent shape: {vl_latent.shape}")

    X_latent_combined = torch.cat((vh_latent, vl_latent), dim=1)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    print(f"결합된 잠재 벡터 Shape: {X_latent_combined.shape}")

    # DataLoader 생성
    full_dataset = TensorDataset(X_latent_combined, y_tensor)
    val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("DataLoader 생성 완료.")

except Exception as e:
    print(f"데이터 준비/잠재벡터 생성 중 오류: {e}"); exit()


# --- 3. 예측 모델 정의 (MLP) ---
print("\n--- 3. 예측 모델 정의 (MLP) ---")
class IsotypePredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(IsotypePredictorMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.relu1(self.layer_1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer_2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

# 모델 인스턴스 생성 및 GPU 이동
predictor_model = IsotypePredictorMLP(
    PREDICTOR_INPUT_DIM, PREDICTOR_HIDDEN_DIM_1, PREDICTOR_HIDDEN_DIM_2, num_classes
).to(device)
print(f"MLP 예측 모델 정의 완료. Device: {device}")
print(predictor_model)


# --- 4. 손실 함수 및 옵티마이저 정의 ---
print("\n--- 4. 손실 함수 및 옵티마이저 정의 ---")
# 클래스 가중치 적용
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device)) # <<< 클래스 가중치 적용
optimizer = optim.Adam(predictor_model.parameters(), lr=LEARNING_RATE)
print(f"손실 함수: CrossEntropyLoss (클래스 가중치 적용됨), 옵티마이저: Adam (lr={LEARNING_RATE})")


# --- 5. 모델 학습 함수 ---
print("\n--- 5. 모델 학습 함수 정의 ---")
def train_predictor_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_filename):
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_accuracy = 0.0 # 정확도 기준 대신 F1 Score 등 다른 지표 고려 가능
    best_val_f1 = 0.0 # 최고 F1 점수 추적
    model_save_path = os.path.join(model_save_dir, model_filename)

    print(f"\n--- 예측 모델 학습 시작 (Epochs: {num_epochs}) ---")
    start_time_train = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 검증 단계
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        # 클래스 불균형 심하므로 Accuracy 외 F1 Score 함께 확인
        epoch_val_accuracy = accuracy_score(all_labels, all_preds) * 100
        epoch_val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) # Macro F1 사용

        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        epoch_end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              # f"Val Accuracy: {epoch_val_accuracy:.2f}%, " # 정확도는 참고용
              f"Val Macro F1: {epoch_val_f1:.4f}, " # Macro F1 출력
              f"Time: {epoch_end_time - epoch_start_time:.2f} sec")

        # 최고 성능 모델 저장 (Macro F1 기준)
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  => Best model saved to {model_save_path} (Val Macro F1: {best_val_f1:.4f})")

    end_time_train = time.time()
    print(f"--- 예측 모델 학습 완료 (총 소요 시간: {end_time_train - start_time_train:.2f} 초) ---")
    return train_losses, val_losses, val_accuracies # val_accuracies 대신 F1 점수 리스트 반환 고려

print("예측 모델 학습 함수 정의 완료.")


# --- 6. 모델 평가 함수 ---
print("\n--- 6. 모델 평가 함수 정의 ---")
def evaluate_predictor_model(model, data_loader, device, label_encoder): # label_encoder 전달
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) # Macro F1 추가
    print("\n--- 최종 검증 세트 평가 결과 ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}") # Macro F1 출력
    print("\nClassification Report:")
    all_labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_
    print(classification_report(y_true, y_pred,
                                labels=all_labels,
                                target_names=target_names,
                                zero_division=0))

print("모델 평가 함수 정의 완료.")


# --- 7. 메인 실행 블록 ---
if __name__ == '__main__':
    # 모델 학습 실행
    train_losses, val_losses, val_accuracies = train_predictor_model(
        predictor_model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device,
        'isotype_predictor_on_latent_best_weighted.pth' # 저장할 모델 파일명 변경
    )

    # 최종 모델 로드 및 평가
    best_model_path = os.path.join(model_save_dir, 'isotype_predictor_on_latent_best_weighted.pth')
    if os.path.exists(best_model_path):
        # 모델 로드 전, num_classes가 정확히 정의되었는지 확인
        if 'num_classes' not in locals() or 'label_encoder' not in locals():
             print("오류: num_classes 또는 label_encoder가 정의되지 않아 모델 로드/평가 불가.")
        else:
             # 로드할 모델 구조와 현재 predictor_model 구조가 동일해야 함
             predictor_model.load_state_dict(torch.load(best_model_path))
             print(f"\nBest predictor model loaded from {best_model_path} for final evaluation.")
             evaluate_predictor_model(predictor_model, val_loader, device, label_encoder) # label_encoder 전달
    else:
        print("\n경고: 저장된 최적 예측 모델 파일을 찾을 수 없어 최종 평가를 건너뜁니다.")

    print("\n--- Latent 기반 Isotype 예측 모델 (Weighted) 학습 및 평가 완료 ---")