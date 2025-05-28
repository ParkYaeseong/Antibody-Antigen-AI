# -*- coding: utf-8 -*-
# train_predictor_on_igbert.py (Uses Pre-saved IgBert Embeddings + Class Weighting)

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

# --- 필요한 라이브러리 ---
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split # stratify 옵션 위해 import
from tqdm.auto import tqdm
try:
    from torch.optim import AdamW # PyTorch 1.8+
except ImportError:
    from transformers import AdamW # Older transformers

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 예측 모델 클래스 정의 ---
# 저장된 모델 로드를 위해 필요
print("--- 예측 모델 클래스 정의 ---")
class IsotypePredictorMLP(nn.Module):
    # 이 클래스 정의는 저장된 모델과 구조가 같아야 함
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, dropout_rate):
        super(IsotypePredictorMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.relu1(self.layer_1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer_2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x
print("IsotypePredictorMLP 클래스 정의 완료.")


# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data') # 분할된 데이터 로드
    model_save_dir = os.path.join(base_dir, '4_trained_models') # 모델 저장 경로
    os.makedirs(model_save_dir, exist_ok=True)

    # --- 사용할 임베딩 파일 경로 ---
    # !!! 중요: 이 파일들이 Exscientia/IgBert 로 생성된 분할 데이터가 맞는지 확인!!!
    vh_embedding_path = os.path.join(split_dir, 'X_train_vh.npy')
    vl_embedding_path = os.path.join(split_dir, 'X_train_vl.npy')
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')

    if not os.path.exists(vh_embedding_path): raise FileNotFoundError(f"VH Embedding 파일 없음: {vh_embedding_path}")
    if not os.path.exists(vl_embedding_path): raise FileNotFoundError(f"VL Embedding 파일 없음: {vl_embedding_path}")
    if not os.path.exists(meta_train_path): raise FileNotFoundError(f"Metadata 파일 없음: {meta_train_path}")

    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")

except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- 모델 및 학습 관련 하이퍼파라미터 ---
TARGET_COLUMN = 'isotype_heavy'
# Exscientia/IgBert 임베딩 차원 확인 필요 (보통 512)
IGBERT_EMBED_DIM = 1024 # <<< Exscientia/IgBert의 임베딩 차원
PREDICTOR_INPUT_DIM = IGBERT_EMBED_DIM * 2 # 512 + 512 = 1024
PREDICTOR_HIDDEN_DIM_1 = 512
PREDICTOR_HIDDEN_DIM_2 = 256
PREDICTOR_DROPOUT_RATE = 0.4

# 학습 파라미터
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100 # Early Stopping 사용 예정
VAL_SPLIT_RATIO = 0.2
EARLY_STOPPING_PATIENCE = 15

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 1. 데이터 로딩 및 전처리 ---
print("\n--- 1. 데이터 로딩 및 전처리 ---")
try:
    # 저장된 IgBert 임베딩 로드
    X_train_vh = np.load(vh_embedding_path)
    X_train_vl = np.load(vl_embedding_path)
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 데이터 로드 완료:")
    print(f"  VH 임베딩 shape: {X_train_vh.shape}")
    print(f"  VL 임베딩 shape: {X_train_vl.shape}")
    print(f"  메타데이터 shape: {metadata_train.shape}")

    # 결측치 처리 (Isotype 기준)
    target_col = TARGET_COLUMN
    if target_col not in metadata_train.columns:
        raise ValueError(f"오류: 메타데이터에 타겟 컬럼 '{target_col}'이 없습니다.")

    original_count = len(metadata_train)
    # 임베딩과 메타데이터 길이가 같은지 확인 (중요)
    if len(X_train_vh) != original_count or len(X_train_vl) != original_count:
        print("경고: 임베딩과 메타데이터 길이가 다릅니다! 데이터 생성 과정 확인 필요.")
        # 길이가 다를 경우, 짧은 쪽에 맞춰 인덱싱 필요 (metadata_train.index 사용 등)
        # 여기서는 일단 진행, 실제로는 원인 파악 및 수정 필요
        min_len = min(len(X_train_vh), len(metadata_train))
        X_train_vh = X_train_vh[:min_len]
        X_train_vl = X_train_vl[:min_len]
        metadata_train = metadata_train.iloc[:min_len]
        original_count = min_len

    valid_indices = metadata_train[target_col].notna()
    metadata_filtered = metadata_train[valid_indices].copy()
    # .loc을 사용하여 boolean indexing 적용 (SettingWithCopyWarning 방지)
    X_train_vh_filtered = X_train_vh[valid_indices.to_numpy(dtype=bool)]
    X_train_vl_filtered = X_train_vl[valid_indices.to_numpy(dtype=bool)]

    num_removed = original_count - len(metadata_filtered)
    print(f"'{target_col}' 결측치 포함 항목 {num_removed}개 제거됨.")
    print(f"최종 사용할 훈련 데이터 수: {len(metadata_filtered)}")

    if len(metadata_filtered) == 0: raise ValueError("유효한 데이터가 없습니다.")

    # 라벨 인코딩
    y_labels = metadata_filtered[target_col].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    num_classes = len(label_encoder.classes_)
    print(f"'{target_col}' 라벨 인코딩 완료. 클래스 개수: {num_classes}")
    print(f"클래스 종류: {list(label_encoder.classes_)}")

    # 클래스 가중치 계산
    print("클래스 가중치 계산 중...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"계산된 클래스 가중치: {class_weights}")

    # VH, VL 임베딩 결합
    X_combined_embeddings = np.concatenate((X_train_vh_filtered, X_train_vl_filtered), axis=1)
    print(f"결합된 IgBert 임베딩 Shape: {X_combined_embeddings.shape}") # (N, 1024)

    # Tensor 변환
    X_tensor = torch.tensor(X_combined_embeddings, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    # DataLoader 생성 (내부 Train/Validation 분할)
    full_dataset = TensorDataset(X_tensor, y_tensor)
    val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
    train_size = len(full_dataset) - val_size
    # train_test_split 사용 시 stratify 적용
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=VAL_SPLIT_RATIO,
        random_state=42,
        stratify=y_tensor.numpy() # stratify 위해 numpy 배열 사용
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"DataLoader 생성 완료: Train={len(train_dataset)}, Validation={len(val_dataset)}")

except FileNotFoundError as e:
    print(e); exit()
except ValueError as e:
    print(e); exit()
except Exception as e:
    print(f"데이터 준비 중 오류: {e}"); import traceback; traceback.print_exc(); exit()


# --- 2. 예측 모델 정의 (MLP) ---
print("\n--- 2. 예측 모델 정의 (MLP) ---")
# MLP 모델 정의는 위에서 완료

# 모델 인스턴스 생성 및 GPU 이동
predictor_model = IsotypePredictorMLP(
    PREDICTOR_INPUT_DIM, PREDICTOR_HIDDEN_DIM_1, PREDICTOR_HIDDEN_DIM_2, num_classes, PREDICTOR_DROPOUT_RATE
).to(device)
print(f"MLP 예측 모델 정의 완료. Input Dim: {PREDICTOR_INPUT_DIM}, Device: {device}")
print(predictor_model)


# --- 3. 손실 함수 및 옵티마이저 정의 ---
print("\n--- 3. 손실 함수 및 옵티마이저 정의 ---")
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # 클래스 가중치 적용
optimizer = AdamW(predictor_model.parameters(), lr=LEARNING_RATE)
print(f"손실 함수: CrossEntropyLoss (클래스 가중치 적용됨), 옵티마이저: AdamW (lr={LEARNING_RATE})")


# --- 4. 모델 학습 함수 정의 (Early Stopping 포함) ---
print("\n--- 4. 모델 학습 함수 정의 ---")
# train_predictor_model 함수 정의 (이전 AntiBERTy 스크립트와 동일)
def train_predictor_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_filename, label_encoder, patience):
    train_losses, val_losses, val_f1s = [], [], [] # F1 점수 추적
    best_val_f1 = 0.0 # 최고 Macro F1 추적
    epochs_no_improve = 0
    model_save_path = os.path.join(model_save_dir, model_filename)

    print(f"\n--- 예측 모델 학습 시작 (Epochs: {num_epochs}, Early Stopping Patience: {patience}) ---")
    start_time_train = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
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
        epoch_val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_val_accuracy = accuracy_score(all_labels, all_preds) * 100
        val_losses.append(epoch_val_loss)
        val_f1s.append(epoch_val_f1)
        epoch_end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Macro F1: {epoch_val_f1:.4f}, "
              f"Val Acc: {epoch_val_accuracy:.2f}%, "
              f"Time: {epoch_end_time - epoch_start_time:.2f} sec")

        # 최고 성능 모델 저장 및 Early Stopping 체크 (Val Macro F1 기준)
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  => Best model saved to {model_save_path} (Val Macro F1: {best_val_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  (No improvement for {epochs_no_improve} epochs)")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    end_time_train = time.time()
    print(f"--- 예측 모델 학습 완료 (총 소요 시간: {end_time_train - start_time_train:.2f} 초) ---")
    return train_losses, val_losses, val_f1s
print("예측 모델 학습 함수 정의 완료.")


# --- 5. 모델 평가 함수 정의 --- (이전과 동일)
print("\n--- 5. 모델 평가 함수 정의 ---")
def evaluate_predictor_model(model, data_loader, device, label_encoder):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print("\n--- 최종 검증 세트 평가 결과 ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print("\nClassification Report:")
    all_labels = np.arange(len(label_encoder.classes_))
    target_names = label_encoder.classes_
    print(classification_report(y_true, y_pred,
                                labels=all_labels,
                                target_names=target_names,
                                zero_division=0))
print("모델 평가 함수 정의 완료.")


# --- 6. 메인 실행 블록 ---
if __name__ == '__main__':
    # 모델 학습 실행
    train_losses, val_losses, val_f1s = train_predictor_model(
        predictor_model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device,
        'isotype_predictor_on_igbert_best_weighted.pth', # <<< 저장 파일명 변경
        label_encoder, # 평가 함수용 전달
        patience=EARLY_STOPPING_PATIENCE
    )

    # 최종 모델 로드 및 평가
    best_model_path = os.path.join(model_save_dir, 'isotype_predictor_on_igbert_best_weighted.pth')
    if os.path.exists(best_model_path):
        try:
            if 'num_classes' not in locals() or 'label_encoder' not in locals():
                 print("오류: num_classes 또는 label_encoder가 정의되지 않아 모델 로드/평가 불가.")
            else:
                 # 로드할 모델 구조와 현재 predictor_model 구조가 동일해야 함
                 predictor_model_loaded = IsotypePredictorMLP(
                     PREDICTOR_INPUT_DIM, PREDICTOR_HIDDEN_DIM_1, PREDICTOR_HIDDEN_DIM_2, num_classes, PREDICTOR_DROPOUT_RATE
                 ).to(device)
                 predictor_model_loaded.load_state_dict(torch.load(best_model_path, map_location=device))
                 print(f"\nBest predictor model loaded from {best_model_path} for final evaluation.")
                 evaluate_predictor_model(predictor_model_loaded, val_loader, device, label_encoder)
        except Exception as e:
            print(f"최적 모델 로드/평가 중 오류: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n경고: 저장된 최적 예측 모델 파일을 찾을 수 없어 최종 평가를 건너뜁니다.")

    print("\n--- IgBert 기반 Isotype 예측 모델 학습 및 평가 완료 ---")