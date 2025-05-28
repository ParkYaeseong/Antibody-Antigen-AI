# -*- coding: utf-8 -*-
# train_predictor_on_antiberty.py (Uses AntiBERTy Embeddings + Class Weighting)

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split # <<< TensorDataset 추가

# --- 필요한 라이브러리 ---
# pip install antiberty torch scikit-learn pandas pyarrow accelerate matplotlib
from antiberty import AntiBERTyRunner # AntiBERTy 사용
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
# accelerate는 선택 사항 (GPU 사용 시 도움)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models') # 예측 모델 저장 경로
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")
except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- 모델 및 학습 관련 하이퍼파라미터 ---
TARGET_COLUMN = 'isotype_heavy'     # 예측 목표
VH_SEQ_COL = 'vh_sequence'
VL_SEQ_COL = 'vl_sequence'

# 예측 MLP 모델 파라미터
PREDICTOR_INPUT_DIM = 512 * 2 # AntiBERTy CLS(512) + CLS(512)
PREDICTOR_HIDDEN_DIM_1 = 512 # 예시 (조절 가능)
PREDICTOR_HIDDEN_DIM_2 = 256 # 예시 (조절 가능)
# 출력 차원은 데이터 로드 후 결정됨 (num_classes)

# 학습 파라미터
LEARNING_RATE = 1e-4
BATCH_SIZE = 32     # AntiBERTy 임베딩 시 메모리 사용량 고려하여 조절
NUM_EPOCHS = 100     # 필요시 조절
VAL_SPLIT_RATIO = 0.2 # 훈련 데이터 내 검증 비율

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. AntiBERTy 모델 로드 ---
# AntiBERTyRunner는 비교적 가벼우므로 여기서 미리 로드
print("\n--- 1. AntiBERTy 모델 로드 ---")
try:
    # AntiBERTyRunner는 내부적으로 모델을 로드 (필요시 GPU 사용 추정)
    antiberty_runner = AntiBERTyRunner()
    # 강제로 GPU 사용 지정 (필요시, runner 내부 구현 확인 필요)
    # if torch.cuda.is_available():
    #     antiberty_runner.model.to(device)
    #     antiberty_runner.tokenizer.to(device) # 토크나이저도? 확인필요
    print("AntiBERTyRunner 생성 완료.")
except NameError:
    print("오류: 'antiberty' 라이브러리가 설치되지 않았거나 import되지 않았습니다.")
    print("pip install antiberty 를 실행해주세요.")
    exit()
except Exception as e:
    print(f"AntiBERTyRunner 생성 중 오류: {e}"); exit()


# --- 2. 데이터 로딩, 전처리 및 AntiBERTy 임베딩 생성 ---
print("\n--- 2. 데이터 로딩, 전처리 및 AntiBERTy 임베딩 생성 ---")
try:
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 메타데이터 로드 완료: {meta_train_path} ({len(metadata_train)} 항목)")

    # 필요한 컬럼 및 결측치 처리
    required_cols = [TARGET_COLUMN, VH_SEQ_COL, VL_SEQ_COL]
    valid_indices = metadata_train[required_cols].notna().all(axis=1)
    metadata_filtered = metadata_train[valid_indices].copy().reset_index(drop=True) # 인덱스 리셋
    num_removed = len(metadata_train) - len(metadata_filtered)
    print(f"결측치 포함 항목 {num_removed}개 제거됨.")
    print(f"최종 사용할 훈련 데이터 수: {len(metadata_filtered)}")

    if len(metadata_filtered) == 0: raise ValueError("유효한 데이터가 없습니다.")

    # 라벨 인코딩
    y_labels = metadata_filtered[TARGET_COLUMN].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    num_classes = len(label_encoder.classes_)
    print(f"'{TARGET_COLUMN}' 라벨 인코딩 완료. 클래스 개수: {num_classes}")
    print(f"클래스 종류: {label_encoder.classes_}")

    # 클래스 가중치 계산
    print("클래스 가중치 계산 중...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"계산된 클래스 가중치: {class_weights}")

    # 서열 리스트 준비
    vh_sequences = metadata_filtered[VH_SEQ_COL].tolist()
    vl_sequences = metadata_filtered[VL_SEQ_COL].tolist()

    # AntiBERTy 임베딩 추출 함수 (배치 처리 포함)
    def get_antiberty_embeddings(sequences, runner, batch_size=32):
        all_cls_embeddings = []
        print(f"총 {len(sequences)}개 서열 임베딩 추출 시작 (배치 크기: {batch_size})...")
        # tqdm으로 진행률 표시
        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding sequences"):
            batch_seqs = sequences[i:i+batch_size]
            # AntiBERTyRunner.embed는 리스트를 반환
            embeddings_list = runner.embed(batch_seqs)
            # 각 임베딩 텐서에서 CLS 토큰 (첫 번째) 임베딩 추출 후 리스트에 추가
            # 결과를 CPU로 옮겨 메모리 관리 용이하게 함
            cls_embeddings_batch = torch.stack([emb[0].cpu() for emb in embeddings_list])
            all_cls_embeddings.append(cls_embeddings_batch)
            # GPU 메모리 정리 (선택적)
            # if device.type == 'cuda': torch.cuda.empty_cache()

        # 모든 배치 결과 합치기
        return torch.cat(all_cls_embeddings, dim=0)

    # VH, VL 임베딩 생성
    vh_embeddings = get_antiberty_embeddings(vh_sequences, antiberty_runner, BATCH_SIZE)
    vl_embeddings = get_antiberty_embeddings(vl_sequences, antiberty_runner, BATCH_SIZE)
    print("VH/VL CLS 임베딩 추출 완료.")
    print(f"VH Embeddings shape: {vh_embeddings.shape}, VL Embeddings shape: {vl_embeddings.shape}")

    # VH, VL 임베딩 결합 (CPU 상에서)
    X_combined_embeddings = torch.cat((vh_embeddings, vl_embeddings), dim=1)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    print(f"결합된 AntiBERTy 임베딩 Shape: {X_combined_embeddings.shape}") # (N, 1024)

    # DataLoader 생성
    full_dataset = TensorDataset(X_combined_embeddings, y_tensor)
    val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("DataLoader 생성 완료.")

except ValueError as e:
    print(e); exit()
except NameError as e:
    print(f"오류: 필요한 라이브러리(예: antiberty)가 import되지 않았을 수 있습니다. {e}"); exit()
except Exception as e:
    print(f"데이터 준비/임베딩 생성 중 오류: {e}"); import traceback; traceback.print_exc(); exit()


# --- 3. 예측 모델 정의 (MLP) ---
print("\n--- 3. 예측 모델 정의 (MLP) ---")
class IsotypePredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(IsotypePredictorMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4) # 드롭아웃 비율 조절 가능
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4) # 드롭아웃 비율 조절 가능
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
print(f"MLP 예측 모델 정의 완료. Input Dim: {PREDICTOR_INPUT_DIM}, Device: {device}")
print(predictor_model)


# --- 4. 손실 함수 및 옵티마이저 정의 ---
print("\n--- 4. 손실 함수 및 옵티마이저 정의 ---")
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device)) # 클래스 가중치 적용
optimizer = AdamW(predictor_model.parameters(), lr=LEARNING_RATE) # AdamW 사용
print(f"손실 함수: CrossEntropyLoss (클래스 가중치 적용됨), 옵티마이저: AdamW (lr={LEARNING_RATE})")


# --- 5. 모델 학습 함수 ---
print("\n--- 5. 모델 학습 함수 정의 ---")
def train_predictor_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_filename, label_encoder):
    train_losses, val_losses, val_f1s = [], [], [] # F1 점수 추적
    best_val_f1 = 0.0
    model_save_path = os.path.join(model_save_dir, model_filename)

    print(f"\n--- 예측 모델 학습 시작 (Epochs: {num_epochs}) ---")
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
        epoch_val_accuracy = accuracy_score(all_labels, all_preds) * 100 # 참고용 정확도
        val_losses.append(epoch_val_loss)
        val_f1s.append(epoch_val_f1)
        epoch_end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Macro F1: {epoch_val_f1:.4f}, "
              f"Val Acc: {epoch_val_accuracy:.2f}%, " # 정확도 추가 출력
              f"Time: {epoch_end_time - epoch_start_time:.2f} sec")

        # 최고 성능 모델 저장 (Val Macro F1 기준)
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"  => Best model saved to {model_save_path} (Val Macro F1: {best_val_f1:.4f})")

    end_time_train = time.time()
    print(f"--- 예측 모델 학습 완료 (총 소요 시간: {end_time_train - start_time_train:.2f} 초) ---")
    return train_losses, val_losses, val_f1s

print("예측 모델 학습 함수 정의 완료.")


# --- 6. 모델 평가 함수 ---
print("\n--- 6. 모델 평가 함수 정의 ---")
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
    # zero_division=1 설정하여 경고 대신 1.0 또는 0.0 출력 (선택사항)
    print(classification_report(y_true, y_pred,
                                labels=all_labels,
                                target_names=target_names,
                                zero_division=0))

print("모델 평가 함수 정의 완료.")


# --- 7. 메인 실행 블록 ---
if __name__ == '__main__':
    # 모델 학습 실행
    # label_encoder 객체가 train_predictor_model 에서 사용되지 않으므로 전달 불필요
    train_losses, val_losses, val_f1s = train_predictor_model(
        predictor_model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device,
        'isotype_predictor_on_antiberty_best_weighted.pth', # 저장할 모델 파일명 변경
        label_encoder # 학습 함수에서는 현재 사용하지 않음
    )

    # 최종 모델 로드 및 평가
    best_model_path = os.path.join(model_save_dir, 'isotype_predictor_on_antiberty_best_weighted.pth')
    if os.path.exists(best_model_path):
        try:
            # 모델 로드 전 num_classes 확인 (위에서 정의됨)
            if 'num_classes' not in locals() or 'label_encoder' not in locals():
                 print("오류: num_classes 또는 label_encoder가 정의되지 않아 모델 로드/평가 불가.")
            else:
                 # 로드할 모델 구조와 현재 predictor_model 구조가 동일해야 함
                 predictor_model.load_state_dict(torch.load(best_model_path, map_location=device))
                 print(f"\nBest predictor model loaded from {best_model_path} for final evaluation.")
                 evaluate_predictor_model(predictor_model, val_loader, device, label_encoder)
        except Exception as e:
            print(f"최적 모델 로드/평가 중 오류: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n경고: 저장된 최적 예측 모델 파일을 찾을 수 없어 최종 평가를 건너뜁니다.")

    print("\n--- AntiBERTy 기반 Isotype 예측 모델 학습 및 평가 완료 ---")