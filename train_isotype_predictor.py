# -*- coding: utf-8 -*-
# train_isotype_predictor.py

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.utils.class_weight import compute_class_weight

# --- 데이터 전처리 및 모델 평가를 위한 라이브러리 ---
from sklearn.model_selection import train_test_split # 내부 검증용 (선택사항)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
# 이전 단계에서 생성된 데이터 경로 (본인 환경에 맞게 확인/수정)
try:
    # Google Drive 경로가 G: 드라이브에 마운트되었다고 가정
    # base_dir = 'G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber'
    # Colab 경로 예시 (실제 환경에 맞게 수정)
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"기본 디렉토리를 찾을 수 없습니다: {base_dir}")

    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models') # 모델 저장 경로
    os.makedirs(model_save_dir, exist_ok=True) # 저장 경로 없으면 생성

    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")

except FileNotFoundError as e:
    print(e)
    exit()
except Exception as e:
    print(f"경로 설정 중 오류 발생: {e}")
    exit()

# --- 학습 관련 하이퍼파라미터 ---
INPUT_DIM = 1024 * 2 # VH(1024) + VL(1024)
# --- 출력 차원 (Isotype 클래스 개수)은 데이터 로드 후 결정 ---
HIDDEN_DIM_1 = 2048 # 첫 번째 은닉층 크기
HIDDEN_DIM_2 = 1024  # 두 번째 은닉층 크기
HIDDEN_DIM_3 = 512
LEARNING_RATE = 1e-5
BATCH_SIZE = 64
NUM_EPOCHS = 200 # 필요에 따라 조절
VAL_SPLIT_RATIO = 0.2 # 훈련 데이터 내에서 검증 세트로 사용할 비율

# --- 1. 데이터 로딩 및 전처리 ---
print("\n--- 1. 데이터 로딩 및 전처리 ---")
try:
    # 분할된 훈련 데이터 로드
    vh_train_path = os.path.join(split_dir, 'X_train_vh.npy')
    vl_train_path = os.path.join(split_dir, 'X_train_vl.npy')
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')

    X_train_vh = np.load(vh_train_path)
    X_train_vl = np.load(vl_train_path)
    metadata_train = pd.read_parquet(meta_train_path)

    print(f"훈련 데이터 로드 완료:")
    print(f"  VH 임베딩 shape: {X_train_vh.shape}")
    print(f"  VL 임베딩 shape: {X_train_vl.shape}")
    print(f"  메타데이터 shape: {metadata_train.shape}")

    # VH, VL 임베딩 결합
    X_train_combined = np.concatenate((X_train_vh, X_train_vl), axis=1)
    print(f"VH, VL 임베딩 결합 완료. Shape: {X_train_combined.shape}")

    # 타겟 변수(isotype_heavy) 추출 및 결측치 처리
    target_col = 'isotype_heavy'
    if target_col not in metadata_train.columns:
        raise ValueError(f"오류: 메타데이터에 타겟 컬럼 '{target_col}'이 없습니다.")

    # 결측치가 있는 행 제거 (타겟 변수 기준)
    original_count = len(metadata_train)
    valid_indices = metadata_train[target_col].notna()
    metadata_train_filtered = metadata_train[valid_indices].copy()
    X_train_combined_filtered = X_train_combined[valid_indices]
    num_removed = original_count - len(metadata_train_filtered)
    print(f"'{target_col}' 결측치 포함 항목 {num_removed}개 제거됨.")
    print(f"최종 사용할 훈련 데이터 수: {len(metadata_train_filtered)}")

    if len(metadata_train_filtered) == 0:
        raise ValueError("오류: 타겟 변수에 유효한 데이터가 없습니다.")

    y_train_labels = metadata_train_filtered[target_col].values

    # 라벨 인코딩 (문자열 -> 숫자)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_labels)
    num_classes = len(label_encoder.classes_)
    print(f"타겟 변수 라벨 인코딩 완료. 클래스 개수: {num_classes}")
    print(f"클래스 종류: {label_encoder.classes_}")

    # --- SMOTE 오버샘플링 적용 ---
    print("\nSMOTE 오버샘플링 적용 중...")
    # k_neighbors 값은 가장 적은 클래스의 샘플 수보다 작아야 함 (최소 1)
    # 현재 데이터에서 'Bulk' 클래스 샘플이 훈련/검증 합쳐서 매우 적을 수 있으므로 확인 필요
    # 전체 훈련 데이터(6510개)에서 각 클래스 개수 확인
    unique, counts = np.unique(y_train_encoded, return_counts=True)
    min_samples = counts.min()
    print(f"SMOTE 적용 전 클래스 분포: {dict(zip(label_encoder.classes_[unique], counts))}")

    # k_neighbors는 최소 클래스 샘플 수보다 작아야 하므로, min_samples-1 또는 적절한 값(최소 1) 설정
    smote_k_neighbors = max(1, min_samples - 1) if min_samples > 1 else 1
    # 만약 min_samples가 1이면 SMOTE 적용 불가 -> RandomOverSampler 고려 또는 해당 클래스 제외
    if min_samples <= smote_k_neighbors :
        print(f"경고: 최소 클래스 샘플 수({min_samples})가 너무 작아 k_neighbors({smote_k_neighbors}) 설정이 어렵습니다. RandomOverSampler로 대체하거나 해당 클래스 제외를 고려하세요.")
        # 여기서는 일단 k_neighbors=1로 진행 (데이터 확인 후 조정 필요)
        smote_k_neighbors = 1

    # SMOTE 객체 생성 시 k_neighbors 설정
    smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined_filtered, y_train_encoded)
    print("SMOTE 오버샘플링 완료.")
    resampled_unique, resampled_counts = np.unique(y_train_resampled, return_counts=True)
    print(f"SMOTE 적용 후 클래스 분포: {dict(zip(label_encoder.classes_[resampled_unique], resampled_counts))}")
    print(f"리샘플링된 훈련 데이터 Shape: X={X_train_resampled.shape}, y={y_train_resampled.shape}")
    # --- SMOTE 적용 끝 ---

    # 데이터를 PyTorch Tensor로 변환 (리샘플링된 데이터 사용)
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32) # <<< 리샘플링된 X 사용
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)  # <<< 리샘플링된 y 사용

    # TensorDataset 생성 (리샘플링된 데이터 사용)
    full_dataset = TensorDataset(X_train_tensor, y_train_tensor) # <<< 리샘플링된 데이터로 생성

    # 훈련/검증 데이터 분할 (DataLoader용) - 리샘플링된 데이터 기준
    val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"리샘플링된 데이터 내부 훈련/검증 분할 완료: Train={train_size}, Validation={val_size}")

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("DataLoader 생성 완료.")

except FileNotFoundError as e:
    print(f"오류: 데이터 파일 로드 실패. 경로 확인 필요: {e}")
    exit()
except ValueError as e:
    print(e)
    exit()
except Exception as e:
    print(f"데이터 준비 중 예상치 못한 오류: {e}")
    exit()


# --- 2. 모델 정의 (MLP) ---
print("\n--- 2. 모델 정의 (MLP) ---")
class IsotypePredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(IsotypePredictorMLP, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3) # 드롭아웃 추가 (과적합 방지)
        self.layer_2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.output_layer = nn.Linear(hidden_dim2, num_classes)
        # CrossEntropyLoss 사용 시 Softmax는 필요 없음 (내부적으로 포함)

    def forward(self, x):
        x = self.relu1(self.layer_1(x))
        x = self.dropout1(x)
        x = self.relu2(self.layer_2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x

# 모델 인스턴스 생성 및 GPU 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IsotypePredictorMLP(INPUT_DIM, HIDDEN_DIM_1, HIDDEN_DIM_2, num_classes).to(device)
print(f"MLP 모델 정의 완료. Device: {device}")
print(model)

# --- 3. 손실 함수 및 옵티마이저 정의 ---
print("\n--- 3. 손실 함수 및 옵티마이저 정의 ---")
criterion = nn.CrossEntropyLoss() # <<< 가중치 제거!
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"손실 함수: CrossEntropyLoss, 옵티마이저: Adam (lr={LEARNING_RATE})") # 로그 메시지도 수정

# --- 4. 모델 학습 함수 ---
print("\n--- 4. 모델 학습 함수 정의 ---")
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_accuracy = 0.0
    model_save_path = os.path.join(model_save_dir, 'isotype_predictor_best.pth')

    print(f"\n--- 모델 학습 시작 (Epochs: {num_epochs}) ---")
    start_time_train = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train() # 학습 모드
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 그래디언트 초기화
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 검증 단계
        model.eval() # 평가 모드
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_accuracy = 100 * correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        epoch_end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Accuracy: {epoch_val_accuracy:.2f}%, "
              f"Time: {epoch_end_time - epoch_start_time:.2f} sec")

        # 최고 성능 모델 저장
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"  => Best model saved to {model_save_path} (Val Acc: {best_val_accuracy:.2f}%)")

    end_time_train = time.time()
    print(f"--- 모델 학습 완료 (총 소요 시간: {end_time_train - start_time_train:.2f} 초) ---")
    return train_losses, val_losses, val_accuracies

print("모델 학습 함수 정의 완료.")

# --- 5. 모델 평가 함수 ---
print("\n--- 5. 모델 평가 함수 정의 ---")
def evaluate_model(model, data_loader, device, label_encoder_classes):
    model.eval() # 평가 모드
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
    f1 = f1_score(y_true, y_pred, average='weighted') # 클래스 불균형 고려
    print("\n--- 최종 검증 세트 평가 결과 ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print("\nClassification Report:")

    # 가능한 모든 클래스의 숫자 라벨 생성 (0부터 num_classes-1 까지)
    all_labels = np.arange(len(label_encoder_classes))
    print(classification_report(y_true, y_pred,
                                labels=all_labels, # <<< labels 인자 추가
                                target_names=label_encoder_classes,
                                zero_division=0))

print("모델 평가 함수 정의 완료.")


# --- 6. 메인 실행 블록 ---
if __name__ == '__main__':
    # 모델 학습 실행
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device
    )

    # 최종 모델 로드 (가장 성능 좋았던 모델)
    best_model_path = os.path.join(model_save_dir, 'isotype_predictor_best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nBest model loaded from {best_model_path} for final evaluation.")
        # 검증 데이터로 최종 평가
        evaluate_model(model, val_loader, device, label_encoder.classes_)
    else:
        print("\n경고: 저장된 최적 모델 파일을 찾을 수 없어 최종 평가를 건너<0xEB><0x9B><0x84>니다.")

    # (선택사항) 학습 곡선 등 시각화 코드 추가 가능
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.legend()
    # plt.title('Loss Curve')
    # plt.subplot(1, 2, 2)
    # plt.plot(val_accuracies, label='Validation Accuracy')
    # plt.legend()
    # plt.title('Accuracy Curve')
    # plt.show()

    print("\n--- Isotype 예측 모델 학습 및 평가 완료 ---")