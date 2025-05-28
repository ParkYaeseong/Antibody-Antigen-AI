# -*- coding: utf-8 -*-
# finetune_alm_isotype.py

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# --- Hugging Face Transformers 및 Scikit-learn 라이브러리 ---
# pip install transformers torch scikit-learn pandas pyarrow accelerate
# accelerate는 GPU 사용 시 학습 속도 향상에 도움 (선택 사항)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models') # 파인튜닝된 모델 저장
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")
except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- 모델 및 학습 관련 하이퍼파라미터 ---
#MODEL_NAME = "Exscientia/AntiBERTy" # 사용할 ALM 모델
TARGET_COLUMN = 'isotype_heavy'     # 예측 목표
VH_SEQ_COL = 'vh_sequence'
VL_SEQ_COL = 'vl_sequence'

# 토크나이저 및 모델 최대 입력 길이 확인 (AntiBERTy는 보통 512 또는 그 이상 지원)
# 실제 모델 설정을 따르는 것이 좋으나, 여기서는 일반적인 값으로 설정
MAX_LENGTH = 256 # VH + [SEP] + VL 길이 고려하여 설정 (모델 최대 길이 이하로)

# 학습 파라미터
LEARNING_RATE = 2e-5 # Transformer 파인튜닝 시 일반적으로 사용되는 작은 학습률
BATCH_SIZE = 16     # GPU 메모리에 맞게 조절 (ALM은 메모리 사용량 큼)
NUM_EPOCHS = 5      # 파인튜닝은 보통 적은 에포크로도 수렴 (데이터셋 크기에 따라 조절)
VAL_SPLIT_RATIO = 0.1 # 훈련 데이터 내 검증 비율

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 데이터 로딩 및 전처리 ---
print("\n--- 1. 데이터 로딩 및 전처리 ---")
try:
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 메타데이터 로드 완료: {meta_train_path} ({len(metadata_train)} 항목)")

    # 필요한 컬럼 및 결측치 처리
    required_cols = [TARGET_COLUMN, VH_SEQ_COL, VL_SEQ_COL]
    valid_indices = metadata_train[required_cols].notna().all(axis=1)
    metadata_filtered = metadata_train[valid_indices].copy()
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

    # VH, VL 서열 리스트 준비
    vh_sequences = metadata_filtered[VH_SEQ_COL].tolist()
    vl_sequences = metadata_filtered[VL_SEQ_COL].tolist()

    # 클래스 가중치 계산
    print("클래스 가중치 계산 중...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_encoded),
        y=y_encoded
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"계산된 클래스 가중치: {class_weights}")

except Exception as e:
    print(f"데이터 준비 중 오류: {e}"); exit()

# --- 2. 토크나이저 로드 및 커스텀 데이터셋 정의 ---
print("\n--- 2. 토크나이저 로드 및 커스텀 데이터셋 정의 ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"'{MODEL_NAME}' 토크나이저 로드 완료.")

    # 특수 토큰 확인 및 추가 (필요시)
    # AntiBERTy는 [PAD], [UNK], [CLS], [SEP], [MASK] 등을 이미 포함
    # print("Tokenizer Special Tokens:", tokenizer.special_tokens_map)

except Exception as e:
    print(f"토크나이저 로딩 중 오류: {e}"); exit()

class AntibodyPairedDataset(Dataset):
    def __init__(self, vh_seqs, vl_seqs, labels, tokenizer, max_len):
        self.vh_seqs = vh_seqs
        self.vl_seqs = vl_seqs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        # [SEP] 토큰 확인
        self.sep_token = tokenizer.sep_token if tokenizer.sep_token else '[SEP]'


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vh = self.vh_seqs[idx]
        vl = self.vl_seqs[idx]
        label = self.labels[idx]

        # VH와 VL을 [SEP] 토큰으로 연결
        # 주의: 모델 입력 형식에 따라 [CLS] 토큰 추가 여부 결정 필요 (일반적으론 추가)
        # sequence = f"[CLS] {vh} [SEP] {vl} [SEP]" # AntiBERTy는 보통 CLS/SEP 사용
        # AntiBERTy가 아미노산 단위 토크나이징을 가정하고, 특수 토큰 처리 방식 확인 필요
        # 간단하게 공백으로 구분 시도
        sequence = " ".join(list(vh)) + f" {self.sep_token} " + " ".join(list(vl))


        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True, # [CLS], [SEP] 자동 추가 (모델에 따라 다를 수 있음)
            max_length=self.max_len,
            padding='max_length',    # 최대 길이에 맞춰 패딩
            truncation=True,         # 최대 길이 초과 시 잘라냄
            return_attention_mask=True,
            return_tensors='pt',     # PyTorch 텐서로 반환
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 데이터셋 생성 및 분할
full_dataset = AntibodyPairedDataset(vh_sequences, vl_sequences, y_encoded, tokenizer, MAX_LENGTH)
val_size = int(len(full_dataset) * VAL_SPLIT_RATIO)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"커스텀 데이터셋 생성 및 분할 완료: Train={train_size}, Validation={val_size}")

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DataLoader 생성 완료.")

# --- 3. 사전 훈련된 모델 로드 (Sequence Classification용) ---
print("\n--- 3. 사전 훈련된 모델 로드 ---")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes # 예측할 클래스 개수 전달
    ).to(device)
    print(f"'{MODEL_NAME}' 모델 로드 완료 (분류 헤드 추가됨).")
except Exception as e:
    print(f"모델 로딩 중 오류: {e}"); exit()

# --- 4. 손실 함수, 옵티마이저, 스케줄러 정의 ---
print("\n--- 4. 손실 함수, 옵티마이저, 스케줄러 정의 ---")
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # 클래스 가중치 적용
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0, # 예열 스텝 수 (선택사항)
    num_training_steps=total_steps
)
print("손실 함수, 옵티마이저(AdamW), 스케줄러(Linear) 정의 완료.")

# --- 5. 모델 학습 함수 ---
print("\n--- 5. 모델 학습 함수 정의 ---")
def train_alm_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses, val_losses, val_f1s = [], [], []
    best_val_f1 = 0.0
    model_save_path = os.path.join(model_save_dir, 'alm_isotype_predictor_best.pth') # 모델 파일명 변경

    print(f"\n--- ALM 파인튜닝 시작 (Epochs: {num_epochs}) ---")
    start_time_train = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss # 모델 출력에서 손실 직접 가져옴
            loss.backward()
            optimizer.step()
            scheduler.step() # 스케줄러 업데이트
            running_loss += loss.item()

        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # 검증 단계
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) # Macro F1 사용
        val_losses.append(epoch_val_loss)
        val_f1s.append(epoch_val_f1)
        epoch_end_time = time.time()

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Macro F1: {epoch_val_f1:.4f}, "
              f"Time: {epoch_end_time - epoch_start_time:.2f} sec")

        # 최고 성능 모델 저장 (Val Macro F1 기준)
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            # 모델 전체 저장 (AutoModel 로딩을 위해) 또는 state_dict 저장
            # model.save_pretrained(model_save_dir) # Hugging Face 방식 저장
            torch.save(model.state_dict(), model_save_path) # state_dict 저장
            print(f"  => Best model saved to {model_save_path} (Val Macro F1: {best_val_f1:.4f})")

    end_time_train = time.time()
    print(f"--- ALM 파인튜닝 완료 (총 소요 시간: {end_time_train - start_time_train:.2f} 초) ---")
    return train_losses, val_losses, val_f1s

print("ALM 학습 함수 정의 완료.")


# --- 6. 모델 평가 함수 ---
print("\n--- 6. 모델 평가 함수 정의 ---")
def evaluate_alm_model(model, data_loader, device, label_encoder):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

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

print("ALM 평가 함수 정의 완료.")


# --- 7. 메인 실행 블록 ---
if __name__ == '__main__':
    # 모델 학습 실행
    train_losses, val_losses, val_f1s = train_alm_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device
    )

    # 최종 모델 로드 및 평가
    best_model_path = os.path.join(model_save_dir, 'alm_isotype_predictor_best.pth')
    if os.path.exists(best_model_path):
        try:
            # 모델 로드 시 num_labels 재확인
            model_reloaded = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=num_classes).to(device) # 구조 먼저 정의
            model_reloaded.load_state_dict(torch.load(best_model_path)) # state_dict 로드
            print(f"\nBest ALM model loaded from {best_model_path} for final evaluation.")
            evaluate_alm_model(model_reloaded, val_loader, device, label_encoder)
        except Exception as e:
            print(f"최적 모델 로드/평가 중 오류: {e}")
    else:
        print("\n경고: 저장된 최적 ALM 모델 파일을 찾을 수 없어 최종 평가를 건너뜁니다.")

    # (선택사항) 학습 곡선 시각화 코드 추가 가능

    print("\n--- ALM 기반 Isotype 예측 모델 학습 및 평가 완료 ---")