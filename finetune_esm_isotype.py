# -*- coding: utf-8 -*-
# finetune_esm_isotype.py (Fine-tuning ALM for Isotype Prediction)

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split 

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
    model_save_dir = os.path.join(base_dir, '4_trained_models', 'esm_isotype_predictor') # 모델 저장 경로 변경
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"모델 저장 경로: {model_save_dir}")
except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- 모델 및 학습 관련 하이퍼파라미터 ---
MODEL_NAME = "facebook/esm2_t6_8M_UR50D" # <<< 사용할 ALM 모델 (ESM-2 Small)
TARGET_COLUMN = 'isotype_heavy'
VH_SEQ_COL = 'vh_sequence'
VL_SEQ_COL = 'vl_sequence'
MAX_LENGTH = 350 # VH + VL + 특수 토큰 고려한 최대 길이 (모델 최대 길이 확인 필요, ESM2는 보통 1024)

# 학습 파라미터
LEARNING_RATE = 2e-5 # Transformer 파인튜닝용 작은 학습률
BATCH_SIZE = 8      # <<< ALM은 메모리를 많이 사용하므로 작게 시작 (GPU 메모리에 맞게 조절)
NUM_EPOCHS = 10     # <<< 파인튜닝은 적은 에포크로도 효과 볼 수 있음 (Early Stopping 사용)
VAL_SPLIT_RATIO = 0.1 # 훈련 데이터 내 검증 비율
EARLY_STOPPING_PATIENCE = 3 # <<< 검증 성능 개선 없을 시 기다릴 에포크 수

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
    metadata_filtered = metadata_train[valid_indices].copy().reset_index(drop=True)
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
    print(f"클래스 종류: {list(label_encoder.classes_)}") # 리스트로 변환하여 출력

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
    print(f"데이터 준비 중 오류: {e}"); import traceback; traceback.print_exc(); exit()


# --- 2. 토크나이저 로드 및 커스텀 데이터셋 정의 ---
print("\n--- 2. 토크나이저 로드 및 커스텀 데이터셋 정의 ---")
try:
    # AutoTokenizer 사용
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"'{MODEL_NAME}' 토크나이저 로드 완료.")

except Exception as e:
    print(f"토크나이저 로딩 중 오류: {e}"); exit()

class AntibodyPairedDataset(Dataset):
    def __init__(self, vh_seqs, vl_seqs, labels, tokenizer, max_len):
        self.vh_seqs = vh_seqs
        self.vl_seqs = vl_seqs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        vh = self.vh_seqs[idx]
        vl = self.vl_seqs[idx]
        label = self.labels[idx]

        # 토크나이저에 두 시퀀스를 직접 전달 (pair=True 와 유사한 효과)
        # 모델이 자동으로 [CLS] seq_vh [SEP] seq_vl [SEP] (또는 모델별 형식)으로 처리
        encoding = self.tokenizer.encode_plus(
            vh,                 # 첫 번째 시퀀스
            vl,                 # 두 번째 시퀀스 (text_pair)
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,    # Truncate sequences if they exceed max_length
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 데이터셋 생성 및 분할
# 훈련/검증 분할 시, 클래스 분포 유지 위해 stratify 사용 고려 (선택 사항)
try:
    train_idx, val_idx = train_test_split(
        range(len(y_encoded)),
        test_size=VAL_SPLIT_RATIO,
        random_state=42,
        stratify=y_encoded # <<< 클래스 비율 유지하며 분할
    )

    train_dataset = AntibodyPairedDataset(
        [vh_sequences[i] for i in train_idx],
        [vl_sequences[i] for i in train_idx],
        y_encoded[train_idx],
        tokenizer, MAX_LENGTH
    )
    val_dataset = AntibodyPairedDataset(
        [vh_sequences[i] for i in val_idx],
        [vl_sequences[i] for i in val_idx],
        y_encoded[val_idx],
        tokenizer, MAX_LENGTH
    )
    print(f"커스텀 데이터셋 생성 및 분할 완료: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("DataLoader 생성 완료.")

except Exception as e:
     print(f"데이터셋/로더 생성 중 오류: {e}"); exit()


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
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
print("손실 함수, 옵티마이저(AdamW), 스케줄러(Linear) 정의 완료.")

# --- 5. 모델 학습 함수 (Early Stopping 추가) ---
print("\n--- 5. 모델 학습 함수 정의 ---")
def train_alm_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_save_dir, patience):
    train_losses, val_losses, val_f1s = [], [], []
    best_val_f1 = 0.0 # 최고 Macro F1 추적
    epochs_no_improve = 0
    # 모델 저장 경로를 디렉토리로 지정 (save_pretrained 사용)
    best_model_save_path = os.path.join(model_save_dir, 'alm_isotype_predictor_best')

    print(f"\n--- ALM 파인튜닝 시작 (Epochs: {num_epochs}, Early Stopping Patience: {patience}) ---")
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
            loss = outputs.loss
            loss.backward()
            # Gradient Clipping (선택 사항, 안정적인 학습에 도움)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
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
            # Hugging Face 모델 저장 방식 사용
            model.save_pretrained(best_model_save_path)
            tokenizer.save_pretrained(best_model_save_path) # 토크나이저도 함께 저장
            print(f"  => Best model saved to {best_model_save_path} (Val Macro F1: {best_val_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  (No improvement for {epochs_no_improve} epochs)")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    end_time_train = time.time()
    print(f"--- ALM 파인튜닝 완료 (총 소요 시간: {end_time_train - start_time_train:.2f} 초) ---")
    return train_losses, val_losses, val_f1s

print("ALM 학습 함수 정의 완료.")


# --- 6. 모델 평가 함수 --- (이전과 유사하나 모델 입력 방식 주의)
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
            # labels를 모델에 넘기지 않음 (evaluation 시)
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
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device,
        model_save_dir, # 저장 경로 (디렉토리) 전달
        EARLY_STOPPING_PATIENCE
    )

    # 최종 모델 로드 및 평가
    best_model_load_path = os.path.join(model_save_dir, 'alm_isotype_predictor_best') # 저장된 디렉토리 경로
    if os.path.exists(best_model_load_path):
        try:
            if 'num_classes' not in locals() or 'label_encoder' not in locals():
                 print("오류: num_classes 또는 label_encoder가 정의되지 않아 모델 로드/평가 불가.")
            else:
                 # AutoModelForSequenceClassification으로 로드
                 model_reloaded = AutoModelForSequenceClassification.from_pretrained(best_model_load_path).to(device)
                 print(f"\nBest ALM model loaded from {best_model_load_path} for final evaluation.")
                 evaluate_alm_model(model_reloaded, val_loader, device, label_encoder)
        except Exception as e:
            print(f"최적 모델 로드/평가 중 오류: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n경고: 저장된 최적 ALM 모델 파일을 찾을 수 없어 최종 평가를 건너뜁니다.")

    print("\n--- ALM 기반 Isotype 예측 모델 파인튜닝 및 평가 완료 ---")