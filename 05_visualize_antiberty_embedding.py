# -*- coding: utf-8 -*-
# visualize_antiberty_embedding.py

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# --- 필요한 라이브러리 ---
# pip install antiberty umap-learn matplotlib pandas pyarrow torch scikit-learn
from antiberty import AntiBERTyRunner
from sklearn.preprocessing import LabelEncoder
import umap # UMAP 설치 필요: pip install umap-learn
import matplotlib.pyplot as plt
import matplotlib.cm as cm # 컬러맵 사용
from tqdm.auto import tqdm

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data')
    # 모델 저장 경로는 여기서는 필요 없음
    visualization_dir = os.path.join(base_dir, '5_visualizations')
    os.makedirs(visualization_dir, exist_ok=True)

    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"시각화 저장 경로: {visualization_dir}")

except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- 설정 ---
TARGET_COLUMN = 'isotype_heavy'     # 색상 구분 기준
VH_SEQ_COL = 'vh_sequence'
VL_SEQ_COL = 'vl_sequence'
BATCH_SIZE = 32     # 임베딩 추출 시 배치 크기 (메모리에 맞게 조절)

# UMAP 파라미터 (조절 가능)
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = 'cosine' # 또는 'euclidean'

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. AntiBERTy 모델 로드 ---
print("\n--- 1. AntiBERTy 모델 로드 ---")
try:
    antiberty_runner = AntiBERTyRunner()
    # 필요시 모델을 GPU로 이동 (runner 내부 구현 확인 필요)
    # if device.type == 'cuda': antiberty_runner.model.to(device)
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
    # 시각화를 위해 훈련 데이터 전체 사용 (또는 일부 샘플링)
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

    # 실제 라벨 저장 (시각화용)
    y_labels = metadata_filtered[TARGET_COLUMN].tolist()
    unique_labels = sorted(list(set(y_labels))) # 고유 라벨 (색상 매핑용)
    print(f"Isotype 클래스 종류: {unique_labels}")

    # 서열 리스트 준비
    vh_sequences = metadata_filtered[VH_SEQ_COL].tolist()
    vl_sequences = metadata_filtered[VL_SEQ_COL].tolist()

    # AntiBERTy 임베딩 추출 함수 (이전 스크립트와 동일)
    def get_antiberty_embeddings(sequences, runner, batch_size):
        all_cls_embeddings = []
        print(f"총 {len(sequences)}개 서열 임베딩 추출 시작 (배치 크기: {batch_size})...")
        for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding sequences"):
            batch_seqs = sequences[i:i+batch_size]
            embeddings_list = runner.embed(batch_seqs)
            # GPU 사용 시 결과를 CPU로 가져와서 저장 (UMAP은 보통 CPU에서 실행)
            cls_embeddings_batch = torch.stack([emb[0].cpu() for emb in embeddings_list])
            all_cls_embeddings.append(cls_embeddings_batch)
        return torch.cat(all_cls_embeddings, dim=0)

    # VH, VL 임베딩 생성
    vh_embeddings = get_antiberty_embeddings(vh_sequences, antiberty_runner, BATCH_SIZE)
    vl_embeddings = get_antiberty_embeddings(vl_sequences, antiberty_runner, BATCH_SIZE)
    print("VH/VL CLS 임베딩 추출 완료.")

    # VH, VL 임베딩 결합
    combined_embeddings = torch.cat((vh_embeddings, vl_embeddings), dim=1).numpy() # Numpy 배열로
    print(f"결합된 AntiBERTy 임베딩 Shape: {combined_embeddings.shape}") # (N, 1024)

except ValueError as e:
    print(e); exit()
except Exception as e:
    print(f"데이터 준비/임베딩 생성 중 오류: {e}"); import traceback; traceback.print_exc(); exit()


# --- 3. UMAP 차원 축소 및 시각화 ---
print("\n--- 3. UMAP 차원 축소 및 시각화 ---")
try:
    print("UMAP 차원 축소 중... (시간이 소요될 수 있습니다)")
    start_time_umap = time.time()
    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST,
                        n_components=2,
                        metric=UMAP_METRIC,
                        random_state=42)
    embedding_2d = reducer.fit_transform(combined_embeddings)
    print(f"UMAP 차원 축소 완료. 소요 시간: {time.time() - start_time_umap:.2f} 초")
    print(f"2D 임베딩 Shape: {embedding_2d.shape}")

    # 시각화
    print("시각화 진행 중...")
    plt.figure(figsize=(12, 10))
    colors = cm.rainbow(np.linspace(0, 1, len(unique_labels))) # 클래스별 색상

    for i, label in enumerate(unique_labels):
        indices = [idx for idx, l in enumerate(y_labels) if l == label]
        if not indices: continue # 해당 라벨 데이터 없으면 건너뛰기
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1],
                    color=colors[i], label=label, s=10, alpha=0.7) # alpha 추가

    plt.title('AntiBERTy Embedding Space Visualization (Colored by Isotype)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Isotype', markerscale=2, loc='best') # 범례 위치 자동 조정
    plt.grid(True, linestyle='--', alpha=0.6)

    # 시각화 결과 저장 및 보기
    save_path = os.path.join(visualization_dir, 'antiberty_embedding_space_isotype.png')
    plt.savefig(save_path, dpi=300)
    print(f"시각화 결과 저장 완료: {save_path}")
    # plt.show() # 인터랙티브 환경에서는 주석 해제

except ImportError:
     print("오류: 시각화에 필요한 라이브러리(umap-learn, matplotlib)가 설치되지 않았습니다.")
     print("pip install umap-learn matplotlib")
except Exception as e:
    print(f"UMAP 시각화 중 오류 발생: {e}")

print("\n--- 스크립트 완료 ---")