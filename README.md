# Antibody-Antigen-AI

# 항체 서열 분석 및 Isotype 예측 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 대규모 항체 서열 데이터셋을 구축하고, 이를 활용하여 다양한 딥러닝 기반 항체 서열 표현(Representation) 방식을 분석합니다. Variational Autoencoder (VAE)를 직접 학습시키고, 사전 훈련된 항체 언어 모델(AntiBERTy, IgBert, ESM-2 등)의 임베딩을 추출하여 이들의 특성을 탐구합니다.

주요 목표는 이러한 가변 영역 서열 기반 표현들이 항체의 불변 영역 특성인 중쇄(Heavy Chain) Isotype을 얼마나 효과적으로 예측할 수 있는지 평가하는 것입니다. Isotype 예측 성능 분석, 잠재 공간 시각화, VAE 기반 신규 서열 생성, 그리고 강화학습을 이용한 서열 최적화 가능성 탐색을 통해 현재 서열 표현 방식의 유용성과 한계를 규명하고자 합니다.

## 2. 주요 기능 및 모듈

* **데이터 구축 및 전처리**: SAbDab, ABSD, OAS, AACDB 등 공개 데이터베이스로부터 인간 유래 VH/VL 쌍 서열 데이터를 통합 및 전처리합니다 (스크립트에서는 전처리된 데이터 사용을 가정).
* **서열 표현 학습**:
    * VH 및 VL 서열에 대한 Variational Autoencoder (VAE) 모델 학습.
    * AntiBERTy, IgBert, ESM-2 등 사전 훈련된 단백질/항체 언어 모델을 활용한 서열 임베딩 추출.
* **Isotype 예측**:
    * 추출된 서열 표현(VAE 잠재 벡터, ALM 임베딩)을 입력으로 사용하여 MLP(Multi-Layer Perceptron) 기반 Isotype 예측 모델 학습.
    * ESM-2와 같은 항체 언어 모델(ALM)을 직접 Isotype 예측 작업에 맞게 파인튜닝.
    * 클래스 불균형 문제 해결을 위한 클래스 가중치(Class Weighting) 및 SMOTE 오버샘플링 기법 적용.
* **잠재 공간 시각화**: VAE로 학습된 잠재 공간과 AntiBERTy 임베딩 공간을 UMAP을 이용하여 2차원으로 시각화하고, Isotype 클래스별 분포 및 분리도 분석.
* **신규 항체 서열 생성**: 학습된 VAE 모델의 잠재 공간에서 샘플링하여 새로운 VH 및 VL 서열 생성.
* **강화학습(RL) 기반 서열 최적화**:
    * 유효한 아미노산 사용과 같은 휴리스틱 보상 함수를 사용한 VH 서열 최적화.
    * 학습된 Isotype 예측 모델의 특정 Isotype(예: IGHG) 예측 확률을 보상 함수로 사용하여 목표 Isotype을 가진 VH 서열 생성 시도.

## 3. 사용된 데이터셋

* **데이터 소스 (언급)**: Structural Antibody Database (SAbDab), Antibody Sequence Database (ABSD), Observed Antibody Space (OAS), Antigen-Antibody Complex Database (AACDB).
* **구축된 데이터셋 (스크립트에서 사용)**: 공개 DB에서 통합 및 전처리된 약 1만 쌍의 인간(Homo sapiens) 유래 VH/VL 서열 데이터. AbNumber로 CDR 영역 주석 처리, CD-HIT로 중복 제거 및 분할된 데이터셋을 사용.
    * 훈련 세트는 약 6,510개의 VH/VL 쌍 및 Isotype 라벨을 포함하며, Isotype 분포는 IGHM이 매우 우세한 불균형 데이터입니다.

## 4. 프로젝트 구조
Antibody_AI_Project/
├── scripts/
│   ├── 02_특성공학 & 데이터 분할.py
│   ├── 03_데이터 확인.py
│   ├── 04_train_vh_vae.py
│   ├── 04_train_vl_vae.py
│   ├── 05_visualize_antiberty_embedding.py
│   ├── 05_visualize_latent_space.py
│   ├── 06_generate_sequences.py
│   ├── train_isotype_predictor.py
│   ├── train_predictor_on_antiberty.py
│   ├── train_predictor_on_igbert.py
│   ├── train_predictor_on_latent.py
│   ├── finetune_alm_isotype.py
│   ├── finetune_esm_isotype.py
│   ├── rl_sequence_optimization.py
│   └── rl_sequence_optimization_predictor.py
├── data_storage/  # 스크립트 내 base_dir ('G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber')에 해당
│   ├── 1_preprocessed/
│   │   └── final_dataset/
│   │       └── antibody_metadata_abnumber.parquet
│   ├── 2_feature_engineered/
│   │   ├── vh_embeddings_IgBert.npy # 예시, 모델에 따라 파일명 변경
│   │   └── vl_embeddings_IgBert.npy # 예시
│   ├── 3_split_data/
│   │   ├── metadata_train.parquet
│   │   ├── metadata_val.parquet
│   │   ├── metadata_test.parquet
│   │   ├── X_train_vh.npy, X_train_vl.npy
│   │   ├── X_val_vh.npy, X_val_vl.npy
│   │   └── X_test_vh.npy, X_test_vl.npy
│   ├── 4_trained_models/
│   │   ├── vh_vae_best.pth
│   │   ├── vl_vae_best.pth
│   │   ├── isotype_predictor_on_antiberty_best_weighted.pth
│   │   ├── isotype_predictor_on_igbert_best_weighted.pth
│   │   ├── isotype_predictor_on_latent_best_weighted.pth
│   │   ├── alm_isotype_predictor_best/ # AntiBERTy 파인튜닝 모델
│   │   ├── esm_isotype_predictor/    # ESM 파인튜닝 모델
│   │   ├── antibody_ppo_seq_heuristic_agent.zip
│   │   └── antibody_ppo_seq_predictor_agent.zip
│   └── 5_visualizations/
│       ├── antiberty_embedding_space_isotype.png
│       └── vae_latent_space_isotype.png
├── temp_data/ # 스크립트 내 temp_dir ('C:/antibody_temp')에 해당
│   └── (CD-HIT 임시 파일 등)
├── requirement.txt
└── README.md

**참고**:
* `data_storage/` 와 `temp_data/` 디렉토리명은 예시이며, 실제로는 각 스크립트에 명시된 `base_dir` 및 `temp_dir` 경로에 해당 데이터가 저장됩니다.
* 일부 전처리 단계(예: 공개 DB에서 초기 데이터 다운로드, AbNumber 실행)는 본 레포지토리의 스크립트에 포함되어 있지 않을 수 있으며, `antibody_metadata_abnumber.parquet` 파일이 이미 전처리된 상태로 존재한다고 가정합니다.

## 5. 설치 및 환경 설정

1.  **Python 환경**: Python 3.8 이상을 권장합니다. 가상 환경(예: conda, venv) 사용을 추천합니다.
2.  **필수 라이브러리 설치**:
    ```bash
    pip install -r requirement.txt
    ```
    `requirement.txt` 파일은 주요 라이브러리 및 버전을 포함합니다. 특정 라이브러리(예: PyTorch)는 사용자의 CUDA 버전에 맞춰 별도 설치가 필요할 수 있습니다 (스크립트 내 주석 참고).

    **주요 라이브러리**:
    * NumPy
    * Pandas
    * PyTorch (GPU 버전 권장)
    * Scikit-learn
    * Transformers, Accelerate (Hugging Face 모델용)
    * Biopython (서열 데이터 처리)
    * CD-HIT (외부 프로그램, 경로 설정 필요, WSL에서 사용 가능)
    * Antiberty (`pip install antiberty`)
    * UMAP-learn, Matplotlib (시각화)
    * Stable-Baselines3, Gymnasium (강화학습)
    * Imbalanced-learn (SMOTE 등)
    * PyArrow (Parquet 파일 처리)

3.  **경로 설정**: 각 Python 스크립트 상단에 있는 `base_dir` (데이터 및 모델 저장 위치) 및 `temp_dir` (CD-HIT 임시 폴더) 변수를 사용자 환경에 맞게 수정해야 합니다.
4.  **CD-HIT 설치**: `02_특성공학 & 데이터 분할.py` 스크립트는 CD-HIT 프로그램을 사용합니다. 시스템에 CD-HIT를 설치하고, 스크립트 내에서 실행 가능하도록 경로를 설정하거나 (Windows의 경우) WSL(Windows Subsystem for Linux) 환경에 CD-HIT를 설치하여 사용해야 합니다.

## 6. 스크립트 실행 가이드

실행 순서는 일반적으로 파일명 앞 숫자 순서를 따르지만, 일부는 독립적으로 실행 가능하거나 특정 파일럿 연구 목적일 수 있습니다.

1.  **`02_특성공학 & 데이터 분할.py`**:
    * `antibody_metadata_abnumber.parquet` 파일로부터 VH/VL 서열을 로드합니다.
    * 지정된 단백질 언어 모델(예: Exscientia/IgBert, facebook/esm2 시리즈)을 사용하여 VH 및 VL 서열에 대한 임베딩을 생성하고 `.npy` 파일로 저장합니다.
    * VH CDR3 서열을 기준으로 CD-HIT 클러스터링을 수행합니다.
    * 클러스터링 결과를 바탕으로 GroupShuffleSplit을 사용하여 데이터(임베딩 및 메타데이터)를 훈련/검증/테스트 세트로 분할하고 Parquet 및 NumPy 파일로 저장합니다.
    * **실행 전**: `base_dir`, `temp_dir`, `model_name` (임베딩 생성용) 변수 확인 및 CD-HIT 설정 필요.

2.  **`03_데이터 확인.py`**:
    * 분할된 `metadata_train.parquet` (또는 다른 Parquet) 파일을 로드하여 기본적인 정보(처음 5줄, 컬럼 목록, 데이터 요약)를 출력합니다. 데이터 확인용 간단한 유틸리티입니다.

3.  **`04_train_vh_vae.py` / `04_train_vl_vae.py`**:
    * `metadata_train.parquet`에서 VH 또는 VL 서열을 로드합니다.
    * LSTM 기반의 Variational Autoencoder (VAE) 모델을 정의하고 학습시킵니다.
    * 학습된 최적의 VAE 모델(state_dict)을 `.pth` 파일로 저장합니다.
    * 학습 곡선 시각화 및 간단한 재구성 예시를 출력합니다.
    * **실행 전**: `SEQ_COLUMN` 변수를 'vh_sequence' 또는 'vl_sequence'로 각각 설정.

4.  **`05_visualize_antiberty_embedding.py` / `05_visualize_latent_space.py`**:
    * `05_visualize_antiberty_embedding.py`: `metadata_train.parquet`에서 VH/VL 서열을 로드하고, AntiBERTy 모델을 사용하여 CLS 토큰 임베딩을 추출합니다. 추출된 VH-VL 결합 임베딩을 UMAP으로 2차원 축소 후 Isotype별로 시각화하여 `.png` 파일로 저장합니다.
    * `05_visualize_latent_space.py`: 학습된 VH 및 VL VAE 모델(`.pth` 파일)과 `metadata_train.parquet`를 로드합니다. VAE 인코더를 사용하여 VH/VL 서열로부터 잠재 벡터(mu)를 추출하고 결합합니다. 이 결합된 잠재 벡터를 UMAP으로 2차원 축소 후 Isotype별로 시각화하여 `.png` 파일로 저장합니다.

5.  **`06_generate_sequences.py`**:
    * 학습된 VH 및 VL VAE 모델(`.pth` 파일)을 로드합니다.
    * 잠재 공간(표준 정규 분포)에서 벡터를 샘플링한 후, VAE 디코더를 사용하여 새로운 VH 및 VL 아미노산 서열을 생성하고 출력합니다.

6.  **Isotype 예측 모델 학습 스크립트**:
    * **`train_isotype_predictor.py`**: pLM 임베딩(예: `02_특성공학 & 데이터 분할.py`의 결과)을 사용하여 MLP 모델로 Isotype을 예측하며 SMOTE 오버샘플링을 적용합니다.
    * **`train_predictor_on_antiberty.py`**: VH/VL 서열에 대해 실시간으로 AntiBERTy 임베딩을 추출하여 MLP 모델의 입력으로 사용하고, 클래스 가중치를 적용하여 Isotype을 예측합니다.
    * **`train_predictor_on_igbert.py`**: 미리 저장된 IgBert VH/VL 임베딩(`.npy` 파일)을 로드하여 MLP 모델 입력으로 사용하며, 클래스 가중치 및 Early Stopping을 적용합니다.
    * **`train_predictor_on_latent.py`**: 학습된 VH/VL VAE 모델로 잠재 벡터를 추출, 결합하여 MLP 모델 입력으로 사용하며, 클래스 가중치를 적용합니다.
    * **`finetune_alm_isotype.py`**: `MODEL_NAME` (기본: "Exscientia/AntiBERTy")으로 지정된 항체 언어 모델을 Isotype 예측 작업에 맞게 파인튜닝하고 클래스 가중치를 적용합니다.
    * **`finetune_esm_isotype.py`**: "facebook/esm2_t6_8M_UR50D" 모델을 Isotype 예측 작업에 맞게 파인튜닝하며, 클래스 가중치 및 Early Stopping을 적용합니다.
    * **공통 사항**: 각 스크립트는 학습된 최적의 모델을 저장하고, 검증 세트에 대한 평가 결과를 출력합니다.

7.  **강화학습 기반 서열 최적화 스크립트**:
    * **`rl_sequence_optimization.py`**: VH 서열을 초기 풀로 사용, Gymnasium 환경에서 VH 서열을 상태로, (위치, 새 아미노산)을 행동으로 설정합니다. 휴리스틱 보상(예: 유효 아미노산 변경)을 사용하여 PPO 에이전트를 학습시킵니다.
    * **`rl_sequence_optimization_predictor.py`**: VH/VL 서열 쌍을 초기 풀로 사용하고, 학습된 Isotype 예측 MLP 모델을 로드합니다. 변경된 VH 서열과 원본 VL 서열의 AntiBERTy 임베딩 기반 예측 모델의 목표 Isotype('IGHG') 예측 확률을 보상으로 사용하여 PPO 에이전트를 학습시킵니다.

## 7. 주요 분석 내용 및 관찰 결과 (스크립트 기반)

* **VAE 재구성 능력**: VAE 모델 학습 스크립트(`04_train_vh_vae.py`, `04_train_vl_vae.py`)는 재구성 손실 및 KLD 손실을 모니터링하며, 학습 후 재구성 예시를 통해 모델이 원본 서열을 얼마나 잘 복원하는지 확인할 수 있습니다.
* **Isotype 예측 성능**: 다수의 예측 모델 학습 스크립트(`train_*.py`, `finetune_*.py`)들은 다양한 임베딩(VAE 잠재 공간, AntiBERTy, IgBert, ESM-2)과 모델 아키텍처(MLP, ALM 파인튜닝)를 사용하여 Isotype 예측을 수행합니다. 각 스크립트는 Macro F1 점수 등의 평가지표를 통해 성능을 보고하며, 클래스 불균형 처리 기법의 효과를 관찰할 수 있습니다.
* **표현 공간 특성**: 시각화 스크립트(`05_visualize_*.py`)는 UMAP을 통해 VAE 잠재 공간 및 AntiBERTy 임베딩 공간에서 Isotype 클래스들이 어떻게 분포하고 군집하는지 보여줍니다. 이를 통해 각 표현 방식이 Isotype 정보를 얼마나 잘 분리해내는지 직관적으로 파악할 수 있습니다.
* **강화학습 기반 최적화**: RL 스크립트(`rl_*.py`)는 휴리스틱 보상 또는 예측 모델 기반 보상을 사용하여 특정 목표(예: 유효한 서열 생성, 특정 Isotype 확률 증가)를 향해 항체 서열을 최적화하는 과정을 보여줍니다. 학습 곡선(평균 보상)을 통해 에이전트의 학습 진행 상황을 확인할 수 있습니다.

## 8. 향후 연구 및 개선 방향 (일반적인 제안)

* 더 다양한 최신 단백질/항체 언어 모델 및 생성 모델(예: Diffusion Model) 탐색.
* 불변 영역 서열 정보, 항체의 3D 구조 정보(예: AlphaFold 예측 구조) 등 멀티모달 정보 통합.
* Isotype 외 다른 항체 기능(예: 항원 결합력, 안정성, 면역원성) 예측 및 최적화로 확장.
* 대규모의 고품질 기능 라벨 데이터를 확보하여 지도 학습 모델의 성능 향상.
* 해석 가능한 AI(Explainable AI) 기법을 적용하여 모델의 예측 근거 이해.

## 9. 참고 기술

* Python, PyTorch, Hugging Face Transformers, Scikit-learn
* Variational Autoencoders (VAE)
* Protein Language Models (pLMs) / Antibody Language Models (ALMs) - AntiBERTy, IgBert, ESM-2
* Multi-Layer Perceptrons (MLP)
* UMAP (Uniform Manifold Approximation and Projection)
* Reinforcement Learning (RL) - PPO (Proximal Policy Optimization)
* CD-HIT (Sequence Clustering)
* Public Antibody Databases (SAbDab, ABSD, OAS, AACDB)
