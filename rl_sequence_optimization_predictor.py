# -*- coding: utf-8 -*-
# rl_sequence_optimization_predictor.py (Uses Predictor Model as Reward, Targets IGHG)

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import random

# --- RL, AntiBERTy, Scikit-learn, PyTorch 라이브러리 ---
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from antiberty import AntiBERTyRunner # <<< AntiBERTy 임베딩 추출용
from sklearn.preprocessing import LabelEncoder # <<< 라벨 인코더 로드/재생성용
from tqdm.auto import tqdm # tqdm 추가
import torch.nn.functional as F # Softmax 사용 위해 추가

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 예측 모델 클래스 정의 (train_predictor_on_antiberty.py 에서 복사) ---
# 저장된 모델 로드를 위해 필요
print("--- 예측 모델 클래스 정의 ---")
class IsotypePredictorMLP(nn.Module):
    # 주의: 이 클래스 정의는 저장된 모델 ('isotype_predictor_on_antiberty_best_weighted.pth')을
    #      학습시킬 때 사용했던 것과 정확히 일치해야 합니다! (입력/은닉/출력 차원, 드롭아웃 등)
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
    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models')
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"RL 모델 저장 경로: {model_save_dir}")

    # --- 불러올 학습된 예측 모델 경로 ---
    predictor_model_path = os.path.join(model_save_dir, 'isotype_predictor_on_antiberty_best_weighted.pth')
    if not os.path.exists(predictor_model_path):
        raise FileNotFoundError("오류: 학습된 Isotype 예측 모델 파일(.pth)을 찾을 수 없습니다.")

except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- 환경 및 학습 관련 하이퍼파라미터 ---
VH_SEQ_COL = 'vh_sequence' # 최적화 대상 서열
VL_SEQ_COL = 'vl_sequence' # 파트너 서열
TARGET_ISOTYPE = 'IGHG'    # <<< 최적화 목표 Isotype
MAX_SEQ_LEN = 150
VOCAB = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB.index('-')
# VALID_AA_INDICES는 예측 모델 기반 보상에서는 직접 사용 안 함 (선택적 페널티 가능)

# 예측 MLP 모델 파라미터 (저장된 모델과 일치)
PREDICTOR_INPUT_DIM = 512 * 2
PREDICTOR_HIDDEN_DIM_1 = 512 # << isotype_predictor 학습 시 사용한 값
PREDICTOR_HIDDEN_DIM_2 = 256 # << isotype_predictor 학습 시 사용한 값
PREDICTOR_DROPOUT_RATE = 0.4 # << isotype_predictor 학습 시 사용한 값
# num_classes는 라벨 인코더로부터 얻음

# RL 관련 설정
MAX_STEPS_PER_EPISODE = 50 # 에피소드 길이 (돌연변이 횟수)
NUM_ENVS = 4               # 병렬 환경 수
TOTAL_TIMESTEPS = 200000   # << 총 학습 타임스텝 (충분히 늘려야 할 수 있음)
PPO_LEARNING_RATE = 1e-4   # << RL 학습률 (조절 가능)
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 64

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 사전 및 인코딩/디코딩 함수 정의
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}

def encode_sequence(seq):
    if not isinstance(seq, str): return []
    return [char_to_int.get(aa, char_to_int['X']) for aa in seq]

def decode_sequence(indices):
     # indices가 numpy 배열일 수 있으므로 .item() 사용 고려 또는 리스트로 변환
     return "".join([int_to_char.get(int(idx), '?') for idx in indices])

# --- 1. 필요 모델 및 데이터 로드 ---
print("\n--- 1. 필요 모델 및 데이터 로드 ---")
try:
    # AntiBERTy 로더 생성
    antiberty_runner = AntiBERTyRunner()
    print("AntiBERTyRunner 생성 완료.")

    # 라벨 인코더 준비
    temp_meta = pd.read_parquet(os.path.join(split_dir, 'metadata_train.parquet'))
    temp_valid_indices = temp_meta[['isotype_heavy', VH_SEQ_COL, VL_SEQ_COL]].notna().all(axis=1)
    temp_filtered = temp_meta[temp_valid_indices]
    label_encoder = LabelEncoder()
    label_encoder.fit(temp_filtered['isotype_heavy'])
    num_classes = len(label_encoder.classes_)
    TARGET_ISOTYPE_INDEX = label_encoder.transform([TARGET_ISOTYPE])[0] # <<< 목표 Isotype 인덱스
    print(f"라벨 인코더 준비 완료. Target='{TARGET_ISOTYPE}' (Index={TARGET_ISOTYPE_INDEX})")

    # 예측 모델 로드
    predictor_model = IsotypePredictorMLP(
        PREDICTOR_INPUT_DIM, PREDICTOR_HIDDEN_DIM_1, PREDICTOR_HIDDEN_DIM_2, num_classes, PREDICTOR_DROPOUT_RATE
    ).to(device)
    predictor_model.load_state_dict(torch.load(predictor_model_path, map_location=device))
    predictor_model.eval() # 평가 모드
    print(f"Isotype 예측 모델 로드 완료: {predictor_model_path}")

    # 초기 서열 풀 로딩 (필터링된 인덱스 기준)
    metadata_train = temp_filtered.reset_index(drop=True) # 필터링 후 인덱스 리셋
    initial_vh_sequences = metadata_train[VH_SEQ_COL].tolist()
    initial_vl_sequences = metadata_train[VL_SEQ_COL].tolist() # VL 서열도 로드
    print(f"초기 VH/VL 서열 풀 로드 완료: {len(initial_vh_sequences)} 쌍")
    if len(initial_vh_sequences) == 0: raise ValueError("유효한 초기 서열 데이터가 없습니다.")

except Exception as e:
    print(f"모델 또는 데이터 로딩 중 오류: {e}"); import traceback; traceback.print_exc(); exit()


# --- 2. 강화학습 환경 정의 (Reward 계산 방식 변경) ---
print("\n--- 2. 강화학습 환경 정의 ---")
class AntibodySequenceEnvPredictor(gym.Env): # 클래스 이름 변경
    metadata = {'render_modes': []}

    # __init__ 수정: 필요한 모델, 인코더, 인덱스, device 등을 받도록 함
    def __init__(self, initial_vh_pool, initial_vl_pool, max_len, vocab_size, pad_idx,
                 antiberty_runner, predictor_model, target_isotype_index, device, max_steps=50):
        super().__init__()

        assert len(initial_vh_pool) == len(initial_vl_pool), "VH/VL 풀 길이가 다릅니다."
        self.initial_vh_pool = initial_vh_pool
        self.initial_vl_pool = initial_vl_pool
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.max_steps = max_steps

        # --- 모델 및 설정 저장 ---
        self.antiberty_runner = antiberty_runner
        self.predictor_model = predictor_model
        self.target_isotype_index = target_isotype_index
        self.device = device

        # Observation Space (VH 서열만 관찰)
        self.observation_space = spaces.MultiDiscrete([vocab_size] * max_len)
        # Action Space (VH 서열 변경: 위치, 새 아미노산 인덱스)
        self.action_space = spaces.MultiDiscrete([max_len, vocab_size])

        self.current_step = 0
        self.current_vh_sequence_encoded = None # 현재 VH 서열 (numpy array)
        self.current_vl_sequence_str = None   # 현재 에피소드의 파트너 VL 서열 (문자열)
        self.current_vl_embedding = None    # 원본 VL 임베딩 저장 (매번 계산 방지)

    def _get_initial_pair(self):
        idx = random.randrange(len(self.initial_vh_pool))
        vh_str = self.initial_vh_pool[idx]
        vl_str = self.initial_vl_pool[idx] # <<< 파트너 VL 서열 가져오기
        encoded_vh = encode_sequence(vh_str)
        padded_vh = encoded_vh[:self.max_len]
        padded_vh.extend([self.pad_idx] * (self.max_len - len(padded_vh)))
        # VL 임베딩도 미리 계산하여 저장 (효율성 증대)
        with torch.no_grad():
             # AntiBERTy 임베딩 계산 시 리스트로 전달 필요
             vl_emb_list = self.antiberty_runner.embed([vl_str])
             vl_cls_emb = vl_emb_list[0][0].cpu() # CLS 토큰 임베딩, CPU에 저장
        return np.array(padded_vh, dtype=np.int64), vl_str, vl_cls_emb # <<< VL 임베딩도 반환

    def _get_obs(self):
        return self.current_vh_sequence_encoded

    def _get_info(self):
        vh_str = decode_sequence(list(self.current_vh_sequence_encoded))
        return {"vh_sequence": vh_str, "vl_sequence": self.current_vl_sequence_str}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset 시 VL 서열과 임베딩도 새로 설정
        self.current_vh_sequence_encoded, self.current_vl_sequence_str, self.current_vl_embedding = self._get_initial_pair()
        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        # 초기 상태의 예측 확률도 info에 포함 (선택 사항)
        try: # 초기 보상 계산 추가
            info['predicted_prob_IGHG'] = self._calculate_reward(self.current_vh_sequence_encoded)
        except Exception:
             info['predicted_prob_IGHG'] = -1.0 # 오류 시
        return observation, info

    def _calculate_reward(self, vh_encoded_sequence):
        """주어진 VH 서열과 저장된 VL 임베딩으로 보상을 계산"""
        reward = 0.0 # 기본 보상
        try:
            new_vh_seq_str = decode_sequence(list(vh_encoded_sequence))

            # AntiBERTy 임베딩 추출 (VH만)
            vh_emb_list = self.antiberty_runner.embed([new_vh_seq_str])
            # <<< vh_cls를 self.device로 이동 >>>
            vh_cls = vh_emb_list[0][0].unsqueeze(0).to(self.device) # (1, 512) on target device

            # 저장된 VL 임베딩 사용
            # <<< vl_cls도 self.device로 이동 >>>
            vl_cls = self.current_vl_embedding.unsqueeze(0).to(self.device) # (1, 512) on target device

            # 두 텐서가 모두 self.device에 있으므로 결합 가능
            combined_emb = torch.cat((vh_cls, vl_cls), dim=1) # (1, 1024) on target device

            # 예측 모델 실행 (입력과 모델이 같은 device에 있음)
            self.predictor_model.eval()
            with torch.no_grad():
                outputs = self.predictor_model(combined_emb)
                probs = torch.softmax(outputs, dim=1)
                reward = probs[0, self.target_isotype_index].item() # 목표 Isotype 확률

        except Exception as e:
            print(f"Reward calculation error: {e}")
            reward = -1.0 # 오류 시 낮은 보상 또는 0점 유지
        return reward

    def step(self, action):
        # 1. Action 적용 (VH 서열 돌연변이)
        position, new_aa_index = action
        position = int(position)
        new_aa_index = int(new_aa_index)

        new_vh_sequence_encoded = self.current_vh_sequence_encoded.copy()
        if 0 <= position < self.max_len:
             # 패딩 영역 변경 시도 시 처리
             if self.current_vh_sequence_encoded[position] == self.pad_idx and new_aa_index != self.pad_idx:
                 reward = -0.1 # 패딩을 바꾸려 하면 작은 페널티 (상태 변경 없음)
             elif self.current_vh_sequence_encoded[position] != self.pad_idx and new_aa_index == self.pad_idx:
                 reward = -0.1 # 아미노산을 패딩으로 바꾸려 하면 작은 페널티 (상태 변경 없음)
             else:
                 # 유효한 변경 시퀀스 업데이트
                 new_vh_sequence_encoded[position] = new_aa_index
                 # 2. 보상 계산 (예측 모델 사용)
                 reward = self._calculate_reward(new_vh_sequence_encoded)
        else: # 유효하지 않은 위치 선택 시
            reward = -0.5 # 위치 오류 시 페널티

        # 상태 업데이트 (유효한 변경시에만 업데이트하도록 수정 가능하나, 일단은 항상 업데이트)
        self.current_vh_sequence_encoded = new_vh_sequence_encoded
        self.current_step += 1

        # 3. 종료 조건 확인
        terminated = self.current_step >= self.max_steps
        truncated = False

        # 4. 정보 반환
        observation = self._get_obs()
        info = self._get_info()
        info['predicted_prob_IGHG'] = reward if reward >= 0 else 0 # 음수 페널티는 제외하고 확률 기록

        return observation, reward, terminated, truncated, info

    def render(self): pass
    def close(self): pass

print("RL 환경 (예측 모델 보상) 정의 완료.")


# --- 3. 병렬 환경 생성 및 에이전트 학습 ---
if __name__ == '__main__':
    print(f"\n--- 3. 병렬 환경 생성 및 학습 시작 (예측 모델 보상, Target: {TARGET_ISOTYPE}) ---")
    print(f"사용할 병렬 환경 수: {NUM_ENVS}")

    # 환경 생성 함수 정의 (필요 인자 전달 확인)
    env_kwargs = {
        'initial_vh_pool': initial_vh_sequences,
        'initial_vl_pool': initial_vl_sequences,
        'max_len': MAX_SEQ_LEN,
        'vocab_size': VOCAB_SIZE,
        'pad_idx': PAD_IDX,
        'max_steps': MAX_STEPS_PER_EPISODE,
        'antiberty_runner': antiberty_runner,
        'predictor_model': predictor_model,
        'target_isotype_index': TARGET_ISOTYPE_INDEX,
        'device': device
    }

    try:
        # 중요: SubprocVecEnv 사용 시 env_kwargs에 포함된 모델/러너가 각 프로세스로
        #       pickle되어 전달됩니다. 모델 크기가 크거나 GPU 객체일 경우 문제가
        #       발생할 수 있습니다. 오류 발생 시 DummyVecEnv로 변경하거나,
        #       각 프로세스에서 모델을 로드하도록 환경 __init__ 수정 필요.
        vec_env = make_vec_env(
            lambda: AntibodySequenceEnvPredictor(**env_kwargs),
            n_envs=NUM_ENVS,
            vec_env_cls=SubprocVecEnv # <<< 문제 발생 시 DummyVecEnv로 변경 시도
        )
        print("병렬 환경 생성 완료.")

        # PPO 에이전트 정의
        rl_model = PPO("MlpPolicy", vec_env, verbose=1, device='auto',
                       n_steps=PPO_N_STEPS, batch_size=PPO_BATCH_SIZE, n_epochs=10, learning_rate=PPO_LEARNING_RATE)
        print("PPO 에이전트 정의 완료.")

        # 에이전트 학습
        print(f"\n--- 강화학습 에이전트 학습 시작 (총 {TOTAL_TIMESTEPS} 타임스텝) ---")
        start_time_rl = time.time()
        rl_model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        end_time_rl = time.time()
        print(f"--- 강화학습 에이전트 학습 완료 (소요 시간: {end_time_rl - start_time_rl:.2f} 초) ---")

        # 학습된 모델 저장 (파일명 변경)
        model_save_path = os.path.join(model_save_dir, "antibody_ppo_seq_predictor_agent")
        rl_model.save(model_save_path)
        print(f"학습된 RL 에이전트 저장 완료: {model_save_path}")

        # 평가 루프
        print("\n--- 학습된 에이전트 평가 (샘플 실행) ---")
        obs = vec_env.reset()
        num_episodes_done = 0
        max_episodes_to_show = NUM_ENVS * 2

        while num_episodes_done < max_episodes_to_show:
             action, _states = rl_model.predict(obs, deterministic=True)
             # step 반환 값 순서 확인: obs, rewards, dones, infos
             obs, rewards, dones, infos = vec_env.step(action)
             for i in range(len(dones)):
                 if dones[i]:
                     num_episodes_done += 1
                     print(f"--- Env {i} terminated (Episode {num_episodes_done}) ---")
                     # print(f"Full info dict for env {i}: {infos[i]}") # 필요시 주석 해제
                     final_info = infos[i].get('final_info', infos[i])
                     final_vh_seq = final_info.get('vh_sequence', 'N/A')
                     final_vl_seq = final_info.get('vl_sequence', 'N/A')
                     predicted_prob = final_info.get('predicted_prob_IGHG', 'N/A')
                     print(f"Final VH Sequence (env {i}): {final_vh_seq}")
                     print(f"Partner VL Sequence (env {i}): {final_vl_seq}")
                     print(f"Predicted IGHG Prob (End): {predicted_prob}")
                     print("-" * 20)
                     if num_episodes_done >= max_episodes_to_show: break
             if num_episodes_done >= max_episodes_to_show: break

        vec_env.close()

    except Exception as e:
        print(f"\n강화학습 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        if 'vec_env' in locals() and vec_env is not None:
            try: vec_env.close(); print("오류 발생 후 병렬 환경 종료 시도.")
            except Exception as close_e: print(f"병렬 환경 종료 중 추가 오류: {close_e}")
        exit()

    print("\n--- 서열 기반 RL (예측 모델 보상) 스크립트 완료 ---")