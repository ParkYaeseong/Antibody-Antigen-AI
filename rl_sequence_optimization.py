# -*- coding: utf-8 -*-
# rl_sequence_optimization.py (Sequence-based RL with Heuristic Reward)

import os
import time
import pandas as pd
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import random

# --- RL 라이브러리 ---
# stable-baselines3[extra], gymnasium 설치 필요
# pip install stable-baselines3[extra] gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

print("--- 필요한 라이브러리 임포트 완료 ---")

# --- 0. 경로 및 설정 ---
print("--- 0. 경로 및 설정 ---")
try:
    base_dir = "G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber"
    split_dir = os.path.join(base_dir, '3_split_data')
    model_save_dir = os.path.join(base_dir, '4_trained_models') # RL 모델 저장용
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"기본 디렉토리: {base_dir}")
    print(f"분할 데이터 로드 경로: {split_dir}")
    print(f"RL 모델 저장 경로: {model_save_dir}")
except Exception as e:
    print(f"경로 설정 중 오류: {e}"); exit()

# --- 환경 및 학습 관련 하이퍼파라미터 ---
SEQ_COLUMN = 'vh_sequence' # VH 서열 사용 (또는 'vl_sequence')
MAX_SEQ_LEN = 150  # VAE 학습 시 사용했던 최대 길이
VOCAB = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'] # VAE와 동일하게 유지
VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB.index('-')
VALID_AA_INDICES = {i for i, aa in enumerate(VOCAB) if aa != '-' and aa != 'X'} # 유효한 AA 인덱스 집합

# RL 관련 설정
MAX_STEPS_PER_EPISODE = 50 # 한 에피소드당 최대 변형 횟수
NUM_ENVS = 4 # 병렬 환경 수 (CPU 코어 수 고려)
TOTAL_TIMESTEPS = 100000 # 총 학습 타임스텝 (조절 필요)

# GPU 설정 (SB3는 자동으로 감지 시도)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}") # SB3가 내부적으로 처리

# 사전 및 인코딩/디코딩 함수 정의
char_to_int = {char: i for i, char in enumerate(VOCAB)}
int_to_char = {i: char for i, char in enumerate(VOCAB)}

def encode_sequence(seq):
    if not isinstance(seq, str): return []
    return [char_to_int.get(aa, char_to_int['X']) for aa in seq]

def decode_sequence(indices):
     return "".join([int_to_char.get(idx, '?') for idx in indices])

# --- 1. 데이터 로딩 (초기 서열 풀) ---
print("\n--- 1. 데이터 로딩 (초기 서열 풀) ---")
try:
    meta_train_path = os.path.join(split_dir, 'metadata_train.parquet')
    metadata_train = pd.read_parquet(meta_train_path)
    print(f"훈련 메타데이터 로드 완료: {meta_train_path} ({len(metadata_train)} 항목)")

    # 서열 데이터 추출 및 결측치 처리
    initial_sequences = metadata_train[SEQ_COLUMN].dropna().tolist()
    print(f"초기 서열 풀로 사용할 {SEQ_COLUMN} 서열 수: {len(initial_sequences)}")
    if len(initial_sequences) == 0: raise ValueError("유효한 초기 서열 데이터가 없습니다.")

except Exception as e:
    print(f"데이터 로딩 중 오류: {e}"); exit()


# --- 2. 강화학습 환경 정의 (Gymnasium Env 상속) ---
print("\n--- 2. 강화학습 환경 정의 ---")
class AntibodySequenceEnv(gym.Env):
    """
    항체 서열 최적화를 위한 커스텀 Gymnasium 환경 (서열 직접 사용).
    State: 현재 항체 서열 (정수 인코딩된 리스트).
    Action: (위치, 새로운 아미노산 인덱스).
    Reward: 휴리스틱 보상 (예: 유효한 아미노산 변경 시 +1, 아니면 -10).
    """
    metadata = {'render_modes': []} # 렌더링은 사용 안 함

    def __init__(self, initial_seq_pool, max_len, vocab_size, pad_idx, valid_aa_indices, max_steps=50):
        super().__init__()

        self.initial_seq_pool = initial_seq_pool
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.valid_aa_indices = valid_aa_indices # 유효한 아미노산 인덱스 집합
        self.max_steps = max_steps

        # Observation Space: 각 위치는 0 ~ vocab_size-1 범위의 정수
        # 예: [2, 5, 10, ..., 0, 0] (길이 max_len)
        self.observation_space = spaces.MultiDiscrete([vocab_size] * max_len)

        # Action Space: 변경할 위치(0 ~ max_len-1), 새로운 아미노산 인덱스(0 ~ vocab_size-1)
        self.action_space = spaces.MultiDiscrete([max_len, vocab_size])

        self.current_step = 0
        self.current_sequence_encoded = None # 현재 서열 (정수 리스트)

    def _get_initial_sequence(self):
        # 풀에서 무작위로 초기 서열 선택 및 인코딩/패딩
        seq_str = random.choice(self.initial_seq_pool)
        encoded = encode_sequence(seq_str)
        # 패딩은 고정 길이 관찰 공간을 위해 필요
        padded = encoded[:self.max_len]
        padded.extend([self.pad_idx] * (self.max_len - len(padded)))
        return np.array(padded, dtype=np.int64) # 상태는 numpy 배열 형태

    def _get_obs(self):
        # 현재 인코딩된 서열 반환
        return self.current_sequence_encoded

    def _get_info(self):
        # 추가 정보 (예: 현재 서열 문자열)
        return {"sequence": decode_sequence(self.current_sequence_encoded)}

    def reset(self, seed=None, options=None):
        # 환경 초기화
        super().reset(seed=seed)
        self.current_sequence_encoded = self._get_initial_sequence()
        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # 1. Action 해석 및 적용 (돌연변이)
        position, new_aa_index = action
        position = int(position) # MultiDiscrete는 int64로 줄 수 있음
        new_aa_index = int(new_aa_index)

        # 위치 유효성 검사 (패딩 영역 변경 방지 - 선택사항)
        # if position >= len(self.current_sequence_encoded) or self.current_sequence_encoded[position] == self.pad_idx:
        #     # 패딩 영역을 변경하려는 경우 페널티 부여 또는 아무것도 안 함
        #     pass # 또는 reward = -1 등

        # 새로운 서열 생성 (복사본 수정)
        new_sequence_encoded = self.current_sequence_encoded.copy()
        if 0 <= position < self.max_len: # 위치 범위 내
            new_sequence_encoded[position] = new_aa_index

        # 2. 휴리스틱 보상 계산
        # 예시: 유효한 아미노산으로 변경되었는지 확인
        # (PAD_IDX(0) 또는 'X'(21) 가 아닌 아미노산으로 변경했는지)
        # (더 복잡한 보상 설계 가능: 길이, 특정 아미노산 비율, 예측 모델 점수 등)
        reward = 0
        if new_aa_index in self.valid_aa_indices:
            reward = 1  # 유효한 아미노산으로 변경 시 +1
        else:
            reward = -10 # 유효하지 않은 문자(PAD, X)로 변경 시 페널티 -10

        # 현재 상태 업데이트
        self.current_sequence_encoded = new_sequence_encoded
        self.current_step += 1

        # 3. 종료 조건 확인
        terminated = self.current_step >= self.max_steps
        truncated = False # 여기서는 사용 안 함

        # 4. 정보 반환
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self): pass
    def close(self): pass

print("RL 환경 정의 완료.")


# --- 3. 병렬 환경 생성 및 에이전트 학습 ---
if __name__ == '__main__': # 멀티프로세싱 사용 시 필요

    print(f"\n--- 3. 병렬 환경 생성 및 학습 시작 ---")
    print(f"사용할 병렬 환경 수: {NUM_ENVS}")

    # 환경 생성 함수 정의
    env_kwargs = {
        'initial_seq_pool': initial_sequences,
        'max_len': MAX_SEQ_LEN,
        'vocab_size': VOCAB_SIZE,
        'pad_idx': PAD_IDX,
        'valid_aa_indices': VALID_AA_INDICES,
        'max_steps': MAX_STEPS_PER_EPISODE
    }

    try:
        # 병렬 환경 생성 (SubprocVecEnv 또는 DummyVecEnv)
        vec_env = make_vec_env(
            lambda: AntibodySequenceEnv(**env_kwargs),
            n_envs=NUM_ENVS,
            vec_env_cls=SubprocVecEnv # 멀티프로세스
            # vec_env_cls=DummyVecEnv # 디버깅용 단일 프로세스
        )
        print("병렬 환경 생성 완료.")

        # PPO 에이전트 정의 (MultiInputPolicy는 관찰 공간이 Dict일 때 사용, 여기서는 MlpPolicy)
        # MultiDiscrete 액션 공간에는 MlpPolicy 사용 가능
        rl_model = PPO("MlpPolicy", vec_env, verbose=1, device='auto', # 'auto'는 GPU 자동 감지
                       n_steps=2048, batch_size=64, n_epochs=10, learning_rate=3e-4)
        print("PPO 에이전트 정의 완료.")

        # 에이전트 학습
        print(f"\n--- 강화학습 에이전트 학습 시작 (총 {TOTAL_TIMESTEPS} 타임스텝) ---")
        start_time_rl = time.time()
        rl_model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
        end_time_rl = time.time()
        print(f"--- 강화학습 에이전트 학습 완료 (소요 시간: {end_time_rl - start_time_rl:.2f} 초) ---")

        # 학습된 모델 저장
        model_save_path = os.path.join(model_save_dir, "antibody_ppo_seq_heuristic_agent")
        rl_model.save(model_save_path)
        print(f"학습된 RL 에이전트 저장 완료: {model_save_path}")

        # (선택사항) 학습된 에이전트로 몇 에피소드 실행해보기
        print("\n--- 학습된 에이전트 평가 (샘플 실행) ---")
        obs = vec_env.reset() # <<< 이 부분은 이전에 수정 완료됨
        num_episodes_done = 0
        max_episodes_to_show = NUM_ENVS * 2 # 예시: 각 환경당 2개 에피소드 정도만 출력

        while num_episodes_done < max_episodes_to_show:
            action, _states = rl_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            # 각 환경(env)의 정보 출력
            for i in range(len(dones)):
                if dones[i]: # 해당 환경의 에피소드가 종료되었는지 확인
                    num_episodes_done += 1
                    print(f"--- Env {i} terminated (Episode {num_episodes_done}) ---")
                    # infos[i]의 전체 내용 출력하여 구조 확인
                    print(f"Full info dict for env {i}: {infos[i]}")

                    # 최종 시퀀스 추출 시도 (SB3 VecEnv 래핑 고려)
                    # 'final_info' 안에 원래 환경의 info가 들어있는 경우가 많음
                    final_info = infos[i].get('final_info')
                    if final_info is not None:
                        final_seq = final_info.get('sequence', 'N/A (Key not found)')
                    else:
                        # 'final_info'가 없으면 infos[i] 자체에서 찾아보기 (대체)
                        final_seq = infos[i].get('sequence', 'N/A (No final_info or key)')

                    print(f"Final Sequence (env {i}): {final_seq}")
                    print("-" * 20)
                    if num_episodes_done >= max_episodes_to_show: break
            if num_episodes_done >= max_episodes_to_show: break

        # 병렬 환경 종료
        vec_env.close()

    except Exception as e:
        print(f"\n강화학습 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        if 'vec_env' in locals() and vec_env is not None:
            try: vec_env.close(); print("오류 발생 후 병렬 환경 종료 시도.")
            except Exception as close_e: print(f"병렬 환경 종료 중 추가 오류: {close_e}")
        exit()

    print("\n--- 서열 기반 RL 스크립트 완료 ---")