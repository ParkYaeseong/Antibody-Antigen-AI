# -*- coding: utf-8 -*-


# --- 라이브러리 설치 안내 ---
# 이 노트북을 실행하기 전에, 사용 중인 Python 가상 환경(예: conda, venv)에
# conda activate antibody_env
# 다음 라이브러리들을 설치해야 합니다. 터미널에서 아래 명령어를 실행하세요.
#
# 1단계: NumPy 버전 고정 (< 2)
# pip install "numpy<2" -q
#
# 2단계: 나머지 주요 라이브러리 설치
# pip install transformers torch torchvision torchaudio pandas biopython scikit-learn pyarrow -q --upgrade
#
# Pytorch GPU 지원 설치 (NVIDIA GPU 사용 시):
# PyTorch 공식 웹사이트(https://pytorch.org/)에서 본인 환경(CUDA 버전 등)에 맞는
# 설치 명령어를 확인하여 설치하세요. (예: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)
# python -u "c:\Users\21\Desktop\항원&항체\특성공학&데이터부할.py"   

import os
import time
import re # 클러스터 파일 파싱용
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer, AutoModel
import torch
import subprocess # 외부 명령어 실행용
import platform # 운영체제 확인용

print("스크립트 실행 시작...")

# --- 기본 경로 설정 (로컬 환경) ---
# 요청하신 로컬 경로로 수정
base_dir = 'G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber' # Windows 경로


# 경로 존재 확인
if not os.path.exists(base_dir):
    print(f"오류: 기본 디렉토리를 찾을 수 없습니다: {base_dir}")
    print("스크립트를 종료합니다. 경로를 확인해주세요.")
    exit()
else:
    print(f"기본 디렉토리 확인: {base_dir}")

# 중간 결과 저장 경로 설정
feature_dir = os.path.join(base_dir, '2_feature_engineered')
split_dir = os.path.join(base_dir, '3_split_data')
temp_dir = 'C:/antibody_temp' # <<< C 드라이브의 새 경로로 지정
# temp_dir = os.path.join(base_dir, 'temp_cdhit') # CD-HIT 임시 파일용

os.makedirs(feature_dir, exist_ok=True)
os.makedirs(split_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True) # 임시 폴더 생성

print(f"특성 저장 디렉토리: {feature_dir}")
print(f"분할 데이터 저장 디렉토리: {split_dir}")
print(f"CD-HIT 임시 디렉토리: {temp_dir}")


# --- 라이브러리 임포트 및 버전 확인 ---
try:
    print("\n--- 필수 라이브러리 임포트 시작 ---")
    # 여기에 필요한 모든 import 문을 모아둘 수 있습니다.
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Torch version: {torch.__version__}")
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
    import Bio
    print(f"Biopython version: {Bio.__version__}")
    print("--- 필수 라이브러리 임포트 완료 ---")
except ImportError as e:
    print(f"오류: 필수 라이브러리 임포트 실패: {e}")
    print("스크립트 상단의 라이브러리 설치 안내를 확인하고 필요한 라이브러리를 설치해주세요.")
    exit()
except Exception as e:
    print(f"오류: 라이브러리 임포트 중 예외 발생: {e}")
    exit()


# --- GPU 상태 확인 ---
print("\n--- 시스템 환경 확인 ---")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'GPU 사용 가능: {torch.cuda.get_device_name(0)}')
    total_mem = torch.cuda.get_device_properties(0).total_memory
    # free_mem = total_mem - torch.cuda.memory_allocated(0) # 시작 시점에서는 allocated가 0일 수 있음
    print(f'GPU 총 메모리: {total_mem / (1024**3):.2f} GB')
    # print(f'GPU 현재 사용 가능 메모리 (추정): {free_mem / (1024**3):.2f} GB') # 동적으로 변함
else:
    device = torch.device("cpu")
    print('GPU 사용 불가, CPU 사용.')
    print("경고: CPU 사용 시 임베딩 생성 속도가 매우 느릴 수 있습니다.")


# --- 데이터 로드 ---
metadata_path = os.path.join(base_dir, '1_preprocessed', 'final_dataset', 'antibody_metadata_abnumber.parquet')
print(f"\n메타데이터 로딩 중: {metadata_path}")
try:
    metadata_df = pd.read_parquet(metadata_path)
    print(f"메타데이터 로드 완료: {len(metadata_df)} 항목")
    required_cols = ['entry_id', 'vh_sequence', 'vl_sequence', 'vh_cdr3']
    if not all(col in metadata_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in metadata_df.columns]
        raise ValueError(f"필수 메타데이터 컬럼 누락: {missing_cols}")
except FileNotFoundError:
    print(f"오류: 메타데이터 파일을 찾을 수 없습니다 - {metadata_path}")
    raise
except Exception as e:
    print(f"메타데이터 로드 중 오류 발생: {e}")
    raise

# 서열 리스트 준비
vh_sequences = metadata_df['vh_sequence'].tolist()
vl_sequences = metadata_df['vl_sequence'].tolist()
print(f"VH/VL 서열 리스트 준비 완료: 각 {len(vh_sequences)} 개")


# --- 모델 선택 및 설정 ---
# 사용할 모델과 배치 크기 설정
# model_name = "facebook/esm2_t33_650M_UR50D"; batch_size_auto = 16
# model_name = "facebook/esm2_t36_3B_UR50D"; batch_size_auto = 2 # VRAM 16GB 이상에서도 OOM 가능성 있음
model_name = "Exscientia/IgBert"; batch_size_auto = 16 # RTX 2060 6GB 에서 OOM 발생 시 8 또는 4로 줄여야 함

effective_batch_size = batch_size_auto
print(f"\n사용할 모델: {model_name}")
print(f"설정된 배치 크기: {effective_batch_size} (메모리 부족 시 줄여야 함)")


# --- 모델 및 토크나이저 로드 ---
# 임베딩 생성에 필요하므로 이 단계는 항상 수행
print("\n모델 및 토크나이저 로딩 중...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval() # 평가 모드 설정
    print("모델 및 토크나이저 로드 완료.")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    raise


# --- 임베딩 생성 함수 정의 ---
def get_embeddings(sequence_list, batch_size=8, model=model, tokenizer=tokenizer, device=device, max_len=1024, progress_interval=50): # progress_interval 증가
    """ pLM 임베딩 생성 함수 (평균 풀링, 패딩 제외) """
    all_embeddings = []
    num_sequences = len(sequence_list)
    num_batches = (num_sequences + batch_size - 1) // batch_size
    start_time_total = time.time()
    print(f"총 서열 수: {num_sequences}, 배치 크기: {batch_size}, 총 배치 수: {num_batches}")
    model.eval()
    for i in range(num_batches):
        start_time_batch = time.time()
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_sequences)
        batch_sequences = sequence_list[start_idx:end_idx]
        if not batch_sequences: continue
        try:
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)
        except Exception as e:
            print(f"오류: 배치 {i+1} 토크나이징 중 오류: {e}")
            return None
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                all_embeddings.extend(mean_embeddings.cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n오류: 배치 {i+1} 처리 중 메모리 부족(OOM)!")
                    print(f"      현재 배치 크기: {batch_size}. 더 작은 값으로 시도하세요.")
                    return None
                else:
                    print(f"\n오류: 배치 {i+1} 모델 실행 중 런타임 오류: {e}")
                    return None
            except Exception as e:
                 print(f"\n오류: 배치 {i+1} 처리 중 예외 발생: {e}")
                 return None
        end_time_batch = time.time()
        if (i + 1) % progress_interval == 0 or (i + 1) == num_batches:
            print(f"  배치 {i+1}/{num_batches} 처리 완료. (배치 당 소요 시간: {end_time_batch - start_time_batch:.2f} 초)")
    end_time_total = time.time()
    print(f"총 임베딩 생성 시간: {end_time_total - start_time_total:.2f} 초")
    return np.array(all_embeddings)


# --- 임베딩 생성 또는 로드 ---
print("\n--- 임베딩 데이터 준비 ---")
# 예상되는 최종 임베딩 파일 경로 정의
model_name_safe = model_name.split('/')[-1].replace('-', '_')
vh_output_filename = f'vh_embeddings_{model_name_safe}.npy'
vl_output_filename = f'vl_embeddings_{model_name_safe}.npy'
vh_output_path = os.path.join(feature_dir, vh_output_filename)
vl_output_path = os.path.join(feature_dir, vl_output_filename)

vh_embeddings = None # 변수 초기화
vl_embeddings = None

# 파일 존재 여부 확인
if os.path.exists(vh_output_path) and os.path.exists(vl_output_path):
    print(f"이미 생성된 임베딩 파일 로딩 시도: {vh_output_path}, {vl_output_path}")
    try:
        vh_embeddings = np.load(vh_output_path)
        vl_embeddings = np.load(vl_output_path)
        # 로드된 임베딩 크기 확인 (데이터 개수 일치 여부)
        if len(vh_embeddings) == len(metadata_df) and len(vl_embeddings) == len(metadata_df):
             print("저장된 임베딩 로드 완료.")
             print(f"VH 임베딩 shape: {vh_embeddings.shape}")
             print(f"VL 임베딩 shape: {vl_embeddings.shape}")
        else:
             print(f"경고: 로드된 임베딩 크기({len(vh_embeddings)})가 메타데이터({len(metadata_df)})와 불일치합니다. 임베딩을 다시 생성합니다.")
             vh_embeddings = None # 불일치 시 재생성 유도
             vl_embeddings = None
    except Exception as e:
        print(f"오류: 저장된 임베딩 파일 로드 실패 ({e}). 임베딩 생성을 다시 시도합니다.")
        vh_embeddings = None # 로드 실패 시 초기화
        vl_embeddings = None
else:
    print("저장된 임베딩 파일을 찾을 수 없습니다. 임베딩 생성을 시작합니다.")

# 임베딩 생성이 필요한 경우 (파일이 없거나 로드 실패/크기 불일치 시)
if vh_embeddings is None or vl_embeddings is None:
    print("\n--- VH 서열 임베딩 생성 시작 ---")
    vh_embeddings = get_embeddings(vh_sequences, batch_size=effective_batch_size, model=model, tokenizer=tokenizer, device=device)

    if vh_embeddings is None:
        print("오류: VH 임베딩 생성 실패. 스크립트를 종료합니다.")
        exit() # VH 생성 실패 시 중단
    else:
        print("\n--- VL 서열 임베딩 생성 시작 ---")
        vl_embeddings = get_embeddings(vl_sequences, batch_size=effective_batch_size, model=model, tokenizer=tokenizer, device=device)

        if vl_embeddings is None:
            print("오류: VL 임베딩 생성 실패. 스크립트를 종료합니다.")
            exit() # VL 생성 실패 시 중단
        else:
            # --- 생성된 결과 저장 ---
            try:
                print(f"\nVH 임베딩 저장 중: {vh_output_path}")
                np.save(vh_output_path, vh_embeddings)
                print(f"VL 임베딩 저장 중: {vl_output_path}")
                np.save(vl_output_path, vl_embeddings)
                print(f"\n임베딩 저장 완료:")
                print(f"VH 임베딩 shape: {vh_embeddings.shape}")
                print(f"VL 임베딩 shape: {vl_embeddings.shape}")
            except Exception as e:
                print(f"오류: 임베딩 저장 중 오류 발생: {e}")
                # 저장 실패 시 이후 단계 진행 어려울 수 있음

# 3. CD-HIT 실행 옵션
identity_threshold = 0.9 # 유사도 임계값
word_size = 5 # 워드 크기

# --- CD-HIT 실행 준비 ---
print("\n--- CD-HIT 실행 준비 ---")
# 1. 클러스터링 기준 서열 선택 및 FASTA 파일 생성
print("CD-HIT 입력 FASTA 파일 생성 중...")
cdr3_sequences_for_cdhit = []
num_invalid_cdr3 = 0
if 'vh_cdr3' in metadata_df.columns:
    for index, row in metadata_df.iterrows():
        entry_id = str(row['entry_id']) # 문자열로 변환
        cdr3_seq = str(row['vh_cdr3']) if pd.notna(row['vh_cdr3']) else ""
        # CDR3 서열 유효성 검사 강화 (예: 표준 아미노산 문자만 허용)
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if cdr3_seq and all(c.upper() in valid_aa for c in cdr3_seq):
            record = SeqRecord(Seq(cdr3_seq.upper()), id=entry_id, description="")
            cdr3_sequences_for_cdhit.append(record)
        else:
            # print(f"경고: Entry ID {entry_id}의 VH CDR3 서열 ('{cdr3_seq}')이 유효하지 않아 제외됩니다.")
            num_invalid_cdr3 += 1
    if num_invalid_cdr3 > 0:
         print(f"경고: 총 {num_invalid_cdr3}개의 유효하지 않은 VH CDR3 서열이 제외되었습니다.")
else:
    print("오류: 'vh_cdr3' 컬럼이 메타데이터에 없습니다.")
    raise ValueError("'vh_cdr3' 컬럼 누락")

# FASTA 파일 저장 (로컬 임시 디렉토리)
input_fasta_filename = 'vh_cdr3_sequences_for_cdhit.fasta'
input_fasta_path = os.path.join(temp_dir, input_fasta_filename)
try:
    with open(input_fasta_path, "w") as output_handle:
        SeqIO.write(cdr3_sequences_for_cdhit, output_handle, "fasta")
    print(f"유효한 VH CDR3 서열 ({len(cdr3_sequences_for_cdhit)}개) FASTA 파일 저장 완료: {input_fasta_path}")
except Exception as e:
    print(f"FASTA 파일 저장 중 오류 발생: {e}")
    raise

# 2. 출력 파일 Prefix 설정 (로컬 임시 디렉토리)
output_prefix_base = f'vh_cdr3_clusters_{int(identity_threshold*100)}' # 파일명에 임계값 포함
output_prefix = os.path.join(temp_dir, output_prefix_base)


# 4. CD-HIT 실행 방식 결정 (WSL 또는 직접 실행)
cd_hit_executable = None
command_list = []

# WSL 사용 여부 확인 (Windows 환경에서)
use_wsl = False
if platform.system() == "Windows":
    # wsl 명령어가 실행 가능한지 간단히 확인
    try:
        subprocess.run(['wsl', '--status'], check=True, capture_output=True)
        print("WSL 확인됨. CD-HIT 실행에 WSL을 사용합니다.")
        use_wsl = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("WSL을 찾을 수 없거나 상태 확인에 실패했습니다. Windows용 CD-HIT 실행을 시도합니다.")

# 명령어 리스트 구성
if use_wsl:
    # --- 경로 변환 함수 (Windows -> WSL) ---
    def windows_to_wsl_path(windows_path):
        path = windows_path.replace('\\', '/')
        drive, path_part = os.path.splitdrive(path)
        if drive: # 드라이브 문자가 있는 경우
             drive_letter = drive.replace(':', '').lower()
             wsl_path = f"/mnt/{drive_letter}{path_part}"
        else: # 상대 경로 등 드라이브 문자 없는 경우
             wsl_path = path # 그대로 사용 시도
        return wsl_path

    cd_hit_executable_name = 'cd-hit' # WSL 내의 실행 파일 이름
    input_fasta_path_exec = windows_to_wsl_path(input_fasta_path)
    output_prefix_exec = windows_to_wsl_path(output_prefix)
    command_list.append('wsl') # WSL 명령 시작
    command_list.append(cd_hit_executable_name)
else:
    # Windows용 CD-HIT 실행 파일 경로 설정 (사용자 환경에 맞게 수정 필요!)
    # 예시: 'C:/Program Files/cd-hit/cd-hit.exe' 또는 PATH에 등록된 'cd-hit.exe'
    cd_hit_executable_path = 'cd-hit.exe' # PATH에 등록된 경우 가정, 아니면 전체 경로 지정
    input_fasta_path_exec = input_fasta_path
    output_prefix_exec = output_prefix
    command_list.append(cd_hit_executable_path)

# 공통 옵션 추가
command_list.extend([
    '-i', input_fasta_path_exec,
    '-o', output_prefix_exec,
    '-c', str(identity_threshold),
    '-n', str(word_size),
    '-M', '0', # 메모리 제한 없음
    '-T', '0' # 모든 스레드 사용
])

# CD-HIT 실행
print(f"\n--- CD-HIT 실행 ({'WSL 사용' if use_wsl else '직접 실행'}) ---")
print(f"실행 명령어: {' '.join(command_list)}")
cluster_file_path = f"{output_prefix}.clstr" # 최종 클러스터 파일 경로 (로컬 기준)

try:
    # 이전 클러스터 파일이 있다면 삭제 (덮어쓰기 오류 방지)
    if os.path.exists(cluster_file_path):
        print(f"기존 클러스터 파일 삭제: {cluster_file_path}")
        os.remove(cluster_file_path)
    if os.path.exists(output_prefix): # .bak 파일 등도 삭제
         print(f"기존 출력 파일 삭제: {output_prefix}")
         os.remove(output_prefix)

    process = subprocess.run(command_list, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')

    print("\n--- CD-HIT 실행 로그 (stdout) ---")
    print(process.stdout)
    print("--- CD-HIT 실행 로그 (stderr) ---")
    print(process.stderr)
    print("---------------------------------")

    if os.path.exists(cluster_file_path):
        print(f"CD-HIT 실행 성공. 클러스터 파일 생성됨: {cluster_file_path}")
    else:
        print(f"오류: CD-HIT 실행 완료되었으나 클러스터 파일({cluster_file_path})이 생성되지 않았습니다.")
        raise FileNotFoundError("CD-HIT 클러스터 파일 생성 실패")

except FileNotFoundError:
    print(f"오류: 실행 파일({command_list[0]})을 찾을 수 없습니다.")
    if use_wsl:
        print("WSL이 설치되어 있고 'cd-hit'가 WSL 내 PATH에 있는지 확인하세요.")
    else:
        print(f"'{cd_hit_executable_path}' 경로가 올바르거나 PATH 환경 변수에 CD-HIT가 등록되었는지 확인하세요.")
    raise
except subprocess.CalledProcessError as e:
    print(f"오류: CD-HIT 실행 실패 (종료 코드: {e.returncode})")
    print("--- 실패 시 stdout ---"); print(e.stdout)
    print("--- 실패 시 stderr ---"); print(e.stderr)
    raise
except Exception as e:
    print(f"CD-HIT 실행 중 예외 발생: {e}")
    raise


# --- 클러스터 파일 파싱 함수 정의 ---
def parse_cdhit_clstr(clstr_file_path):
    """ CD-HIT .clstr 파일 파싱 함수 """
    clusters = {}
    current_cluster_id = -1
    try:
        with open(clstr_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('>Cluster'):
                    try:
                        current_cluster_id = int(line.strip().split()[-1])
                    except (IndexError, ValueError):
                         print(f"경고: 클러스터 ID 파싱 오류 - {line.strip()}")
                         current_cluster_id = -1
                elif line.strip():
                    # 정규식 개선: '>' 이후 공백 아닌 문자열(\S+)을 entry_id로 추출하고, '...' 확인
                    match = re.search(r'>(\S+)\.{3}', line)
                    if match and current_cluster_id != -1:
                        entry_id = match.group(1)
                        clusters[entry_id] = current_cluster_id
                    elif line.strip().endswith('*'): # 대표 서열 라인도 ID 추출 시도
                         match_rep = re.search(r'>(\S+)\.{3}', line)
                         if match_rep and current_cluster_id != -1:
                              entry_id = match_rep.group(1)
                              if entry_id not in clusters: # 아직 매핑 안된 경우만 추가
                                   clusters[entry_id] = current_cluster_id
                         else:
                              print(f"경고: 대표 서열 ID 파싱 오류 또는 유효하지 않은 클러스터 ID - {line.strip()}")
                    # else: # 파싱 실패 케이스 로깅 감소 (필요시 주석 해제)
                    #      print(f"경고: 서열 ID 파싱 오류 추정 - {line.strip()}")
        print(f"클러스터 파일 파싱 완료: 총 {len(set(clusters.values()))}개 클러스터, {len(clusters)}개 서열 매핑됨.")
        # 파싱된 서열 수와 입력 서열 수 비교
        if len(clusters) != len(cdr3_sequences_for_cdhit):
             print(f"경고: 입력 FASTA 서열 수({len(cdr3_sequences_for_cdhit)})와 파싱된 서열 수({len(clusters)})가 다릅니다.")
        return clusters
    except FileNotFoundError:
        print(f"오류: 클러스터 파일을 찾을 수 없습니다 - {clstr_file_path}")
        return None
    except Exception as e:
        print(f"클러스터 파일 파싱 중 오류 발생: {e}")
        return None


# --- 클러스터 파일 파싱 실행 ---
print(f"\n--- 클러스터 파일 파싱 시작: {cluster_file_path} ---")
entry_to_cluster = parse_cdhit_clstr(cluster_file_path)

if entry_to_cluster is None:
    print("오류: 클러스터 정보 파싱 실패. 데이터 분할을 진행할 수 없습니다.")
    raise ValueError("클러스터 파싱 실패")
else:
    # --- 그룹 정보 생성 ---
    print("\n--- 그룹 정보 생성 및 데이터 분할 준비 ---")
    groups = metadata_df['entry_id'].astype(str).map(entry_to_cluster).fillna(-1).astype(int)
    metadata_df['cluster_id'] = groups
    num_unclustered = (groups == -1).sum()
    if num_unclustered > 0:
        print(f"경고: {num_unclustered}개의 서열이 클러스터링되지 않았습니다 (그룹 ID: -1).")

    # --- 데이터 분할 실행 (GroupShuffleSplit 사용) ---
    # 이 단계에서는 이전에 로드했거나 새로 생성한 vh_embeddings, vl_embeddings 사용
    if 'vh_embeddings' not in locals() or 'vl_embeddings' not in locals() or \
       vh_embeddings is None or vl_embeddings is None:
        print("오류: 데이터 분할에 필요한 임베딩 변수가 준비되지 않았습니다.")
        raise NameError("임베딩 변수(vh_embeddings, vl_embeddings) 정의/생성 오류")

    indices = np.arange(len(metadata_df))

    # 훈련/테스트 분할
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    print("훈련/테스트 데이터 분할 중...")
    train_val_idx, test_idx = next(gss_test.split(indices, groups=groups))

    # 훈련/검증 분할
    validation_split_ratio = 0.125
    gss_val = GroupShuffleSplit(n_splits=1, test_size=validation_split_ratio, random_state=42)
    print("훈련/검증 데이터 분할 중...")
    train_rel_idx, val_rel_idx = next(gss_val.split(indices[train_val_idx], groups=groups[train_val_idx]))

    original_train_idx = train_val_idx[train_rel_idx]
    original_val_idx = train_val_idx[val_rel_idx]
    original_test_idx = test_idx

    # 분할된 데이터 생성
    X_train_vh, X_val_vh, X_test_vh = vh_embeddings[original_train_idx], vh_embeddings[original_val_idx], vh_embeddings[original_test_idx]
    X_train_vl, X_val_vl, X_test_vl = vl_embeddings[original_train_idx], vl_embeddings[original_val_idx], vl_embeddings[original_test_idx]
    metadata_train = metadata_df.iloc[original_train_idx].copy()
    metadata_val = metadata_df.iloc[original_val_idx].copy()
    metadata_test = metadata_df.iloc[original_test_idx].copy()

    print("\n--- 데이터 분할 결과 ---")
    print(f"훈련 세트 크기: {len(metadata_train)} (VH: {X_train_vh.shape}, VL: {X_train_vl.shape})")
    print(f"검증 세트 크기: {len(metadata_val)} (VH: {X_val_vh.shape}, VL: {X_val_vl.shape})")
    print(f"테스트 세트 크기: {len(metadata_test)} (VH: {X_test_vh.shape}, VL: {X_test_vl.shape})")
    total_split = len(metadata_train) + len(metadata_val) + len(metadata_test)
    print(f"총합: {total_split} (원본: {len(metadata_df)})")
    if total_split != len(metadata_df):
         print("경고: 분할된 데이터 총합이 원본 데이터 크기와 다릅니다!")
    total_size = len(metadata_df)
    print(f"분할 비율 (Train/Val/Test): {len(metadata_train)/total_size:.2f} / {len(metadata_val)/total_size:.2f} / {len(metadata_test)/total_size:.2f}")

    # --- 분할된 데이터 저장 ---
    print(f"\n--- 분할된 데이터 저장 시작 ({split_dir}) ---")
    try:
        np.save(os.path.join(split_dir, 'X_train_vh.npy'), X_train_vh)
        np.save(os.path.join(split_dir, 'X_val_vh.npy'), X_val_vh)
        np.save(os.path.join(split_dir, 'X_test_vh.npy'), X_test_vh)
        np.save(os.path.join(split_dir, 'X_train_vl.npy'), X_train_vl)
        np.save(os.path.join(split_dir, 'X_val_vl.npy'), X_val_vl)
        np.save(os.path.join(split_dir, 'X_test_vl.npy'), X_test_vl)
        print("임베딩 데이터 (Train/Val/Test) 저장 완료.")

        metadata_train.to_parquet(os.path.join(split_dir, 'metadata_train.parquet'))
        metadata_val.to_parquet(os.path.join(split_dir, 'metadata_val.parquet'))
        metadata_test.to_parquet(os.path.join(split_dir, 'metadata_test.parquet'))
        print("메타데이터 (Train/Val/Test) 저장 완료.")
        print("\n모든 프로세스 완료.")

    except Exception as e:
        print(f"분할된 데이터 저장 중 오류 발생: {e}")