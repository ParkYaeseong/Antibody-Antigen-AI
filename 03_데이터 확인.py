import pandas as pd

# metadata_train.parquet 파일 경로 (정확한 경로로 수정)
metadata_path = 'G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber/3_split_data/metadata_train.parquet'
# metadata_path = 'G:/내 드라이브/Antibody_AI_Data_Stage1_AbNumber/1_preprocessed/final_dataset/antibody_metadata_abnumber.parquet'

try:
    df = pd.read_parquet(metadata_path)
    print("--- 데이터 첫 5줄 ---")
    print(df.head())
    print("\n--- 컬럼 목록 ---")
    print(df.columns)
    print("\n--- 데이터 정보 요약 ---")
    df.info()
    # 숫자형 데이터 통계 (결합력, 점수 등 확인에 유용)
    # print("\n--- 숫자형 데이터 통계 ---")
    # print(df.describe())
except FileNotFoundError:
    print(f"오류: 파일을 찾을 수 없습니다 - {metadata_path}")
except Exception as e:
    print(f"파일 로딩 중 오류: {e}")