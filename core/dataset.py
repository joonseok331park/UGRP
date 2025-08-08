# core/dataset.py (의존성 제거 및 이중 클래스 구조 적용 최종 버전)

import re
import torch
import random
import pandas as pd
from typing import List, Dict, Tuple
from itertools import chain
from torch.utils.data import Dataset

# 이 파일은 같은 core 폴더 내의 tokenizer만 의존합니다.
from core.tokenizer import CANTokenizer

def _load_and_parse_log(file_path: str) -> pd.DataFrame:
    """
    CAN-MIRGU 형식의 로그 파일을 읽고 파싱하여 DataFrame으로 변환하는 내부 헬퍼 함수.
    (기존 utils.data_loader.load_can_data의 역할을 대체합니다)

    :param file_path: .log 파일의 경로.
    :return: 'CAN_ID', 'Data', 'Label' 컬럼을 가진 pandas DataFrame.
    """
    # 정규 표현식을 사용하여 로그 라인을 효율적으로 파싱합니다.
    # 포맷: (timestamp) can0 CAN_ID#DATA LABEL
    log_pattern = re.compile(r'\((?P<timestamp>\d+\.\d+)\)\s+can0\s+(?P<can_id>[0-9A-Fa-f]{3})#(?P<data>[0-9A-Fa-f]{0,16})\s+(?P<label>[01])')
    
    parsed_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = log_pattern.match(line)
            if match:
                d = match.groupdict()
                
                # 데이터 페이로드를 8바이트(16진수 16글자)로 패딩합니다.
                # Jo & Kim (2024)의 <VOID> 토큰 개념과 LSF-IDM (2023)의 '00' 패딩 개념을
                # 실제 데이터 처리 로직으로 구현한 것입니다.
                padded_data = d['data'].ljust(16, '0')
                
                # 2글자씩 잘라 8바이트 리스트로 만듭니다.
                data_bytes = [padded_data[i:i+2] for i in range(0, 16, 2)]
                
                parsed_data.append({
                    'CAN_ID': d['can_id'],
                    'Data': data_bytes,
                    'Label': int(d['label'])
                })
    return pd.DataFrame(parsed_data)


# class MLMDataset(Dataset):
#     """
#     BERT 모델의 MLM 사전 훈련을 위한 데이터셋.
#     CAN-MIRGU 보고서에 명시된 '정상(Benign)' 데이터 파일 처리에 최적화되어 있습니다.
#     """
#     def __init__(self, file_path: str, tokenizer: CANTokenizer, seq_len: int, mask_prob: float = 0.15):
#         self.tokenizer = tokenizer
#         self.seq_len = seq_len
#         self.mask_prob = mask_prob
        
#         print(f"MLMDataset: Loading and tokenizing data from file: {file_path}...")
#         can_df = _load_and_parse_log(file_path)
        
#         if can_df.empty:
#             print(f"Warning: No data loaded from {file_path}. MLMDataset will be empty.")
#             self.token_id_stream = []
#         else:
#             # DataFrame을 토큰 리스트의 리스트로 변환 (Label 컬럼은 의도적으로 무시)
#             all_frames_as_tokens = []
#             for _, row in can_df.iterrows():
#                 can_id_token = str(int(row['CAN_ID'], 16) + self.tokenizer.ID_OFFSET)
#                 frame_tokens = [can_id_token] + row['Data']
#                 all_frames_as_tokens.append(frame_tokens)
            
#             token_stream = list(chain.from_iterable(all_frames_as_tokens))
#             self.token_id_stream = self.tokenizer.encode(token_stream)

#         # 특수 토큰 ID 미리 저장
#         self.mask_token_id = tokenizer.token_to_id['<MASK>']
#         self.pad_token_id = tokenizer.token_to_id['<PAD>']
#         self.vocab_size = tokenizer.get_vocab_size()
#         self.special_token_ids = {tokenizer.token_to_id[token] for token in tokenizer.special_tokens}
        
#         print(f"MLMDataset: Init complete. Token stream length: {len(self.token_id_stream):,}")

#     def __len__(self) -> int:
#         if not self.token_id_stream or len(self.token_id_stream) < self.seq_len:
#             return 0
#         return len(self.token_id_stream) - self.seq_len + 1

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         sequence = self.token_id_stream[idx : idx + self.seq_len]
#         input_ids = list(sequence)
#         labels = [-100] * self.seq_len

#         candidate_indices = [i for i, token_id in enumerate(input_ids) if token_id not in self.special_token_ids]
#         num_to_mask = int(len(candidate_indices) * self.mask_prob)
#         if num_to_mask > 0:
#             masked_indices = random.sample(candidate_indices, num_to_mask)
#             for i in masked_indices:
#                 labels[i] = input_ids[i]
#                 rand = random.random()
#                 if rand < 0.8:
#                     input_ids[i] = self.mask_token_id
#                 elif rand < 0.9:
#                     random_token_id = random.randint(len(self.special_token_ids), self.vocab_size - 1)
#                     input_ids[i] = random_token_id
        
#         attention_mask = [1] * self.seq_len

#         return {
#             'input_ids': torch.tensor(input_ids, dtype=torch.long),
#             'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }

class ClassificationDataset(Dataset):
    """
    미세 조정(Fine-tuning) 및 지식 증류(Knowledge Distillation)를 위한 분류 데이터셋.
    CAN-MIRGU 보고서에 명시된 '공격(Attack)' 데이터 처리에 적합합니다.
    """
    def __init__(self, file_path: str, tokenizer: CANTokenizer, seq_len: int, stride: int = 1):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.sequences = []
        self.labels = []
        
        print(f"ClassificationDataset: Loading and preparing sequences from: {file_path}...")
        can_df = _load_and_parse_log(file_path)
        
        if not can_df.empty:
            all_frames_as_tokens = []
            frame_labels = []
            for _, row in can_df.iterrows():
                can_id_token = str(int(row['CAN_ID'], 16) + self.tokenizer.ID_OFFSET)
                frame_tokens = [can_id_token] + row['Data']
                all_frames_as_tokens.append(frame_tokens)
                frame_labels.append(row['Label'])

            # 각 프레임은 9개의 토큰으로 구성됨
            num_tokens_per_frame = 9
            token_stream = list(chain.from_iterable(all_frames_as_tokens))
            encoded_stream = self.tokenizer.encode(token_stream)
            
            # 슬라이딩 윈도우로 시퀀스와 레이블 생성
            for i in range(len(encoded_stream) - seq_len + 1, stride):
                self.sequences.append(encoded_stream[i : i + seq_len])
                
                # 시퀀스에 해당하는 원본 프레임의 레이블 범위를 계산
                start_frame_idx = i // num_tokens_per_frame
                end_frame_idx = (i + seq_len -1) // num_tokens_per_frame + 1
                
                # 시퀀스 내에 공격 프레임이 하나라도 있으면 시퀀스의 레이블은 1(공격)
                sequence_label = 1 if any(frame_labels[start_frame_idx:end_frame_idx]) else 0
                self.labels.append(sequence_label)
        
        print(f"ClassificationDataset: Init complete. Number of sequences: {len(self.sequences):,}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'attention_mask': torch.tensor([1] * self.seq_len, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

if __name__ == '__main__':
    # --- 실제 데이터 기반 테스트 환경 설정 ---
    import os
    import numpy as np
    from pathlib import Path

    # 1. 실제 데이터 파일 경로 지정
    #    [주의] 이 파일이 실제로 존재해야 합니다. 경로가 다르다면 수정해 주십시오.
    #    우선, 보고서에 명시된 Benign 데이터 중 하나를 대상으로 합니다.
    try:
        # 프로젝트 루트 디렉토리를 기준으로 상대 경로 설정
        project_root = Path(__file__).parent.parent 
        # 실제 파일명을 정확히 알 수 없으므로, 일반적인 이름으로 가정합니다.
        # 연구원님의 실제 파일명으로 이 부분을 수정해야 할 수 있습니다.
        real_data_path = project_root / "data" / "raw" / "can_mirgu" / "Benign" / "Day_1" / "Benign_day1_file1.log"
        NUM_LINES_TO_TEST = 100000

        if not real_data_path.exists():
            raise FileNotFoundError

    except FileNotFoundError:
        print("="*60)
        print(f"⚠️  테스트 경고: 실제 데이터 파일을 찾을 수 없습니다.")
        print(f"   - 확인한 경로: {real_data_path.resolve()}")
        print(f"   - `dataset.py`의 테스트 코드를 실행하려면 위 경로에 실제 로그 파일이 필요합니다.")
        print("="*60)
        # 실제 파일이 없으면 테스트를 진행할 수 없으므로 종료합니다.
        exit()


    # 2. 실제 데이터 일부를 읽어 가상의 테스트 로그 파일 생성
    temp_dir = Path("./temp_test")
    temp_dir.mkdir(exist_ok=True)
    test_file_path = temp_dir / "test_real_data_snippet.log"

    with open(real_data_path, 'r', encoding='utf-8') as f_real:
        lines = [f_real.readline() for _ in range(NUM_LINES_TO_TEST)]

    with open(test_file_path, "w", encoding="utf-8") as f_temp:
        f_temp.writelines(lines)

    print(f"--- `_load_and_parse_log` 실제 데이터 기반 테스트 ---")
    print(f"대상 파일: {real_data_path.name}")
    print(f"앞부분 {NUM_LINES_TO_TEST}줄을 임시 파일로 복사하여 테스트합니다.")

    # 3. 파서 함수 실행
    try:
        df = _load_and_parse_log(test_file_path)
        
        # 4. 결과 검증
        print("\n검증 시작...")

        # 4.1. 데이터가 성공적으로 로드되었는지 확인
        assert not df.empty, "오류: 파싱 후 DataFrame이 비어있습니다. 파일 내용이나 파싱 로직을 확인하세요."
        print(f"✅ [성공] 데이터 로드 완료 ({len(df)}/{NUM_LINES_TO_TEST} 라인 파싱)")
        
        print("\n파싱 결과 일부 (상위 5개 행):")
        print(df.head())

        # 4.2. DataFrame 구조 및 타입 검증
        assert all(col in df.columns for col in ['CAN_ID', 'Data', 'Label']), "오류: 필수 컬럼이 누락되었습니다."
        print("✅ [성공] 필수 컬럼(CAN_ID, Data, Label) 존재 여부 검증 완료.")
        
        first_row = df.iloc[0]
        assert isinstance(first_row['CAN_ID'], str) and len(first_row['CAN_ID']) == 3, "오류: CAN_ID 형식이 올바르지 않습니다 (3자리 문자열이어야 함)."
        assert isinstance(first_row['Data'], list) and len(first_row['Data']) == 8, "오류: Data 형식이 올바르지 않습니다 (8개 요소를 가진 리스트여야 함)."
        assert isinstance(first_row['Label'], np.integer), "오류: Label 형식이 올바르지 않습니다 (정수여야 함)."
        print("✅ [성공] 데이터 타입 및 형식 검증 완료.")

        print("\n🎉 모든 테스트 통과! `_load_and_parse_log` 함수가 실제 데이터에 대해 정상적으로 동작합니다.")

    finally:
        # 5. 테스트 종료 후 가상 파일 및 디렉토리 삭제
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        print(f"\n테스트 완료 후 임시 파일 및 디렉토리 삭제됨.")