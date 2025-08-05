# core/dataset.py (메모리 효율적으로 개선된 최종 버전)
import torch
import random
import pandas as pd
from typing import List, Dict
from itertools import chain

# 이 파일은 다른 파일에 의존하므로, 임포트 경로는 그대로 둡니다.
from core.tokenizer import CANTokenizer
from utils.data_loader import load_can_data

class MLMDataset(torch.utils.data.Dataset):
    """
    BERT 모델의 MLM 훈련을 위한 메모리 효율적인 데이터셋.
    파일 경로를 직접 받아 필요한 데이터만 스트리밍 방식으로 처리합니다.
    """
    def __init__(self, file_path: str, tokenizer: CANTokenizer, seq_len: int, dataset_type: str = 'candump', mask_prob: float = 0.15):
        """
        데이터셋을 초기화하고, 모든 토큰을 하나의 긴 스트림으로 변환하여 저장합니다.

        :param file_path: 훈련 데이터 파일의 경로.
        :param tokenizer: 훈련된 CANTokenizer 객체.
        :param seq_len: 모델에 입력될 시퀀스의 길이.
        :param dataset_type: 데이터 로더가 사용할 데이터셋 유형.
        :param mask_prob: 각 토큰을 마스킹할 확률.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        
        # 1. 데이터 로드
        print("Dataset: Loading and tokenizing data from file...")
        can_df = load_can_data(file_path, dataset_type)
        if can_df.empty:
            raise ValueError(f"No data loaded from {file_path}")

        # 2. DataFrame을 토큰 리스트의 리스트로 변환 (ID 오프셋 적용)
        all_frames_as_tokens = []
        for _, row in can_df.iterrows():
            can_id_token = str(int(row['CAN_ID'], 16) + self.tokenizer.ID_OFFSET)
            frame_tokens = [can_id_token] + row['Data']
            all_frames_as_tokens.append(frame_tokens)
        
        # 3. 모든 토큰을 하나의 거대한 1차원 리스트로 만듭니다. (메모리 효율적)
        token_stream = list(chain.from_iterable(all_frames_as_tokens))
        
        # 4. 토큰 스트림을 숫자 ID 스트림으로 인코딩하여 저장합니다.
        # 이 단계 이후로는 거대한 문자열 리스트 대신 정수 리스트만 메모리에 남습니다.
        self.token_id_stream = self.tokenizer.encode(token_stream)
        
        # 5. 특수 토큰 ID 미리 저장
        self.mask_token_id = tokenizer.token_to_id.get('<MASK>', -1)
        self.pad_token_id = tokenizer.token_to_id.get('<PAD>', -1)
        self.vocab_size = len(tokenizer.token_to_id)
        self.special_token_ids = {tokenizer.token_to_id[token] for token in tokenizer.special_tokens if token in tokenizer.token_to_id}
        
        print(f"Dataset: Initialization complete. Token stream length: {len(self.token_id_stream):,}")


    def __len__(self) -> int:
        """전체 데이터셋에서 생성 가능한 시퀀스의 총 개수를 반환합니다."""
        return len(self.token_id_stream) - self.seq_len + 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        하나의 데이터 샘플을 동적으로 생성하고 마스킹합니다.
        
        :param idx: 생성할 시퀀스의 시작 인덱스.
        :return: input_ids, attention_mask, labels를 포함하는 딕셔너리.
        """
        # 1. 필요한 만큼의 시퀀스만 스트림에서 잘라냅니다.
        sequence = self.token_id_stream[idx : idx + self.seq_len]
        
        input_ids = list(sequence)
        labels = [-100] * self.seq_len

        # 2. 동적 마스킹 수행
        candidate_indices = [i for i, token_id in enumerate(input_ids) if token_id not in self.special_token_ids]
        num_to_mask = int(len(candidate_indices) * self.mask_prob)
        masked_indices = random.sample(candidate_indices, num_to_mask)

        for i in masked_indices:
            labels[i] = input_ids[i]
            rand = random.random()
            if rand < 0.8:
                input_ids[i] = self.mask_token_id
            elif rand < 0.9:
                random_token_id = random.randint(len(self.special_token_ids), self.vocab_size - 1)
                input_ids[i] = random_token_id
        
        attention_mask = [1] * self.seq_len # 패딩이 없으므로 항상 1

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }