import torch
import random
from typing import List, Dict

# 테스트를 위해 core.tokenizer에서 CANTokenizer를 임포트합니다.
# 실제 프로젝트 구조에 따라 경로를 조정해야 할 수 있습니다.
from core.tokenizer import CANTokenizer
import pandas as pd

class MLMDataset(torch.utils.data.Dataset):
    """
    BERT 모델의 Masked Language Model(MLM) 훈련을 위한 데이터셋.
    torch.utils.data.Dataset을 상속받습니다.
    """
    def __init__(self, sequences: List[List[int]], tokenizer: CANTokenizer, mask_prob: float = 0.15):
        """
        데이터셋을 초기화합니다.

        :param sequences: 정수 시퀀스의 리스트 (List[List[int]]).
        :param tokenizer: 훈련된 CANTokenizer 객체.
        :param mask_prob: 각 토큰을 마스킹할 확률 (기본값: 0.15).
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.vocab_size = len(tokenizer.token_to_id)

        # 마스킹에 필요한 특수 토큰 ID를 미리 저장합니다.
        self.mask_token_id = tokenizer.token_to_id['<MASK>']
        self.pad_token_id = tokenizer.token_to_id['<PAD>']
        
        # 마스킹에서 제외할 특수 토큰 ID 집합
        self.special_token_ids = {tokenizer.token_to_id[token] for token in tokenizer.special_tokens}

    def __len__(self) -> int:
        """전체 시퀀스의 개수를 반환합니다."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        하나의 데이터 샘플을 동적 마스킹과 함께 생성합니다.
        BERT 표준 규칙에 따라 전체 토큰의 15%를 마스킹합니다.

        :param idx: 가져올 시퀀스의 인덱스.
        :return: input_ids, attention_mask, labels를 포함하는 딕셔너리.
        """
        sequence = self.sequences[idx]
        
        # 1. input_ids와 labels 생성
        input_ids = list(sequence)
        labels = [-100] * len(sequence)

        # 2. 마스킹 후보 인덱스 식별 (특수 토큰 제외)
        candidate_indices = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in self.special_token_ids
        ]

        # 3. 마스킹할 토큰 수 계산 및 인덱스 무작위 선택
        num_to_mask = int(len(candidate_indices) * self.mask_prob)
        masked_indices = random.sample(candidate_indices, num_to_mask)

        # 4. 선택된 인덱스에 마스킹 적용
        for i in masked_indices:
            # 4a. labels에 원래 토큰 ID 저장
            labels[i] = input_ids[i]

            # 4b. BERT 마스킹 규칙 적용
            rand = random.random()
            if rand < 0.8:
                # 80% 확률: <MASK> 토큰으로 교체
                input_ids[i] = self.mask_token_id
            elif rand < 0.9:
                # 10% 확률: 어휘집 내 임의의 토큰으로 교체
                # 특수 토큰이 아닌 토큰 중에서 랜덤하게 선택
                random_token_id = random.randint(len(self.special_token_ids), self.vocab_size - 1)
                input_ids[i] = random_token_id
            # else: 10% 확률로 원래 토큰 유지 (변경 없음)

        # 5. attention_mask 생성
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in input_ids]

        # 6. 결과를 torch.Tensor로 변환하여 딕셔너리로 반환
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

if __name__ == '__main__':
    # --- MLMDataset 클래스 테스트 코드 ---
    print("--- MLMDataset Test ---")

    # 1. 가상의 CANTokenizer와 어휘집 생성
    tokenizer = CANTokenizer()
    sample_data = {
        'CAN_ID': ['1A1', '2B2', '3C3'],
        'Data': [
            ['00', '11', '22', '33', '44', '55', '66', '77'],
            ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', '88', '99'],
            ['F0', 'E1', 'D2', 'C3', 'B4', 'A5', '96', '87']
        ]
    }
    df_sample = pd.DataFrame(sample_data)
    tokenizer.build_vocab(df_sample)
    print("\n1. Virtual CANTokenizer and vocabulary created.")
    print(f"   Vocabulary size: {len(tokenizer.token_to_id)}")

    # 2. 더미 숫자 시퀀스 생성 (인코딩된 시퀀스)
    id_offset = tokenizer.ID_OFFSET
    sequences = [
        tokenizer.encode([str(int('1A1', 16) + id_offset)] + ['11', '22', '33', '44', '55', '66', '77', '88'] + ['<PAD>'] * 2),
        tokenizer.encode([str(int('2B2', 16) + id_offset)] + ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', '88', '99'] + ['<PAD>'] * 2),
        tokenizer.encode([str(int('3C3', 16) + id_offset)] + ['F0', 'E1', 'D2', 'C3', 'B4', 'A5', '96', '87'] + ['<PAD>'] * 2)
    ]
    print("\n2. Dummy encoded sequences created.")
    print("   Original sequence (decoded):", tokenizer.decode(sequences[0]))


    # 3. MLMDataset 인스턴스화
    dataset = MLMDataset(sequences, tokenizer, mask_prob=0.15)
    print("\n3. MLMDataset instantiated.")

    # 4. 첫 번째 아이템 가져와서 결과 확인
    print("\n4. Fetching the first item from the dataset (dataset[0]):")
    data_item = dataset[0]
    
    input_ids = data_item['input_ids']
    attention_mask = data_item['attention_mask']
    labels = data_item['labels']

    print(f"\n   - Input IDs (shape: {input_ids.shape}):")
    print(f"     {input_ids.tolist()}")
    print(f"     Decoded: {tokenizer.decode(input_ids.tolist())}")

    print(f"\n   - Attention Mask (shape: {attention_mask.shape}):")
    print(f"     {attention_mask.tolist()}")

    print(f"\n   - Labels (shape: {labels.shape}):")
    print(f"     {labels.tolist()}")
    
    print("\n--- Verification ---")
    masked_count = 0
    correctly_labeled_count = 0
    for i in range(len(labels)):
        if labels[i] != -100:
            masked_count += 1
            # 레이블이 원래 시퀀스의 토큰과 일치하는지 확인
            if labels[i] == sequences[0][i]:
                correctly_labeled_count += 1
            # input_ids의 해당 위치가 마스킹되었거나, 랜덤 토큰이거나, 원래 토큰인지 확인
            is_masked = input_ids[i] == dataset.mask_token_id
            is_random = input_ids[i] != sequences[0][i] and not is_masked
            is_original = input_ids[i] == sequences[0][i]
            
            print(f"  - Position {i}: Original='{tokenizer.decode([sequences[0][i]])[0]}', "
                  f"Input='{tokenizer.decode([input_ids[i].item()])[0]}', Label='{tokenizer.decode([labels[i].item()])[0]}'")

    if masked_count > 0:
        print(f"\nTotal {masked_count} tokens were masked.")
        assert masked_count == correctly_labeled_count
        print("Verification successful: All masked positions have correct labels.")
    else:
        print("\nNo tokens were masked in this run (due to probability). Run the test again.")

    print("\nTest finished.")