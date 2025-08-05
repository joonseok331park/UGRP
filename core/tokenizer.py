# [수정 사항] ID_OFFSET 값을 260에서 256으로 변경하여 Jo & Kim (2024) 논문과 완벽히 일치시켰습니다.
# 이 외의 모든 로직은 설계 의도에 완벽하게 부합하여 그대로 유지합니다.

import json
import pandas as pd
from itertools import chain

class CANTokenizer:
    """
    CAN 데이터의 어휘집을 관리하고 토큰-인덱스 변환을 처리합니다.
    Jo & Kim (2024) 논문에서 설명된 오프셋 기반 어휘집을 구현합니다.
    """

    def __init__(self):
        """
        특수 토큰과 ID 오프셋을 포함하여 토크나이저를 초기화합니다.
        <PAD> 토큰은 인덱스 0으로 설정됩니다.
        """
        self.token_to_id = {}
        self.id_to_token = {}
        # Jo & Kim (2024) 논문의 방법론에 따라 ID 오프셋을 256으로 명확히 설정합니다. 
        # 이는 0~255 범위의 데이터 토큰과의 충돌을 방지하기 위함입니다.
        self.ID_OFFSET = 256

        # 특수 토큰을 추가합니다. <PAD>는 반드시 인덱스 0이어야 합니다.
        self.special_tokens = ['<PAD>', '<UNK>', '<MASK>', '<VOID>']
        for token in self.special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> None:
        """어휘집에 토큰을 추가하는 보조 함수입니다."""
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

    def build_vocab(self, df: pd.DataFrame) -> None:
        """
        DataFrame으로부터 오프셋 기반 통합 어휘집을 구축합니다.
        1. 데이터 토큰: '00'부터 'FF'까지 (256개).
        2. ID 토큰: DataFrame의 고유한 CAN ID에 self.ID_OFFSET을 더한 값.
        """
        # 1. 데이터 토큰('00' ~ 'FF') 추가
        for i in range(256):
            self._add_token(f'{i:02X}')

        # 2. 오프셋을 적용한 ID 토큰 추가
        unique_ids = df['CAN_ID'].unique()
        for can_id in unique_ids:
            # 토큰은 (ID의 16진수 값 + 오프셋)의 문자열 표현입니다.
            # 이 방식은 Jo & Kim (2024) 논문에서 기술된 '어휘집 통합' 방법론과 일치합니다. 
            token = str(int(can_id, 16) + self.ID_OFFSET)
            self._add_token(token)

    def save_vocab(self, file_path: str) -> None:
        """token_to_id 사전을 JSON 파일로 저장합니다."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=4)

    def load_vocab(self, file_path: str) -> None:
        """JSON 파일로부터 어휘집을 로드합니다."""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, tokens: list[str]) -> list[int]:
        """토큰 리스트를 정수 ID 리스트로 변환합니다."""
        unk_token_id = self.token_to_id['<UNK>']
        return [self.token_to_id.get(token, unk_token_id) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """정수 ID 리스트를 다시 토큰 리스트로 변환합니다."""
        return [self.id_to_token.get(id, '<UNK>') for id in ids]

    def get_vocab_size(self) -> int:
        """어휘집의 크기를 반환합니다."""
        return len(self.token_to_id)


class CANSequencer:
    """
    CANTokenizer를 사용하여 CAN 메시지 DataFrame을 고정 길이 시퀀스로 변환합니다.
    """

    def __init__(self, tokenizer: CANTokenizer, seq_len: int = 126):
        """
        시퀀서를 초기화합니다.
        :param tokenizer: CANTokenizer의 인스턴스.
        :param seq_len: 각 시퀀스의 고정 길이.
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def transform(self, df: pd.DataFrame) -> list[list[int]]:
        """
        DataFrame을 인코딩된 시퀀스 리스트로 변환합니다.
        'Data' 컬럼은 16진수 문자열의 리스트 형태여야 합니다.
        """
        # 1. 프레임 수준 토큰화 (ID + Payload)
        all_tokens = []
        for _, row in df.iterrows():
            # 토크나이저의 오프셋을 사용하여 일관성을 유지합니다.
            can_id_token = str(int(row['CAN_ID'], 16) + self.tokenizer.ID_OFFSET)
            
            # 'Data' 컬럼을 직접 사용합니다. (문자열 리스트 형태)
            data_bytes = row['Data']
            
            # 각 프레임은 9개의 토큰(ID 1개 + 데이터 바이트 8개)으로 구성됩니다.
            # 이 'ID+Payload 통합 시퀀싱'은 Jo & Kim (2024)의 핵심 방법론입니다. [cite: 1751]
            frame_tokens = [can_id_token] + data_bytes
            all_tokens.append(frame_tokens)

        # 2. 단일 스트림 생성
        token_stream = list(chain.from_iterable(all_tokens))

        # 3. 슬라이딩 윈도우
        sequences = []
        stride = 1 # 한 칸씩 이동하며 시퀀스를 생성합니다.
        if len(token_stream) >= self.seq_len:
            for i in range(0, len(token_stream) - self.seq_len + 1, stride):
                sequence = token_stream[i:i + self.seq_len]
                sequences.append(sequence)

        # 4. 인코딩
        encoded_sequences = [self.tokenizer.encode(seq) for seq in sequences]

        return encoded_sequences


if __name__ == '__main__':
    # 이 테스트 코드는 파일의 독립적인 기능 검증을 위해 매우 유용합니다.
    # 그대로 유지하여 향후 발생할 수 있는 문제를 사전에 방지하는 데 사용하겠습니다.
    sample_data = {
        'CAN_ID': ['01A', '02B', '01A', '03C'],
        'Data': [
            ['11', '22', '33', '44', '55', '66', '77', '88'],
            ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', '00', '11'],
            ['99', '88', '77', '66', '55', '44', '33', '22'],
            ['DE', 'AD', 'BE', 'EF', 'CA', 'FE', 'BA', 'BE']
        ]
    }
    df_sample = pd.DataFrame(sample_data)

    print("--- CAN Tokenizer 및 Sequencer 테스트 ---")
    print("\n1. 샘플 DataFrame:")
    print(df_sample)

    tokenizer = CANTokenizer()
    tokenizer.build_vocab(df_sample)
    print("\n2. 어휘집 구축 완료.")
    print(f"   - 어휘집 크기: {tokenizer.get_vocab_size()}")

    print("\n3. 어휘집 내용 확인:")
    print(f"  - ID 오프셋: {tokenizer.ID_OFFSET}")
    id_tokens_to_check = {str(int(id, 16) + tokenizer.ID_OFFSET) for id in df_sample['CAN_ID'].unique()}
    print(f"  - ID 토큰 (오프셋 적용됨):", {k: v for k, v in tokenizer.token_to_id.items() if k in id_tokens_to_check})
    
    vocab_file = 'vocab_test.json'
    tokenizer.save_vocab(vocab_file)
    print(f"\n4. 어휘집이 '{vocab_file}'에 저장되었습니다.")
    
    new_tokenizer = CANTokenizer()
    new_tokenizer.load_vocab(vocab_file)
    print(f"   '{vocab_file}'에서 어휘집을 성공적으로 로드했습니다.")
    assert tokenizer.token_to_id == new_tokenizer.token_to_id
    print("   로드된 어휘집이 원본과 일치합니다.")

    sequence_length = 10
    sequencer = CANSequencer(tokenizer=tokenizer, seq_len=sequence_length)
    encoded_sequences = sequencer.transform(df_sample)
    print(f"\n5. {len(encoded_sequences)}개의 인코딩된 시퀀스가 생성되었습니다.")

    print("\n--- 검증 ---")
    if encoded_sequences:
        first_sequence_decoded = tokenizer.decode(encoded_sequences[0])
        print("\n첫 번째 인코딩된 시퀀스:", encoded_sequences[0])
        print("첫 번째 디코딩된 시퀀스:", first_sequence_decoded)
        id_01A_token = str(int('01A', 16) + tokenizer.ID_OFFSET)
        assert first_sequence_decoded[0] == id_01A_token, "오프셋 적용된 ID 토큰 검증 실패"
        assert first_sequence_decoded[1] == '11', "데이터 토큰 검증 실패"
        print("\n검증 성공!")