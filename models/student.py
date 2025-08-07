# UGRP/models/student.py

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

def get_student_hyperparams() -> Dict[str, Any]:
    """
    specification.md 및 검토 보고서에 따라 확정된 학생 모델의 하이퍼파라미터를 반환합니다.
    """
    # embedding_dim을 64로 조정하여 경량화 강화
    return {
        "embedding_dim": 64,       # hidden_size와 통일
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
    }

class BiLSTMStudent(nn.Module):
    """
    UGRP 프로젝트의 학생 모델(Student Model)을 위한 BiLSTM 기반 아키텍처입니다.
    (v1.1: 검토 보고서의 피드백을 반영하여 경량화 및 가변 길이 시퀀스 처리 기능 강화)
    """
    def __init__(self, vocab_size: int, num_classes: int, pad_token_id: int):
        """
        BiLSTM 학생 모델을 초기화합니다.

        :param vocab_size: 토크나이저의 전체 어휘집 크기.
        :param num_classes: 최종 분류할 클래스의 수.
        :param pad_token_id: 패딩 토큰의 ID.
        """
        super().__init__()
        
        self.hyperparams = get_student_hyperparams()
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.hyperparams["embedding_dim"],
            padding_idx=self.pad_token_id # 패딩 토큰의 임베딩은 0으로 고정
        )
        
        self.lstm = nn.LSTM(
            input_size=self.hyperparams["embedding_dim"],
            hidden_size=self.hyperparams["hidden_size"],
            num_layers=self.hyperparams["num_layers"],
            bidirectional=True,
            batch_first=True,
            dropout=self.hyperparams["dropout"] if self.hyperparams["num_layers"] > 1 else 0
        )
        
        self.dropout = nn.Dropout(self.hyperparams["dropout"])
        
        self.classifier = nn.Linear(
            self.hyperparams["hidden_size"] * 2,
            num_classes
        )

    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor = None
               ) -> torch.Tensor:
        """
        모델의 순전파 로직을 정의합니다. 가변 길이 시퀀스를 지원합니다.

        :param input_ids: 입력 텐서 (batch_size, sequence_length).
        :param attention_mask: 어텐션 마스크 텐서 (batch_size, sequence_length).
        :return: 분류 결과 로짓 (batch_size, num_classes).
        """
        # (batch, seq_len) -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # 실제 시퀀스 길이 계산
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1).cpu()
            # 모든 시퀀스가 패딩인 경우 처리
            if seq_lengths.max() == 0:
                # 모든 시퀀스가 패딩이면 LSTM 건너뛰고 0으로 초기화
                batch_size = input_ids.size(0)
                hidden_size = self.hyperparams["hidden_size"]
                num_layers = self.hyperparams["num_layers"]
                hidden = torch.zeros(num_layers * 2, batch_size, hidden_size, device=input_ids.device)
                cell = torch.zeros(num_layers * 2, batch_size, hidden_size, device=input_ids.device)
            else:
                # 패딩된 시퀀스를 압축
                packed_embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, seq_lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, (hidden, cell) = self.lstm(packed_embedded)
        else:
            # 어텐션 마스크가 없으면 모든 시퀀스가 최대 길이라고 가정
            lstm_out, (hidden, cell) = self.lstm(embedded)

        # 마지막 타임스텝의 hidden state 추출 및 결합
        final_hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        dropped_out = self.dropout(final_hidden_state)
        logits = self.classifier(dropped_out)
        
        return logits


if __name__ == '__main__':
    # --- 학생 모델(BiLSTM) 아키텍처 검증 스크립트 v1.1 ---
    DUMMY_VOCAB_SIZE = 2309
    NUM_CLASSES = 2
    PAD_TOKEN_ID = 0 # 일반적인 PAD 토큰 ID
    
    print("--- 학생 모델(BiLSTM) 아키텍처 검증을 시작합니다. (v1.1) ---")

    student_model = BiLSTMStudent(DUMMY_VOCAB_SIZE, NUM_CLASSES, PAD_TOKEN_ID)
    print("✅ [성공] 개선된 모델 생성 완료.")

    # 1. 파라미터 수 재계산 (embedding_dim=64 적용)
    num_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"\n[훈련 가능한 파라미터 수]: {num_params:,}")
    # embedding(2309*64) + lstm + classifier = 약 25만개 수준으로 감소 예상
    if num_params < 320000:
         print("✅ [검증] 파라미터 수가 더욱 감소하여 경량화 목표에 매우 부합합니다.")
    else:
        print("⚠️ [경고] 파라미터 수가 예상보다 많습니다.")

    # 2. 가변 길이 시퀀스 처리 기능 검증
    print("\n[가변 길이 시퀀스 처리 기능 검증]:")
    
    dummy_input = torch.tensor([
        [10, 20, 30, 40, 50],
        [15, 25, 35, PAD_TOKEN_ID, PAD_TOKEN_ID], # 패딩 포함
        [18, 28, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]
    ])
    dummy_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0]
    ], dtype=torch.long)

    try:
        with torch.no_grad():
            output = student_model(dummy_input, attention_mask=dummy_mask)
        print(f"✅ [성공] 패딩된 입력 처리 완료. 출력 Shape: {output.shape}")
        assert output.shape == (3, NUM_CLASSES)
    except Exception as e:
        print(f"❌ [실패] 패딩된 입력 처리 중 오류 발생: {e}")

    print("\n--- 검증 완료 ---")