# UGRP/models/teacher.py

from typing import Dict, Any
from transformers import BertConfig, BertForMaskedLM

def get_teacher_config(vocab_size: int) -> BertConfig:
    """
    specification.md에 명시된 교사 모델의 하이퍼파라미터를 기반으로
    BertConfig 객체를 생성하여 반환합니다. 이 함수는 설계를 코드와 동기화하는
    단일 진실 공급원(Single Source of Truth) 역할을 합니다.

    :param vocab_size: 토크나이저의 전체 어휘집 크기.
    :return: 교사 모델의 아키텍처 설정이 담긴 BertConfig 객체.
    """
    # Alkhatib et al. (2022) 논문의 Table II 및 specification.md 7.1. 명세 근거
    teacher_hyperparams: Dict[str, Any] = {
        "vocab_size": vocab_size,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 1,
        "intermediate_size": 512,
        "hidden_act": "relu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 256, # CAN-BERT의 최대 시퀀스 길이 T보다 크게 설정
        "type_vocab_size": 1, # 본 프로젝트에서는 세그먼트 임베딩을 사용하지 않음
    }
    return BertConfig(**teacher_hyperparams)

class CANBert(BertForMaskedLM):
    """
    UGRP 프로젝트의 교사 모델(CAN-BERT)을 위한 클래스입니다.
    Hugging Face의 BertForMaskedLM를 상속받아 프로젝트 맞춤형으로 확장합니다.
    """
    def __init__(self, config: BertConfig):
        """
        CANBert 모델을 초기화합니다.

        :param config: 모델의 아키텍처를 정의하는 BertConfig 객체.
        """
        super().__init__(config)

    @classmethod
    def from_spec(cls, vocab_size: int) -> "CANBert":
        """
        프로젝트 명세(specification.md)에 정의된 기본 설정으로
        CANBert 모델 인스턴스를 생성하는 팩토리 메서드입니다.

        :param vocab_size: 토크나이저의 어휘집 크기.
        :return: 명세에 따라 초기화된 CANBert 모델 객체.
        """
        config = get_teacher_config(vocab_size)
        return cls(config)

if __name__ == '__main__':
    # --- 교사 모델(CAN-BERT) 아키텍처 검증 스크립트 ---
    # 이 스크립트는 models/teacher.py가 specification.md의 요구사항과
    # Alkhatib et al. (2022) 논문의 설계를 올바르게 반영하는지 검증합니다.

    # 가상 어휘집 크기 (실제 훈련 시에는 core.tokenizer.py에서 로드)
    # Jo & Kim (2024)의 '오프셋 기반 통합 어휘집'에 따라 Payload(0-255),
    # CAN ID(256~), 특수 토큰(PAD, MASK 등)을 포함한 크기를 가정합니다.
    DUMMY_VOCAB_SIZE = 2309

    print("--- 교사 모델(CAN-BERT) 아키텍처 검증을 시작합니다. ---")

    # 1. 명세서(specification.md) 기반 모델 생성
    try:
        teacher_model = CANBert.from_spec(vocab_size=DUMMY_VOCAB_SIZE)
        print("✅ [성공] 명세서 기반 모델 생성 완료.")
    except Exception as e:
        print(f"❌ [실패] 모델 생성 중 오류 발생: {e}")
        exit()

    # 2. 모델 설정(Configuration) 출력
    print("\n[모델 설정 (BertConfig)]: ")
    print(teacher_model.config)

    # 3. 훈련 가능한 파라미터 수 계산 및 출력
    num_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print(f"\n[훈련 가능한 파라미터 수]: {num_params:,}")

    # 4. 아키텍처 검증 (Alkhatib et al., 2022, Table III 참조)
    # 논문에서는 어휘집 크기에 따라 약 2-3백만개의 파라미터를 가집니다.
    # 우리 모델이 유사한 규모를 갖는지 확인하여 아키텍처 설정의 정확성을 검증합니다.
    if 2_000_000 < num_params < 4_000_000:
        print("✅ [검증] 파라미터 수가 Alkhatib et al. (2022) 논문의 모델 규모와 유사합니다.")
    else:
        print("⚠️ [경고] 파라미터 수가 예상 범위를 벗어납니다. 아키텍처 설정을 재확인하십시오.")

    print("\n--- 검증 완료 ---")