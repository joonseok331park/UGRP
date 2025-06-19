import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertLMPredictionHead

class CANBertForMaskedLM(nn.Module):
    """
    CAN-BERT 논문에 기반한 교사 모델 아키텍처.
    Hugging Face Transformers 라이브러리를 사용하여 구현되었습니다.
    """
    def __init__(self, config: BertConfig):
        """
        모델을 초기화합니다.

        Args:
            config (BertConfig): 모델 설정을 위한 BertConfig 객체.
        """
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.cls = BertLMPredictionHead(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        모델의 순전파를 수행합니다.

        Args:
            input_ids (torch.Tensor): 입력 토큰 ID 텐서.
            attention_mask (torch.Tensor, optional): 어텐션 마스크 텐서. Defaults to None.
            labels (torch.Tensor, optional): MLM을 위한 레이블 텐서. Defaults to None.

        Returns:
            tuple: Hugging Face 표준 출력 형식에 따른 결과.
                   (loss, prediction_scores) 또는 (prediction_scores,)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,)
        return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

if __name__ == '__main__':
    # 1. 필요한 라이브러리 임포트
    import pandas as pd
    from core.tokenizer import CANTokenizer

    # 2. CANTokenizer 객체 생성 및 가상 어휘집 구축
    # 테스트를 위한 가상 데이터프레임 생성
    sample_data = {
        'CAN_ID': ['01A', '02B', '01A', '03C', '1F0', '2A5'],
        'Data': [
            ['11', '22', '33', '44', '55', '66', '77', '88'],
            ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', '00', '11'],
            ['99', '88', '77', '66', '55', '44', '33', '22'],
            ['DE', 'AD', 'BE', 'EF', 'CA', 'FE', 'BA', 'BE'],
            ['01', '02', '03', '04', '05', '06', '07', '08'],
            ['F1', 'E2', 'D3', 'C4', 'B5', 'A6', '97', '88']
        ]
    }
    df_sample = pd.DataFrame(sample_data)

    tokenizer = CANTokenizer()
    tokenizer.build_vocab(df_sample)
    
    # 3. 토크나이저로부터 vocab_size 얻기
    vocab_size = len(tokenizer.token_to_id)
    print(f"--- Tokenizer and Model Config ---")
    print(f"Dynamically determined vocab_size: {vocab_size}")

    # 4. BertConfig 객체 생성
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=256,              # d_model
        num_hidden_layers=4,          # L
        num_attention_heads=1,        # h
        intermediate_size=512,        # d_ff
        hidden_dropout_prob=0.1,      # P_drop
        attention_probs_dropout_prob=0.1, # P_drop
    )
    print("BertConfig created successfully.")
    print("-" * 32)

    # 5. CANBertForMaskedLM 모델 인스턴스화
    model = CANBertForMaskedLM(config)

    # 6. 모델 구조 출력
    print("\n--- Model Architecture ---")
    print(model)
    
    # 7. 총 파라미터 수 계산 및 출력
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal Parameters: {total_params:,}")
    print("-" * 26)

    # 8. 더미 입력 텐서 생성
    batch_size = 4
    seq_len = 126
    # 실제 어휘집 크기 내에서 유효한 ID로 텐서 생성
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # 9. 더미 입력을 모델에 통과
    model.eval() # 평가 모드로 설정
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 10. 출력 로짓 차원 확인
    prediction_scores = outputs[0]
    print(f"\n--- Output Shape Verification ---")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {prediction_scores.shape}")
    
    expected_shape = (batch_size, seq_len, vocab_size)
    print(f"Expected shape: {expected_shape}")
    
    assert prediction_scores.shape == expected_shape, \
        f"Shape mismatch! Got {prediction_scores.shape}, expected {expected_shape}"
    
    print("\nOutput shape verification successful!")
    print("-" * 31)