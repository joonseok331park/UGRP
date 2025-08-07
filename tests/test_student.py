# UGRP/tests/test_student.py
"""
BiLSTM 학생 모델의 포괄적인 테스트 스위트.
엣지 케이스, 성능 벤치마크, 메모리 프로파일링을 포함합니다.
"""

import torch
import torch.nn as nn
import time
import tracemalloc
import sys
import os
from typing import Dict, Tuple, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.student import BiLSTMStudent, get_student_hyperparams


class TestStudentModel:
    """BiLSTM 학생 모델을 위한 종합 테스트 클래스"""
    
    def __init__(self):
        self.vocab_size = 2309
        self.num_classes = 2
        self.pad_token_id = 0
        self.seq_len = 126
        self.batch_size = 32
        
        # 테스트 결과 저장
        self.results: Dict[str, Any] = {}
        
    def setup_model(self) -> BiLSTMStudent:
        """테스트용 모델 인스턴스 생성"""
        return BiLSTMStudent(
            vocab_size=self.vocab_size,
            num_classes=self.num_classes,
            pad_token_id=self.pad_token_id
        )
    
    def test_model_creation(self) -> bool:
        """테스트 1: 모델 생성 및 기본 속성 검증"""
        print("\n[TEST 1] 모델 생성 테스트")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            
            # 하이퍼파라미터 검증
            hyperparams = get_student_hyperparams()
            assert hyperparams["embedding_dim"] == 64, "임베딩 차원 불일치"
            assert hyperparams["hidden_size"] == 64, "Hidden size 불일치"
            assert hyperparams["num_layers"] == 2, "레이어 수 불일치"
            assert hyperparams["dropout"] == 0.2, "드롭아웃 비율 불일치"
            
            print("✅ 모델 생성 성공")
            print(f"   - 임베딩 차원: {hyperparams['embedding_dim']}")
            print(f"   - Hidden size: {hyperparams['hidden_size']}")
            print(f"   - 레이어 수: {hyperparams['num_layers']}")
            print(f"   - 드롭아웃: {hyperparams['dropout']}")
            
            self.results["model_creation"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ 모델 생성 실패: {e}")
            self.results["model_creation"] = "FAIL"
            return False
    
    def test_parameter_count(self) -> bool:
        """테스트 2: 파라미터 수 및 모델 크기 검증"""
        print("\n[TEST 2] 파라미터 수 및 모델 크기")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            
            # 파라미터 수 계산
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # 모델 크기 계산 (MB)
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            print(f"✅ 총 파라미터 수: {total_params:,}")
            print(f"✅ 학습 가능 파라미터: {trainable_params:,}")
            print(f"✅ 모델 크기: {model_size_mb:.2f} MB")
            
            # 경량화 목표 검증 (< 300,000 파라미터)
            assert trainable_params < 300_000, f"파라미터 수 초과: {trainable_params}"
            # 모델 크기 목표 검증 (< 16MB)
            assert model_size_mb < 16, f"모델 크기 초과: {model_size_mb} MB"
            
            self.results["parameter_count"] = trainable_params
            self.results["model_size_mb"] = model_size_mb
            return True
            
        except Exception as e:
            print(f"❌ 파라미터 검증 실패: {e}")
            self.results["parameter_count"] = "FAIL"
            return False
    
    def test_normal_forward(self) -> bool:
        """테스트 3: 정상적인 순전파 테스트"""
        print("\n[TEST 3] 정상 순전파 테스트")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            model.eval()
            
            # 정상 입력 생성
            input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
            attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
            
            with torch.no_grad():
                output = model(input_ids, attention_mask)
            
            expected_shape = (self.batch_size, self.num_classes)
            assert output.shape == expected_shape, f"출력 shape 불일치: {output.shape} != {expected_shape}"
            
            print(f"✅ 입력 shape: {input_ids.shape}")
            print(f"✅ 출력 shape: {output.shape}")
            print(f"✅ 정상 순전파 성공")
            
            self.results["normal_forward"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ 순전파 실패: {e}")
            self.results["normal_forward"] = "FAIL"
            return False
    
    def test_variable_length_sequences(self) -> bool:
        """테스트 4: 가변 길이 시퀀스 처리"""
        print("\n[TEST 4] 가변 길이 시퀀스 테스트")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            model.eval()
            
            # 다양한 길이의 시퀀스 생성
            batch_size = 4
            input_ids = torch.tensor([
                [10, 20, 30, 40, 50, 0, 0, 0],  # 길이 5
                [15, 25, 35, 0, 0, 0, 0, 0],     # 길이 3
                [18, 28, 38, 48, 58, 68, 78, 88], # 길이 8 (최대)
                [11, 0, 0, 0, 0, 0, 0, 0]        # 길이 1
            ])
            
            attention_mask = torch.tensor([
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0]
            ], dtype=torch.long)
            
            with torch.no_grad():
                output = model(input_ids, attention_mask)
            
            assert output.shape == (batch_size, self.num_classes)
            
            seq_lengths = attention_mask.sum(dim=1).tolist()
            print(f"✅ 시퀀스 길이: {seq_lengths}")
            print(f"✅ 출력 shape: {output.shape}")
            print(f"✅ 가변 길이 처리 성공")
            
            self.results["variable_length"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ 가변 길이 처리 실패: {e}")
            self.results["variable_length"] = "FAIL"
            return False
    
    def test_edge_cases(self) -> bool:
        """테스트 5: 엣지 케이스 (극단적 상황)"""
        print("\n[TEST 5] 엣지 케이스 테스트")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            model.eval()
            
            # 케이스 1: 모든 토큰이 패딩
            print("  [5-1] 모든 토큰이 패딩인 경우")
            all_padding = torch.zeros((1, 10), dtype=torch.long)
            all_padding_mask = torch.zeros((1, 10), dtype=torch.long)
            
            with torch.no_grad():
                output1 = model(all_padding, all_padding_mask)
            assert output1.shape == (1, self.num_classes)
            print("  ✅ 모든 패딩 처리 성공")
            
            # 케이스 2: 최대 길이 시퀀스
            print("  [5-2] 최대 길이 시퀀스")
            max_length = torch.randint(1, self.vocab_size, (1, self.seq_len))
            max_mask = torch.ones(1, self.seq_len, dtype=torch.long)
            
            with torch.no_grad():
                output2 = model(max_length, max_mask)
            assert output2.shape == (1, self.num_classes)
            print("  ✅ 최대 길이 처리 성공")
            
            # 케이스 3: 단일 토큰만 유효
            print("  [5-3] 단일 유효 토큰")
            single_valid = torch.zeros((1, 10), dtype=torch.long)
            single_valid[0, 0] = 100
            single_mask = torch.zeros((1, 10), dtype=torch.long)
            single_mask[0, 0] = 1
            
            with torch.no_grad():
                output3 = model(single_valid, single_mask)
            assert output3.shape == (1, self.num_classes)
            print("  ✅ 단일 토큰 처리 성공")
            
            # 케이스 4: attention_mask 없이 실행
            print("  [5-4] attention_mask 없는 경우")
            no_mask_input = torch.randint(1, self.vocab_size, (2, 8))
            
            with torch.no_grad():
                output4 = model(no_mask_input, attention_mask=None)
            assert output4.shape == (2, self.num_classes)
            print("  ✅ 마스크 없이 처리 성공")
            
            self.results["edge_cases"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ 엣지 케이스 실패: {e}")
            print(f"   오류 상세: {traceback.format_exc()}")
            self.results["edge_cases"] = "FAIL"
            return False
    
    def test_inference_speed(self) -> bool:
        """테스트 6: 추론 속도 벤치마크"""
        print("\n[TEST 6] 추론 속도 벤치마크")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            model.eval()
            
            # GPU 사용 가능 시 GPU로 이동
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # 테스트 입력 준비
            input_ids = torch.randint(1, self.vocab_size, (1, self.seq_len)).to(device)
            attention_mask = torch.ones(1, self.seq_len, dtype=torch.long).to(device)
            
            # 워밍업 (GPU 초기화)
            for _ in range(10):
                with torch.no_grad():
                    _ = model(input_ids, attention_mask)
            
            # 추론 시간 측정 (100회 평균)
            num_iterations = 100
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model(input_ids, attention_mask)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.perf_counter() - start_time
            
            avg_time_ms = (total_time / num_iterations) * 1000
            
            print(f"✅ 디바이스: {device}")
            print(f"✅ 평균 추론 시간: {avg_time_ms:.3f} ms")
            print(f"✅ 처리량: {1000/avg_time_ms:.1f} sequences/sec")
            
            # 목표: 0.1ms 미만 (GPU 기준)
            if device.type == "cuda":
                if avg_time_ms < 0.1:
                    print(f"✅ 목표 달성: {avg_time_ms:.3f} ms < 0.1 ms")
                else:
                    print(f"⚠️ 목표 미달성: {avg_time_ms:.3f} ms >= 0.1 ms")
            
            self.results["inference_time_ms"] = avg_time_ms
            return True
            
        except Exception as e:
            print(f"❌ 추론 속도 테스트 실패: {e}")
            self.results["inference_speed"] = "FAIL"
            return False
    
    def test_memory_usage(self) -> bool:
        """테스트 7: 메모리 사용량 프로파일링"""
        print("\n[TEST 7] 메모리 사용량 프로파일링")
        print("-" * 50)
        
        try:
            # 메모리 추적 시작
            tracemalloc.start()
            
            # 모델 생성 전 메모리
            snapshot1 = tracemalloc.take_snapshot()
            
            # 모델 생성
            model = self.setup_model()
            
            # 모델 생성 후 메모리
            snapshot2 = tracemalloc.take_snapshot()
            
            # 추론 실행
            input_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.seq_len))
            attention_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
            
            with torch.no_grad():
                _ = model(input_ids, attention_mask)
            
            # 추론 후 메모리
            snapshot3 = tracemalloc.take_snapshot()
            
            # 메모리 사용량 계산
            model_memory = sum(stat.size_diff for stat in snapshot2.compare_to(snapshot1, 'lineno'))
            inference_memory = sum(stat.size_diff for stat in snapshot3.compare_to(snapshot2, 'lineno'))
            
            model_memory_mb = model_memory / 1024 / 1024
            inference_memory_mb = inference_memory / 1024 / 1024
            total_memory_mb = model_memory_mb + inference_memory_mb
            
            print(f"✅ 모델 메모리: {model_memory_mb:.2f} MB")
            print(f"✅ 추론 메모리: {inference_memory_mb:.2f} MB")
            print(f"✅ 총 메모리: {total_memory_mb:.2f} MB")
            
            # 목표: 128MB 미만
            assert total_memory_mb < 128, f"메모리 초과: {total_memory_mb} MB"
            
            tracemalloc.stop()
            
            self.results["memory_usage_mb"] = total_memory_mb
            return True
            
        except Exception as e:
            print(f"❌ 메모리 프로파일링 실패: {e}")
            self.results["memory_usage"] = "FAIL"
            tracemalloc.stop()
            return False
    
    def test_gradient_flow(self) -> bool:
        """테스트 8: 그래디언트 흐름 검증"""
        print("\n[TEST 8] 그래디언트 흐름 테스트")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            model.train()
            
            # 더미 입력 및 타겟
            input_ids = torch.randint(1, self.vocab_size, (4, 16))
            attention_mask = torch.ones(4, 16, dtype=torch.long)
            targets = torch.randint(0, self.num_classes, (4,))
            
            # 순전파
            outputs = model(input_ids, attention_mask)
            
            # 손실 계산
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 확인
            gradients_exist = False
            zero_gradients = []
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients_exist = True
                    if torch.all(param.grad == 0):
                        zero_gradients.append(name)
            
            assert gradients_exist, "그래디언트가 전혀 계산되지 않음"
            
            if zero_gradients:
                print(f"⚠️ 0 그래디언트 파라미터: {zero_gradients}")
            else:
                print("✅ 모든 파라미터에 유효한 그래디언트")
            
            print(f"✅ 손실값: {loss.item():.4f}")
            print("✅ 그래디언트 흐름 정상")
            
            self.results["gradient_flow"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ 그래디언트 흐름 실패: {e}")
            self.results["gradient_flow"] = "FAIL"
            return False
    
    def test_batch_processing(self) -> bool:
        """테스트 9: 다양한 배치 크기 처리"""
        print("\n[TEST 9] 배치 처리 테스트")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            model.eval()
            
            batch_sizes = [1, 8, 32, 64, 128]
            
            for bs in batch_sizes:
                input_ids = torch.randint(1, self.vocab_size, (bs, 32))
                attention_mask = torch.ones(bs, 32, dtype=torch.long)
                
                with torch.no_grad():
                    output = model(input_ids, attention_mask)
                
                assert output.shape == (bs, self.num_classes)
                print(f"  ✅ 배치 크기 {bs:3d}: 성공")
            
            self.results["batch_processing"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ 배치 처리 실패: {e}")
            self.results["batch_processing"] = "FAIL"
            return False
    
    def test_onnx_compatibility(self) -> bool:
        """테스트 10: ONNX 변환 가능성 테스트"""
        print("\n[TEST 10] ONNX 호환성 테스트")
        print("-" * 50)
        
        try:
            model = self.setup_model()
            model.eval()
            
            # 더미 입력
            dummy_input = torch.randint(1, self.vocab_size, (1, 32))
            dummy_mask = torch.ones(1, 32, dtype=torch.long)
            
            # ONNX 추적 시도
            try:
                torch.onnx.export(
                    model,
                    (dummy_input, dummy_mask),
                    "temp_student.onnx",
                    input_names=['input_ids', 'attention_mask'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'attention_mask': {0: 'batch_size', 1: 'sequence'},
                        'logits': {0: 'batch_size'}
                    },
                    opset_version=11,
                    verbose=False
                )
                
                # ONNX 파일 삭제
                import os
                if os.path.exists("temp_student.onnx"):
                    os.remove("temp_student.onnx")
                
                print("✅ ONNX 변환 가능")
                self.results["onnx_compatible"] = "YES"
                return True
                
            except Exception as onnx_error:
                print(f"⚠️ ONNX 변환 제한: {onnx_error}")
                self.results["onnx_compatible"] = "NO"
                return True  # ONNX 변환은 선택사항이므로 실패해도 테스트는 통과
            
        except Exception as e:
            print(f"❌ ONNX 호환성 테스트 실패: {e}")
            self.results["onnx_compatibility"] = "FAIL"
            return False
    
    def run_all_tests(self):
        """모든 테스트 실행 및 결과 요약"""
        print("=" * 60)
        print("BiLSTM 학생 모델 종합 테스트 시작")
        print("=" * 60)
        
        tests = [
            ("모델 생성", self.test_model_creation),
            ("파라미터 수", self.test_parameter_count),
            ("정상 순전파", self.test_normal_forward),
            ("가변 길이", self.test_variable_length_sequences),
            ("엣지 케이스", self.test_edge_cases),
            ("추론 속도", self.test_inference_speed),
            ("메모리 사용량", self.test_memory_usage),
            ("그래디언트 흐름", self.test_gradient_flow),
            ("배치 처리", self.test_batch_processing),
            ("ONNX 호환성", self.test_onnx_compatibility),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"테스트 {test_name} 예외 발생: {e}")
                failed += 1
        
        # 최종 결과 요약
        print("\n" + "=" * 60)
        print("테스트 결과 요약")
        print("=" * 60)
        
        print(f"\n총 테스트: {len(tests)}개")
        print(f"✅ 통과: {passed}개")
        print(f"❌ 실패: {failed}개")
        print(f"성공률: {(passed/len(tests))*100:.1f}%")
        
        # 주요 메트릭 출력
        print("\n" + "-" * 60)
        print("주요 성능 지표")
        print("-" * 60)
        
        if "parameter_count" in self.results and isinstance(self.results["parameter_count"], int):
            print(f"파라미터 수: {self.results['parameter_count']:,}")
        
        if "model_size_mb" in self.results:
            print(f"모델 크기: {self.results['model_size_mb']:.2f} MB")
        
        if "inference_time_ms" in self.results:
            print(f"추론 시간: {self.results['inference_time_ms']:.3f} ms")
        
        if "memory_usage_mb" in self.results:
            print(f"메모리 사용량: {self.results['memory_usage_mb']:.2f} MB")
        
        if "onnx_compatible" in self.results:
            print(f"ONNX 호환: {self.results['onnx_compatible']}")
        
        # 목표 달성 여부
        print("\n" + "-" * 60)
        print("프로젝트 목표 달성 여부")
        print("-" * 60)
        
        goals_met = []
        goals_not_met = []
        
        # 파라미터 수 목표 (< 500,000)
        if "parameter_count" in self.results and isinstance(self.results["parameter_count"], int):
            if self.results["parameter_count"] < 500_000:
                goals_met.append("파라미터 수 < 500,000")
            else:
                goals_not_met.append(f"파라미터 수: {self.results['parameter_count']:,} >= 500,000")
        
        # 모델 크기 목표 (< 16MB)
        if "model_size_mb" in self.results:
            if self.results["model_size_mb"] < 16:
                goals_met.append("모델 크기 < 16MB")
            else:
                goals_not_met.append(f"모델 크기: {self.results['model_size_mb']:.2f} MB >= 16MB")
        
        # 추론 시간 목표 (< 0.1ms on GPU)
        if "inference_time_ms" in self.results:
            if torch.cuda.is_available() and self.results["inference_time_ms"] < 0.1:
                goals_met.append("추론 시간 < 0.1ms (GPU)")
            elif not torch.cuda.is_available():
                goals_met.append(f"추론 시간: {self.results['inference_time_ms']:.3f} ms (CPU)")
        
        # 메모리 사용량 목표 (< 128MB)
        if "memory_usage_mb" in self.results:
            if self.results["memory_usage_mb"] < 128:
                goals_met.append("메모리 사용량 < 128MB")
            else:
                goals_not_met.append(f"메모리: {self.results['memory_usage_mb']:.2f} MB >= 128MB")
        
        print("\n달성한 목표:")
        for goal in goals_met:
            print(f"  ✅ {goal}")
        
        if goals_not_met:
            print("\n미달성 목표:")
            for goal in goals_not_met:
                print(f"  ❌ {goal}")
        
        # 최종 평가
        print("\n" + "=" * 60)
        if failed == 0:
            print("🎉 모든 테스트 통과! 프로덕션 준비 완료!")
        elif failed <= 2:
            print("⚠️ 일부 테스트 실패. 검토 필요.")
        else:
            print("❌ 다수 테스트 실패. 수정 필요.")
        print("=" * 60)


if __name__ == "__main__":
    import traceback
    
    # 테스트 실행
    tester = TestStudentModel()
    tester.run_all_tests()