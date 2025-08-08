# UGRP/scripts/prepare_dataset.py

from pathlib import Path
from tqdm import tqdm
import logging

# 로깅 설정: 스크립트 실행에 대한 명확한 기록을 남깁니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def aggregate_can_logs(source_dir: Path, output_path: Path, dataset_name: str):
    """
    지정된 소스 디렉토리 내의 모든 .log 파일을 찾아 하나의 파일로 병합하는 함수.

    :param source_dir: 검색을 시작할 최상위 디렉토리 경로 (Path 객체).
    :param output_path: 병합된 로그를 저장할 파일 경로 (Path 객체).
    :param dataset_name: 로깅을 위한 데이터셋의 이름 (e.g., "Benign", "Attack").
    """
    logging.info(f"'{dataset_name}' 데이터셋 병합 작업을 시작합니다.")
    logging.info(f"소스 디렉토리: {source_dir}")

    # 1. 모든 .log 파일 재귀적으로 검색 (Pathlib 사용)
    # .rglob('*.log')는 지정된 디렉토리와 모든 하위 디렉토리에서 .log 파일을 찾습니다.
    log_files = list(source_dir.rglob('*.log'))

    if not log_files:
        logging.error(f"'{source_dir}'에서 .log 파일을 찾을 수 없습니다.")
        logging.error("specification.md에 정의된 'data/raw/can_mirgu/' 구조를 확인해주십시오.")
        return

    logging.info(f"총 {len(log_files)}개의 '{dataset_name}' 로그 파일을 찾았습니다.")
    
    # 2. 목적지 디렉토리 생성 (필요한 경우)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"병합된 파일 저장 위치: {output_path}")

    # 3. 모든 로그 파일을 하나의 파일로 병합
    # tqdm으로 시각적인 진행 상황 표시
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for log_file in tqdm(log_files, desc=f"{dataset_name} 파일 병합 중"):
                outfile.write(log_file.read_text(encoding='utf-8'))
                outfile.write('\n') # 파일 간 구분을 위한 줄바꿈
        
        logging.info(f"'{dataset_name}' 데이터셋 병합 완료! 성공적으로 '{output_path}'에 저장되었습니다.")
    except IOError as e:
        logging.error(f"파일 쓰기 중 오류 발생: {e}")

def main():
    """
    프로젝트 데이터 준비 스크립트의 메인 실행 함수.
    Benign 데이터와 Attack 데이터를 각각 별도의 파일로 병합합니다.
    """
    # specification.md 5.1절에 명시된 프로젝트 디렉토리 구조를 따릅니다.
    BASE_RAW_DIR = Path("data/raw/can_mirgu")
    PROCESSED_DIR = Path("data/processed")

    # 처리할 데이터셋 정보: (소스 하위 디렉토리, 결과 파일명, 데이터셋 이름)
    datasets_to_process = [
        ("Benign", "can_mirgu_benign.log", "Benign (정상 주행)"),
        ("Attack", "can_mirgu_attack.log", "Attack (공격)")
    ]

    logging.info("="*50)
    logging.info("UGRP 데이터셋 준비 스크립트를 시작합니다.")
    logging.info("="*50)

    for sub_dir, out_file, name in datasets_to_process:
        source_directory = BASE_RAW_DIR / sub_dir
        output_file_path = PROCESSED_DIR / out_file
        aggregate_can_logs(source_directory, output_file_path, name)
        logging.info("-" * 50)

    logging.info("모든 데이터 준비 작업이 완료되었습니다.")


if __name__ == '__main__':
    main()