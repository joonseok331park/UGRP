# scripts/aggregate_data.py
import os
import glob
from tqdm import tqdm

def aggregate_logs():
    """
    연구원님의 CAN-MIRGU 데이터셋 구조에 맞춰,
    여러 하위 폴더에 나뉘어 있는 모든 .log 파일을 찾아 하나의 파일로 병합합니다.
    """
    # 1. 데이터 소스 경로 설정
    # 연구원님께서 보여주신 'CAN-MIRGU(train)/Benign/' 폴더를 정확히 가리킵니다.
    # 이 스크립트는 'can-ids-project' 폴더 내에서 실행되므로 상대 경로를 사용합니다.
    source_dir = 'dataset/CAN-MIRGU(train)/Benign/'

    # 2. 병합된 파일이 저장될 목적지 경로 설정
    output_dir = 'data/HCRL_dataset/'
    output_filename = 'train_aggregated.log'
    output_path = os.path.join(output_dir, output_filename)

    # 목적지 폴더가 없는 경우, 안전하게 생성합니다.
    os.makedirs(output_dir, exist_ok=True)

    # 3. 모든 .log 파일 검색
    # glob.glob과 recursive=True 옵션을 사용합니다.
    # 이렇게 하면 'source_dir' 아래의 모든 하위 폴더(Day_1, Day_2 등)를 전부 탐색하여
    # .log로 끝나는 모든 파일의 경로를 리스트로 가져옵니다.
    search_pattern = os.path.join(source_dir, '**/*.log')
    log_files = glob.glob(search_pattern, recursive=True)

    if not log_files:
        print(f"오류: '{source_dir}' 디렉토리에서 .log 파일을 찾을 수 없습니다.")
        print("프로젝트 최상위 폴더에 'dataset/CAN-MIRGU(train)/Benign/' 구조로 데이터가 있는지 확인해주세요.")
        return

    print(f"총 {len(log_files)}개의 .log 파일을 찾았습니다. 병합을 시작합니다...")
    print(f"대상 파일: {output_path}")

    # 4. 모든 로그 파일을 하나의 파일로 병합
    # tqdm을 사용하여 진행 상황을 시각적으로 보여줍니다.
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for filename in tqdm(log_files, desc="파일 병합 중"):
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            # 각 파일의 내용이 끝나면 줄바꿈 문자를 추가하여,
            # 파일들이 서로 붙어버리는 문제를 방지합니다.
            outfile.write('\n')

    print(f"\n병합 완료! 모든 데이터가 성공적으로 '{output_path}' 파일에 저장되었습니다.")

if __name__ == '__main__':
    # 이 스크립트를 직접 실행하면 aggregate_logs 함수가 호출됩니다.
    aggregate_logs()