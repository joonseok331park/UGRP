import pandas as pd
from typing import List

def load_can_data(file_path: str, dataset_type: str = 'hcrl') -> pd.DataFrame:
    """
    지정된 파일에서 원시 CAN 버스 데이터를 로드하여 표준화된 Pandas DataFrame으로 변환합니다.

    매개변수:
        file_path (str): 원시 데이터 파일(.log, .csv 등)의 전체 경로.
        dataset_type (str): 데이터셋 유형을 나타내는 문자열. 형식의 미세한 차이를 처리하기 위함. 
                            기본값은 'hcrl' (HCRL/Car-Hacking 데이터셋용).

    반환값:
        pd.DataFrame: 파싱된 CAN 데이터를 담고 있는 표준화된 스키마의 DataFrame.
    """
    # 표준 스키마에 맞춘 데이터 저장을 위한 리스트 초기화
    parsed_data = []

    # 현재는 'hcrl' 타입만 지원
    if dataset_type == 'hcrl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 라인을 공백 기준으로 분리
                    parts = line.strip().split()

                    # 필드 개수가 부족한 경우 건너뛰기 (최소 3개: Timestamp, ID#DLC, Label)
                    if len(parts) < 3:
                        continue

                    # 1. Timestamp 파싱
                    timestamp = float(parts[0])

                    # 2. CAN_ID 및 DLC 파싱
                    can_id_dlc = parts[1].split('#')
                    if len(can_id_dlc) != 2:
                        continue # ID#DLC 형식이 아니면 건너뛰기
                    can_id = can_id_dlc[0]
                    dlc = int(can_id_dlc[1])

                    # 3. Label 파싱
                    label_str = parts[-1]
                    if label_str == 'T':
                        label = 1
                    elif label_str == 'R':
                        label = 0
                    else:
                        # 'T' 또는 'R'이 아니면 유효하지 않은 라인으로 간주하고 건너뛰기
                        continue

                    # 4. Data 파싱 및 패딩
                    # 데이터 필드는 parts[2]부터 parts[-2]까지
                    data = parts[2:-1]
                    
                    # (중요) 명세에 따라 데이터 필드를 8바이트로 패딩
                    padded_data = data + ['00'] * (8 - len(data))
                    
                    # 파싱된 데이터를 딕셔너리 형태로 추가
                    parsed_data.append({
                        'Timestamp': timestamp,
                        'CAN_ID': can_id,
                        'DLC': dlc,
                        'Data': padded_data,
                        'Label': label
                    })
                except (ValueError, IndexError):
                    # 파싱 중 에러 발생 시 (예: float 변환 실패, 인덱스 오류 등)
                    # 해당 라인은 건너뛰고 계속 진행
                    continue
    else:
        # 지원하지 않는 데이터셋 타입에 대한 처리
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    # 리스트로부터 DataFrame 생성
    if not parsed_data:
        # 파싱된 데이터가 없으면 빈 DataFrame 반환
        return pd.DataFrame(columns=['Timestamp', 'CAN_ID', 'DLC', 'Data', 'Label'])

    df = pd.DataFrame(parsed_data)
    
    # 최종 스키마에 맞게 컬럼 순서 및 타입 고정
    schema = {
        'Timestamp': 'float64',
        'CAN_ID': 'str',
        'DLC': 'int32',
        'Data': 'object', # 리스트를 담기 위해 object 타입 사용
        'Label': 'int32'
    }
    df = df.astype(schema)
    
    return df[['Timestamp', 'CAN_ID', 'DLC', 'Data', 'Label']]

if __name__ == '__main__':
    # 스크립트를 직접 실행할 때를 위한 테스트 코드
    import os

    # 가상의 HCRL 데이터 파일 생성
    dummy_data = [
        "1478191234.567890 0545#4 d8 00 00 8b T",
        "1478191234.567990 018F#8 00 00 00 00 00 00 00 00 R",
        "invalid line", # 잘못된 형식의 라인
        "1478191234.568090 0333#2 11 22 T",
        "1478191234.568190 0444#8 11 22 33 44 55 66 77 88 R",
        "1478191234.568290 0555#0 T", # 데이터가 없는 경우
        "1478191234.568390 0666#1 1a G", # 잘못된 레이블
    ]
    dummy_file_path = 'dummy_can_data.log'
    with open(dummy_file_path, 'w', encoding='utf-8') as f:
        for line in dummy_data:
            f.write(line + '\n')

    print(f"'{dummy_file_path}' 파일을 생성하여 테스트를 시작합니다.")
    
    # 함수 테스트
    try:
        can_df = load_can_data(dummy_file_path)
        print("\n[성공] CAN 데이터를 성공적으로 로드하고 파싱했습니다:")
        print(can_df)
        
        print("\n[정보] DataFrame 정보:")
        can_df.info()
        
        if not can_df.empty:
            print("\n[확인] 첫 번째 행의 'Data' 필드 (패딩 확인):")
            print(can_df.iloc[0]['Data'])
        
    except Exception as e:
        print(f"\n[에러] 테스트 중 오류가 발생했습니다: {e}")
    
    finally:
        # 테스트 파일 삭제
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)
            print(f"\n테스트 완료 후 '{dummy_file_path}' 파일을 삭제했습니다.")