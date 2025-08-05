# utils/data_loader.py (candump 형식을 지원하는 최종 버전)
import pandas as pd
import re

def parse_candump_line(line):
    """'candump' 형식의 로그 한 줄을 파싱합니다."""
    try:
        # 정규표현식을 사용하여 '(timestamp) canX ID#PAYLOAD' 형식 추출
        match = re.match(r'\((\d+\.\d+)\)\s+\w+\s+([0-9A-F]+)#([0-9A-F]*)', line)
        if not match:
            return None

        timestamp = float(match.group(1))
        can_id = match.group(2)
        payload_str = match.group(3)

        # Payload 문자열을 2글자씩 나누어 데이터 리스트 생성
        data = [payload_str[i:i+2] for i in range(0, len(payload_str), 2)]
        dlc = len(data)
        
        # Benign 데이터이므로 라벨은 0
        label = 0

        # 데이터 필드를 8바이트로 패딩
        padded_data = data + ['00'] * (8 - dlc)
        
        return {
            'Timestamp': timestamp, 'CAN_ID': can_id, 'DLC': dlc,
            'Data': padded_data, 'Label': label
        }
    except (ValueError, IndexError):
        return None

def load_can_data(file_path: str, dataset_type: str = 'candump') -> pd.DataFrame:
    """
    지정된 파일에서 원시 CAN 버스 데이터를 로드하여 표준화된 Pandas DataFrame으로 변환합니다.
    'candump' 형식을 기본으로 지원합니다.
    """
    parsed_data = []
    print(f"Loading data from: {file_path} (type: {dataset_type})")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if dataset_type == 'candump':
                    parsed_line = parse_candump_line(line)
                    if parsed_line:
                        parsed_data.append(parsed_line)
                # (참고) 이전에 만들었던 다른 형식 파서도 여기에 추가할 수 있습니다.
                # elif dataset_type == 'hcrl':
                #    ...
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return pd.DataFrame()

    if not parsed_data:
        print(f"경고: '{file_path}' 파일에서 유효한 CAN 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()
        
    df = pd.DataFrame(parsed_data)
    schema = {'Timestamp': 'float64', 'CAN_ID': 'str', 'DLC': 'int32', 'Data': 'object', 'Label': 'int32'}
    df = df.astype(schema)
    return df[['Timestamp', 'CAN_ID', 'DLC', 'Data', 'Label']]