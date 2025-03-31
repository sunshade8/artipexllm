#!/usr/bin/env python
"""
예제 데이터 다운로드 스크립트
이 스크립트는 Adaptive RAG Ko 시스템 테스트를 위한 예제 데이터를 다운로드합니다.
"""
import os
import sys
import requests
from pathlib import Path

# 패키지 루트 찾기
script_dir = Path(__file__).parent.absolute()
package_root = script_dir.parent

# 데이터 저장 경로
data_dir = os.path.join(package_root, "data")
os.makedirs(data_dir, exist_ok=True)

# 다운로드할 파일 목록 (URL, 저장 경로)
files_to_download = [
    # 이곳에 실제 다운로드할 파일 목록 추가
    # 예: ("https://example.com/sample1.pdf", "sample1.pdf")
    ("https://www.verywellmind.com/thmb/ouKmyAWLUQQbqn7tkICk1OtVNpc=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/color-psychology-2795824-01-08d07ea8afee48698f7f70b1398703c5.png", "color_psychology_image.png"),
    ("https://www.verywellmind.com/thmb/KWSet8AmMCJKf8lNG2eDazn_Z-U=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/the-color-psychology-of-blue-2795815-5c22e535c9e77c000110ec83.png", "blue_psychology.png"),
]

def download_file(url, destination):
    """파일을 다운로드하여 지정된 경로에 저장합니다."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        full_path = os.path.join(data_dir, destination)
        
        with open(full_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"다운로드 완료: {destination}")
        return True
    
    except Exception as e:
        print(f"다운로드 실패 ({url}): {str(e)}")
        return False

def main():
    print("Adaptive RAG Ko 예제 데이터 다운로드 시작...")
    
    success_count = 0
    for url, destination in files_to_download:
        if download_file(url, destination):
            success_count += 1
    
    print(f"다운로드 완료: {success_count}/{len(files_to_download)} 파일")
    print(f"데이터 저장 경로: {data_dir}")

if __name__ == "__main__":
    main()
