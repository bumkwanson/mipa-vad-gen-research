"""여러 파일을 받아서 transcript 길이를 비교, 대사 많은 것 선별"""
import sys, json, os
sys.path.insert(0, '/home/son/seamless_interaction/src')
from seamless_interaction.fs import SeamlessInteractionFS, DatasetConfig
from pathlib import Path

config = DatasetConfig(
    label='improvised',
    split='dev',
    local_dir=Path.home() / 'vad-gen/data/seamless',
)
fs = SeamlessInteractionFS(config=config)

# 후보 8개 랜덤 샘플해서 실제 다운로드
candidates = fs.sample_random_file_ids(num_samples=8)
print(f"{len(candidates)}개 다운로드 후 transcript 비교...\n")

results = []
for fid in candidates:
    try:
        fs.gather_file_id_data_from_s3(fid)
        # 받은 json에서 transcript 확인
        paths = fs.get_path_list_for_file_id_local(fid)
        json_path = [p for p in paths if p.endswith('.json')]
        if not json_path:
            continue
        with open(json_path[0]) as f:
            meta = json.load(f)
        transcript = meta.get('metadata:transcript', [])
        word_count = sum(len(seg.get('words', [])) for seg in transcript)
        full_text = ' '.join(seg.get('transcript', '') for seg in transcript)
        results.append((fid, word_count, full_text))
        print(f"{fid}: 단어 {word_count}개")
    except Exception as e:
        print(f"{fid}: 실패 ({e})")

results.sort(key=lambda x: -x[1])
print("\n=== 대사 많은 순 ===")
for fid, wc, text in results:
    preview = text[:80] + ('...' if len(text) > 80 else '')
    print(f"\n[{fid}] 단어 {wc}개")
    print(f"  \"{preview}\"")

if results:
    print(f"\n>>> 추천: {results[0][0]} (단어 {results[0][1]}개)")
