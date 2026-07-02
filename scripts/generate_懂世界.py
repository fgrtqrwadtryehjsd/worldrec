import sys
from pathlib import Path
import subprocess

out = Path('dataset/懂世界.jsonl')
out.parent.mkdir(exist_ok=True)

cmd = [
    sys.executable, 'baseline/demo/convertv2.py',
    '--input', 'baseline/data/OneReason_General/',
    '--output', str(out),
    '--report',
    '--summary', 'dataset/懂世界_summary.json',
]

print('Running:', ' '.join(cmd))
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
out_text = result.stdout.decode('utf-8', errors='replace')
print(out_text)
print('Return code:', result.returncode)

if out.exists():
    print(f'Output: {out} size={out.stat().st_size / 1024**3:.2f} GB')
    lines = sum(1 for _ in open(out, encoding='utf-8'))
    print(f'Lines: {lines}')
