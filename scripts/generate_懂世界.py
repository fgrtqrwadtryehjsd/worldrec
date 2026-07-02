"""把 baseline convertv2.py 输出的 Alpaca JSONL 转成 dataset 兼容格式。

一步到位：parquet → Alpaca JSONL (convertv2.py) → dataset 格式

dataset 格式：[{"system": "...", "prompt": "...", "response": "..."}]
"""
import sys, json, subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Step 1: convertv2.py parquet → Alpaca JSONL
alpaca_path = Path('dataset/懂世界_alpaca.jsonl')
alpaca_path.parent.mkdir(exist_ok=True)

cmd = [
    sys.executable, 'baseline/demo/convertv2.py',
    '--input', 'baseline/data/OneReason_General/',
    '--output', str(alpaca_path),
    '--summary', 'dataset/懂世界_summary.json',
]
print('Step 1: parquet → Alpaca JSONL')
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.stdout.decode('utf-8', errors='replace')[-2000:])

# Step 2: Alpaca → dataset 格式
dst_path = Path('dataset/懂世界.jsonl')
total = multi_turn = empty_sys = 0

print('Step 2: Alpaca → dataset 格式')
with open(alpaca_path, encoding='utf-8') as fin, open(dst_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        alpaca = json.loads(line)
        system = alpaca.get('instruction', '') or ''
        prompt = alpaca.get('input', '') or ''
        response = alpaca.get('output', '') or ''
        history = alpaca.get('history', []) or []

        if not system:
            empty_sys += 1
        if history:
            multi_turn += 1
            prefix = []
            for h_user, h_assistant in history:
                prefix.append(f'用户：{h_user}')
                prefix.append(f'助手：{h_assistant}')
            prompt = '\n'.join(prefix) + '\n用户：' + prompt

        sample = {'system': system, 'prompt': prompt, 'response': response}
        fout.write(json.dumps([sample], ensure_ascii=False) + '\n')
        total += 1

alpaca_path.unlink()  # 删除中间文件
print(f'完成: {total} 条 (多轮 {multi_turn}, 无system {empty_sys})')
print(f'输出: {dst_path} ({dst_path.stat().st_size / 1024**3:.2f} GB)')
