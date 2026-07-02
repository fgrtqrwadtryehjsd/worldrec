"""从 baseline 原始数据组装训练样本（懂物料 / 懂用户 / 懂推荐）。

数据源：
  baseline/data/OneReason_Pid2Sid     PID → 三段语义 ID
  baseline/data/OneReason_Pid2Caption PID → 文本描述
  baseline/data/OneReason_UserProfile 用户多域行为序列

输出（与 dataset 格式一致）：
  dataset_reconstructed/懂物料_ext.jsonl
  dataset_reconstructed/懂用户_ext.jsonl
  dataset_reconstructed/懂推荐_ext.jsonl

用法：python scripts/generate_from_baseline.py [--max-users N] [--max-materials N]
"""
import sys, json, random, argparse
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path('baseline/data')
OUT = Path('dataset_reconstructed')
OUT.mkdir(exist_ok=True)

DOMAIN_PREFIX = {
    'goods':       '<|prod_begin|>',
    'live':        '<|living_begin|>',
    'video/video':'<|video_begin|>',
    'video/ad':    '<|ad_begin|>',
}
DOMAIN_CN = {
    'goods':       '商品',
    'live':        '主播',
    'video/video':'短视频',
    'video/ad':    '广告',
}


def sid_token(domain, sid):
    p = DOMAIN_PREFIX.get(domain, '')
    return f"{p}<s_a_{int(sid[0])}><s_b_{int(sid[1])}><s_c_{int(sid[2])}>"


def load_pid2sid(max_files=None):
    pid2sid = {}
    files = sorted((BASE / 'OneReason_Pid2Sid').glob('*.parquet'))
    if max_files: files = files[:max_files]
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            pid2sid[(int(row['pid']), row['domain'])] = list(row['sid_three'])
    print(f'  pid2sid: {len(pid2sid)} entries')
    return pid2sid


def load_pid2caption(max_files=None):
    pid2cap = {}
    files = sorted((BASE / 'OneReason_Pid2Caption').glob('*.parquet'))
    if max_files: files = files[:max_files]
    for f in files:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            pid2cap[(int(row['pid']), row['domain'])] = row['caption']
    print(f'  pid2caption: {len(pid2cap)} entries')
    return pid2cap


def summarize_caption(caption, max_len=50):
    """从 caption 提取简短关键词摘要，放在 think 块里。"""
    # 简单提取：取前几个分句
    for sep in ['。', '；', '，', ' ']:
        if sep in caption:
            parts = caption.split(sep)[:3]
            return sep.join(parts)[:max_len]
    return caption[:max_len]


def gen_materials(pid2sid, pid2cap, max_per_task=5000):
    """组装懂物料 part1-7。"""
    rng = random.Random(42)
    candidates = [(k, v) for k, v in pid2cap.items() if k in pid2sid]
    rng.shuffle(candidates)

    samples = []

    # part1-4: desc -> token
    systems = {
        'goods':       '作为商品标识生成助手，你需要根据给定的商品描述输出匹配的商品token。',
        'live':        '作为主播标识生成助手，你需要根据给定的主播描述输出匹配的主播token。',
        'video/ad':    '你擅长根据广告内容、风格和主题描述，输出对应的广告token。',
        'video/video':'作为短视频标识生成助手，你需要根据给定的短视频描述输出匹配的短视频token。',
    }
    prompts = {
        'goods':       '下面是一段商品描述，请返回匹配的商品token：',
        'live':        '请从以下主播描述中推断并生成对应的主播token：',
        'video/ad':    '请根据这段广告dense caption生成匹配的广告token：',
        'video/video':'基于这段短视频描述，生成最匹配的短视频token：',
    }
    for (pid, domain), caption in candidates[:max_per_task]:
        token = sid_token(domain, pid2sid[(pid, domain)])
        summary = summarize_caption(caption)
        samples.append({
            'system': systems[domain],
            'prompt': f'{prompts[domain]}{caption}/think',
            'response': f'imd{summary}',
        })

    # part5-7: token -> desc
    systems_rev = {
        'goods':       '你是一个智能商品解说助手，可以根据输入的商品token创建生动、准确的商品描述。',
        'live':        '你是一名主播内容理解助手，请根据给定的主播token生成准确、自然的主播内容描述。',
        'video/ad':    '你是一名广告内容理解助手，请根据给定的广告token生成准确、自然的广告内容描述。',
        'video/video':'作为视频内容解析助手，你需要根据提供的短视频标识给出精准的视频内容描述。',
    }
    for (pid, domain), caption in candidates[:max_per_task]:
        token = sid_token(domain, pid2sid[(pid, domain)])
        cn = DOMAIN_CN[domain]
        summary = summarize_caption(caption)
        samples.append({
            'system': systems_rev[domain],
            'prompt': f'给定{cn}token{token}这件{cn}有什么特点？/think',
            'response': f'imd{summary}',
        })

    return samples


def gen_user_and_rec(pid2sid, max_users=10000):
    """从 UserProfile 组装懂用户 + 懂推荐。"""
    up_files = sorted((BASE / 'OneReason_UserProfile').glob('*.parquet'))
    rng = random.Random(42)

    懂用户 = []
    懂推荐 = []

    for f in up_files:
        df = pd.read_parquet(f, columns=[
            'live_hist_author_id_list', 'live_hist_timestamp_list',
            'video_sampled_pid_list', 'video_ts_list',
            'ec_good_click_item_id_list_extend',
            'outer_loop_history_action_pid_list_click',
        ])
        for _, row in df.iterrows():
            def safe_list(col):
                v = row.get(col)
                if v is None or (hasattr(v, '__len__') and len(v) == 0):
                    return []
                return list(v)

            # 直播
            live_pids = safe_list('live_hist_author_id_list')
            live_ts   = safe_list('live_hist_timestamp_list')
            live_tokens = []
            for pid in live_pids[:20]:
                if (pid, 'live') in pid2sid:
                    live_tokens.append(sid_token('live', pid2sid[(pid, 'live')]))

            # 视频
            video_pids = safe_list('video_sampled_pid_list')
            video_tokens = []
            for pid in video_pids[:20]:
                if (pid, 'video/video') in pid2sid:
                    video_tokens.append(sid_token('video/video', pid2sid[(pid, 'video/video')]))

            # 电商
            ec_pids = safe_list('ec_good_click_item_id_list_extend')
            ec_tokens = []
            for pid in ec_pids[:20]:
                if (pid, 'goods') in pid2sid:
                    ec_tokens.append(sid_token('goods', pid2sid[(pid, 'goods')]))

            # 广告
            ad_pids = safe_list('outer_loop_history_action_pid_list_click')
            ad_tokens = []
            for pid in ad_pids[:20]:
                if (pid, 'video/ad') in pid2sid:
                    ad_tokens.append(sid_token('video/ad', pid2sid[(pid, 'video/ad')]))

            all_tokens = live_tokens + video_tokens + ec_tokens + ad_tokens
            if len(all_tokens) < 3:
                continue

            # === 懂用户：历史 -> 下一个预测 ===
            events = []
            for i, (ts, tok) in enumerate(zip(live_ts, live_tokens)):
                events.append((str(ts), '直播-关注', tok))
            for i, tok in enumerate(video_tokens):
                events.append((f'video_{i}', '视频-观看', tok))
            for i, tok in enumerate(ec_tokens):
                events.append((f'ec_{i}', '商品-购买', tok))
            for i, tok in enumerate(ad_tokens):
                events.append((f'ad_{i}', '广告-点击', tok))

            rng.shuffle(events)

            history_lines = []
            for ts, action, tok in events[:-1]:
                history_lines.append(f'【{ts}】\n  --:-- [{action}] {tok}')
            target = events[-1][2] if events else all_tokens[-1]

            懂用户.append({
                'system': '',
                'prompt': '【用户交互历史】：\n' + '\n'.join(history_lines[:20]),
                'response': f'imd',
            })

            # === 懂推荐：多域行为 -> 兴趣分析 ===
            lines = []
            if live_tokens:
                lines.append(f'用户在直播域: 关注了主播 {", ".join(live_tokens[:8])}')
            if video_tokens:
                lines.append(f'用户视频行为: 深度观看了 {", ".join(video_tokens[:8])}')
            if ec_tokens:
                lines.append(f'用户购物行为: 浏览了商品 {", ".join(ec_tokens[:8])}')
            if ad_tokens:
                lines.append(f'用户广告行为: 点击了 {", ".join(ad_tokens[:5])}')

            # 推荐目标：从各域各取 1-2 个
            rec_targets = []
            if live_tokens:  rec_targets.append(live_tokens[0])
            if video_tokens: rec_targets.append(video_tokens[0])
            if ec_tokens:    rec_targets.append(ec_tokens[0])
            rng.shuffle(rec_targets)

            懂推荐.append({
                'system': '你负责根据用户多域行为理解用户兴趣偏好，并输出该用户在各场景中的目标内容。',
                'prompt': '以下是一个用户的多域历史行为信息：\n' + '\n'.join(lines),
                'response': f'imd根据用户多域行为分析，推荐相关内容。',
            })

            if len(懂用户) >= max_users and len(懂推荐) >= max_users:
                break
        if len(懂用户) >= max_users and len(懂推荐) >= max_users:
            break

    return 懂用户, 懂推荐


def save_jsonl(samples, path):
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps([s], ensure_ascii=False) + '\n')
    print(f'  {path.name}: {len(samples)} samples ({path.stat().st_size / 1024**2:.1f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-users', type=int, default=20000, help='每个用户类任务最大样本数')
    ap.add_argument('--max-materials', type=int, default=10000, help='每个物料子任务最大样本数')
    ap.add_argument('--max-sid-files', type=int, default=None, help='pid2sid 加载文件数限制')
    ap.add_argument('--max-cap-files', type=int, default=None, help='pid2caption 加载文件数限制')
    args = ap.parse_args()

    print('Loading pid2sid ...')
    pid2sid = load_pid2sid(args.max_sid_files)
    print('Loading pid2caption ...')
    pid2cap = load_pid2caption(args.max_cap_files)

    print('\nGenerating 懂物料 ...')
    materials = gen_materials(pid2sid, pid2cap, args.max_materials)
    save_jsonl(materials, OUT / '懂物料_ext.jsonl')

    print('\nGenerating 懂用户 + 懂推荐 ...')
    懂用户, 懂推荐 = gen_user_and_rec(pid2sid, args.max_users)
    save_jsonl(懂用户, OUT / '懂用户_ext.jsonl')
    save_jsonl(懂推荐, OUT / '懂推荐_ext.jsonl')

    total = len(materials) + len(懂用户) + len(懂推荐)
    print(f'\nDone! Total {total} samples in {OUT}/')


if __name__ == '__main__':
    main()
