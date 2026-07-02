import sys, json, random, re
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
BASE = Path('baseline/data')
OUT = Path('baseline_reconstructed')
OUT.mkdir(exist_ok=True)

random.seed(42)

# ===== 加载映射表（采样足够覆盖即可）=====
print('Loading pid2sid ...')
pid2sid = {}
for f in sorted((BASE / 'OneReason_Pid2Sid').glob('*.parquet')):
    df = pd.read_parquet(f)
    for _, row in df.iterrows():
        pid2sid[(int(row['pid']), row['domain'])] = [int(x) for x in row['sid_three']]
print(f'  pid2sid entries: {len(pid2sid)}')

print('Loading pid2caption ...')
pid2cap = {}
for f in sorted((BASE / 'OneReason_Pid2Caption').glob('*.parquet')):
    df = pd.read_parquet(f)
    for _, row in df.iterrows():
        pid2cap[(int(row['pid']), row['domain'])] = row['caption']
print(f'  pid2cap entries: {len(pid2cap)}')

def sid_token(domain, sid):
    prefix = {'goods':'<|prod_begin|>', 'video/video':'<|video_begin|>',
              'video/ad':'<|ad_begin|>', 'live':'<|living_begin|>'}.get(domain, '')
    return f"{prefix}<s_a_{sid[0]}><s_b_{sid[1]}><s_c_{sid[2]}>"

# ===== 1. 懂物料：desc <-> token 双向 =====
print('Generating 懂物料 ...')
懂物料 = []
# part1-4: desc -> token
candidates = list(pid2cap.items())
random.shuffle(candidates)
for (pid, domain), caption in candidates[:2000]:
    if (pid, domain) not in pid2sid: continue
    token = sid_token(domain, pid2sid[(pid, domain)])
    domain_cn = {'goods':'商品', 'live':'主播', 'video/ad':'广告', 'video/video':'短视频'}[domain]
    懂物料.append({
        'system': f'作为{domain_cn}标识生成助手，你需要根据给定的{domain_cn}描述输出匹配的{domain_cn}token。',
        'prompt': f'下面是一段{domain_cn}描述，请返回匹配的{domain_cn}token：{caption}',
        'response': f'<think>{caption[:50]}</think>\n{token}'
    })
# part5-7: token -> desc
for (pid, domain), caption in candidates[:2000]:
    if (pid, domain) not in pid2sid: continue
    token = sid_token(domain, pid2sid[(pid, domain)])
    domain_cn = {'goods':'商品', 'live':'直播', 'video/ad':'广告', 'video/video':'短视频'}[domain]
    if domain == 'goods':
        system = '你是一个智能商品解说助手，可以根据输入的商品token创建生动、准确的商品描述。'
    elif domain == 'video/ad':
        system = '你是一名广告内容理解助手，请根据给定的广告token生成准确、自然的广告内容描述。'
    elif domain == 'live':
        system = '你是一名直播内容理解助手，请根据给定的直播token生成准确、自然的直播内容描述。'
    else:
        system = '作为视频内容解析助手，你需要根据提供的短视频标识给出精准的视频内容描述。'
    懂物料.append({
        'system': system,
        'prompt': f'给定{domain_cn}token{token}这件{domain_cn}有什么特点？/think',
        'response': f'<think>{caption[:50]}</think>\n{caption}'
    })

# ===== 2. 懂推荐 / 懂用户：UserProfile =====
print('Generating 懂推荐/懂用户 from UserProfile ...')
懂推荐 = []
懂用户 = []
up_files = sorted((BASE / 'OneReason_UserProfile').glob('*.parquet'))
for f in up_files[:1]:  # 只用一个 part，约 5万 用户
    df = pd.read_parquet(f)
    for _, row in df.iterrows():
        # 直播历史
        live_pids = row.get('live_hist_author_id_list', []) or []
        live_tokens = []
        for pid in live_pids[:10]:
            if (pid, 'live') in pid2sid:
                live_tokens.append(sid_token('live', pid2sid[(pid, 'live')]))
        # 视频历史
        video_pids = row.get('video_sampled_pid_list', []) or []
        video_tokens = []
        for pid in video_pids[:10]:
            if (pid, 'video/video') in pid2sid:
                video_tokens.append(sid_token('video/video', pid2sid[(pid, 'video/video')]))
        # 电商历史
        ec_pids = row.get('ec_good_click_item_id_list_extend', []) or []
        ec_tokens = []
        for pid in ec_pids[:10]:
            if (pid, 'goods') in pid2sid:
                ec_tokens.append(sid_token('goods', pid2sid[(pid, 'goods')]))

        if not (live_tokens or video_tokens or ec_tokens):
            continue

        # 懂用户：历史 -> 下一个推荐
        all_tokens = live_tokens + video_tokens + ec_tokens
        if len(all_tokens) >= 2:
            懂用户.append({
                'system': '',
                'prompt': '【用户交互历史】：\n' + '\n'.join([f'  --:-- [行为] {t}' for t in all_tokens[:-1]]) + '\n请预测该用户下一个可能交互的内容。',
                'response': f'<think>\n</think>\n{json.dumps([all_tokens[-1]])}'
            })

        # 懂推荐：多域行为 -> 兴趣分析
        lines = []
        if live_tokens: lines.append(f'用户直播行为: 观看了 {", ".join(live_tokens[:5])}')
        if video_tokens: lines.append(f'用户视频行为: 深度观看了 {", ".join(video_tokens[:5])}')
        if ec_tokens: lines.append(f'用户购物行为: 浏览了商品 {", ".join(ec_tokens[:5])}')
        懂推荐.append({
            'system': '你负责根据用户多域行为理解用户兴趣偏好，并输出该用户在各场景中的目标内容。',
            'prompt': '以下是一个用户的多域历史行为信息：\n' + '\n'.join(lines),
            'response': f'<think>根据用户多域行为，偏好相关直播/视频/商品内容。</think>\n该用户最近喜欢的内容有: {all_tokens[-1]}'
        })

        if len(懂用户) >= 200 and len(懂推荐) >= 200:
            break

# ===== 保存 =====
for name, samples in [('懂物料_recon', 懂物料), ('懂用户_recon', 懂用户), ('懂推荐_recon', 懂推荐)]:
    with open(OUT / f'{name}.jsonl', 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps([s], ensure_ascii=False) + '\n')
    print(f'{name}: {len(samples)} samples')

print('Done')
