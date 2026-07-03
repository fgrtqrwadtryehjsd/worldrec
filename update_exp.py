import json
from pathlib import Path

exp_file = Path("experiments.md")
content = exp_file.read_text(encoding="utf-8")

new_table = """
| 指标 | Baseline | Exp01 | **Exp02** | 备注/变化(vs Exp01) |
|------|----------|-------|-------|------|
| **总分** | 0.6731 | 0.7256 | **0.7910** | 🚀 大幅提升 **+0.0654** |
| 懂物料 | 0.1533 | 0.1226 | **0.1840** | 🚀 恢复并超越 Baseline (+0.0614) |
| 懂用户-1 | 0.0000 | 0.0867 | **0.0514** | ⚠️ 下降 (-0.0353) |
| 懂用户-2 | 0.0055 | 0.0377 | **0.0394** | 稳定/略升 (+0.0017) |
| 懂推荐-1 | 0.0960 | 0.0488 | **0.0384** | ⚠️ 继续下降 (-0.0104) |
| 懂推荐-2 | 0.0544 | 0.0955 | **0.1088** | 🚀 提升 (+0.0133) |
| 懂推荐-3 | 0.1330 | 0.1233 | **0.1260** | 稳定/略升 (+0.0027) |
| 懂推荐-4 | 0.0900 | 0.0900 | **0.1044** | 🚀 提升 (+0.0144) |
| 懂世界 | 0.1409 | 0.1123 | **0.1337** | 🚀 显著恢复 (+0.0214) |
"""

if "| **Exp02** |" not in content:
    # 替换原本空白的评测结果表格
    old_table_start = content.find("### 评测结果", content.find("## Exp02"))
    old_table_start = content.find("| 指标 |", old_table_start)
    old_table_end = content.find("### 📈 Exp02 结果诊断", old_table_start)
    
    if old_table_start != -1 and old_table_end != -1:
        content = content[:old_table_start] + new_table.strip() + "\n\n" + content[old_table_end:]
        exp_file.write_text(content, encoding="utf-8")
        print("Updated experiments.md with Exp02 evaluation results.")
    else:
        print("Could not find the table to replace in experiments.md")
else:
    print("Table already updated in experiments.md")
