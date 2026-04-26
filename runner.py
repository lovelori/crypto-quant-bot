#!/usr/bin/env python3
"""
GitHub Actions 定时运行入口
每 4h 执行: 拉数据 → 训练 → 推信号 → 飞书推送
"""

import os, sys, json

# 导入主模块 (必须先导入, 才能访问 CONFIG)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypto_quant import *

# ─── 配置: CI模式 (CPU, 快速) ────────────────────────────────────
CONFIG['model_type'] = 'gru'
CONFIG['epochs'] = 100
CONFIG['hidden_dim'] = 32  # 更小模型加速CPU推理
CONFIG['total_candles'] = 5000
CONFIG['device'] = 'cpu'

print(f"CI模式: model={CONFIG['model_type']}, "
      f"epochs={CONFIG['epochs']}, "
      f"device={CONFIG['device']}")

# 从环境变量读取飞书 Webhook
FEISHU_WEBHOOK = os.environ.get('FEISHU_WEBHOOK', '')

def feishu_notify(signals: dict, backtest_result=None):
    """发送信号结果到飞书机器人"""
    if not FEISHU_WEBHOOK:
        print("FEISHU_WEBHOOK not set, skipping notification")
        return

    # 构建卡片消息
    lines = []
    for sym, sig in signals.items():
        lines.append(f"{sym} | {sig['signal']} | {sig['prediction']:+.4f} | {sig['price']:.2f}")

    # 使用飞书消息卡片
    card = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": "🤖 币圈量化信号"},
                "template": "blue"
            },
            "elements": [
                {
                    "tag": "div",
                    "text": {
                        "tag": "lark_md",
                        "content": f"**运行时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
                    }
                },
                {"tag": "hr"},
            ]
        }
    }

    # 每个币种一条
    for sym, sig in signals.items():
        direction_emoji = "🟢" if sig['prediction'] > 0 else "🔴" if sig['prediction'] < 0 else "⚪"
        card_text = (
            f"**{direction_emoji} {sym}**\n"
            f"预测值: `{sig['prediction']:+.4f}`\n"
            f"价格: ${sig['price']:.2f}\n"
            f"信号: {sig['signal']} ({sig['confidence']})"
        )
        card["card"]["elements"].append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": card_text}
        })

    # 添加回测摘要
    if backtest_result:
        card["card"]["elements"].append({"tag": "hr"})
        bt_text = (
            f"**回测绩效 (Walk-Forward)**\n"
            f"近 {backtest_result.num_trades} 笔交易 | "
            f"胜率 {backtest_result.win_rate:.1%} | "
            f"夏普 {backtest_result.sharpe_ratio:.2f} | "
            f"最大回撤 {backtest_result.max_drawdown:.1%}"
        )
        card["card"]["elements"].append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": bt_text}
        })

    # 发送
    try:
        resp = requests.post(FEISHU_WEBHOOK, json=card, timeout=10)
        print(f"  Feishu notify: {resp.status_code}")
    except Exception as e:
        print(f"  Feishu notify failed: {e}")


def run_auto():
    """自动运行: 训练 + 回测 + 信号 + 推送"""
    print("=" * 60)
    print(f"Crypto Quant Auto Run — {datetime.now().isoformat()}")
    print("=" * 60)

    try:
        # 1. 拉数据 + 训练
        print("\n[1] Training...")
        # Reset config for longer training
        CONFIG['total_candles'] = 5000
        main_fit()

        # 2. 生成信号 (所有币种)
        print("\n[2] Generating signals...")
        signals = {}
        for sym in CONFIG['symbols']:
            print(f"\n  --- {sym} ---")
            sig = generate_signal(sym, 'crypto_model.pt', num_candles=300)
            if sig:
                signals[sym] = sig

        # 3. 飞书推送
        print("\n[3] Feishu notification...")
        feishu_notify(signals)

        print("\n✅ Done!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

        # 出错也通知
        if FEISHU_WEBHOOK:
            try:
                card = {
                    "msg_type": "interactive",
                    "card": {
                        "header": {
                            "title": {"tag": "plain_text", "content": "❌ 量化运行失败"},
                            "template": "red"
                        },
                        "elements": [{
                            "tag": "div",
                            "text": {"tag": "lark_md", "content": f"错误: {e}"}
                        }]
                    }
                }
                requests.post(FEISHU_WEBHOOK, json=card, timeout=10)
            except:
                pass
        sys.exit(1)


if __name__ == '__main__':
    run_auto()
