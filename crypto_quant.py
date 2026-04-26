#!/usr/bin/env python3
"""
币圈量化交易系统 v2
====================
4h OHLCV + 技术指标 → Transformer/GRU → 预测涨跌[-1,1]

新特性:
  - 技术指标特征 (RSI, MACD, BB% B, ATR, OBV)
  - Transformer + GRU 双模型可选
  - Walk-Forward 回测框架 (夏普/索提诺/最大回撤/胜率)
  - 实时交易信号输出
  - 损失: -prediction * actual_return
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    'symbols':        ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'LINKUSDT'],
    'interval':       '4h',
    'seq_len':        50,
    'limit_fetch':    1000,
    'total_candles':  5000,
    'batch_size':     64,
    'epochs':         200,
    'lr':             1e-3,
    'weight_decay':   1e-5,
    'hidden_dim':     64,
    'num_layers':     2,
    'dropout':        0.25,
    'model_type':     'transformer',     # 'gru' | 'transformer'
    'nhead':          4,                  # transformer heads
    'train_ratio':    0.7,
    'val_ratio':      0.15,
    'device':         'cuda' if torch.cuda.is_available() else 'cpu',
    'seed':           42,
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
print(f"设备: {CONFIG['device']}")
print(f"模型类型: {CONFIG['model_type']}")
print(f"配置: {json.dumps(CONFIG, indent=2)}")

# ═══════════════════════════════════════════════════════════════
#  数据获取
# ═══════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str, limit=1000,
                         total=2000) -> pd.DataFrame:
    BASE = 'https://data-api.binance.vision/api/v3/klines'
    all_rows, end_time = [], None
    pbar = tqdm(total=total, desc=f"{symbol}", unit="candles", leave=False)
    while len(all_rows) < total:
        params = {'symbol': symbol, 'interval': interval,
                  'limit': min(limit, total - len(all_rows))}
        if end_time: params['endTime'] = end_time
        resp = requests.get(BASE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data: break
        all_rows.extend(data)
        pbar.update(len(data))
        end_time = data[0][0] - 1
        if len(data) < limit // 2: break
    pbar.close()
    df = pd.DataFrame(all_rows, columns=[
        'open_time','open','high','low','close','volume','close_time',
        'quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df[['open_time','open','high','low','close','volume']]
    for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
    df.sort_values('open_time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ═══════════════════════════════════════════════════════════════
#  合成数据
# ═══════════════════════════════════════════════════════════════

def generate_multi_symbol_data() -> dict:
    params = {
        'BTCUSDT': {'base': 50000, 'vol': 0.012, 'seed': 42},
        'ETHUSDT': {'base': 2800,  'vol': 0.015, 'seed': 123},
        'DOGEUSDT':{'base': 0.12,  'vol': 0.035, 'seed': 456},
        'LINKUSDT':{'base': 14,    'vol': 0.022, 'seed': 789},
    }
    data = {}
    for sym, p in params.items():
        rng = np.random.RandomState(p['seed'])
        price, vol = p['base'], p['vol']
        prices = []
        for i in range(CONFIG['total_candles'] + 10):
            vol = 0.82*vol + 0.12*abs(rng.randn()*p['vol']) + 0.06*p['vol']*0.5
            ret = rng.standard_t(df=3)*vol + 0.00003*np.sin(2*np.pi*i/300)
            if sym == 'DOGEUSDT':
                ret += 0.0001*(1 if (i % 400) < 20 else -0.3 if (i % 800) < 10 else 0)
            price *= (1 + ret)
            prices.append(price)
        prices = np.array(prices)
        n = len(prices)
        data[sym] = pd.DataFrame({
            'open_time': pd.date_range('2020-01-01', periods=n, freq='4h'),
            'open':  prices*(1+rng.randn(n)*0.002),
            'high':  prices*(1+abs(rng.randn(n))*0.005),
            'low':   prices*(1-abs(rng.randn(n))*0.005),
            'close': prices.copy(),
            'volume': np.exp(rng.randn(n)*0.9+10),
        })
    return data

def fetch_data() -> dict:
    mirrors = [
        'https://data-api.binance.vision',
        'https://api.binance.com',
        'https://api1.binance.com',
    ]
    base_url = None
    for m in mirrors:
        try:
            requests.get(f'{m}/api/v3/ping', timeout=5)
            base_url = m
            break
        except Exception:
            continue
    
    if base_url:
        print(f"  Binance OK ({base_url}), pulling real data...")
        return {sym: fetch_binance_klines(sym, CONFIG['interval'],
                  CONFIG['limit_fetch'], CONFIG['total_candles'])
                for sym in CONFIG['symbols']}
    else:
        print(f"  All Binance endpoints unreachable, using synthetic data...")
        return generate_multi_symbol_data()

# ═══════════════════════════════════════════════════════════════
#  技术指标 + 特征工程
# ═══════════════════════════════════════════════════════════════

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加 RSI, MACD, BB%B, ATR, OBV 技术指标"""
    close, high, low, vol = df['close'], df['high'], df['low'], df['volume']

    # — RSI(14) —
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # — MACD(12,26,9) —
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # — BB%B(20,2) —
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_bband'] = bb_mid + 2*bb_std          # upper band
    df['bb_pctb'] = (close - bb_mid) / (bb_mid+2*bb_std - (bb_mid-2*bb_std)).replace(0, np.nan)

    # — ATR(14) —
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # — OBV —
    df['obv'] = (vol * np.sign(close.diff())).fillna(0).cumsum()

    # — Williams %R(14) —
    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    df['williams_r'] = -100 * (hh14 - close) / (hh14 - ll14).replace(0, np.nan)

    # — 成交量变化率 —
    df['vol_ratio'] = vol / vol.rolling(5).mean()

    return df


FEATURE_COLS = [
    'ret_open', 'ret_high', 'ret_low', 'ret_close', 'ret_volume',
    'rsi', 'macd_hist', 'bb_pctb', 'atr', 'obv_norm',
    'williams_r', 'vol_ratio_norm',
]

def preprocess_data(df: pd.DataFrame, seq_len=50, fit_params=None):
    """
    完整预处理: 技术指标 → 对数收益率 → Z-score归一化 → 序列化
    
    Returns:
        X, y, feat_cols, scaler_params (fit_params for inference)
    """
    df = df.copy()
    df = add_technical_indicators(df)

    # 对数收益率 (OHLCV)
    for c in ['open','high','low','close']:
        df[f'ret_{c}'] = np.log(df[c]).diff()
    df['ret_volume'] = np.log(df['volume'].clip(1e-8)).diff()

    # OBV 和 vol_ratio 归一化
    df['obv_norm'] = (df['obv'] - df['obv'].rolling(200).mean()) / df['obv'].rolling(200).std().clip(1e-8)
    df['vol_ratio_norm'] = (df['vol_ratio'] - 1.0) / df['vol_ratio'].rolling(200).std().clip(1e-8)

    # 去掉NaN
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # Z-score 归一化 (滚动窗口)
    normed = pd.DataFrame(index=df.index)
    is_training = fit_params is None
    if is_training:
        fit_params = {}
        for c in FEATURE_COLS:
            mean = df[c].rolling(200, min_periods=50).mean()
            std  = df[c].rolling(200, min_periods=50).std().clip(1e-8)
            normed[c] = (df[c] - mean) / std
            # Save last valid mean/std for inference
            last_valid = mean.last_valid_index()
            if last_valid is not None:
                fit_params[c] = {
                    'mean': float(mean.loc[last_valid]),
                    'std': float(std.loc[last_valid]),
                }
        normed = normed.dropna().reset_index(drop=True)
    else:
        for c in FEATURE_COLS:
            mean, std = fit_params[c]['mean'], fit_params[c]['std']
            normed[c] = (df[c].iloc[-200:] - mean) / std
        normed = normed.iloc[-seq_len:].reset_index(drop=True) if len(normed) >= seq_len else normed

    close_vals = df['close'].values
    if is_training:
        label_idx = normed.index + seq_len
        valid = label_idx < len(close_vals)
        normed = normed.iloc[valid].reset_index(drop=True)
        label_idx = label_idx[valid]
        labels = close_vals[label_idx] / close_vals[label_idx - 1] - 1.0
        labels = np.clip(labels, -0.15, 0.15)
    else:
        labels = None

    arr = normed.values.astype(np.float32)
    if is_training:
        total = len(arr) - seq_len
        X = np.array([arr[i:i+seq_len] for i in range(total)], dtype=np.float32)
        y = np.array(labels[:total], dtype=np.float32)
    else:
        X = arr[-seq_len:].reshape(1, seq_len, -1) if len(arr) >= seq_len else arr.reshape(1, *arr.shape)
        y = None

    return X, y, FEATURE_COLS, fit_params


def get_fit_params_for_inference(fit_params):
    """将拟合参数转换为可序列化格式"""
    serializable = {}
    for c in FEATURE_COLS:
        serializable[c] = {
            'mean': float(fit_params[c]['mean']),
            'std': float(fit_params[c]['std']),
        }
    return serializable

# ═══════════════════════════════════════════════════════════════
#  模型定义 — GRU + Transformer
# ═══════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class CryptoTransformer(nn.Module):
    """Transformer Encoder 序列预测"""
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.25, max_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)                 # → (batch, seq, d_model)
        x = self.pos_enc(x)                    # + positional encoding
        x = self.transformer(x)                # → (batch, seq, d_model)
        x = x.mean(dim=1)                      # global average pooling over time
        x = self.norm(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.tanh(x).squeeze(-1)        # (batch,)


class CryptoGRU(nn.Module):
    """GRU 序列预测"""
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        _, h_n = self.gru(x)
        h = h_n[-1]
        h = self.dropout(h)
        h = self.relu(self.fc1(h))
        h = self.dropout(h)
        return self.tanh(self.fc2(h)).squeeze(-1)


def build_model(input_dim: int) -> nn.Module:
    mt = CONFIG['model_type']
    if mt == 'transformer':
        model = CryptoTransformer(
            input_dim=input_dim,
            d_model=CONFIG['hidden_dim'],
            nhead=CONFIG['nhead'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout'],
        )
    else:
        model = CryptoGRU(
            input_dim=input_dim,
            hidden_dim=CONFIG['hidden_dim'],
            num_layers=CONFIG['num_layers'],
            dropout=CONFIG['dropout'],
        )
    return model


# ═══════════════════════════════════════════════════════════════
#  自定义损失函数
# ═══════════════════════════════════════════════════════════════

class DirectionalProfitLoss(nn.Module):
    """
    Loss = -prediction * actual_return
    方向正确 => 负损失; 方向错误 => 正损失
    """
    def forward(self, pred, actual_ret):
        return -torch.mean(pred * actual_ret)


# ═══════════════════════════════════════════════════════════════
#  训练与评估
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_actuals = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * X_batch.size(0)
        total_samples += X_batch.size(0)
        all_preds.append(pred.cpu())
        all_actuals.append(y_batch.cpu())
    preds = torch.cat(all_preds).numpy()
    actuals = torch.cat(all_actuals).numpy()
    direction_acc = np.mean((preds > 0) == (actuals > 0))
    strat_ret = np.mean(np.sign(preds) * actuals)
    return total_loss / total_samples, direction_acc, strat_ret, preds, actuals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════
#  回测框架 (Walk-Forward)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    entry_time: datetime
    entry_price: float
    direction: int          # 1 = long, -1 = short
    pred_strength: float
    exit_time: datetime
    exit_price: float
    return_pct: float       # realized return including direction
    hold_periods: int

@dataclass
class BacktestResult:
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    num_trades: int = 0
    calmar_ratio: float = 0.0


def run_walkforward_backtest(
    model: nn.Module, X_all: np.ndarray, y_all: np.ndarray,
    df_all: pd.DataFrame, seq_len: int = 50,
    window_size: int = 500, step: int = 50,
    confidence_threshold: float = 0.0,
) -> BacktestResult:
    """
    Walk-Forward 回测:
    - 每 step 步, 用前 window_size 个样本重新训练模型
    - 在接下来的 step 个样本上验证交易
    
    Returns:
        BacktestResult with full metrics
    """
    print(f"\n  Walk-Forward 回测: 窗口={window_size}, 步长={step}, "
          f"置信阈值={confidence_threshold}")
    N = len(X_all)
    result = BacktestResult()
    position = 0          # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    entry_time = None
    entry_pred = 0.0
    entry_idx = 0
    equity = [1.0]        # start at 1.0
    timestamps = [df_all['open_time'].iloc[seq_len-1]]

    # 需要用原始 close 价格来计算实际交易收益
    close_prices = df_all['close'].values

    # 找出所有时间步对应的原始索引
    # y_all[i] 对应从 df 的第 i 步到 i+seq_len 步
    # 交易在 i+seq_len 时开盘

    start_idx = 0
    while start_idx + window_size + step <= N:
        train_slice = slice(start_idx, start_idx + window_size)
        test_slice = slice(start_idx + window_size,
                          min(start_idx + window_size + step, N - 1))

        # 训练
        X_train = torch.FloatTensor(X_all[train_slice])
        y_train = torch.FloatTensor(y_all[train_slice])
        train_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)), batch_size=64, shuffle=True)

        local_model = build_model(X_all.shape[2]).to(CONFIG['device'])
        local_model.load_state_dict(model.state_dict())  # warm-start
        opt = torch.optim.AdamW(local_model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = DirectionalProfitLoss()

        local_model.train()
        for _ in range(5):  # 5 fine-tune epochs per window
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(CONFIG['device']), yb.to(CONFIG['device'])
                opt.zero_grad()
                loss = criterion(local_model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                opt.step()

        # 测试
        local_model.eval()
        X_test = torch.FloatTensor(X_all[test_slice]).to(CONFIG['device'])
        with torch.no_grad():
            preds = local_model(X_test).cpu().numpy()
        actual_rets = y_all[test_slice]

        # 模拟交易
        for t in range(len(preds)):
            global_idx = test_slice.start + t
            pred = preds[t]
            actual_ret = actual_rets[t]

            # 当前K线收盘价 (entry at next candle open ~= close of current candle)
            # 假设我们在当前K线结束时以收盘价开仓
            current_close = close_prices[global_idx + seq_len]

            # 平仓检查
            if position != 0:
                # 如果信号反转或过强反方向, 平仓
                if (position > 0 and pred < 0) or (position < 0 and pred > 0):
                    exit_price = current_close
                    hold = global_idx - entry_idx
                    ret = (exit_price / entry_price - 1) * position
                    result.trades.append(TradeRecord(
                        entry_time=df_all['open_time'].iloc[entry_idx + seq_len],
                        entry_price=float(entry_price),
                        direction=position,
                        pred_strength=float(entry_pred),
                        exit_time=df_all['open_time'].iloc[global_idx + seq_len],
                        exit_price=float(exit_price),
                        return_pct=float(ret),
                        hold_periods=hold,
                    ))
                    equity.append(equity[-1] * (1 + ret))
                    timestamps.append(df_all['open_time'].iloc[global_idx + seq_len])
                    position = 0

            # 开仓检查
            if position == 0 and abs(pred) > confidence_threshold:
                position = 1 if pred > 0 else -1
                entry_price = current_close
                entry_pred = pred
                entry_idx = global_idx

        start_idx += step

    # 平掉最后持仓
    if position != 0:
        exit_price = close_prices[-1]
        ret = (exit_price / entry_price - 1) * position
        result.trades.append(TradeRecord(
            entry_time=df_all['open_time'].iloc[entry_idx + seq_len],
            entry_price=float(entry_price),
            direction=position,
            pred_strength=float(entry_pred),
            exit_time=df_all['open_time'].iloc[-1],
            exit_price=float(exit_price),
            return_pct=float(ret),
            hold_periods=N - entry_idx,
        ))
        equity.append(equity[-1] * (1 + ret))
        timestamps.append(df_all['open_time'].iloc[-1])

    result.equity_curve = list(zip(
        [t.timestamp() for t in timestamps], equity))

    # 计算指标
    result = _calc_backtest_metrics(result, equity, timestamps)
    return result


def _calc_backtest_metrics(result: BacktestResult, equity: list,
                           timestamps: list) -> BacktestResult:
    trades = result.trades
    result.num_trades = len(trades)
    if not trades:
        return result

    returns = [t.return_pct for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    result.win_rate = len(wins) / len(trades) if trades else 0
    result.avg_win = np.mean(wins) if wins else 0
    result.avg_loss = np.mean(losses) if losses else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-10
    result.profit_factor = gross_profit / gross_loss

    result.total_return = equity[-1] - 1.0

    # 年化收益 (4h = 2190 bars/year)
    n_bars = len(equity) - 1
    years = n_bars / 2190
    result.annual_return = (equity[-1] ** (1 / years)) - 1 if years > 0 else result.total_return

    # 日收益率 (for Sharpe)
    eq = np.array(equity)
    daily_rets = np.diff(eq) / eq[:-1]
    if len(daily_rets) > 1:
        result.sharpe_ratio = np.sqrt(2190) * np.mean(daily_rets) / (np.std(daily_rets) + 1e-10)
        # Sortino: only downside deviation
        downside = daily_rets[daily_rets < 0]
        result.sortino_ratio = np.sqrt(2190) * np.mean(daily_rets) / (np.std(downside) + 1e-10)

    # 最大回撤
    peak = np.maximum.accumulate(eq)
    drawdown = (eq - peak) / peak
    result.max_drawdown = abs(drawdown.min())
    result.calmar_ratio = result.annual_return / (result.max_drawdown + 1e-10)

    return result


def plot_backtest(result: BacktestResult, symbol: str, save_path: str):
    """回测结果可视化"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12),
                             gridspec_kw={'height_ratios': [3, 1, 1.5]})

    # — 权益曲线 + 回撤 —
    ax = axes[0]
    ts = [datetime.fromtimestamp(t) for t, _ in result.equity_curve]
    eq = [e for _, e in result.equity_curve]
    ax.plot(ts, eq, 'b-', linewidth=1.2, label='Equity')
    # Buy & hold 对比
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, label='Baseline (1.0)')
    # Max drawdown annotation
    peak = np.maximum.accumulate(eq)
    dd = (np.array(eq) - peak) / peak
    dd_peak_idx = np.argmin(dd)
    ax.fill_between(ts, peak, eq, alpha=0.15, color='red', label='Drawdown')
    ax.axvline(x=ts[dd_peak_idx], color='red', linestyle=':', alpha=0.3)
    ax.set_ylabel('Equity')
    ax.set_title(f'{symbol} Walk-Forward Backtest')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # — 回撤曲线 —
    ax = axes[1]
    ax.fill_between(ts, 0, dd*100, alpha=0.4, color='red')
    ax.plot(ts, dd*100, 'r-', linewidth=0.8)
    ax.set_ylabel('Drawdown (%)')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.2)

    # — 交易分布 —
    ax = axes[2]
    trades = result.trades
    if trades:
        win_rets = [t.return_pct*100 for t in trades if t.return_pct > 0]
        loss_rets = [t.return_pct*100 for t in trades if t.return_pct <= 0]
        bins = 30
        if win_rets:
            ax.hist(win_rets, bins=bins, alpha=0.6, color='green', label=f'Wins ({len(win_rets)})')
        if loss_rets:
            ax.hist(loss_rets, bins=bins, alpha=0.6, color='red', label=f'Losses ({len(loss_rets)})')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Trade Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

        # Stats box
        stats_text = (
            f"Total Return: {result.total_return:.1%}\n"
            f"Ann Return: {result.annual_return:.1%}\n"
            f"Sharpe: {result.sharpe_ratio:.2f}\n"
            f"Max DD: {result.max_drawdown:.1%}\n"
            f"Win Rate: {result.win_rate:.1%}\n"
            f"Trades: {result.num_trades}\n"
            f"Profit Factor: {result.profit_factor:.2f}"
        )
        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  实时信号生成
# ═══════════════════════════════════════════════════════════════

def generate_signal(
    symbol: str = 'BTCUSDT',
    model_path: str = 'crypto_model.pt',
    num_candles: int = 250,
    interval: str = '4h',
):
    """
    生成实时交易信号
    
    步骤:
    1. 拉取最新 250 根 4h K线
    2. 计算技术指标
    3. 与训练时相同的预处理
    4. 模型推理 → 输出信号
    """
    # 加载模型
    if not os.path.exists(model_path):
        print(f"  模型文件 {model_path} 不存在, 请先训练")
        return

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    old_config = ckpt.get('config', CONFIG)
    input_dim = ckpt.get('input_dim', len(FEATURE_COLS))

    # 重建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mt = old_config.get('model_type', 'gru')
    if mt == 'transformer':
        model = CryptoTransformer(input_dim=input_dim,
            d_model=old_config.get('hidden_dim', 64),
            nhead=old_config.get('nhead', 4),
            num_layers=old_config.get('num_layers', 2),
            dropout=old_config.get('dropout', 0.25))
    else:
        model = CryptoGRU(input_dim=input_dim,
            hidden_dim=old_config.get('hidden_dim', 64),
            num_layers=old_config.get('num_layers', 2),
            dropout=old_config.get('dropout', 0.25))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    fit_params = ckpt.get('fit_params', None)

    # 获取数据
    if HAS_REQUESTS:
        try:
            df = fetch_binance_klines(symbol, interval, 500, num_candles)
        except Exception as e:
            print(f"  无法获取 {symbol} 数据: {e}")
            # 用合成数据演示
            data_all = generate_multi_symbol_data()
            df = data_all.get(symbol)
            if df is None:
                print("  No data available")
                return
    else:
        df = generate_multi_symbol_data().get(symbol)
        if df is None:
            print("  无可用数据")
            return

    seq_len = old_config.get('seq_len', 50)
    X, _, _, _ = preprocess_data(df, seq_len, fit_params)

    if len(X) == 0:
        print("  预处理后无有效序列")
        return

    # 推理
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X[-1:]).to(device)
        pred = model(X_tensor).cpu().numpy()[0]

    # 当前价格
    last_close = df['close'].iloc[-1]
    last_time = df['open_time'].iloc[-1]
    next_time = last_time + timedelta(hours=4)

    # 输出信号
    print(f"\n{'='*50}")
    print(f"  {symbol} 实时信号")
    print(f"{'='*50}")
    print(f"  当前时间:      {last_time}")
    print(f"  最新收盘价:    {last_close:.2f}")
    print(f"  下一K线:       {next_time}")
    print(f"  ─────────────────────────────")
    print(f"  预测值:        {pred:+.4f}   (范围 [-1, +1])")

    if pred > 0.3:
        signal = "🟢 强烈做多"
        confidence = "高"
    elif pred > 0.1:
        signal = "🟢 温和做多"
        confidence = "中"
    elif pred < -0.3:
        signal = "🔴 强烈做空"
        confidence = "高"
    elif pred < -0.1:
        signal = "🔴 温和做空"
        confidence = "中"
    else:
        signal = "⚪ 观望"
        confidence = "低 (信号弱)"

    print(f"  信号:          {signal}")
    print(f"  置信度:        {confidence}")
    print(f"  信号强度:      {abs(pred):.2f}/1.00")
    print(f"  ─────────────────────────────")

    # 模型历史表现提示
    if 'test_acc' in ckpt:
        print(f"  模型历史准确率: {ckpt['test_acc']:.1%}")
    if 'test_strat_ret' in ckpt:
        print(f"  模型历史均收益: {ckpt['test_strat_ret']:+.4f}")
    print(f"{'='*50}")

    return {
        'symbol': symbol,
        'timestamp': last_time,
        'price': last_close,
        'prediction': float(pred),
        'signal': signal,
        'confidence': confidence,
    }


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

def main_fit():
    """训练 + 评估 + 回测 + 信号 (可被 runner.py 调用)"""
    title = f"币圈量化交易 — {CONFIG['model_type'].upper()}预测系统"
    print("=" * 60)
    print(title)
    print("策略: 50根4h K线 + 技术指标 → 涨跌 [-1,1]")
    print(f"损失: Loss = -prediction × actual_return")
    print(f"特征: {', '.join(FEATURE_COLS)}")
    print("=" * 60)

    all_X, all_y = [], []
    label_info = {}
    all_dfs = {}

    # ── 1. 获取数据 ──
    print("\n[1/5] 拉取数据 + 技术指标计算...")
    all_data = fetch_data()
    for symbol in CONFIG['symbols']:
        df = all_data[symbol]
        all_dfs[symbol] = df
        print(f"  {symbol}: {len(df)} candles  "
              f"{df['open_time'].iloc[0].date()} → {df['open_time'].iloc[-1].date()}")

        X, y, feat_cols, fp = preprocess_data(df, CONFIG['seq_len'])
        print(f"  样本: {len(X)}  (特征: {len(feat_cols)})")
        all_X.append(X)
        all_y.append(y)
        fit_params_from_training = fp  # use last coin's scaler params
        label_info[symbol] = {
            'candles': len(df), 'samples': len(X),
            'mean_ret': float(np.mean(y)), 'std_ret': float(np.std(y)),
            'pos_ratio': float(np.mean(y > 0)),
        }

    print("\n标签统计:")
    for sym, info in label_info.items():
        print(f"  {sym:>8s}: 样本={info['samples']:>5d}  "
              f"均值={info['mean_ret']:+.4f}  std={info['std_ret']:.4f}  "
              f"上涨率={info['pos_ratio']:.1%}")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"\n总样本: {len(X_all)}  输入: (batch, {CONFIG['seq_len']}, {X_all.shape[2]})")

    # ── 6b. 时序划分 (不shuffle — 防止未来数据泄露) ──
    # 每个币种独立做时间划分, 再按split合并
    N = len(all_X[0])  # 每个币种样本数相同
    train_end = int(N * CONFIG['train_ratio'])
    val_end   = int(N * (CONFIG['train_ratio'] + CONFIG['val_ratio']))

    X_train = np.concatenate([all_X[i][:train_end] for i in range(len(all_X))], axis=0)
    y_train = np.concatenate([all_y[i][:train_end] for i in range(len(all_X))], axis=0)
    X_val   = np.concatenate([all_X[i][train_end:val_end] for i in range(len(all_X))], axis=0)
    y_val   = np.concatenate([all_y[i][train_end:val_end] for i in range(len(all_X))], axis=0)
    X_test  = np.concatenate([all_X[i][val_end:] for i in range(len(all_X))], axis=0)
    y_test  = np.concatenate([all_y[i][val_end:] for i in range(len(all_X))], axis=0)
    print(f"划分: 训练={len(X_train)}  验证={len(X_val)}  测试={len(X_test)}")

    # ── 3. 模型 ──
    input_dim = X_all.shape[2]
    model = build_model(input_dim).to(CONFIG['device'])
    print(f"\n[2/5] 模型参数: {count_parameters(model):,}")
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                  weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion = DirectionalProfitLoss()

    train_loader = torch.utils.data.DataLoader(
        list(zip(X_train, y_train)), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        list(zip(X_val, y_val)), batch_size=CONFIG['batch_size']*2)
    test_loader = torch.utils.data.DataLoader(
        list(zip(X_test, y_test)), batch_size=CONFIG['batch_size']*2)

    # ── 4. 训练 ──
    print(f"\n[3/5] 训练 {CONFIG['epochs']} epochs...")
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_strat_ret': []}

    for epoch in range(1, CONFIG['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        val_loss, val_acc, val_strat_ret, _, _ = evaluate(
            model, val_loader, criterion, CONFIG['device'])
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_strat_ret'].append(val_strat_ret)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if epoch == 1 or epoch % 20 == 0 or epoch == CONFIG['epochs']:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{CONFIG['epochs']}  "
                  f"train={train_loss:+.6f}  val={val_loss:+.6f}  "
                  f"acc={val_acc:.1%}  strat={val_strat_ret:+.4f}  "
                  f"lr={lr_now:.2e}")

    # ── 5. 测试评估 ──
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_strat_ret, test_preds, test_actuals = evaluate(
        model, test_loader, criterion, CONFIG['device'])

    print(f"\n[4/5] 测试集结果:")
    print(f"{'='*50}")
    print(f"  测试损失:       {test_loss:+.6f}")
    print(f"  方向准确率:     {test_acc:.1%}")
    print(f"  策略平均收益:   {test_strat_ret:+.4f}")
    print(f"  预测范围:       [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"{'='*50}")

    # 单币种
    print(f"\n单币种性能:")
    for i, sym in enumerate(CONFIG['symbols']):
        loader = torch.utils.data.DataLoader(
            list(zip(all_X[i], all_y[i])), batch_size=CONFIG['batch_size']*2)
        ls, ac, sr, _, _ = evaluate(model, loader, criterion, CONFIG['device'])
        print(f"  {sym:>8s}:  loss={ls:+.6f}  acc={ac:.1%}  strat={sr:+.4f}")

    # ── 6. Walk-Forward 回测 ──
    print(f"\n[5/5] Walk-Forward 回测...")
    # 使用最后一个币种 (LINK) 的数据做回测 — 保持时序完整性
    bt_symbol = CONFIG['symbols'][-1]
    bt_df = all_dfs[bt_symbol]
    bt_X, bt_y, _, _ = preprocess_data(bt_df, CONFIG['seq_len'])
    bt_result = run_walkforward_backtest(
        model, bt_X, bt_y, bt_df,
        seq_len=CONFIG['seq_len'],
        window_size=min(500, len(bt_X)//3),
        step=min(100, len(bt_X)//10),
        confidence_threshold=0.05,
    )

    print(f"\n  {bt_symbol} 回测结果:")
    print(f"  {'─'*40}")
    print(f"  总交易数:       {bt_result.num_trades}")
    print(f"  总收益率:       {bt_result.total_return:+.2%}")
    print(f"  年化收益率:     {bt_result.annual_return:+.2%}")
    print(f"  夏普比率:       {bt_result.sharpe_ratio:.2f}")
    print(f"  索提诺比率:     {bt_result.sortino_ratio:.2f}")
    print(f"  最大回撤:       {bt_result.max_drawdown:.2%}")
    print(f"  卡尔玛比率:     {bt_result.calmar_ratio:.2f}")
    print(f"  胜率:           {bt_result.win_rate:.1%}")
    print(f"  盈亏比:         {bt_result.profit_factor:.2f}")
    print(f"  平均盈利:       {bt_result.avg_win:+.4f}")
    print(f"  平均亏损:       {bt_result.avg_loss:+.4f}")

    plot_backtest(bt_result, bt_symbol, 'crypto_backtest.png')
    print(f"\n  回测图表: crypto_backtest.png")

    # ── 7. 保存模型 + 可视化训练 ──
    plot_training(history, test_preds, test_actuals, bt_result)
    print(f"\n训练图表: crypto_training.png")

    torch.save({
        'model_state_dict': best_model_state,
        'config': CONFIG,
        'input_dim': input_dim,
        'fit_params': fit_params_from_training,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_strat_ret': test_strat_ret,
    }, 'crypto_model.pt')
    print(f"模型: crypto_model.pt")

    # ── 8. 生成实时信号 ──
    print(f"\n{'='*60}")
    print("  实时信号示例")
    print(f"{'='*60}")
    generate_signal(CONFIG['symbols'][0], 'crypto_model.pt')

    print(f"\n{'='*60}")
    print("  完成!")
    print(f"{'='*60}")


def plot_training(history, test_preds, test_actuals, bt_result=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Crypto Quant — {CONFIG["model_type"].upper()} Training', fontsize=14)

    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train', alpha=0.8)
    ax.plot(history['val_loss'], label='Val', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (-pred * ret)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('Loss Curve')

    # Accuracy
    ax = axes[0, 1]
    ax.plot(history['val_acc'], color='green')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3, label='Random')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Direction Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('Validation Accuracy')

    # Strategy Return
    ax = axes[0, 2]
    cum_ret = np.cumsum(history['val_strat_ret'])
    ax.plot(cum_ret, color='purple')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cumulative Strat Ret')
    ax.grid(True, alpha=0.3)
    ax.set_title('Cumulative Strategy Return')

    # Test scatter
    ax = axes[1, 0]
    ax.scatter(test_actuals, test_preds, s=5, alpha=0.4, c='blue')
    lim = max(abs(test_actuals).max(), abs(test_preds).max(), 0.01) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'r--', alpha=0.3, label='Perfect')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel('Actual Return'); ax.set_ylabel('Prediction')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('Test: Prediction vs Actual')

    # Backtest equity
    ax = axes[1, 1]
    if bt_result and bt_result.equity_curve:
        ts = [datetime.fromtimestamp(t) for t, _ in bt_result.equity_curve]
        eq = [e for _, e in bt_result.equity_curve]
        ax.plot(ts, eq, 'b-', linewidth=1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4)
        ax.set_title('Backtest Equity Curve')
        ax.set_ylabel('Equity')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

    # Metrics table
    ax = axes[1, 2]
    ax.axis('off')
    metrics = [
        f"Model: {CONFIG['model_type'].upper()}",
        f"Params: {count_parameters(build_model(len(FEATURE_COLS))):,}",
        f"Test Acc: {history['val_acc'][-1]:.1%}",
        f"Test Strat: {history['val_strat_ret'][-1]:+.4f}",
    ]
    if bt_result:
        metrics += [
            "",
            f"Backtest: {bt_result.num_trades} trades",
            f"Return: {bt_result.total_return:+.1%}",
            f"Sharpe: {bt_result.sharpe_ratio:.2f}",
            f"Max DD: {bt_result.max_drawdown:.1%}",
            f"Win Rate: {bt_result.win_rate:.1%}",
        ]
    ax.text(0.05, 0.95, '\n'.join(metrics), transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('crypto_training.png', dpi=150)
    plt.close()


# ─── 命令行入口 ──────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Crypto Quant Trading System')
    parser.add_argument('--mode', choices=['train', 'signal'], default='train',
                       help='运行模式: train(训练+回测) 或 signal(实时信号)')
    parser.add_argument('--symbol', default='BTCUSDT', help='信号币种')
    parser.add_argument('--model', default='crypto_model.pt', help='模型路径')
    parser.add_argument('--model-type', choices=['gru', 'transformer'],
                       default=None, help='覆盖模型类型')
    args = parser.parse_args()

    if args.model_type:
        CONFIG['model_type'] = args.model_type

    if args.mode == 'signal':
        generate_signal(args.symbol, args.model)
    else:
        main_fit()
