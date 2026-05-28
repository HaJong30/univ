"""
backtest.py
학습된 DQN 에이전트의 백테스트 (Out-of-sample 검증)

[목적]
1. 학습에 사용되지 않은 데이터(2022~2025)로 진짜 성능 검증
2. Buy-and-Hold 전략과 비교
3. 매매 패턴 분석 (특히 VIX와의 관계)

[강의 연결]
- 제안서 평가 계획: "Buy-and-Hold 베이스라인 대비 유의미한 초과 달성"
- 학습 시 training=True (ε-greedy), 백테스트 시 training=False (greedy only)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

import config
from environment import TradingEnvironment
from dqn_agent import DQNAgent


def run_dqn_backtest(agent, env):
    """
    학습된 DQN 에이전트로 백테스트를 실행한다.
    
    Args:
        agent: 로드된 DQNAgent
        env: TradingEnvironment (테스트 데이터)
    
    Returns:
        백테스트 기록 dict
    """
    state = env.reset()
    
    # 시계열 기록
    records = {
        "step": [],
        "date": [],
        "price": [],
        "vix": [],
        "action": [],
        "portfolio_value": [],
        "cash": [],
        "shares": [],
    }
    
    step = 0
    while not env.done:
        # 평가 모드: ε=0, 항상 Q값 최대 행동 선택
        action = agent.select_action(state, training=False)
        
        next_state, reward, done, info = env.step(action)
        
        # 기록 (env.current_step은 step() 후 증가된 상태)
        current_idx = env.current_step
        records["step"].append(step)
        records["date"].append(env.data.loc[current_idx, "Date"])
        records["price"].append(info["price"])
        records["vix"].append(env.data.loc[current_idx, "vix_close"])
        records["action"].append(action)
        records["portfolio_value"].append(info["portfolio_value"])
        records["cash"].append(info["cash"])
        records["shares"].append(info["shares"])
        
        state = next_state
        step += 1
    
    return records


def run_buy_and_hold(env_data):
    """
    Buy-and-Hold 베이스라인 전략을 시뮬레이션한다.
    첫날 전 자본금으로 SPY를 매수하고 마지막까지 보유.
    
    Args:
        env_data: 테스트 데이터 DataFrame
    
    Returns:
        Buy-and-Hold 포트폴리오 시계열
    """
    # 첫날 가격 (window_size 이후부터 시작, 환경과 동일)
    start_idx = config.WINDOW_SIZE
    start_price = env_data.loc[start_idx, "spy_close"]
    
    # 수수료 1번 차감하고 전 자본을 SPY로 변환
    initial_cash = config.INITIAL_CASH
    fee = initial_cash * config.TRANSACTION_FEE
    shares = (initial_cash - fee) / start_price
    
    # 매일 포트폴리오 가치 기록
    portfolio_values = []
    dates = []
    prices = []
    
    for i in range(start_idx + 1, len(env_data)):
        price = env_data.loc[i, "spy_close"]
        portfolio_values.append(shares * price)
        dates.append(env_data.loc[i, "Date"])
        prices.append(price)
    
    return {
        "date": dates,
        "price": prices,
        "portfolio_value": portfolio_values,
    }


def calculate_metrics(portfolio_values, initial_value=config.INITIAL_CASH):
    """
    주요 성능 지표를 계산한다.
    
    [지표 설명]
    - 총 수익률: 최종 / 초기 - 1
    - 최대 낙폭(MDD): 고점 대비 최대 하락폭 (위험 지표)
    - 샤프 비율: 수익률 / 변동성 (위험 대비 수익)
    - 승률: 일별 수익이 양수인 날의 비율
    """
    values = np.array(portfolio_values)
    
    # 1. 총 수익률
    total_return = (values[-1] / initial_value - 1) * 100
    
    # 2. 일별 수익률
    daily_returns = np.diff(values) / values[:-1]
    
    # 3. 연환산 수익률 (대략 252 거래일/년 기준)
    n_days = len(values)
    n_years = n_days / 252
    annualized_return = ((values[-1] / initial_value) ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    
    # 4. 변동성 (연환산)
    volatility = np.std(daily_returns) * np.sqrt(252) * 100
    
    # 5. 샤프 비율 (무위험 이자율 0으로 가정)
    sharpe = annualized_return / volatility if volatility > 1e-8 else 0
    
    # 6. 최대 낙폭 (MDD)
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / cummax
    mdd = drawdown.min() * 100
    
    # 7. 승률 (일별)
    win_rate = (daily_returns > 0).mean() * 100
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "final_value": values[-1],
    }


def plot_backtest_results(dqn_records, bnh_records, dqn_metrics, bnh_metrics):
    """
    백테스트 결과를 4개 패널로 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("DQN vs Buy-and-Hold 백테스트 (2022~2025)", fontsize=16, fontweight="bold")
    
    # 날짜 변환
    dqn_dates = pd.to_datetime(dqn_records["date"])
    bnh_dates = pd.to_datetime(bnh_records["date"])
    
    # ====================================================
    # (1) 포트폴리오 가치 비교
    # ====================================================
    ax = axes[0, 0]
    ax.plot(dqn_dates, dqn_records["portfolio_value"], 
            label=f"DQN ({dqn_metrics['total_return']:+.1f}%)", 
            color="navy", linewidth=2)
    ax.plot(bnh_dates, bnh_records["portfolio_value"], 
            label=f"Buy-and-Hold ({bnh_metrics['total_return']:+.1f}%)", 
            color="orange", linewidth=2, linestyle="--")
    ax.axhline(y=config.INITIAL_CASH, color="gray", linestyle=":", alpha=0.5, label="초기 자본")
    ax.set_ylabel("포트폴리오 가치 ($)")
    ax.set_title("포트폴리오 가치 변화")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    
    # ====================================================
    # (2) SPY 가격 + DQN 매매 시점
    # ====================================================
    ax = axes[0, 1]
    prices = np.array(dqn_records["price"])
    actions = np.array(dqn_records["action"])
    
    ax.plot(dqn_dates, prices, color="black", linewidth=1, alpha=0.7, label="SPY 가격")
    
    # 매수 시점 (action=1) — 초록 화살표
    buy_idx = np.where(actions == 1)[0]
    if len(buy_idx) > 0:
        ax.scatter(dqn_dates.values[buy_idx], prices[buy_idx], 
                   color="green", marker="^", s=40, alpha=0.6, label=f"Buy ({len(buy_idx)}회)")
    
    # 매도 시점 (action=2) — 빨강 화살표
    sell_idx = np.where(actions == 2)[0]
    if len(sell_idx) > 0:
        ax.scatter(dqn_dates.values[sell_idx], prices[sell_idx], 
                   color="red", marker="v", s=40, alpha=0.6, label=f"Sell ({len(sell_idx)}회)")
    
    ax.set_ylabel("SPY 가격 ($)")
    ax.set_title("DQN 매매 시점")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    
    # ====================================================
    # (3) VIX와 매매의 관계 (이 프로젝트의 핵심!)
    # ====================================================
    ax = axes[1, 0]
    vix = np.array(dqn_records["vix"])
    
    ax.plot(dqn_dates, vix, color="purple", linewidth=1, alpha=0.7, label="VIX")
    ax.axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="VIX=20 (평상시)")
    ax.axhline(y=30, color="orange", linestyle="--", alpha=0.5, label="VIX=30 (불안)")
    
    # 매수/매도 시점의 VIX 값
    if len(buy_idx) > 0:
        ax.scatter(dqn_dates.values[buy_idx], vix[buy_idx], 
                   color="green", marker="^", s=40, alpha=0.6, label="Buy")
    if len(sell_idx) > 0:
        ax.scatter(dqn_dates.values[sell_idx], vix[sell_idx], 
                   color="red", marker="v", s=40, alpha=0.6, label="Sell")
    
    ax.set_ylabel("VIX")
    ax.set_title("VIX와 매매 의사결정 (역발상 매매 확인)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(axis='x', rotation=30)
    
    # ====================================================
    # (4) 행동 분포 + 성능 비교 표
    # ====================================================
    ax = axes[1, 1]
    ax.axis("off")
    
    # 행동 분포
    n_hold = (actions == 0).sum()
    n_buy = (actions == 1).sum()
    n_sell = (actions == 2).sum()
    total = len(actions)
    
    # 성능 비교 텍스트
    summary = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       성능 비교 요약
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    [DQN 에이전트]
      • 총 수익률:       {dqn_metrics['total_return']:+8.2f}%
      • 연환산 수익률:   {dqn_metrics['annualized_return']:+8.2f}%
      • 변동성:          {dqn_metrics['volatility']:8.2f}%
      • 샤프 비율:       {dqn_metrics['sharpe']:8.3f}
      • 최대 낙폭(MDD):  {dqn_metrics['mdd']:8.2f}%
      • 일별 승률:       {dqn_metrics['win_rate']:8.2f}%
    
    [Buy-and-Hold]
      • 총 수익률:       {bnh_metrics['total_return']:+8.2f}%
      • 연환산 수익률:   {bnh_metrics['annualized_return']:+8.2f}%
      • 변동성:          {bnh_metrics['volatility']:8.2f}%
      • 샤프 비율:       {bnh_metrics['sharpe']:8.3f}
      • 최대 낙폭(MDD):  {bnh_metrics['mdd']:8.2f}%
      • 일별 승률:       {bnh_metrics['win_rate']:8.2f}%
    
    [초과 성과 (DQN - B&H)]
      • 수익률 차이:     {dqn_metrics['total_return'] - bnh_metrics['total_return']:+8.2f}%p
      • MDD 차이:        {dqn_metrics['mdd'] - bnh_metrics['mdd']:+8.2f}%p
    
    [DQN 행동 분포]
      • Hold: {n_hold:4d} ({n_hold/total*100:5.1f}%)
      • Buy:  {n_buy:4d} ({n_buy/total*100:5.1f}%)
      • Sell: {n_sell:4d} ({n_sell/total*100:5.1f}%)
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))
    
    plt.tight_layout()
    
    output_path = "backtest_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[그래프 저장 완료] {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print(" 백테스트 시작 (Out-of-Sample 검증)")
    print("=" * 70)
    
    # ====================================================
    # 1. 학습된 모델 로드
    # ====================================================
    print("\n[1/4] 학습된 모델 로드 중...")
    model_path = "checkpoints/dqn_final.pt"
    if not os.path.exists(model_path):
        print(f"  모델 파일이 없습니다: {model_path}")
        print(f"  train.py를 먼저 실행하세요.")
        return
    
    # 임시 환경에서 상태 차원 확인
    test_df = pd.read_csv("test_data.csv")
    env = TradingEnvironment(test_df)
    
    agent = DQNAgent(env.state_dim, env.NUM_ACTIONS)
    agent.load(model_path)
    
    # 평가 모드: ε을 명시적으로 0으로 설정 (혹시 모를 무작위성 차단)
    agent.epsilon = 0.0
    
    print(f"  → 테스트 데이터: {len(test_df)}일")
    
    # ====================================================
    # 2. DQN 백테스트 실행
    # ====================================================
    print("\n[2/4] DQN 백테스트 실행 중...")
    dqn_records = run_dqn_backtest(agent, env)
    print(f"  → 총 {len(dqn_records['action'])}일 시뮬레이션 완료")
    
    # ====================================================
    # 3. Buy-and-Hold 베이스라인
    # ====================================================
    print("\n[3/4] Buy-and-Hold 베이스라인 계산 중...")
    bnh_records = run_buy_and_hold(test_df)
    print(f"  → 총 {len(bnh_records['portfolio_value'])}일 시뮬레이션 완료")
    
    # ====================================================
    # 4. 성능 지표 계산 + 시각화
    # ====================================================
    print("\n[4/4] 성능 분석 중...")
    dqn_metrics = calculate_metrics(dqn_records["portfolio_value"])
    bnh_metrics = calculate_metrics(bnh_records["portfolio_value"])
    
    # 콘솔 출력
    print("\n" + "=" * 70)
    print(" 백테스트 결과")
    print("=" * 70)
    
    print(f"\n{'지표':<20} {'DQN':>15} {'Buy-and-Hold':>15} {'차이':>15}")
    print("-" * 70)
    
    metrics_to_show = [
        ("총 수익률 (%)", "total_return"),
        ("연환산 수익률 (%)", "annualized_return"),
        ("변동성 (%)", "volatility"),
        ("샤프 비율", "sharpe"),
        ("최대 낙폭 (%)", "mdd"),
        ("일별 승률 (%)", "win_rate"),
        ("최종 가치 ($)", "final_value"),
    ]
    
    for label, key in metrics_to_show:
        dqn_v = dqn_metrics[key]
        bnh_v = bnh_metrics[key]
        diff = dqn_v - bnh_v
        if key == "final_value":
            print(f"{label:<20} {dqn_v:>15,.2f} {bnh_v:>15,.2f} {diff:>+15,.2f}")
        else:
            print(f"{label:<20} {dqn_v:>15.3f} {bnh_v:>15.3f} {diff:>+15.3f}")
    
    # 결론
    print("\n" + "=" * 70)
    excess = dqn_metrics["total_return"] - bnh_metrics["total_return"]
    if excess > 0:
        print(f"  DQN이 Buy-and-Hold 대비 {excess:+.2f}%p 초과 수익을 달성했습니다!")
    else:
        print(f"  DQN이 Buy-and-Hold 대비 {excess:+.2f}%p 부족했습니다.")
    
    if dqn_metrics["mdd"] > bnh_metrics["mdd"]:
        print(f"  최대 낙폭도 {dqn_metrics['mdd'] - bnh_metrics['mdd']:+.2f}%p 더 작아 위험 관리에도 우수합니다.")
    print("=" * 70)
    
    # 시각화
    plot_backtest_results(dqn_records, bnh_records, dqn_metrics, bnh_metrics)
    
    # 상세 기록 CSV 저장 (발표용 분석 자료)
    df_records = pd.DataFrame(dqn_records)
    df_records.to_csv("backtest_dqn_records.csv", index=False)
    print(f"\n[상세 기록 저장] backtest_dqn_records.csv")


if __name__ == "__main__":
    main()