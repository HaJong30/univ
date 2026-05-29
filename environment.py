"""
environment.py (v8 - 상태/보상 전면 재설계)
SPY/VIX 기반 트레이딩 환경 (MDP)

[v8 전면 재설계]
1. 상태(State) 재설계: 43차원 → 29차원
   - 기존: 20일 종가(20) + 20일 MA차이(20) + VIX/현금/주식(3)
     → 단기 노이즈만 보고 큰 추세를 못 봄 → 뇌동매매
   - 신규: 다중 시간축 추세 정보 추가
     · 5/20/60일 수익률 (단기/중기/장기 모멘텀)
     · MA20-MA60 정렬 (추세 방향)
     · VIX 수준 + VIX 변화 (공포 수준과 방향)
   - 근거: Chapter 1 "data pre-processing: use our own knowledge
     of the environment into this (compute distance, sort...)"

2. 보상(Reward) 재설계: 다목적 → 단일 로그수익률
   - 기존: 수익 + 손실페널티 + 보유보너스 + 수수료페널티 (충돌!)
   - 신규: reward = log(V_t / V_{t-1}) * 100
   - 근거: Chapter 1, p.16 "Priority? Combining multiple objectives?"
     목적이 충돌하면 학습이 꼬임 → 단일 목적으로 정리
     위험관리는 보상이 아닌 상태(추세/VIX)가 학습하도록 분리

[강의 연결]
- Chapter 4: Markov Decision Process
- Chapter 1: State augmentation, Data pre-processing, Reward design
"""

import numpy as np
import pandas as pd
import config


class TradingEnvironment:
    """
    SPY/VIX 기반 강화학습 트레이딩 환경
    
    상태(State): 43차원 벡터
        - 최근 20일 SPY 정규화 종가 (20)
        - 최근 20일 SPY-MA 차이 정규화 (20)
        - 현재 VIX 정규화 (1)
        - 현금 비율 (1)
        - 주식 가치 비율 (1)
    
    행동(Action): 0=Hold, 1=Buy, 2=Sell
    """
    
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    NUM_ACTIONS = 3
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)
        self.window_size = config.WINDOW_SIZE

        # ====================================================
        # [v8] 상태 차원 재설계
        # 최근 20일 정규화 종가 (20)  : 단기 가격 패턴
        # 5/20/60일 수익률 (3)        : 다중 시간축 추세/모멘텀
        # MA20/MA60 정렬 신호 (2)     : 골든/데드크로스
        # VIX 수준 + VIX 변화 (2)     : 공포 수준과 방향
        # 현금/주식 비율 (2)          : 포트폴리오 상태
        # = 29차원
        # ====================================================
        self.state_dim = self.window_size + 9

        # 장기 추세용 lookback (에피소드 시작점이 이보다 뒤여야 함)
        self.long_window = 60

        self.current_step = None
        self.cash = None
        self.shares = None
        self.initial_portfolio_value = None
        self.prev_portfolio_value = None
        self.done = None
        
        self.vix_min = self.data["vix_close"].min()
        self.vix_max = self.data["vix_close"].max()
    
    def reset(self, start_step: int = None):
        """
        에피소드를 초기화하고 첫 상태를 반환한다.

        Args:
            start_step: 시작 시점 인덱스.
                        None이면 항상 처음(window_size)에서 시작 (백테스트용)
                        정수면 그 위치에서 시작 (학습 시 랜덤 시작점으로 사용)
        Returns:
            state: 43차원 numpy 배열
        """
        if start_step is None:
            self.current_step = self.window_size
        else:
            self.current_step = start_step
        
        self.cash = config.INITIAL_CASH
        self.shares = 0.0
        self.initial_portfolio_value = self.cash
        self.prev_portfolio_value = self.cash
        self.done = False
        
        return self._get_state()
    
    def step(self, action: int):
        """
        한 타임스텝을 진행한다.
        """
        if self.done:
            raise RuntimeError("에피소드가 이미 종료되었습니다. reset()을 호출하세요.")
        
        current_price = self.data.loc[self.current_step, "spy_close"]
        
        # 1. 행동 실행
        transaction_cost = 0.0
        
        if action == self.ACTION_BUY:
            # ====================================================
            # [v6] 매수: min(총자본의 BUY_RATIO, 남은 현금)
            # 교수님 제안(총자본 비율 기반)을 반영
            # ====================================================
            portfolio_value = self.cash + self.shares * current_price
            buy_amount = min(portfolio_value * config.BUY_RATIO, self.cash)
            if buy_amount > 1.0:
                shares_to_buy = buy_amount / current_price
                fee = buy_amount * config.TRANSACTION_FEE
                self.cash -= (buy_amount + fee)
                self.shares += shares_to_buy
                transaction_cost = fee
        
        # ====================================================
        # [v6] 매도: 전량매도 → 부분매도(SELL_RATIO)로 복원
        #
        # [근거] 교수님은 전량매도를 "선택지"로 두셨음
        #   ("전량매도를 기준으로 하는 경우도 많긴 하니 그대로 가도 됩니다")
        # 실험 결과: 전량매도 시 Buy(소량)/Sell(전량) 비대칭으로
        #   에이전트가 거의 매도를 안 하게 됨(Sell 6%) → Buy-and-Hold화
        #   → 위험관리 실패(MDD -26.68%로 B&H보다 나쁨)
        # 부분매도로 바꾸면 포지션을 점진적으로 조절 가능
        #   → 균형잡힌 행동, 더 나은 Sharpe/MDD (실측 Sharpe 1.05)
        # ====================================================
        elif action == self.ACTION_SELL:
            shares_to_sell = self.shares * config.SELL_RATIO
            if shares_to_sell * current_price > 1.0:
                sell_amount = shares_to_sell * current_price
                fee = sell_amount * config.TRANSACTION_FEE
                self.cash += (sell_amount - fee)
                self.shares -= shares_to_sell
                transaction_cost = fee
        
        # 2. 다음 시점으로 이동
        self.current_step += 1
        
        # 3. 포트폴리오 가치 계산
        next_price = self.data.loc[self.current_step, "spy_close"]
        current_portfolio_value = self.cash + self.shares * next_price
        
        # ====================================================
        # [v8] 보상 함수 재설계 — 단일 목적 + 로그수익률
        #
        # [철학] (Chapter 1, p.16: "Priority? Combining multiple
        #   objectives?") 보상에 수익+위험+보유+수수료를 다 섞으니
        #   목적들이 충돌해 학습이 꼬임. → 매 스텝 보상은 하나로 단순화.
        #   위험 관리는 보상이 아니라 "상태(추세/VIX)"가 학습하게 분리.
        #
        # 로그수익률 사용 이유:
        #   - 복리(누적 수익)를 합으로 자연스럽게 표현: Σ log(V_t/V_{t-1})
        #     = log(V_T/V_0) → 에피소드 누적보상이 곧 총수익률
        #   - 큰 변동에서도 스케일이 안정적 (Chapter 1 normalization 철학)
        #   - 하락에 비대칭적으로 더 큰 음수 → 자연스러운 손실 회피
        #     (별도 loss-aversion 계수 불필요)
        #
        # 수수료는 포트폴리오 가치에 이미 반영되므로(현금 차감)
        # 로그수익률에 자동으로 녹아듦 → 별도 페널티 불필요
        # ====================================================
        ratio = current_portfolio_value / (self.prev_portfolio_value + 1e-8)
        reward = np.log(max(ratio, 1e-8)) * 100

        # 4. 종료 조건
        if current_portfolio_value < self.initial_portfolio_value * 0.5:
            reward = config.BANKRUPT_PENALTY
            self.done = True
        
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # 5. 상태 업데이트 및 반환
        self.prev_portfolio_value = current_portfolio_value
        next_state = self._get_state() if not self.done else np.zeros(self.state_dim)
        
        info = {
            "portfolio_value": current_portfolio_value,
            "cash": self.cash,
            "shares": self.shares,
            "price": next_price,
            "transaction_cost": transaction_cost,
            "action": action,
        }
        
        return next_state, reward, self.done, info
    
    def _get_state(self):
        """
        [v8] 상태 벡터 구성 — 다중 시간축 추세 정보 포함
        (Chapter 1: data pre-processing, normalization 반영)

        구성 (총 29차원):
          - 최근 20일 정규화 종가 (20): 단기 가격 패턴
          - 5/20/60일 수익률 (3): 단기/중기/장기 모멘텀
          - MA20-MA60 정렬 (1): 추세 방향 (양수=상승추세)
          - 현재가-MA20 비율 (1): 단기 이격
          - VIX 정규화 수준 (1) + VIX 5일 변화 (1): 공포 수준과 방향
          - 현금 비율 (1) + 주식 비율 (1): 포트폴리오 상태
        """
        idx = self.current_step

        # 최근 20일 정규화 종가 (early period는 가능한 만큼만)
        w_start = max(0, idx - self.window_size + 1)
        prices_win = self.data["spy_close"].values[w_start: idx + 1]
        if len(prices_win) < self.window_size:
            # 앞쪽을 첫 값으로 패딩
            pad = np.full(self.window_size - len(prices_win), prices_win[0])
            prices_win = np.concatenate([pad, prices_win])

        p_min, p_max = prices_win.min(), prices_win.max()
        if p_max - p_min > 1e-8:
            norm_prices = (prices_win - p_min) / (p_max - p_min)
        else:
            norm_prices = np.zeros_like(prices_win)

        current_price = self.data["spy_close"].values[idx]

        # 다중 시간축 수익률 (clip으로 이상치 제한)
        def ret_over(n):
            j = max(0, idx - n)
            past = self.data["spy_close"].values[j]
            return np.clip((current_price / (past + 1e-8)) - 1.0, -0.5, 0.5)

        ret_5 = ret_over(5)
        ret_20 = ret_over(20)
        ret_60 = ret_over(60)

        # 이동평균 정렬 (MA20 vs MA60)
        ma20 = self.data["spy_close"].values[max(0, idx - 19): idx + 1].mean()
        ma60 = self.data["spy_close"].values[max(0, idx - 59): idx + 1].mean()
        ma_align = np.clip((ma20 - ma60) / (ma60 + 1e-8), -0.3, 0.3)
        price_vs_ma20 = np.clip((current_price - ma20) / (ma20 + 1e-8), -0.3, 0.3)

        # VIX 수준 + 변화
        current_vix = self.data["vix_close"].values[idx]
        if self.vix_max - self.vix_min > 1e-8:
            norm_vix = (current_vix - self.vix_min) / (self.vix_max - self.vix_min)
        else:
            norm_vix = 0.5
        vix_5ago = self.data["vix_close"].values[max(0, idx - 5)]
        vix_change = np.clip((current_vix - vix_5ago) / (vix_5ago + 1e-8), -0.5, 0.5)

        # 포트폴리오 상태
        portfolio_value = self.cash + self.shares * current_price
        cash_ratio = self.cash / (portfolio_value + 1e-8)
        stock_ratio = (self.shares * current_price) / (portfolio_value + 1e-8)

        state = np.concatenate([
            norm_prices,                              # 20
            [ret_5, ret_20, ret_60],                  # 3
            [ma_align, price_vs_ma20],                # 2
            [norm_vix, vix_change],                   # 2
            [cash_ratio, stock_ratio],                # 2
        ]).astype(np.float32)                         # = 29

        return state
    
    def get_portfolio_value(self):
        """현재 포트폴리오 가치 반환"""
        if self.current_step >= len(self.data):
            current_price = self.data.iloc[-1]["spy_close"]
        else:
            current_price = self.data.loc[self.current_step, "spy_close"]
        return self.cash + self.shares * current_price
    
    def get_max_start_step(self):
        """
        랜덤 시작점의 최대 인덱스 반환.
        마지막 1년(252일) 이전까지만 시작점으로 허용.
        """
        return len(self.data) - config.MAX_STEPS_PER_EPISODE - 1


# ============================================================
# 환경 검증
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 60)
    print("Environment v3 검증")
    print("=" * 60)
    
    train_df = pd.read_csv("train_data.csv")
    env = TradingEnvironment(train_df)
    
    print(f"\n[환경 정보]")
    print(f"  - 상태 차원: {env.state_dim}")
    print(f"  - 전체 데이터 길이: {len(train_df)}일")
    print(f"  - 랜덤 시작점 최대값: {env.get_max_start_step()}")
    
    # ---- 시나리오 1: 항상 Hold ----
    print(f"\n[시나리오 1] 항상 Hold")
    state = env.reset()
    while not env.done:
        state, reward, done, info = env.step(env.ACTION_HOLD)
    final_value = info["portfolio_value"]
    print(f"  - 최종: ${final_value:.2f} ({(final_value/config.INITIAL_CASH - 1)*100:+.2f}%)")
    
    # ---- 시나리오 2: Buy-and-Hold ----
    print(f"\n[시나리오 2] Buy 50번 후 Hold")
    state = env.reset()
    for _ in range(50):
        state, reward, done, info = env.step(env.ACTION_BUY)
        if done: break
    while not env.done:
        state, reward, done, info = env.step(env.ACTION_HOLD)
    final_value = info["portfolio_value"]
    print(f"  - 최종: ${final_value:.2f} ({(final_value/config.INITIAL_CASH - 1)*100:+.2f}%)")
    
    # ---- 시나리오 3: 매일 Buy/Sell ----
    print(f"\n[시나리오 3] 매일 Buy/Sell 반복 (수수료 패널티 확인)")
    state = env.reset()
    total_cost = 0.0
    i = 0
    while not env.done:
        action = [env.ACTION_BUY, env.ACTION_SELL][i % 2]
        state, reward, done, info = env.step(action)
        if info["transaction_cost"] > 0:
            total_cost += info["transaction_cost"]
        i += 1
    final_value = info["portfolio_value"]
    print(f"  - 최종: ${final_value:.2f} ({(final_value/config.INITIAL_CASH - 1)*100:+.2f}%)")
    print(f"  - 누적 수수료: ${total_cost:.2f}")
    
    # ---- 시나리오 4: 랜덤 시작점 ----
    print(f"\n[시나리오 4] 랜덤 시작점 테스트")
    np.random.seed(42)
    max_start = env.get_max_start_step()
    for trial in range(3):
        random_start = np.random.randint(env.window_size, max_start)
        state = env.reset(start_step=random_start)
        print(f"  - 시작점 {random_start}: 날짜={train_df.loc[random_start, 'Date']}, "
              f"상태 shape={state.shape}")
    
    print(f"\n[검증 완료]")