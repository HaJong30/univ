"""
environment.py (v4 - 개선 버전)
SPY/VIX 기반 트레이딩 환경 (MDP)

[v3 → v4 변경 사항]
1. 보상 스케일 축소: * 10 → * 1
   - 이유: reward * 10 + GAMMA=0.999 조합에서
           Q값 ≈ reward / (1-γ) = 0.01/0.001 = 10
           → TD 오차 ~10, Loss = 오차² ~100 누적 → 10^3 발산
           보상을 1배로 줄여 Q값 스케일 안정화

2. 수수료 패널티 추가 완화: 10 → 3
   - 보상 스케일이 작아진 만큼 패널티도 비례해서 조정

[v2 → v3 변경 사항 유지]
- 매도 방식: 전량 매도 → 50% 비율 매도
- Loss Aversion 제거
- 수수료 패널티 150 → 10 → 3

[강의 연결]
- Chapter 4: Markov Decision Process
- Chapter 1: State augmentation, Data pre-processing
- 교수님 피드백: 매수=현금 비율, 매도=전량, 수수료로 glitch 방지
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
        self.state_dim = self.window_size * 2 + 3
        
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
            buy_amount = self.cash * config.BUY_RATIO
            if buy_amount > 1.0:
                shares_to_buy = buy_amount / current_price
                fee = buy_amount * config.TRANSACTION_FEE
                self.cash -= (buy_amount + fee)
                self.shares += shares_to_buy
                transaction_cost = fee
        
        # ====================================================
        # [v3 변경 1] 전량 매도 → 50% 비율 매도
        # ====================================================
        elif action == self.ACTION_SELL:
            sell_ratio = 0.5                          # 보유 주식의 50%만 매도
            shares_to_sell = self.shares * sell_ratio
            if shares_to_sell * current_price > 1.0:
                sell_amount = shares_to_sell * current_price
                fee = sell_amount * config.TRANSACTION_FEE
                self.cash += (sell_amount - fee)
                self.shares -= shares_to_sell         # 기존: self.shares = 0.0
                transaction_cost = fee
        
        # 2. 다음 시점으로 이동
        self.current_step += 1
        
        # 3. 포트폴리오 가치 계산
        next_price = self.data.loc[self.current_step, "spy_close"]
        current_portfolio_value = self.cash + self.shares * next_price
        
        # ====================================================
        # [v3 변경 2] 보상 함수 단순화
        #   - Loss Aversion (2배 패널티) 제거
        #   - 수수료 패널티 150 → 10 완화
        # ====================================================
        portfolio_return = (
            (current_portfolio_value - self.prev_portfolio_value)
            / self.prev_portfolio_value
        )

        # ====================================================
        # [v4 변경] 보상 스케일 축소: * 10 → * 1
        # Q값 ≈ reward / (1-γ) 이므로 reward가 크면 Q값, TD 오차,
        # Loss가 연쇄적으로 커짐. 스케일 축소로 Loss 발산 억제.
        # ====================================================
        reward = portfolio_return * 1

        # 수수료 패널티 (보상 스케일 축소에 비례해서 조정)
        if transaction_cost > 0:
            transaction_penalty = (transaction_cost / (current_portfolio_value + 1e-8)) * 3
            reward -= transaction_penalty
        
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
        """상태 벡터 구성 (Chapter 1의 normalization 반영)"""
        start = self.current_step - self.window_size + 1
        end = self.current_step + 1
        window = self.data.iloc[start:end]
        
        prices = window["spy_close"].values
        mas = window["spy_ma"].values
        current_vix = self.data.loc[self.current_step, "vix_close"]
        current_price = prices[-1]
        
        price_min = prices.min()
        price_max = prices.max()
        if price_max - price_min > 1e-8:
            normalized_prices = (prices - price_min) / (price_max - price_min)
        else:
            normalized_prices = np.zeros_like(prices)
        
        ma_diff_ratio = (prices - mas) / (mas + 1e-8)
        
        if self.vix_max - self.vix_min > 1e-8:
            normalized_vix = (current_vix - self.vix_min) / (self.vix_max - self.vix_min)
        else:
            normalized_vix = 0.5
        
        portfolio_value = self.cash + self.shares * current_price
        cash_ratio = self.cash / (portfolio_value + 1e-8)
        stock_ratio = (self.shares * current_price) / (portfolio_value + 1e-8)
        
        state = np.concatenate([
            normalized_prices,
            ma_diff_ratio,
            [normalized_vix],
            [cash_ratio],
            [stock_ratio],
        ]).astype(np.float32)
        
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