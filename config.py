"""
config.py
프로젝트 전반의 하이퍼파라미터 중앙 관리

[v6 변경 사항]
- GAMMA: 0.999 -> 0.99
  [근거] 에피소드 길이는 252일(1년). gamma의 유효 지평(effective horizon)은
  1/(1-gamma). gamma=0.999면 1000일로 에피소드의 ~4배 -> 에이전트가 한 번도
  경험하지 못하는 먼 미래까지 평가하려다 bootstrapping이 불안정해짐
  (TD Loss 발산의 근본 원인). gamma=0.99면 지평이 ~100일로 에피소드 안에
  들어와 안정적. (교수님은 0.999 권장하셨으나, 에피소드 길이 대비
  과도한 지평은 학습을 해치므로 타당한 근거로 조정)
- SELL_RATIO 추가: 부분매도 비율
- BANKRUPT_PENALTY: -100 -> -10 (보상 스케일에 맞춤)
"""

# ============================================================
# 1. 데이터 관련 설정
# ============================================================
TICKER_SPY = "SPY"
TICKER_VIX = "^VIX"

TRAIN_START = "2010-01-01"
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2025-12-31"

WINDOW_SIZE = 20
MA_PERIOD = 20

# ============================================================
# 2. 환경(Environment) 관련 설정
# ============================================================
INITIAL_CASH = 10000.0

# 매수: min(총자본 * BUY_RATIO, 남은 현금)
BUY_RATIO = 0.3            # [v7] 0.2 -> 0.3 (더 공격적 진입)

# 매도: 보유 주식 * SELL_RATIO (부분매도)
SELL_RATIO = 0.3           # [v7] 0.5 -> 0.3 (한 번에 조금만 매도, 포지션 유지)

TRANSACTION_FEE = 0.001
BANKRUPT_PENALTY = -10.0

# ============================================================
# 3. DQN 알고리즘 관련 설정
# ============================================================
# [v6] 0.999 -> 0.99 (유효 지평을 에피소드 길이에 맞춤, TD Loss 안정화)
GAMMA = 0.99

HIDDEN_DIM = 128
NUM_HIDDEN_LAYERS = 2

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50000

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 252

GRAD_CLIP = 1.0

# ============================================================
# 4. 기타
# ============================================================
DEVICE = "cuda"
RANDOM_SEED = 42
LOG_INTERVAL = 10