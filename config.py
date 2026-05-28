"""
config.py
프로젝트 전반의 하이퍼파라미터 중앙 관리

[v4 변경 사항]
- GAMMA: 0.999 → 0.99  ← TD Loss 발산의 핵심 원인
  Q값 ≈ reward / (1-γ). 0.999면 1000배 증폭, 0.99면 100배로 안정화
- GRAD_CLIP: 10.0 → 1.0  ← gradient 폭주 방지 실효성 확보
- TARGET_UPDATE_FREQ: 100 → 500  ← 타겟 안정성 향상
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
BUY_RATIO = 0.3             # 0.5 → 0.3 (매수 강도 완화)
TRANSACTION_FEE = 0.001
BANKRUPT_PENALTY = -10.0    # -100 → -10 (보상 스케일 축소에 맞게 조정)

# ============================================================
# 3. DQN 알고리즘 관련 설정
# ============================================================
# [v4 핵심 변경] GAMMA 0.999 → 0.99
# Q값 ≈ reward / (1-γ). γ=0.999면 Q값이 1000배 증폭되어 Loss 발산
# γ=0.99면 100배로 줄어들어 안정적인 학습 가능
GAMMA = 0.99

HIDDEN_DIM = 128
NUM_HIDDEN_LAYERS = 2

LEARNING_RATE = 1e-4
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000

# [v4 변경] 100 → 500: 타겟 네트워크를 더 천천히 업데이트해서 안정성 확보
TARGET_UPDATE_FREQ = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 50000

NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 252

# [v4 변경] 10.0 → 1.0: 실효성 있는 gradient clipping
GRAD_CLIP = 1.0

# ============================================================
# 4. 기타
# ============================================================
DEVICE = "cuda"
RANDOM_SEED = 42
LOG_INTERVAL = 10