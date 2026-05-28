"""
replay_buffer.py
경험 재현 버퍼 (Experience Replay Buffer)

[강의 연결]
- Chapter 1, 13페이지: 데이터 효율성 문제 해결
  - "If the agent observe a not interesting data, it should not learn anything 
     but still update the parameters. It goes into vicious cycle"
- 핵심 아이디어:
  1. 경험을 저장해서 재사용 → 데이터 효율성 향상
  2. 무작위 샘플링 → 시간적 상관성 제거
  3. 안정적인 그래디언트 업데이트
"""

import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    DQN 학습을 위한 경험 재현 버퍼
    
    저장 단위: (state, action, reward, next_state, done) 튜플
    이를 'transition' 또는 'experience'라고 부른다.
    
    Chapter 3의 7페이지에서 다룬 데이터 (i, u, c, j)와 동일한 개념
    (i=state, u=action, c=cost/reward, j=next_state)
    """
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: 버퍼 최대 크기 (config.REPLAY_BUFFER_SIZE)
                      가득 차면 가장 오래된 경험부터 삭제됨 (FIFO)
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        새로운 경험을 버퍼에 추가한다.
        
        Args:
            state: 현재 상태 (numpy array, shape=(state_dim,))
            action: 취한 행동 (int: 0, 1, 2)
            reward: 받은 보상 (float)
            next_state: 다음 상태 (numpy array)
            done: 에피소드 종료 여부 (bool)
        """
        # 명시적으로 float32/int로 변환하여 메모리 효율성 확보
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))
    
    def sample(self, batch_size: int):
        """
        버퍼에서 batch_size개의 경험을 무작위로 샘플링한다.
        
        무작위 샘플링이 핵심:
        - 연속된 경험은 매우 유사 → 학습 편향 발생
        - 무작위로 섞어야 IID(독립항등분포) 가정 근사적 만족
        
        Args:
            batch_size: 샘플링할 경험 수
        
        Returns:
            5개의 numpy 배열 튜플: (states, actions, rewards, next_states, dones)
            각 배열의 shape는 (batch_size, ...)
        """
        # 무작위로 batch_size개 선택
        batch = random.sample(self.buffer, batch_size)
        
        # 5개의 리스트로 분리 후 numpy 배열로 변환
        # zip(*batch)는 [(s1,a1,r1,...), (s2,a2,r2,...)] -> [(s1,s2), (a1,a2), ...]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),        # (B, state_dim)
            np.array(actions, dtype=np.int64),         # (B,)
            np.array(rewards, dtype=np.float32),       # (B,)
            np.array(next_states, dtype=np.float32),   # (B, state_dim)
            np.array(dones, dtype=np.float32),         # (B,) - 0 또는 1
        )
    
    def __len__(self):
        """현재 저장된 경험 수"""
        return len(self.buffer)
    
    def is_ready(self, min_size: int):
        """학습 시작 가능 여부 (최소 데이터 수집량 충족 확인)"""
        return len(self.buffer) >= min_size


# ============================================================
# 단독 실행 시 동작 확인
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ReplayBuffer 검증")
    print("=" * 60)
    
    # 작은 버퍼 생성
    buffer = ReplayBuffer(capacity=100)
    
    # 가짜 경험 추가
    for i in range(50):
        state = np.random.randn(43).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(43).astype(np.float32)
        done = (i == 49)  # 마지막만 done=True
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"\n[버퍼 상태]")
    print(f"  - 현재 크기: {len(buffer)}")
    print(f"  - 최대 용량: {buffer.capacity}")
    print(f"  - 학습 준비 (min=30): {buffer.is_ready(30)}")
    print(f"  - 학습 준비 (min=100): {buffer.is_ready(100)}")
    
    # 샘플링 테스트
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=16)
    
    print(f"\n[샘플링 결과 (batch_size=16)]")
    print(f"  - states shape:      {states.shape}")
    print(f"  - actions shape:     {actions.shape}")
    print(f"  - rewards shape:     {rewards.shape}")
    print(f"  - next_states shape: {next_states.shape}")
    print(f"  - dones shape:       {dones.shape}")
    print(f"\n  - actions 예시: {actions[:5]}")
    print(f"  - rewards 예시: {rewards[:5]}")
    
    # 용량 초과 테스트 (FIFO 동작 확인)
    print(f"\n[용량 초과 테스트]")
    for i in range(80):  # 50 + 80 = 130개 시도 (capacity=100)
        buffer.push(
            np.zeros(43, dtype=np.float32), 0, 0.0,
            np.zeros(43, dtype=np.float32), False
        )
    print(f"  - 130개 push 후 크기: {len(buffer)} (예상: 100)")
    
    print("\n[검증 완료]")