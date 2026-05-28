"""
dqn_agent.py
Deep Q-Network 에이전트

[v2 변경 사항]
1. MSE Loss → Huber Loss (SmoothL1Loss)
   - MSE: 큰 오차에 오차² 적용 → 이상치에 민감, Loss 폭발
   - Huber: 작은 오차는 MSE, 큰 오차는 MAE → 이상치에 강건
   - 결과: TD Loss 발산 억제

[강의 연결]
- Chapter 3, 10페이지: Q-learning TD update
- Chapter 3, 12페이지: Q-learning을 gradient descent로 해석
- Chapter 4, 23페이지: ε-greedy exploration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config


# ============================================================
# Q-Network 신경망 구조
# ============================================================
class QNetwork(nn.Module):
    """
    Q-function을 근사하는 신경망
    
    입력: 상태 벡터 (state_dim 차원)
    출력: 각 행동의 Q값 (action_dim 차원)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        return self.network(state)


# ============================================================
# DQN 에이전트
# ============================================================
class DQNAgent:
    
    def __init__(self, state_dim: int, action_dim: int):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.DEVICE == "cuda" else "cpu"
        )
        print(f"[DQN] 사용 디바이스: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Online + Target network
        self.q_network = QNetwork(
            state_dim, action_dim, config.HIDDEN_DIM
        ).to(self.device)
        
        self.target_network = QNetwork(
            state_dim, action_dim, config.HIDDEN_DIM
        ).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.LEARNING_RATE
        )
        
        # ε-greedy
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = (
            (config.EPSILON_START - config.EPSILON_END) 
            / config.EPSILON_DECAY_STEPS
        )
        
        self.total_steps = 0
    
    def select_action(self, state, training: bool = True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def update(self, batch):
        """
        배치 데이터로 신경망을 한 번 업데이트한다.
        
        [v2 변경] MSE Loss → Huber Loss (SmoothL1Loss)
        
        MSE Loss = (target - pred)²
          → 큰 오차일수록 제곱으로 폭발적 증가
          → 학습 초반 이상치 하나가 Loss를 10^3 단위로 올려버림
        
        Huber Loss = |error| < δ 이면 0.5 * error²  (MSE처럼)
                     |error| >= δ 이면 δ * (|error| - 0.5δ)  (MAE처럼)
          → 큰 오차를 선형으로 처리해서 Loss 스케일 안정화
          → 이상치에 강건하면서도 작은 오차에서는 미분 가능
        """
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 현재 Q값: Q(s, a)
        current_q_values = self.q_network(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)
        
        # TD 타겟: r + γ·max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            td_target = rewards + config.GAMMA * next_q_values * (1 - dones)
        
        # ====================================================
        # [v2 변경] MSE → Huber Loss
        # ====================================================
        # 기존: loss = nn.functional.mse_loss(current_q_values, td_target)
        loss = nn.functional.smooth_l1_loss(current_q_values, td_target)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=config.GRAD_CLIP
        )
        
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.total_steps += 1
    
    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }, path)
        print(f"[저장 완료] {path}")
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        print(f"[로드 완료] {path}")