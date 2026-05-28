"""
train.py
DQN 에이전트 학습 메인 루프

[강의 연결]
- Chapter 3, 7페이지: 시뮬레이션 데이터 수집
  [x_0, u_0, c_0, x_1, u_1, c_1, ...] 시계열 데이터 생성
- Chapter 3, 9-10페이지: TD learning 업데이트
- Chapter 4, 23페이지: ε-greedy 탐색

[학습 절차]
1. 환경 reset → 초기 상태 얻기
2. ε-greedy로 행동 선택
3. 환경에 행동 적용 → 다음 상태, 보상 얻기
4. (s, a, r, s', done)을 buffer에 저장
5. buffer에서 배치 샘플링 → 신경망 업데이트
6. 주기적으로 target network 동기화
7. 에피소드 종료 시 누적 보상 기록
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import config
from environment import TradingEnvironment
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent


def set_seed(seed: int):
    """재현성을 위한 시드 고정"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    """
    DQN 에이전트를 학습시키는 메인 함수
    
    Returns:
        agent: 학습된 DQNAgent
        history: 학습 통계 dict
    """
    # ====================================================
    # 1. 초기화
    # ====================================================
    print("=" * 70)
    print(" DQN 학습 시작")
    print("=" * 70)
    
    set_seed(config.RANDOM_SEED)
    
    # 데이터 로드
    print("\n[1/4] 데이터 로드 중...")
    if not os.path.exists("train_data.csv"):
        print("  train_data.csv가 없습니다. data_loader.py를 먼저 실행하세요.")
        return None, None
    
    train_df = pd.read_csv("train_data.csv")
    print(f"  → 학습 데이터: {len(train_df)}일")
    
    # 환경 생성
    print("\n[2/4] 환경 생성 중...")
    env = TradingEnvironment(train_df)
    print(f"  → 상태 차원: {env.state_dim}, 행동 수: {env.NUM_ACTIONS}")
    
    # 에이전트 + 버퍼 생성
    print("\n[3/4] 에이전트 생성 중...")
    agent = DQNAgent(env.state_dim, env.NUM_ACTIONS)
    buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
    
    # 학습 통계 기록용
    history = {
        "episode_rewards": [],      # 에피소드별 누적 보상
        "episode_returns": [],      # 에피소드별 수익률(%)
        "episode_losses": [],       # 에피소드별 평균 손실
        "epsilons": [],             # 에피소드별 ε
        "portfolio_values": [],     # 에피소드별 최종 포트폴리오
    }
    
    print(f"\n[4/4] 학습 시작 (총 {config.NUM_EPISODES} 에피소드)")
    print("-" * 70)
    
    # ====================================================
    # 2. 메인 학습 루프
    # ====================================================
    start_time = time.time()
    target_update_counter = 0
    
    for episode in range(1, config.NUM_EPISODES + 1):
        # 에피소드 초기화 (랜덤 시작점 적용 - 다양한 시장 국면 학습)
        max_start = env.get_max_start_step()
        random_start = np.random.randint(env.window_size, max_start)
        state = env.reset(start_step=random_start)
        episode_reward = 0.0
        episode_losses = []
        step_count = 0
        
        # ----- 에피소드 내부 루프 -----
        while not env.done and step_count < config.MAX_STEPS_PER_EPISODE:
            # (1) 행동 선택 (ε-greedy)
            action = agent.select_action(state, training=True)
            
            # (2) 환경 step
            next_state, reward, done, info = env.step(action)
            
            # (3) 경험 저장
            buffer.push(state, action, reward, next_state, done)
            
            # (4) 학습 (버퍼에 충분한 데이터가 쌓였을 때만)
            if buffer.is_ready(config.MIN_REPLAY_SIZE):
                batch = buffer.sample(config.BATCH_SIZE)
                loss = agent.update(batch)
                episode_losses.append(loss)
                
                # (5) Target network 주기적 동기화
                target_update_counter += 1
                if target_update_counter % config.TARGET_UPDATE_FREQ == 0:
                    agent.update_target_network()
            
            # (6) ε 감소 (글로벌 스텝 기준)
            agent.decay_epsilon()
            
            # 상태 업데이트
            state = next_state
            episode_reward += reward
            step_count += 1
        
        # ----- 에피소드 종료 후 통계 기록 -----
        final_value = env.get_portfolio_value()
        episode_return = (final_value / config.INITIAL_CASH - 1) * 100
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        
        history["episode_rewards"].append(episode_reward)
        history["episode_returns"].append(episode_return)
        history["episode_losses"].append(avg_loss)
        history["epsilons"].append(agent.epsilon)
        history["portfolio_values"].append(final_value)
        
        # ----- 주기적 로그 출력 -----
        if episode % config.LOG_INTERVAL == 0 or episode == 1:
            elapsed = time.time() - start_time
            recent_rewards = history["episode_rewards"][-config.LOG_INTERVAL:]
            recent_returns = history["episode_returns"][-config.LOG_INTERVAL:]
            
            print(
                f"Ep {episode:4d}/{config.NUM_EPISODES} | "
                f"Reward: {np.mean(recent_rewards):+7.2f} | "
                f"Return: {np.mean(recent_returns):+6.2f}% | "
                f"Loss: {avg_loss:.4f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Buffer: {len(buffer):5d} | "
                f"Time: {elapsed:.0f}s"
            )
    
    # ====================================================
    # 3. 학습 종료 후 처리
    # ====================================================
    total_time = time.time() - start_time
    print("-" * 70)
    print(f"\n[학습 완료] 총 소요 시간: {total_time/60:.1f}분")
    
    # 모델 저장
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/dqn_final.pt"
    agent.save(model_path)
    
    # 히스토리 저장 (나중에 분석용)
    np.savez(
        "checkpoints/training_history.npz",
        rewards=history["episode_rewards"],
        returns=history["episode_returns"],
        losses=history["episode_losses"],
        epsilons=history["epsilons"],
        portfolio_values=history["portfolio_values"],
    )
    
    # ====================================================
    # 4. 학습 곡선 시각화
    # ====================================================
    plot_training_curves(history)
    
    return agent, history


def plot_training_curves(history: dict):
    """
    학습 통계를 4개 패널 그래프로 시각화

    이 그래프가 발표에서 가장 중요한 자료가 된다.
    """
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DQN Training Curves", fontsize=16, fontweight="bold")
    
    episodes = range(1, len(history["episode_rewards"]) + 1)
    
    # (1) 누적 보상
    ax = axes[0, 0]
    ax.plot(episodes, history["episode_rewards"], alpha=0.3, color="steelblue", label="Per episode")
    # 이동평균 (트렌드 파악용)
    window = min(20, len(episodes) // 5)
    if window > 1:
        moving_avg = pd.Series(history["episode_rewards"]).rolling(window).mean()
        ax.plot(episodes, moving_avg, color="navy", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Episode Rewards (수렴 확인)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (2) 포트폴리오 수익률
    ax = axes[0, 1]
    ax.plot(episodes, history["episode_returns"], alpha=0.3, color="green", label="Per episode")
    if window > 1:
        moving_avg = pd.Series(history["episode_returns"]).rolling(window).mean()
        ax.plot(episodes, moving_avg, color="darkgreen", linewidth=2, label=f"{window}-ep moving avg")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Break-even")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return (%)")
    ax.set_title("Portfolio Returns")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # (3) 손실
    ax = axes[1, 0]
    ax.plot(episodes, history["episode_losses"], alpha=0.5, color="orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average TD Loss")
    ax.set_title("TD Loss (학습 안정성)")
    ax.set_yscale("log")  # 손실은 로그 스케일이 보기 좋음
    ax.grid(alpha=0.3, which="both")
    
    # (4) ε 감소
    ax = axes[1, 1]
    ax.plot(episodes, history["epsilons"], color="purple", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (Exploration Rate)")
    ax.set_title("Epsilon Decay (탐색→착취)")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[그래프 저장 완료] {output_path}")
    plt.close()


if __name__ == "__main__":
    agent, history = train()
    
    if history is not None:
        # 최종 통계 요약
        print("\n" + "=" * 70)
        print(" 최종 학습 결과 요약")
        print("=" * 70)
        
        # 마지막 20% 에피소드 (수렴 후) 통계
        n = len(history["episode_returns"])
        last_n = max(n // 5, 1)
        recent_returns = history["episode_returns"][-last_n:]
        
        print(f"\n[전체 학습]")
        print(f"  - 평균 수익률: {np.mean(history['episode_returns']):+.2f}%")
        print(f"  - 최고 수익률: {np.max(history['episode_returns']):+.2f}%")
        print(f"  - 최저 수익률: {np.min(history['episode_returns']):+.2f}%")
        
        print(f"\n[최근 {last_n} 에피소드 (수렴 후)]")
        print(f"  - 평균 수익률: {np.mean(recent_returns):+.2f}%")
        print(f"  - 표준편차:    {np.std(recent_returns):.2f}%")
        print(f"  - 최종 ε:      {history['epsilons'][-1]:.3f}")