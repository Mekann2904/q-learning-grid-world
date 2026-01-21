import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorld
from q_agent import QAgent

plt.rcParams["font.family"] = "Hiragino Sans"


def train_agent(episodes=1000, max_steps=50, seed=42):
    env = GridWorld(seed=seed)
    agent = QAgent(state_size=25, action_size=4, discount_factor=0.99)

    rewards_per_episode = []
    steps_per_episode = []

    print("訓練開始...")
    print(f"障害物位置: {env.obstacles}\n")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(
                f"エピソード {episode + 1}: 平均報酬 = {avg_reward:.2f}, ε = {agent.epsilon:.3f}"
            )

    print("\n訓練完了")
    return agent, env, rewards_per_episode, steps_per_episode


def visualize_policy(agent, env):
    print("\n学習された方策:")
    policy_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    for r in range(env.size):
        for c in range(env.size):
            state = (r, c)

            if state in env.obstacles:
                print(" X ", end="")
            elif state == env.goal:
                print(" G ", end="")
            elif state == env.start:
                print(" S ", end="")
            else:
                state_idx = agent._state_to_index(state)
                best_action = np.argmax(agent.q_table[state_idx])
                print(f" {policy_map[best_action]} ", end="")
        print()


def plot_results(rewards, steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    episodes = np.arange(1, len(rewards) + 1)

    ax1.plot(episodes, rewards, alpha=0.3, label="各エピソード")
    ax1.plot(
        episodes,
        np.convolve(rewards, np.ones(50) / 50, mode="same"),
        "r-",
        label="移動平均（50）",
    )
    ax1.set_xlabel("エピソード")
    ax1.set_ylabel("総報酬")
    ax1.set_title("学習曲線（報酬）")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(episodes, steps, alpha=0.3, label="各エピソード")
    ax2.plot(
        episodes,
        np.convolve(steps, np.ones(50) / 50, mode="same"),
        "r-",
        label="移動平均（50）",
    )
    ax2.set_xlabel("エピソード")
    ax2.set_ylabel("ステップ数")
    ax2.set_title("ステップ数の変化")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("\nグラフを 'training_results.png' に保存しました")


def test_agent(agent, env, num_tests=5):
    print("\nテスト（ε=0で最適方策を使用）:")

    for test in range(num_tests):
        state = env.reset()
        steps = 0
        path = [state]

        for _ in range(50):
            action = np.argmax(agent.q_table[agent._state_to_index(state)])
            next_state, reward, done = env.step(action)
            path.append(next_state)
            steps += 1
            state = next_state

            if done:
                break

        print(f"テスト {test + 1}: ステップ数 = {steps}, 経路 = {path}")


if __name__ == "__main__":
    agent, env, rewards, steps = train_agent(episodes=1000, max_steps=50, seed=42)

    visualize_policy(agent, env)

    plot_results(rewards, steps)

    test_agent(agent, env, num_tests=3)
