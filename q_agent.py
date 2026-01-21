import numpy as np


class QAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size))

    def _state_to_index(self, state):
        return state[0] * 5 + state[1]

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_idx = self._state_to_index(state)
        return np.argmax(self.q_table[state_idx])

    def learn(self, state, action, reward, next_state, done):
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)

        best_next_action = np.argmax(self.q_table[next_state_idx])
        td_target = reward + self.discount_factor * self.q_table[next_state_idx][
            best_next_action
        ] * (1 - done)
        td_error = td_target - self.q_table[state_idx][action]
        self.q_table[state_idx][action] += self.learning_rate * td_error

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
