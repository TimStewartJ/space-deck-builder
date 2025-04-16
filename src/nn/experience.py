import torch


class Experience:
    def __init__(self, state: torch.FloatTensor, action: int, reward: float, next_state: torch.FloatTensor, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __repr__(self):
        return f"Episode(state={self.state}, action={self.action}, reward={self.reward}, next_state={self.next_state}, done={self.done})"