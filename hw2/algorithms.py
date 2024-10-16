import numpy as np
import json
from collections import deque

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        returns = {state: [] for state in range(self.state_space)}  # Store returns
        while self.episode_counter < self.max_episode:
            current_state = self.grid_world.reset()
            state_trace = []
            reward_trace = []
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                state_trace.append(current_state)
                reward_trace.append(reward)
                current_state = next_state

            G = 0
            # for each step of episode, t from T-1 to 0:
            #     G <- gamma * G + R_{t+1}
            #     unless the pair S_t, A_t appears in S_0, A_0, S_1, A_1, ..., S_{t-1}, A_{t-1}:
            #         Append G to Returns(S_t, A_t)
            #         Q(S_t, A_t) <- Average(Returns(S_t, A_t))
            for t in range(len(state_trace)-1, -1, -1):
                G = self.discount_factor * G + reward_trace[t]
                s = state_trace[t]
                if s not in state_trace[:t]:
                    returns[s].append(G)
                    self.values[s] = np.mean(returns[s])
            # wandb record state value bias and variance according to prediction_GT.npy
            # wandb.log({"state_value_bias": np.mean((self.values - gt)**2), "state_value_variance": np.var(self.values)})


class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        while self.episode_counter < self.max_episode:
            current_state = self.grid_world.reset()
            done = False
            while not done:
                next_state, reward, done = self.collect_data()
                # be careful with the done flag
                if done: 
                    td_target = reward
                else:
                    td_target = reward + self.discount_factor * self.values[next_state]

                td_error = td_target - self.values[current_state]
                self.values[current_state] += self.lr * td_error  # Update value per step
                current_state = next_state
            # wandb.log({"state_value_bias": np.mean((self.values - gt)**2), "state_value_variance": np.var(self.values)})
class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the N-step TD algorithm until max_episode"""
        while self.episode_counter < self.max_episode:
            current_state = self.grid_world.reset()
            T = np.inf  # Set T to infinity initially
            tau = 0  # Initialize tau
            t = 0  # Time step
            rewards = [0]  # To store the rewards
            states = [current_state]
            while True:
                if t < T:
                    next_state, reward, done = self.collect_data()
                    states.append(next_state)
                    rewards.append(reward)
                    if done:
                        T = t + 1
                tau = t - self.n + 1
                if tau >= 0:
                    G = sum([self.discount_factor ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + self.n, T)+1)])
                    if tau + self.n < T:
                        G += self.discount_factor ** self.n * self.values[states[tau + self.n]]
                    td_error = G - self.values[states[tau]]
                    self.values[states[tau]] += self.lr * td_error
                if tau == T - 1:
                    break
                t += 1

# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)      
        self.episodic_rewards = []
        self.losses = []                    # deterministic policy

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, state_trace, action_trace, reward_trace) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)
        G = 0
        # every visit MC
        loss = []
        for t in range(len(state_trace)-1, -1, -1):
            G = self.discount_factor * G + reward_trace[t]
            s = state_trace[t]
            estimation_loss = G - self.q_values[s, action_trace[t]]
            self.q_values[s, action_trace[t]] += self.lr * (estimation_loss)
            loss.append(abs(estimation_loss))
        self.losses.append(np.mean(loss))
        
    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy
        for state in range(self.state_space):
            best_action = np.argmax(self.q_values[state])
            self.policy[state] = np.ones(self.action_space) * self.epsilon / self.action_space
            self.policy[state][best_action] += (1.0 - self.epsilon)

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()

        while iter_episode < max_episode:
            # TODO: write your code here
            current_state = self.grid_world.get_current_state()
            state_trace   = []
            action_trace  = []
            reward_trace  = []
            # hint: self.grid_world.reset() is NOT needed here
            iter_episode += 1
            done = False
            while not done:
                action = np.random.choice(self.action_space, p=self.policy[current_state])
                next_state, reward, done = self.grid_world.step(action)
                action_trace.append(action)
                reward_trace.append(reward)
                state_trace.append(current_state)
                current_state = next_state
            self.policy_evaluation(state_trace, action_trace, reward_trace)
            self.policy_improvement()
            self.episodic_rewards.append(sum(reward_trace)/len(reward_trace))
            # if len(self.episodic_rewards) >= 10:
            #     avg_reward = sum(self.episodic_rewards[-10:]) / 10
            #     wandb.log({"Average Episodic Reward": avg_reward, "Episode": iter_episode})
            # if len(self.losses) >= 10:
            #     avg_loss = sum(self.losses[-10:]) / 10
            #     wandb.log({"Average Absolute Estimation Loss": avg_loss, "Episode": iter_episode})
            
class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon
    def epsilon_greedy_policy(self, state):
        """Generates an epsilon-greedy policy based on Q-values for the given state."""
        policy = np.ones(self.action_space) * self.epsilon / self.action_space
        best_action = np.argmax(self.q_values[state])
        policy[best_action] += (1.0 - self.epsilon)
        return policy
    
    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        if is_done:
            target = r
        else:
            target = r + self.discount_factor * self.q_values[s2, a2]
        td_error = target - self.q_values[s, a]
        self.q_values[s, a] += self.lr * td_error
        return abs(td_error)

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        is_done = False
        # TODO: write your code here
        # hint: self.grid_world.reset() is NOT needed here
        while iter_episode < max_episode:
            current_state = self.grid_world.get_current_state()
            iter_episode += 1
            # current_state = self.grid_world.reset()  # Start a new episode
            # action = self.choose_action(current_state)
            action_probs = self.epsilon_greedy_policy(current_state)
            action = np.random.choice(np.arange(self.action_space), p=action_probs)
            is_done = False
            loss = []
            reward_trace = []

            while not is_done:
                next_state, reward, is_done = self.grid_world.step(action)
                # next_action = np.random.choice(self.action_space) if np.random.rand() < self.epsilon else np.argmax(self.q_values[next_state])
                reward_trace.append(reward)
                next_action_probs = self.epsilon_greedy_policy(next_state)
                next_action = np.random.choice(np.arange(self.action_space), p=next_action_probs)
                # Update Q-values using SARSA
                loss.append(self.policy_eval_improve(current_state, action, reward, next_state, next_action, is_done))
                current_state = next_state
                action = next_action
            self.episodic_rewards.append(sum(reward_trace)/len(reward_trace))
            self.losses.append(np.mean(loss))
            # if len(self.episodic_rewards) >= 10:
            #     avg_reward = sum(self.episodic_rewards[-10:]) / 10
            #     wandb.log({"Average Episodic Reward": avg_reward, "Episode": iter_episode})
            # if len(self.losses) >= 10:
            #     avg_loss = sum(self.losses[-10:]) / 10
            #     wandb.log({"Average Absolute Estimation Loss": avg_loss, "Episode": iter_episode})
            
class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        self.buffer.append((s, a, r, s2, d))

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        # uniform random sampling self.sample_batch_size indexes from the buffer
        # return the batch of transitions
        indices = np.random.choice(len(self.buffer), self.sample_batch_size)
        return [self.buffer[i] for i in indices]

    def policy_eval_improve(self, s, a, r, s2, is_done):
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        if is_done:
            td_target = r  # If terminal state, no future rewards
        else:
            td_target = r + self.discount_factor * np.max(self.q_values[s2])  # Q-learning target: max_a' Q(S', a')
        td_error = td_target - self.q_values[s, a]
        self.q_values[s, a] += self.lr * td_error
        abs_error = abs(td_error)
        return abs_error

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        transition_count = 0
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            current_state = self.grid_world.get_current_state()
            iter_episode += 1
            is_done = False
            loss = []
            reward_trace = []

            while not is_done:
                # Epsilon-greedy action selection
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.action_space)
                else:
                    action = np.argmax(self.q_values[current_state])
                # Take a step in the environment
                next_state, reward, is_done = self.grid_world.step(action)
                reward_trace.append(reward)
                loss.append(self.policy_eval_improve(current_state, action, reward, next_state, is_done))
                # Store the transition (S, A, R, S', done) in the replay buffer
                self.add_buffer(current_state, action, reward, next_state, is_done)
                transition_count += 1
                B = []
                if transition_count % self.update_frequency == 0:
                    B = self.sample_batch()
                for (s, a, r, s2, is_done) in B:
                    self.policy_eval_improve(s, a, r, s2, is_done)
                current_state = next_state
            self.episodic_rewards.append(sum(reward_trace)/len(reward_trace))
            # print(f"np.mean(loss): {np.mean(loss)}, iter_episode: {iter_episode}")
            self.losses.append(np.mean(loss))
            # if len(self.episodic_rewards) >= 10:
            #     avg_reward = sum(self.episodic_rewards[-10:]) / 10
            #     wandb.log({"Average Episodic Reward": avg_reward, "Episode": iter_episode})
            # if len(self.losses) >= 10:
            #     avg_loss = sum(self.losses[-10:]) / 10
            #     # print(f"avg_loss: {avg_loss}")
            #     wandb.log({"Average Absolute Estimation Loss": avg_loss, "Episode": iter_episode})
            