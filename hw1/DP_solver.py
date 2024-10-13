import numpy as np
import heapq
from gridworld import GridWorld


class DynamicProgramming:
    """Base class for dynamic programming algorithms"""

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for DynamicProgramming

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.threshold = 1e-4  # default threshold for convergence
        self.values = np.zeros(grid_world.get_state_space())  # V(s)
        self.policy = np.zeros(grid_world.get_state_space(), dtype=int)  # pi(s)

    def set_threshold(self, threshold: float) -> None:
        """Set the threshold for convergence

        Args:
            threshold (float): threshold for convergence
        """
        self.threshold = threshold

    def get_policy(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy
        """
        return self.policy

    def get_values(self) -> np.ndarray:
        """Return the values

        Returns:
            np.ndarray: values
        """
        return self.values

    def get_q_value(self, state: int, action: int) -> float:
        """Get the q-value for a state and action

        Args:
            state (int)
            action (int)

        Returns:
            float
        """
        # TODO: Get reward from the environment and calculate the q-value
        next_state, reward, done = self.grid_world.step(state, action)
        q_value =  reward + (self.discount_factor * self.values[next_state] ) * (1 - done)
        return q_value

class IterativePolicyEvaluation(DynamicProgramming):
    def __init__(
        self, grid_world: GridWorld, policy: np.ndarray, discount_factor: float
    ):
        """Constructor for IterativePolicyEvaluation

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): policy (probability distribution state_spacex4)
            discount (float): discount factor gamma
        """
        super().__init__(grid_world, discount_factor)
        self.policy = policy

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float: value
        """
        # TODO: Get the value for a state by calculating the q-values
        value = 0
        for action in range(self.grid_world.get_action_space()):
            value += self.policy[state][action] * self.get_q_value(state, action)
        return value

    def evaluate(self):
        """Evaluate the policy and update the values for one iteration"""
        # TODO: Implement the policy evaluation step
        delta = 0
        updated_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            v = self.values[state]
            updated_values[state] = self.get_state_value(state)
            delta = max(delta, abs(v - updated_values[state]))
        self.values = updated_values
        return delta

    def run(self) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the iterative policy evaluation algorithm until convergence

        # Initialize V(s) for all states
        self.values = np.zeros(self.grid_world.get_state_space())
        while True:
            if self.evaluate() < self.threshold:
                break

class PolicyIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for PolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values

        value = self.get_q_value(state, self.policy[state])
        return value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        delta = 0
        updated_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            v = self.values[state]
            updated_values[state] = self.get_state_value(state)
            delta = max(delta, abs(v - updated_values[state]))
        self.values = updated_values
        return delta
        

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        policy_stable = True
        for state in range(self.grid_world.get_state_space()):
            old_action = self.policy[state]
            self.policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the policy iteration algorithm until convergence

        while True:
            while True:
                if self.policy_evaluation() < self.threshold:
                    break
            if self.policy_improvement():
                # V ~ v* and pi ~ pi*
                break
            

class ValueIteration(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)

    def get_state_value(self, state: int) -> float:
        """Get the value for a state

        Args:
            state (int)

        Returns:
            float
        """
        # TODO: Get the value for a state by calculating the q-values
        # for all actions and return the maximum value
        value = max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
        return value

    def policy_evaluation(self):
        """Evaluate the policy and update the values"""
        # TODO: Implement the policy evaluation step
        delta = 0
        updated_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            v = self.values[state]
            updated_values[state] = self.get_state_value(state)
            delta = max(delta, abs(v - updated_values[state]))
        self.values = updated_values
        return delta

    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the value iteration algorithm until convergence
        while True:
            if self.policy_evaluation() < self.threshold:
                break
        self.policy_improvement()


class AsyncDynamicProgramming(DynamicProgramming):
    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """Constructor for ValueIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        super().__init__(grid_world, discount_factor)
        self.predecessors = [set() for _ in range(grid_world.get_state_space())]
        self._compute_predecessors()
        self.theta = 0.1

    def policy_evaluation(self):
        delta = 0
        # updated_values = np.zeros(self.grid_world.get_state_space())
        for state in range(self.grid_world.get_state_space()):
            v = self.values[state]
            # updated_values[state] = self.get_state_value(state)
            # delta = max(delta, abs(v - updated_values[state]))
            self.values[state] = max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
            delta = max(delta, abs(v - self.values[state]))
        # self.values = updated_values
        return delta
    
    def _compute_predecessors(self):
        """Compute predecessors of each state."""
        for state in range(self.grid_world.get_state_space()):
            for action in range(self.grid_world.get_action_space()):
                next_state, _, _ = self.grid_world.step(state, action)
                self.predecessors[next_state].add(state)
    
    def get_state_value(self, state):
        value = max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
        return value
    
    def policy_improvement(self):
        """Improve the policy based on the evaluated values"""
        # TODO: Implement the policy improvement step
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])

    def run(self) -> None:
        """Run the algorithm until convergence"""
        # TODO: Implement the async dynamic programming algorithm until convergence
        # 1. In-place dynamic programming
        while True:
            delta = 0
            for state in range(self.grid_world.get_state_space()):
                v = self.values[state]
                self.values[state] = max([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])
                delta = max(delta, abs(v - self.values[state]))
            if delta < self.threshold:
                break
        for state in range(self.grid_world.get_state_space()):
            self.policy[state] = np.argmax([self.get_q_value(state, action) for action in range(self.grid_world.get_action_space())])



        # 2. Prioritized sweeping        
        # priority_queue = []
        # for state in range(self.grid_world.get_state_space()):
        #     error = abs(self.get_state_value(state) - self.values[state])
        #     if error > self.theta:
        #         heapq.heappush(priority_queue, (-error, state))

        # while priority_queue:
        #     if self.policy_evaluation() < self.threshold:
        #         break
        #     _, state = heapq.heappop(priority_queue)
        #     self.values[state] = self.get_state_value(state)
        #     for pre_state in self.predecessors[state]:
        #         old_pre_value = self.values[pre_state]
        #         pre_value = self.get_state_value(pre_state)
        #         self.values[pre_state] = pre_value
        #         pre_error = abs(pre_value - old_pre_value)
                
        #         if pre_error > self.theta:
        #             heapq.heappush(priority_queue, (-pre_error, pre_state))
        # self.policy_improvement()  



        # # 3. Real time DP
        # for i in range(self.grid_world.get_state_space()):
        #     state = self.grid_world.get_state_space() - i - 1
        #     if self.policy_evaluation() < self.threshold:
        #         break
        #     while True:
        #         action = np.argmax([self.get_q_value(state, a) for a in range(self.grid_world.get_action_space())])
        #         self.values[state] = self.get_state_value(state)
        #         next_state, reward, done = self.grid_world.step(state, action)
        #         if done:
        #             break
        #         for pred in self.predecessors[state]:
        #             self.values[pred] = self.get_state_value(pred)
        #         state = next_state
        # self.policy_improvement()



        # # # 4. My own implementation
        # sorted_states = np.argsort(self.values)[::-1]
        # for state in sorted_states:
        #     if self.policy_evaluation() < self.threshold:
        #         break
        #     while True:
        #         action = np.argmax([self.get_q_value(state, a) for a in range(self.grid_world.get_action_space())])
        #         self.values[state] = self.get_state_value(state)
        #         next_state, reward, done = self.grid_world.step(state, action)
        #         if done:
        #             break
        #         for pred in self.predecessors[state]:
        #             self.values[pred] = self.get_state_value(pred)
        #         state = next_state

        # self.policy_improvement()
