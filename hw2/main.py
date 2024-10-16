import random
import numpy as np
import json
import wandb

from algorithms import (
    MonteCarloPrediction,
    TDPrediction,
    NstepTDPrediction,
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 2-1
STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
POLICY          = None
MAX_EPISODE     = 300
LEARNING_RATE   = 0.01
NUM_STEP        = 3
# 2-2
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500

def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_MC_prediction(grid_world: GridWorld,seed):
    print(f"Run MC prediction. Seed:{seed}")
    prediction = MonteCarloPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"Monte Carlo Prediction",
        show=False,
        filename=f"MC_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_TD_prediction(grid_world: GridWorld, seed):
    print(f"Run TD(0) prediction. Seed:{seed}")
    prediction = TDPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        learning_rate=LEARNING_RATE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"TD(0) Prediction",
        show=False,
        filename=f"TD0_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_NstepTD_prediction(grid_world: GridWorld,seed):
    print(f"Run N-step TD prediction. Seed:{seed}")
    prediction = NstepTDPrediction(
        grid_world,
        learning_rate=LEARNING_RATE,
        num_step=NUM_STEP,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed=seed,
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"N-step TD Prediction",
        show=False,
        filename=f"NstepTD_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"MC Policy Iteration",
        show=False,
        filename=f"MC_policy_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

def run_SARSA(grid_world: GridWorld, iter_num: int):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"SARSA",
        show=False,
        filename=f"SARSA_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

if __name__ == "__main__":
    seed = 1
    grid_world = init_grid_world("maze.txt",INIT_POS)
    run_MC_prediction(grid_world,seed)
    run_TD_prediction(grid_world,seed)
    run_NstepTD_prediction(grid_world,seed)

    # # 2-2
    grid_world = init_grid_world("maze.txt")
    run_MC_policy_iteration(grid_world, 512000)
    run_SARSA(grid_world, 512000)
    run_Q_Learning(grid_world, 50000)

    # # random generate 50 seeds
    # seeds = np.random.randint(0, 10000, 50)
    # MC_state_values = np.array([run_MC_prediction(grid_world,seed) for seed in seeds])
    # TD_state_values = np.array([run_TD_prediction(grid_world,seed) for seed in seeds])
    # # calculate the mean and variation of the state values
    # np.save("./MC_state_values.npy", MC_state_values)
    # np.save("./TD_state_values.npy", TD_state_values)
    # MC_mean = np.mean(MC_state_values, axis=0)
    # TD_mean = np.mean(TD_state_values, axis=0)
    # # var = sum(state value - mean)^2 / n
    # MC_var = np.sum((MC_state_values - MC_mean) ** 2, axis=0) / len(seeds)
    # TD_var = np.sum((TD_state_values - TD_mean) ** 2, axis=0) / len(seeds)

    # # save the mean and standard deviation of the state values
    # np.save("./MC_mean.npy", MC_mean)
    # np.save("./MC_var.npy", MC_var)
    # np.save("./TD_mean.npy", TD_mean)
    # np.save("./TD_var.npy", TD_var)

    # # calculate the bias(mean - gt)
    # gt = np.load("./sample_solutions/prediction_GT.npy")
    # mc_mean = np.load("./MC_mean.npy")
    # td_mean = np.load("./TD_mean.npy")
    # mc_var = np.load("./MC_var.npy")
    # td_var = np.load("./TD_var.npy")
    # print(f"GT: {gt}\n")
    # print(f"MC mean: {mc_mean}\n")
    # print(f"TD mean: {td_mean}\n")
    # print(f"MC var: {mc_var}\n")
    # print(f"TD var: {td_var}\n")
    
    
    # MC_bias = MC_mean - gt
    # TD_bias = TD_mean - gt
    # # visualize the mean and standard deviation of the state values
    # grid_world.visualize(MC_bias, title=f"MC Prediction Bias", show=False, filename=f"MC_prediction_bias.png")
    # grid_world.visualize(MC_var, title=f"MC Prediction var", show=False, filename=f"MC_prediction_var.png")
    # grid_world.visualize(TD_bias, title=f"TD Prediction Bias", show=False, filename=f"TD_prediction_bias.png")
    # grid_world.visualize(TD_var, title=f"TD Prediction var", show=False, filename=f"TD_prediction_var.png")

    # for seed in seeds:
        # wandb.init(project = "hw2", config = {"algorithm":"MC", "seed": seed})
        # run_MC_prediction(grid_world,seed)
        # wandb.finish()
        # wandb.init(project = "hw2", config = {"algorithm":"TD", "seed": seed})
        # run_TD_prediction(grid_world,seed)
        # wandb.finish()

    # Discuss and plot learning curves under 𝜖 values of (0.1, 0.2, 0.3, 0.4) on MC, SARSA, and Q-Learning (4%)
    # grid_world = init_grid_world("maze.txt")
    # for epsilon in [0.1, 0.2, 0.3, 0.4]:
    # # for epsilon in [0.1]:
    #     # wandb.init(project = "hw2", config = {"algorithm":"MC", "epsilon": epsilon})
    #     # run_MC_policy_iteration(grid_world, 512000)
    #     # wandb.finish()
    #     # wandb.init(project = "hw2", config = {"algorithm":"SARSA", "epsilon": epsilon})
    #     # run_SARSA(grid_world, 512000)
    #     # wandb.finish()
    #     wandb.init(project = "hw2", config = {"algorithm":"Q_Learning", "epsilon": epsilon})
    #     run_Q_Learning(grid_world, 512000)
    #     wandb.finish()