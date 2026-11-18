import gymnasium as gym
import numpy as np
import neat
import pickle
import os

from environment import make_placement_env
from clears_reward import ClearRewardWrapper

SEED = 42
CONFIG_PATH = "neat_config.txt"
CHECKPOINT_DIR = "neat_checkpoints"

def create_env():
    env = make_placement_env(render_mode="rgb_array")
    #env = gym.wrappers.HumanRendering(env)
    env = ClearRewardWrapper(env)
    return env

def get_inputs(observation):
    return np.asarray(observation, dtype=np.float32).flatten().tolist()

def evaluate_genomes(genomes, config):
    env = create_env()
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation, info = env.reset(seed=SEED)
        fitness = 0.0
        done = False
        while not done:
            inputs = get_inputs(observation)
            ouput = net.activate(inputs)
            action = int(np.argmax(ouput))
            observation, reward, terminated, truncated, info = env.step(action)
            fitness += reward
            done = terminated or truncated
        genome.fitness = fitness
        env.close()

def main(model_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    # Get input/output dims
    test_env = create_env()
    observation, info = test_env.reset(seed=SEED)
    num_inputs = len(get_inputs(observation))
    num_outputs = test_env.action_space.n
    test_env.close()

    assert config.genome_config.num_inputs == num_inputs
    assert config.genome_config.num_outputs == num_outputs

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config, seed=SEED)

    # Training stats
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Save checkpoints every 10 generations
    p.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(CHECKPOINT_DIR, "neat-checkpoint-")
        )
    )

    # Run evolution
    winner = p.run(evaluate_genomes, n=1000)

    # Save winning genome
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:")
    print(winner)
    return winner

def continue_training(checkpoint_path, generations=100):
    population = neat.Checkpointer.restore.checkpoint(
        checkpoint_path
    )
    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(
        evaluate_genomes,
        generations
    )

if __name__ == "__main__":
    main()

    