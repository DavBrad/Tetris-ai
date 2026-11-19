import gymnasium as gym
import numpy as np
import neat
import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx

from environment import make_placement_env
from clears_reward import ClearRewardWrapper

SEED = 42
EPISODES_PER_GENOME = 1 # 3 by default
EPISODE_CHANGE_INTERVAL = 10000 # Change seeds every x generations

CurrentGen = 0

CONFIG_PATH = "neat_config.txt"
CHECKPOINT_DIR = "neat_checkpoints"

def create_env():
    env = make_placement_env(render_mode="rgb_array")
    #env = gym.wrappers.HumanRendering(env)
    env = ClearRewardWrapper(env)
    return env

def get_inputs(observation):
    return np.asarray(observation, dtype=np.float32).flatten().tolist()

def choose_action(net, observation, action_mask):
    scores = np.full(len(action_mask), -np.inf)
    for i, candidate in enumerate(observation):
        if not action_mask[i]:
            continue
        inputs = get_candidate_inputs(candidate)
        scores[i] = net.activate(inputs)[0]

    return int(np.argmax(scores)) # return highest value output

def get_candidate_inputs(candidate):
    board = candidate[:20, 4:14]
    board = (board != 0).astype(np.float32)
    return board.flatten().tolist()

def get_generation_seeds(generation):
    generation = generation // EPISODE_CHANGE_INTERVAL
    rng = np.random.default_rng(SEED + generation)
    return rng.integers(0, 1_000_000, size=EPISODES_PER_GENOME).tolist()

def evaluate_genomes(genomes, config):
    global CurrentGen

    seeds = get_generation_seeds(CurrentGen)
    print(
        f"\nEvaluating generation {CurrentGen} "
        f"with seeds: {seeds}"
    )

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        episode_rewards = []

        for seed in seeds:
            env = create_env()
            observation, info = env.reset(seed=int(seed))

            action_mask = info["action_mask"]

            total_reward = 0.0
            done = False

            while not done:
                action = choose_action(net, observation, action_mask)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                action_mask = info["action_mask"]
                done = terminated or truncated
            episode_rewards.append(total_reward)
            env.close()
        genome.fitness = float(np.mean(episode_rewards))
        env.close()

    CurrentGen += 1
    best_genome_id, best_genome = max(
        genomes,
        key=lambda x: x[1].fitness
    )
    show_network(best_genome, config, CurrentGen)

def main(model_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         CONFIG_PATH)

    # Get input/output dims
    #test_env = create_env()
    #observation, info = test_env.reset(seed=SEED)
    #num_inputs = len(get_inputs(observation))
   #num_outputs = test_env.action_space.n
    #test_env.close()

    #print("Input neurons:", num_inputs)
    #print("Output neurons:", num_outputs)

    #assert config.genome_config.num_inputs == num_inputs
    #assert config.genome_config.num_outputs == num_outputs

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Training stats
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #best_reporter = BestGenomeReporter()
    #p.add_reporter(best_reporter)

    # Save checkpoints every 10 generations
    p.add_reporter(
        neat.Checkpointer(
            generation_interval=10,
            filename_prefix=os.path.join(CHECKPOINT_DIR, f"{model_name}-checkpoint-")
        )
    )

    # Run evolution
    winner = p.run(evaluate_genomes, n=1000)

    # Save winning genome
    with open(f"{model_name}_best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nBest genome:")
    print(winner)
    return winner

def continue_training(model_name, generations=100):
    population = neat.Checkpointer.restore_checkpoint(
        model_name
    )
    population.add_reporter(neat.StdOutReporter(True))

    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(
        evaluate_genomes,
        generations
    )

    with open(f"{model_name}_best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner


plt.ion()
fig, ax = plt.subplots(figsize=(14, 8))
def show_network(genome, config, generation):
    ax.clear()
    G = nx.DiGraph()
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    # Add nodes
    for key in input_keys:
        G.add_node(key, kind="input")

    for key in genome.nodes:
        kind = "output" if key in output_keys else "hidden"
        G.add_node(key, kind=kind)

    # Add enabled connections
    for connection in genome.connections.values():
        if connection.enabled:
            source, target = connection.key
            G.add_edge(
                source,
                target,
                weight=connection.weight
            )

    # Assign positions by node type
    pos = {}

    for i, key in enumerate(input_keys):
        pos[key] = (0, -i)

    hidden = [
        key for key in genome.nodes
        if key not in output_keys
    ]

    for i, key in enumerate(hidden):
        pos[key] = (1, -i)

    for i, key in enumerate(output_keys):
        pos[key] = (2, -i)

    # Draw nodes
    input_nodes = [
        n for n in G.nodes
        if G.nodes[n]["kind"] == "input"
    ]

    hidden_nodes = [
        n for n in G.nodes
        if G.nodes[n]["kind"] == "hidden"
    ]

    output_nodes = [
        n for n in G.nodes
        if G.nodes[n]["kind"] == "output"
    ]

    nx.draw_networkx_nodes(
        G, pos, nodelist=input_nodes,
        node_color="skyblue", node_size=100, ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=hidden_nodes,
        node_color="plum", node_size=300, ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=output_nodes,
        node_color="lightgreen", node_size=500, ax=ax
    )

    # Draw connections
    edges = list(G.edges(data=True))

    positive = [
        (u, v) for u, v, d in edges
        if d["weight"] >= 0
    ]

    negative = [
        (u, v) for u, v, d in edges
        if d["weight"] < 0
    ]

    nx.draw_networkx_edges(
        G, pos, edgelist=positive,
        edge_color="blue", alpha=0.5,
        arrows=True, ax=ax
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=negative,
        edge_color="red", alpha=0.5,
        arrows=True, ax=ax
    )

    ax.set_title(
        f"Generation {generation} | "
        f"Fitness: {genome.fitness:.2f} | "
        f"Hidden nodes: {len(hidden)}"
    )

    ax.axis("off")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

if __name__ == "__main__":
    #main("neat_m2")
    #main("neat_m3") # introduce episode seed change interval
    main("neat_oneseed") # try with only the same seed for 1k gens