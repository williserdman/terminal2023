import neat
import os
import pickle
import subprocess
import json
import time

# gptd not checked
def get_winner(): 
    folder_path = "replays"
    file_extension = ".replay"

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Filter files with the specified extension
    replay_files = [file for file in files if file.endswith(file_extension)]

    if replay_files:
        # Sort the files by modification time (most recent first)
        replay_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

        # Get the path to the most recent replay file
        most_recent_file_path = os.path.join(folder_path, replay_files[0])

        # Read the last line of the file
        with open(most_recent_file_path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1].strip()

        # print("Last line of the most recent replay file:")
        return last_line
    else:
        # print("No replay files found in the specified folder.")
        return ""

# gptd not checked
def clear_replays():
    import os

    folder_path = "replays"  # Specify the folder path

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file and delete them
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print("All files in the folder have been deleted.")

def run_match(g1, g2, config):
    # Set up each agent
    with open("agent-1/g.pickle", "wb") as f:
        pickle.dump([g1, config], f)
    with open("agent-2/g.pickle", "wb") as f:
        pickle.dump([g2, config], f)

    # Run the match
    os.system("python3 scripts/run_match.py agent-1 agent-2")

    # Find the winner and do fitnesses
    # agent-1 will be p1 in replays
    last_frame = json.loads(get_winner())


    p1_stats = last_frame["p1Stats"]
    p1_health = p1_stats[0]
    p2_stats = last_frame["p2Stats"]
    p2_health = p2_stats[0]

    g1.fitness += p1_health - p2_health
    g2.fitness += p2_health - p1_health

    clear_replays()

    

def eval_genomes(genomes, config):
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            run_match(genome1, genome2, config)

def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-7')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(3))

    winner = p.run(eval_genomes, 500)

    with open("winner.pickle", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    print("has to be run from the terminal2023 dir, make sure java is installed")
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    run_neat(config)