import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import neat
import pickle
import numpy as np
import os
import copy

# arbitrary number that determines the size of input/output, higher is more possible data
ARB = 30

# have to grab genome and config


#nn_in = json.loads("""{"p2Units":[[],[],[],[],[],[],[],[]],"turnInfo":[0,0,-1,0],"p1Stats":[30.0,40.0,5.0,0],"p1Units":[[],[],[],[],[],[],[],[]],"p2Stats":[30.0,40.0,5.0,0],"events":{"selfDestruct":[],"breach":[],"damage":[],"shield":[],"move":[],"spawn":[],"death":[],"attack":[],"melee":[]}}""")

def flatten_array(arr):
    flattened_array = []
    for item in arr:
        if isinstance(item, list):
            flattened_array.extend(flatten_array(item))
        else:
            flattened_array.append(item)
    return flattened_array


def remove_string_from_list(lst):
    st = type("")
    filtered_list = [item for item in lst if type(item) != st]
    return filtered_list

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))
        self.nn_in = "{}"
        self.genome = ""
        self.config = ""

        local_dir = os.path.dirname(__file__)
        p_path = os.path.join(local_dir, "g.pickle")
        with open(p_path, "rb") as f:
            [self.genome, self.config] = pickle.load(f)

        self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

    def moves(self):
        #gamelib.debug_write(self.nn_in)
        invaljson = json.loads(self.nn_in)
        master = []
        if invaljson == {}:
            gamelib.debug_write("skipping")
            return [[0, 0, 0] * 60]
        master += (invaljson["p1Stats"])
        master += (invaljson["p2Stats"])
        for i in range(8):
            x = invaljson["p1Units"][i]
            l = len(x)
            if l >= ARB:
                master += (x[:ARB])
            else:
                master += (x)
                master += ([[0, 0, 0, "-1"]]*(ARB-l))
        for i in range(8):
            x = invaljson["p2Units"][i]
            l = len(x)
            if l >= ARB:
                master += (x[:ARB])
            else:
                master += (x)
                master += ([[0, 0, 0, "-1"]]*(ARB-l))
        o = np.array(self.net.activate(remove_string_from_list(flatten_array(master))))
        return o.reshape(60, 3)

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        # This is a good place to do initial setup
        self.scored_on_locations = []

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        self.strategy(game_state)

        game_state.submit_turn()

    def strategy(self, game_state):
        #nn will create moves
        #do nothing, rem, upgrade, wall, support, turret, scout, destructor, interceptor
        outlayer = self.moves()
        # outlayer = [[0, 0, 0] * 60]
        for i in outlayer:
            c = round(i[0])
            if c == 0:
                pass
            elif c == 1:
                game_state.attempt_remove([round(i[1]), round(i[2])])
            elif c == 2:
                game_state.attempt_upgrade([round(i[1]), round(i[2])])
            elif c == 3:
                game_state.attempt_spawn(WALL, [round(i[1]), round(i[2])])
            elif c == 4:
                game_state.attempt_spawn(SUPPORT, [round(i[1]), round(i[2])])
            elif c == 5:
                game_state.attempt_spawn(TURRET, [round(i[1]), round(i[2])])
            elif c == 6:
                game_state.attempt_spawn(SCOUT, [round(i[1]), round(i[2])])
            elif c == 7:
                game_state.attempt_spawn(DEMOLISHER, [round(i[1]), round(i[2])])
            elif c == 8:
                game_state.attempt_spawn(INTERCEPTOR, [round(i[1]), round(i[2])])


    def on_action_frame(self, turn_string):
        # Let's record at what position we get scored on
        #state = json.loads(turn_string)
        #if state["turnInfo"][2] == -1:
         #   self.nn_in = turn_string
         #   gamelib.debug_write(self.nn_in)
        self.nn_in = turn_string


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
