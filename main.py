# -*- coding: utf-8 -*-
import neat
import random
import pickle
import itertools
import visualize
import os
import re

"""
Created on Sat Oct 29 22:19:09 2022

@author: Mmdrlol.fr
"""
"""
an AI using neat for 2048
"""


def game(board, direction, points=0):
    """here is the game of 2048 I tried to make as optimized as possible"""
    # direction: 0=up 1=down 2=left 3=right
    """test to know if the board is blocked"""
    if 0 not in board:
        for n in range(len(board)-1):
            # check if two same numbers are next to each other
            if board[n] == board[n+1]:  # horizontally
                break
            elif n <= len(board)-5 and board[n] == board[n+4]:  # vertically
                break
        else:
            # if no 0 is in the board and if no merger is possible
            return False, 0

    """cut the board into 4 lines to facilitate the merger"""
    if direction == 0 or direction == 1:
        # make 4 lists of columns to merge upwards or downwards
        list_lines = [[board[i+j*4] for j in range(4)] for i in range(4)]

    elif direction == 2 or direction == 3:
        # make 4 lists of lines to merge leftwards or rightwards
        list_lines = [board[i:i+4] for i in range(0, 16, 4)]

    else:
        print("an error occurred")
        return False, 0

    """inverse the lines if its are not in the right direction"""
    # to merge the list from right to left
    if direction == 1 or direction == 3:
        list_lines = [lines[::-1] for lines in list_lines]

    """merge the squares"""
    for i, lines in enumerate(list_lines):
        if not all(square == 0 for square in lines):
            """delete all the 0 in the lines to put its to the right"""
            remove = [0 for rm in range(lines.count(0))]  # count the removed 0
            lines = [n for n in lines if n != 0]  # delete the 0
            lines.extend(remove)  # replace the removed 0 to the right

            """add to numbers next to when its are the same"""
            for n in range(len(lines)-1):
                # check if two same numbers (except 0) are next to each other
                if lines[n] != 0 and lines[n] == lines[n+1]:
                    lines[n:n+2] = [lines[n]*2, 0]  # merge the two numbers
                    points += lines[n]  # add the result to the score

            """delete all the 0 in the lines to put its at the end"""
            remove = [0 for rm in range(lines.count(0))]
            lines = [n for n in lines if n != 0]
            lines.extend(remove)

            """inverse the lines if its have been inversed"""
            if direction == 1 or direction == 3:
                lines = lines[::-1]

            list_lines[i] = lines

    """conbine the lines to make the board"""
    if direction == 0 or direction == 1:
        # reassemble the columns horizontally to remake the board
        new_board = list(itertools.chain.from_iterable(list(zip(*list_lines))))
    elif direction == 2 or direction == 3:
        # reassemble the lines vertically to remake the board
        new_board = list(itertools.chain.from_iterable(list_lines))

    """check if the decision has moved the board"""
    if board == new_board:
        return False, 0

    """put randomly a 2 or a 4 in a square which is empty"""
    square = -1
    # count the number of 0 in the board and take randomly one
    for i in range(random.randint(1, new_board.count(0))):
        square = new_board.index(0, square+1)
    if random.randint(0, 10) == 1:
        num = 4
        points += 4
    else:
        num = 2
    new_board[square] = num

    return new_board, points


def eval_genomes(genomes, config):
    """test the genomes in the 2048 game and give the fitness of the genome"""
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        run = True
        board = [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
            ]
        if random.randint(0, 10) == 1:
            num = 4
        else:
            num = 2
        # put the first random number
        board[random.randint(0, len(board))] = num
        genome.fitness = num

        while run:
            output = net.activate(board)
            # return the direction the genome has taken
            decision = output.index(max(output))
            board, points = game(board, decision)
            genome.fitness += points

            # if the board is blocked or if the move hasn't moved it
            if board is False:
                run = False


def eval_genome(genome, config):
    """same as eval_genomes but for parallelism or theading,
    it takes only one genome at once and return the fitness of the genome
    instead of adding it to genome.fitness"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    run = True
    board = [
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
        ]
    if random.randint(0, 10) == 1:
        num = 4
    else:
        num = 2
    board[random.randint(0, len(board)-1)] = num
    fitness = num

    while run:
        output = net.activate(board)
        decision = output.index(max(output))
        board, points = game(board, decision)
        fitness += points
        if board is False:
            run = False
            return fitness


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '2048config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    """automaticly reloads to the biggest checkpoint when there is one"""
    list_checkpoint = [item for item in os.listdir()
                       if item.startswith('neat-checkpoint-')]
    if len(list_checkpoint) == 0:
        p = neat.Population(config)
    else:
        """warning: check if the last checkpoint isn't empty
        or it will return an error"""
        p = neat.Checkpointer.restore_checkpoint(
            max(list_checkpoint,
                key=lambda num: int(re.findall(r"\d+$", num)[0])))

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000))

    "------------------normal-------------------"
    # winner = p.run(eval_genomes)

    "----------------parallelism----------------"
    # advice: parallelism is faster
    # personnaly, I have put the number of core of my cpu
    pe = neat.ParallelEvaluator(6, eval_genome)
    winner = p.run(pe.evaluate)

    "-----------------theading------------------"
    # pe = neat.ThreadedEvaluator(6, eval_genome)
    # winner = p.run(pe.evaluate)
    # pe.stop()

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    node_names = {
        0: "up", 1: "down", 2: "left", 3: "right",

        -1:  "a1", -2:  "b1", -3:  "c1", -4:  "d1",
        -5:  "a2", -6:  "b2", -7:  "c2", -8:  "d2",
        -9:  "a3", -10: "b3", -11: "c3", -12: "d3",
        -13: "a4", -14: "b4", -15: "c4", -16: "d4"}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    # save the best genome to be used in test_2048.py
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
