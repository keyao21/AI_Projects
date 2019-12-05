from enum import Enum
import numpy as np 
import pdb 
import os
import re
import time
from utils import read_sample, switch_perspective
from Player import * 

class DisplayType(Enum):
    NONE = 1
    SIMPLE = 2
    PRETTY = 3

class Checkers(object): 
    def __init__(self, init_board, p1, p2):
        self.history = [init_board]
        self.noncapture_ct = 0
        self.p1 = p1
        self.p2 = p2 
        self.log_path = 'log.txt'

    def parse_command(self, ply):
        while True:
            response = input('Enter command: ')
            if response == '':
                return ply + 1
            elif response == 'e':
                return ply + 1000
            elif response.startswith("n"):
                num = response.split("n")[1].strip()
                if num == "":
                    return ply + 1
                elif num.isdigit():
                    return ply + int(num.strip())
            elif response.startswith("b"):
                num = response.split("b")[1].strip()
                if num == "":
                    return ply - 1
                elif num.isdigit():
                    return ply - int(num.strip())
            elif response.isdigit():
                return int(response.strip())
            elif response == "eval" or response == "v":
                print(self.p1.name, " heuristic: ", self.p1.evaluate(self.history[ply]))
                print(self.p2.name, " heuristic: ", self.p2.evaluate(switch_perspective(self.history[ply])))
            else:
                print("Invalid command!")

    def get_player(self, ply):
        return self.p1 if (ply % 2 == 1) else self.p2

    def pretty_print_board(self, board):
        pretty_mapping = {
            0: "  ",
            1: " X",
            2: "XX",
            -1: " O",
            -2: "OO"
        }
        for j in range(7,-1,-1):
            print("┼────" * 8 + "┼")
            for i in range(7,-1,-1):
                c = pretty_mapping.get(board[j, i])
                print("│ " + c + " ", end = "")
            print("│ " + str(j+1))
        print("┼────" * 8 + "┼")
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        bottom_axis = "".join("  " + letter + "  " for letter in letters) 
        print( bottom_axis )
 
    def display_board(self, disp_mode, ply, end=False):
        print( '='*40 )
        if disp_mode != DisplayType.NONE:
            print()
            print('Ply: ', ply)
            player = self.get_player(ply + 1)
            if disp_mode == DisplayType.SIMPLE:
                print(self.history[ply])
                print("Player to Move: ", player.player_num, "\n")
            elif disp_mode == DisplayType.PRETTY:
                self.pretty_print_board(self.history[ply])
                if end == False: print("Player to Move: ", player.description, "\n")

    def get_other_player(self, player):
       return self.p2 if (player == self.p1) else self.p1

    def run(self, disp_mode = DisplayType.PRETTY, command_mode = True):
        if command_mode:
            os.system("cls")

        ply = 0
        while True:
            # Calculate the history needed to display the chosen ply
            while len(self.history) - 1 < ply:
                ply_to_play = len(self.history)

                # Helper for debugging very long games
                if ply_to_play > 500:
                    if (disp_mode == DisplayType.NONE):
                        disp_mode = DisplayType.SIMPLE
                    pdb.set_trace()


                player = self.get_player(ply_to_play)
                self.display_board(disp_mode, ply_to_play - 1)
                new_board_state = player.select_move(self.history[ply_to_play - 1])
                self.history.append(new_board_state)

                game_result = self.check_winner(player, new_board_state, ply_to_play)
                if (game_result is not None):
                    winner, ending_ply = game_result
                    print(ending_ply, 'plies to end of game', '\n')
                    if (disp_mode != DisplayType.NONE):
                        print("Final board state:")
                    self.display_board(disp_mode, ending_ply, end=True)

                    # if command_mode:
                    #     input("Press enter to continue to next game...")

                    return game_result


            # self.display_board(disp_mode, ply)
            if command_mode:
                ply = max(0, self.parse_command(ply))
            else:
                ply += 1

    def total_pieces(self, board): 
        return np.sum( np.where(board!=0,1,0) )

    def has_pieces(self, board, player):
        ct = 0
        players_pieces = (1, 2) if (player.player_num == 1) else (-1, -2)
        for j in range(8):
            for i in range(8):
                if (board[j, i] in players_pieces):
                    return True
        return False

    def check_winner(self, player, board, ply):
        other_player = self.get_other_player(player)

        # If the current player (i.e the one we're selecting a move for) has no valid moves, they lose
        if board is None:             
            print('{0} wins! ({1} out of valid moves)'.format(other_player.description, player.name))
            return other_player, ply - 1
        # If the other player has no more pieces, the current player (who just ) wins
        elif (not self.has_pieces(board, other_player)):
            print('{0} wins!'.format(player.description))
            return player, ply
        elif self.total_pieces(board) == self.total_pieces(self.history[-1]): 
            self.noncapture_ct += 1
            if self.noncapture_ct == 160: 
                print('Draw!')
                return "Draw", ply
        else:
            return None

if __name__ == '__main__':
    # Note that this init board state is backwards in j (columns)
    init_board = np.asarray([
        [ 1, 0, 1, 0, 1, 0, 1, 0],
        [ 0, 1, 0, 1, 0, 1, 0, 1],
        [ 1, 0, 1, 0, 1, 0, 1, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0,-1, 0,-1, 0,-1, 0,-1],
        [-1, 0,-1, 0,-1, 0,-1, 0],
        [ 0,-1, 0,-1, 0,-1, 0,-1]
    ])

    # init_board = np.asarray([
    #     [ 0, 0, 1, 0, 1, 0, 1, 0],
    #     [ 0, 1, 0, 0, 0, 1, 0, 1],
    #     [ 1, 0, 1, 0, 1, 0, 0, 0],
    #     [ 0, 1, 0, 0, 0, 0, 0, 0],
    #     [ 0, 0, 0, 0, 0, 0, 0, 0],
    #     [ 0,-1, 0,-1, 0,-1, 0, 0],
    #     [-1, 0,-1, 0,-1, 0,-1, 0],
    #     [ 0,-1, 0,-1, 0, 0, 0, 0]
    # ])

    p = dict() # map from player number to Player object 
    input_file = input("Input filename (Hit Enter if none): ")
    if input_file: 
        init_board, user_num, search_time_out = read_sample(input_file)
        ai_num = 1 if user_num != 1 else 2
        p[ai_num] = BoardPositionEndGameAI(ai_num, max_search_depth = 100, max_search_time = search_time_out, weights = [1e6, 1e4, 1e2, 1])
        p[user_num] = User(user_num)
    else: 
        search_time_out = int(input("Search time (sec): "))
        user_flag = int(input("Enter 1 for user vs. ai, 2 for ai vs. ai: ")) 
        if user_flag==1: 
            user_num = int(input("Enter user player number (1 or 2): "))
            ai_num = 1 if user_num != 1 else 2
            p[ai_num] = BoardPositionEndGameAI(ai_num, max_search_depth = 100, max_search_time = search_time_out, weights = [1e6, 1e4, 1e2, 1])
            p[user_num] = User(user_num)
        else: 
            p[1] = BoardPositionEndGameAI(1, max_search_depth = 100, max_search_time = search_time_out, weights = [1e6, 1e4, 1e2, 1])
            p[2] = BoardPositionEndGameAI(2, max_search_depth = 100, max_search_time = search_time_out, weights = [1e6, 1e4, 1e2, 1])
    

    # p2 = BoardPositionAI(2, max_search_depth = 4, max_search_time = search_time_out,  weights = [10, 1])
    # p2 = BoardPositionAI(2, max_search_depth = 4, max_search_time = search_time_out,  weights = [10, 1])
    # p1 = BoardPositionEndGameAI(1, max_search_depth = 100, max_search_time = search_time_out, weights = [1e6, 1e4, 1e2, 1])
    # p2 = User(2)

    wins = { p[1].name: 0, p[2].name: 0, 'Draw': 0 }
    ply_cts = []
    times = []

    num_games = 4
    for i in range(num_games):
        print("Running Game ", i, " of ", num_games)
        test_game = Checkers(init_board, p[1], p[2])

        start = time.process_time()
        winner_player_num, ply_ct = test_game.run(disp_mode=DisplayType.PRETTY, command_mode=False)
        input("Press enter to continue to next game...")
        end = time.process_time()
        if winner_player_num == 'Draw': 
            wins['Draw'] += 1
        else: 
            wins[winner_player_num.name] += 1
        ply_cts = []
        ply_cts.append(ply_ct)
        times.append(end - start)

    print('Testing ')
    print("Wins: ", wins)
    print("Average Plies to Win: ", sum(ply_cts) / len(ply_cts))
    print("Average Time to Win: ", sum(times) / len(times))
