import numpy as np 
import random 
import time
from utils import switch_perspective

class Player(object): 
    """
    Player class for checkers
    Implements basic functionality for interaction with board object 
    Methods for calculating moves for AI should be extended
    """
    def __init__(self, player_num, name='uh'): 
        # player state (p_state)
        self.name = name
        self.player_num = 1 if player_num == 1 else -1
        self.player_symbol = "X" if (player_num == 1) else "O"
        self.description = "{0} ({1})".format(name, self.player_symbol)
        self.locations = [] # list of tuples of tile, position tuple 
        self.moves_boards = [] 

    def _locations(self, board): 
        locations = [] 
        for j,row in enumerate(board): 
            for i,tile in enumerate(row): 
                if tile>0 : locations.append((tile,(i,j)))
        return locations

    def move(self, curr_loc, new_loc, board): 
        (i,j) = curr_loc 
        (new_i, new_j) = new_loc 
        new_board_state = board.copy()
        if new_j == 7: # promote to King 
            unit_type = 2 
        else: 
            unit_type = new_board_state[j,i]
        new_board_state[j,i] = 0 # move unit from old loc n
        new_board_state[new_j, new_i] = unit_type # move unit to new loc
        return new_board_state


    def remove(self, loc, board): 
        (i, j) = loc 
        new_board_state = board.copy()
        pc_type = new_board_state[j, i]
        new_board_state[j,i] = 0 
        return new_board_state

    def check_board(self, i, j, board): 
        if i > 7 or j > 7: return None 
        if i < 0 or j < 0: return None 
        return board[j, i]

    def _get_moves(self, board): 
        # returns dict : tuple of move path -> end board in first person perspective
        # result moves and recalc
        # self.moves_test = defaultdict(list)
        # self.moves = defaultdict(list)
        locations = self._locations(board)
        moves_boards = dict()
        pawn_dirs = [(1,1),(-1,1)]
        king_dirs = [(1,1),(-1,1),(1,-1),(-1,-1)]
        
        import copy 
        def jump_moves(i,j,board,depth,dirs,move_list=[]): 
            moves_found = False
            _move_list = copy.deepcopy(move_list) 
            _move_list.append((i,j))
            # if (depth > 0): dirs = king_dirs # second jump of multijump can be any direction
            for (di,dj) in dirs: 
                if self.check_board(i+di,j+dj,board) in (-1,-2) and self.check_board(i+di*2,j+dj*2,board)==0: 
                    new_board_state = self.move((i,j), (i+di*2,j+dj*2), board)
                    new_board_state = self.remove((i+di,j+dj), new_board_state)
                    jump_moves(i+di*2,j+dj*2,new_board_state,depth+1,dirs,_move_list[:])
                    moves_found = True
            if not moves_found and depth>0: 
                # moves_boards.append( [((i0,j0),(i,j)), board] )
                moves_boards[tuple(_move_list)] = board
        
        # first check if can jump 
        for unit_type, (i,j) in locations: 
            dirs = pawn_dirs if unit_type==1 else king_dirs
            jump_moves(i,j,board,depth=0,dirs=dirs)

        # import pdb;pdb.set_trace()
        # cant jump 
        if len(moves_boards)==0: 
            for unit_type, (i,j) in locations: 
                dirs = pawn_dirs if unit_type==1 else king_dirs
                for (di,dj) in dirs: 
                    if self.check_board(i+di,j+dj,board)==0: 
                        new_board_state = self.move((i,j), (i+di,j+dj), board)
                        # moves_boards.append(((i0,j0),(i+di,j+dj)),new_board_state)
                        moves_boards[((i,j),(i+di,j+dj))] = new_board_state
        # print(moves_boards)
        return moves_boards

    def select_move(self, board): 
        # change this per instance of derived class of player 
        try: 
            new_board = random.choice(self.moves_boards)
            if self.player_num != 1: 
                return switch_perspective(new_board)
            else: 
                return new_board
        except IndexError: 
            return None

class Node:    
    def __init__(self, board_state): 
        self.board_state = board_state
        self.children = []

class User(Player): 
    def __init__(self, player_num): 
        Player.__init__(self, player_num, name = "user-" + str(player_num))
    

    def translate(move, perspective): 
        # move: ([1-8], [1-8]) 
        # returns ([A-H], [1-8])
        if perspective != 1:
            row = move[0]
            col = move[1] 
            return
        else: 
            row = move[0]
            col = move[1] 
            return

    def select_move(self, board): 
        if self.player_num != 1: 
            board = switch_perspective(board)
        
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        print('Legal moves: ')
        enum_new_boards = [] # order new boards
        for i,(move_path, board_state) in enumerate(self._get_moves(board).items()): 
            # print( move_path )
            enum_new_boards.append(board_state)
            if self.player_num != 1:
                print(str(i) + ": " + "->".join( letters[col]+str(8-row) for (col,row) in move_path  ))
            else:
                print(str(i) + ": " + "->".join( letters[7-col]+str(row+1) for (col,row) in move_path ))
            
        new_board = None
        response = input('Enter move: ')
        while new_board is None: 
            try: 
                new_board = enum_new_boards[int(response)]
            except: 
                print('Not a valid move!')
                response = input('Enter move (between {0} and {1}):'.format(0, len(enum_new_boards)-1))

        if self.player_num != 1: 
            return switch_perspective(new_board)
        else: 
            return new_board

class AI(Player): 
    def __init__(self, player_num, max_search_depth, max_search_time): 
        Player.__init__(self, player_num, name = "ai-" + str(player_num))
        self.max_search_depth = max_search_depth
        self.max_search_time = max_search_time

    def evaluate(self,board): 
        return 0

    def alphabeta(self, node, depth):
        def maximize(node, depth, a, b):
            if depth==1: # calculate frontier nodes using _get_moves(), insert into tree data structure
                node.children = [Node(board_state) for board_state in self._get_moves(node.board_state).values() ]
            if depth==0 or len(node.children)==0:
                return self.evaluate(node.board_state)
            val = -1e9
            for child in node.children: 
                val = max(val, minimize(child, depth-1, a, b))
                a = max(a, val) 
                if a >= b: return val 
            return val 

        def minimize(node, depth, a, b):
            if depth==1: # calculate frontier nodes using _get_moves(), insert into tree data structure
                node.children = [Node(switch_perspective(board_state)) for board_state in self._get_moves(switch_perspective(node.board_state)).values() ]
            if depth==0 or len(node.children)==0:
                return self.evaluate(node.board_state)
            val = 1e9 
            for child in node.children:
                val = min(val, maximize(child, depth-1, a,b))
                b = min(b, val) 
                if a >= b: return val 
            return val 

        if depth==1: # calculate frontier nodes using _get_moves(), insert into tree data structure
            node.children = [Node(board_state) for board_state in self._get_moves(node.board_state).values() ]
        next_board_state = None
        val = -1e9
        a, b = -1e9, 1e9
        for child in node.children: 
            val = max(val, minimize(child, depth-1, a, b))
            if val > a: 
                a = val 
                next_board_state = child.board_state

        return next_board_state

    def select_move(self, board): 
        if self.player_num != 1: 
            root = Node(switch_perspective(board))
        else: 
            root = Node(board)

        # import pdb;pdb.set_trace()
        start_time = time.time()
        for curr_depth in range(1, self.max_search_depth+1): 
            selected_move = self.alphabeta(root, curr_depth)
            elapsed_time = (time.time()-start_time)
            if elapsed_time*2  >  self.max_search_time:  
                break
            # TODO: put time restriction in to break out

        print("depth: ", curr_depth)

        if self.player_num != 1: 
            return switch_perspective(selected_move)
        else: 
            return selected_move


class BoardSumAI(AI): 
    def __init__(self, player_num, max_search_depth, max_search_time): 
        AI.__init__(self, player_num,max_search_depth, max_search_time)

    def evaluate(self, board):
        # if (np.sum(board) > 0): print(board)
        # if (np.sum(board) > 0): print(np.sum(board))
        return np.sum(board) + .5 * random.uniform(0,1)


class BoardPositionAI(AI): 
    def __init__(self, player_num, max_search_depth, max_search_time, weights): 
        AI.__init__(self, player_num, max_search_depth, max_search_time)
        self.weights = weights

    def average_row(self, board):
        locs = self._locations(board)
        summ = 0
        non_kings = 0
        for unit_type, (i, j) in locs: 
            if (unit_type == 1):
                non_kings += 1
                summ += 7 if (i == 0) else i
        if (non_kings == 0):return 0
        return summ / non_kings

    def evaluate(self, board):
        return self.weights[0] * np.sum(board) \
                + self.weights[1] * self.average_row(board) \
                + .5 * random.uniform(0,1)
                

class BoardPositionEndGameAI(BoardPositionAI):
    def __init__(self, player_num, max_search_depth, max_search_time, weights):
        BoardPositionAI.__init__(self, player_num, max_search_depth, max_search_time, weights)
        
    def total_pieces(self, board): 
        return np.sum( np.where(board!=0,1,0) )
    
    def l2_norm(self, p1, p2): 
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(1/2)
    
    def maximum_distance(self, board): 
        # this might be pretty inefficient... 
        # we can instead only calculate norm for the "outer" pieces? 
        player_idxs = [[i,j] for _,(i,j) in self._locations(board)]
        other_idxs = [[7-i,7-j] for _,(i,j) in self._locations(switch_perspective(board))]
        # l2_norm = lambda p1,p2: ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(1/2)
        max_distance = 0
        min_distance = 1e9
        for player_idx in player_idxs: 
            for other_idx in other_idxs: 
                max_distance = max(self.l2_norm(player_idx, other_idx), max_distance)
        return max_distance

    def avg_dist_to_center(self, board): 
        player_idxs = [[i,j] for _,(i,j) in self._locations(board)]
        players_ct = len(player_idxs) + 1
        avg = sum( [self.l2_norm(player_idx, [3.5,3.5]) for player_idx in player_idxs] )/players_ct
        if avg == 0: 
            return 0.1
        else: 
            return avg

    def evaluate(self, board):
        # print( self.average_row(board))
        # print(np.sum(board)**3)
        board_sum = np.sum(board)
        total_pieces = (self.total_pieces(board))
        end_game_criterion = total_pieces < 12

        max_dist = 0.0
        if end_game_criterion:
            if board_sum >= 0: # winning position, be aggressive and penalize max distance 
                max_dist = (-self.maximum_distance(board))
            else: # losing position, be defensive and reward max distance
                max_dist = (self.maximum_distance(board))

        return (self.weights[0] * board_sum \
                + self.weights[1] * self.average_row(board) \
                + self.weights[2] * max_dist \
                + self.weights[3] * 1/self.avg_dist_to_center(board) \
                + .2 * random.uniform(0,1))                     

