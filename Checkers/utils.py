import numpy as np 

def read_sample(filename): 
    # filename = 'sampleCheckers.txt'
    np_board = np.zeros((8,8))
    player_num = None
    time_limit = None
    def translate_pieces(str_i): 
        trans_dict = {
            ' ': 0,
            '0': 0,
            '1': 1, 
            '2': -1, 
            '3': 2, 
            '4': -2
        }
        return trans_dict[str_i]

    with open(filename, 'r') as f: 
        for i,line in enumerate(f):
            if i<8: 
                if i%2==0: 
                    np_board[ 7-i , : ] = np.asarray([translate_pieces(line[i]) for i in range(0,16,2)])
                    # print(np_board)
                else:
                    np_board[ 7-i , : ] = np.asarray([translate_pieces(line[i]) for i in range(0,14,2)] + [0])
                    # print(np_board)
            elif i==8: 
                player_num = int(line[0])
            elif i==9: 
                time_limit = int(line[0])

    print(np_board)
    print(player_num)
    print(time_limit)
    return np_board, player_num, time_limit

def switch_perspective(board):
    # switch sign, rotate 180 degs 
    return None if (board is None) else np.rot90(board.copy()*-1, 2)

