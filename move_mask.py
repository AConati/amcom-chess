import chess
import numpy
ranks = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}

# Return the list of legal moves, vector of move activations for each legal move
def move_mask(pout):
    board = chess.Board()
    legal_moves = board.legal_moves

    # moves in form eg. 'g1h3'
    legal_moves = [chess.Move.uci(move) for move in legal_moves]
    move_values = []

    for move in legal_moves:
        x = ranks[move[0]]
        y = int(move[1])
        plane = find_plane(move)
        vector_pos = find_vector_pos(plane, x, y)
        move_values.append(pout[vector_pos])

    return legal_moves, move_values


# return plane
def find_plane(move):
    #Initialize plane to -1 to catch errors
    plane = -1

    # such a minor thing to irk me but the way we have it set up,
    # "positive" distances actually indicate moving backwards/left
    h_dist = ranks[move[0]] - ranks[move[2]]
    v_dist = int(move[1]) - int(move[3])

    # Promotion Moves: underpromotions encompass planes 65-73
    # Order: Knight promotions, Bishop promotions, Rook promotions
    # 3 planes for each (left diagonal capture, forward move, right diagonal capture)

    if len(move) > 4:
        if move[-1] == 'q':
            # Queen promotions are included in queen move planes
            # Add once queen plane order is decided
            pass
        elif move[-1] == 'n':
            # kNight promotion
            plane = 65 + h_dist
        elif move[-1] == 'b':
            # Bishop promotion
            plane = 68 + h_dist
        elif move[-1] == 'r':
            # Rook promotion
            plane = 71 + h_dist
        else:
            plane = -2 # Unique error code


#Knight moves. First Knight plane is 57, I assigned them clockwise
    if h_dist != 0 and v_dist != 0 and h_dist != v_dist:
        if v_dist == 2:
            if h_dist == 1:
                plane = 56
            else:
                plane = 63
        if h_dist == 2:
            if v_dist == 1:
                plane = 57
            else:
                plane = 58
        if v_dist == -2:
            if h_dist == 1:
                plane = 59
            else:
                plane = 60
        if h_dist == -2:
            if v_dist == -1:
                plane = 61
            else:
                plane = 62
            #unique error code
        else:
            plane = -3
    
    # Queen move- planes 0 through 55
    else:
        # mentally, I'm going y axis - from bottom to top (ie backwards 7 to forwards 7)
        # although I understand that our distance definitions are reversed
        if h_dist == 0:
            if v_dist < 0:
                plane = v_dist + 7 # map "down" moves to 0 - 6
            else:
                plane = v_dist + 6 # map "up" from 7-13
        elif v_dist == 0:  # horizontal moves from left to right, mapped 14-27
            if h_dist < 0:
                plane = h_dist + 21 # map "left" from 14-20
            else:
                plane = h_dist + 20 # map "right" from 21-27
        elif v_dist * h_dist > 0:   # indicates same sign i.e. either down&left or up&right
            # map these from 28-41
            if v_dist < 0:      # arbitrary, could be h_dist < 0, same thing
                plane = v_dist + 35   # map down left from 28-34
            else:
                plane = v_dist + 34   # map up right from 35-41
        else:           # up left to bottom right moves mapped 42-55
            if h_dist < 0:
                plane = h_dist + 49   # map up left from 42-48
            else:
                plane = h_dist + 48   # map down right from 43-49

    return plane

def find_vector_pos(plane, x, y):
    return plane*64 + 8*(8-y) + x


# We have a stack 8*8*73 where each 8*8 plane represents a move(eg knight 2 left 1 up). Flattened to 4672*1. 
# Need to define the move for each plane(eg top plane move up 1, second move up 2, etc.)
# Need to convert from legal moves to bitmask in the vector representing planes
# Then in forward function zero out fake moves and softmax

# Also maybe need function to go from selected move to a move format that we can then push in python chess to make the move because we need to do that in order to make the move which we need to do in order to progress the game which we need to progress in order to get lcoser to the end which we need to get to so that the game can end which we need to happen so that we can train but sometimes if the game is taking too long maybe we just dont always end we will see.
