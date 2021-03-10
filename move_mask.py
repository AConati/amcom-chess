import chess

ranks = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}

# Return the list of legal moves, vector of move probabilities for each legal move
def move_mask(pout):
    board = chess.Board()
    legal_moves = board.legal_moves

    # moves in form eg. 'g1h3'
    legal_moves = [chess.Move.uci(move) for move in legal_moves]
    move_probabilities = []

    for move in legal_moves:
        x = ranks[move[0]]
        y = int(move[1])
        plane = find_plane(move)

    return legal_moves, move_probabilities


# return plane
def find_plane(move):

    #Initialize plane to -1 to catch errors
    plane = -1

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
            plane = 66 + h_dist
        elif move[-1] == 'b':
            # Bishop promotion
            plane = 69 + h_dist
        elif move[-1] == 'r':
            # Rook promotion
            plane = 72 + h_dist
        else:
            plane = -2 # Unique error code


#Knight moves. First Knight plane is 57, I assigned them clockwise
    if h_dist != 0 and v_dist != 0 and h_dist != v_dist:
        if v_dist == 2:
            if h_dist == 1:
                plane = 57
            else:
                plane = 64
        if h_dist == 2:
            if v_dist == 1:
                plane = 58
            else:
                plane = 59
        if v_dist == -2:
            if h_dist == 1:
                plane = 60
            else:
                plane = 61
        if h_dist == -2:
            if v_dist == -1:
                plane = 62
            else:
                plane = 63
    # Queen move

    if h_dist == 0 and v_dist > 0:
        pass
    return plane



# We have a stack 8*8*73 where each 8*8 plane represents a move(eg knight 2 left 1 up). Flattened to 4672*1. 
# Need to define the move for each plane(eg top plane move up 1, second move up 2, etc.)
# Need to convert from legal moves to bitmask in the vector representing planes
# Then in forward function zero out fake moves and softmax

# Also maybe need function to go from selected move to a move format that we can then push in python chess to make the move because we need to do that in order to make the move which we need to do in order to progress the game which we need to progress in order to get lcoser to the end which we need to get to so that the game can end which we need to happen so that we can train but sometimes if the game is taking too long maybe we just dont always end we will see.
