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
    if(len(move) > 4):
        # promotion
        pass

    h_dist = ranks[move[0]] - ranks[move[2]]
    v_dist = int(move[1]) - int(move[3])

    if h_dist != 0 and v_dist != 0 and h_dist != v_dist:
        # knight move
        pass

    # Queen move

    if h_dist == 0 and v_dist > 0:
        pass




# We have a stack 8*8*73 where each 8*8 plane represents a move(eg knight 2 left 1 up). Flattened to 4672*1. 
# Need to define the move for each plane(eg top plane move up 1, second move up 2, etc.)
# Need to convert from legal moves to bitmask in the vector representing planes
# Then in forward function zero out fake moves and softmax

# Also maybe need function to go from selected move to a move format that we can then push in python chess to make the move because we need to do that in order to make the move which we need to do in order to progress the game which we need to progress in order to get lcoser to the end which we need to get to so that the game can end which we need to happen so that we can train but sometimes if the game is taking too long maybe we just dont always end we will see.
