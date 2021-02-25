import chess

board = chess.Board()
legal_moves = board.legal_moves

# moves in form eg. 'g1h3'
for move in legal_moves:
    chess.Move.uci(move)


#We have a stack 8*8*73 where each 8*8 plane represents a move(eg knight 2 left 1 up). Flattened to 4672*1. 
#Need to define the move for each plane(eg top plane move up 1, second move up 2, etc.)
#Need to convert from legal moves to bitmask in the vector representing planes
#Then in forward function zero out fake moves and softmax

#Also maybe need function to go from selected move to a move format that we can then push in python chess to make the move because we need to do that in order to make the move which we need to do in order to progress the game which we need to progress in order to get lcoser to the end which we need to get to so that the game can end which we need to happen so that we can train but sometimes if the game is taking too long maybe we just dont always end we will see.
