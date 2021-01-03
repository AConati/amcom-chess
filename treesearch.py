# Monte Carlo Tree Search . py

import chess
import random

class Node(object):
    def __init__(self, board, data):
        self.parent = []
        self.children = []
        self.board = board
        self.visit_ct = data[0] 
        self.total_val = data[1]
        self.mean_val = data[2]
        self.prior_p = data[3]

# Outputs a fake probability dist p for legal moves and a fake value v for the state
def fakeNN(board):
    num_moves = board.legal_moves.count()
    randomlist = random.sample(range(0, 100), num_moves)

    total = sum(randomlist)

    randomlist = [number / total for number in randomlist]
    return randomlist, random.uniform(-1, 1)


board = chess.Board()
print(board)
prior_p, state_val = fakeNN(board)
print(prior_p, state_val)
