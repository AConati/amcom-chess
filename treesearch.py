# Monte Carlo Tree Search . py

import chess

class Node(object):
    def __init__(self, board, data):
        self.children = []
        self.board = board
        self.visit_ct = data[0] 
        self.total_val = data[1]
        self.mean_val = data[2]
        self.prior_p = data[3]
        



board = chess.Board()
print(board)