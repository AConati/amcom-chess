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


def MCTS(Node n):
    children = n.children
    if(children ==[]):
        prior_p, state_val = fakeNN(n.board)
        children_moves = board.legal_moves
        for x in range(0, len(children_moves)):
            move = n.board.push_san(children_moves[x])
            node = Node(move, [0, 0, 0, prior_p[x]])
            n.children.append(node)
            node.parent.append(n)
            n.visit_ct += 1
            n.total_val += state_val
            n.mean_val = n.total_val/n.visit_ct
            return state_val
    #Keep recursing. First pick child to maximize Q+U. Second rootnode +=MCTS(child node), visit_ct +1,update total_val and mean_val.
    else():
        

