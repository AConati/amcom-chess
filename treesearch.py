# Monte Carlo Tree Search . py

import chess
import random
import math
#cpuct controls exploration. 4 is optimal according to this:
#https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
cpuct = 4 #wowee this is glooobal


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
    if total !=0:
        randomlist = [number / total for number in randomlist]
        return randomlist, random.uniform(-1, 1)
    else:
        return []


def MCTS(n, verbose):
    children = n.children
    if(children == []):
        prior_p, state_val = fakeNN(n.board)
        moveList = [move for move in n.board.legal_moves]
        for x in range(0, len(moveList)):
            moveToTake = str(moveList[x])
            n.board.push_san(moveToTake)
            node = Node(n.board.copy(), [0, 0, 0, prior_p[x]])
            n.children.append(node)
            node.parent.append(n)
            node.prior_p = prior_p[x]
            n.board.pop()
            if verbose==True:
               print(node.board)
        n.visit_ct +=1
        return state_val
    #Keep recursing. First pick child to maximize Q+U. Second rootnode +=MCTS(child node), visit_ct +1,update total_val and mean_val.
    else:
        maxQU = -1
        bestChildIndex = -1
        for x in range(0, len(children)):
            nodeQ = children[x].mean_val
            nodeU = children[x].prior_p * cpuct * math.sqrt(n.visit_ct)/(1+children[x].visit_ct)
            nodeQU = nodeQ + nodeU
            if (nodeQU>maxQU):
                bestChildIndex = x
                maxQU = nodeQU
        n.visit_ct += 1
        n.total_val += MCTS(children[bestChildIndex], verbose)
        n.mean_val = n.total_val/n.visit_ct
        return n.mean_val

board = chess.Board()
node = Node(board, [0,0,0,0])
for x in range(16000):
    MCTS(node, False)
MCTS(node, True)
