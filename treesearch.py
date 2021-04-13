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


class GameState(object):
    def __init__(self, node, mcProb, endVal):
        self.node = node
        self.mcProb = mcProb
        self.endVal = endVal

# Outputs a fake probability dist p for legal moves and a fake value v for the state
def fakeNN(moveList, debug = False):
    num_moves = len(moveList)
    randomlist = random.sample(range(1, 100), num_moves)
    if debug:
        print("num moves")
        print(num_moves)
        print("random list")
        print(randomlist)
        print(sum(randomlist))
    total = sum(randomlist)
    if total !=0:
        randomlist = [number / total for number in randomlist]
        return randomlist, random.uniform(-1, 1)
    if num_moves == 0:
        return [], -10



def convert_to_int(board):
    mapped = {
        'P': 1,     # White Pawn
        'p': -1,    # Black Pawn
        'N': 3,     # White Knight
        'n': -3,    # Black Knight
        'B': 3,     # White Bishop
        'b': -3,    # Black Bishop
        'R': 5,     # White Rook
        'r': -5,    # Black Rook
        'Q': 9,     # White Queen
        'q': -9,    # Black Queen
        'K': 100,     # White King
        'k': -100     # Black King
        }
    epd_string = board.epd()
    list_int = []
    for i in epd_string:
        if i == " ":
            return list_int
        elif i != "/":
            if i in mapped.keys():
                list_int.append(mapped[i])
            else:
                for counter in range(0, int(i)):
                    list_int.append(0)

def boardScore(board):
    score = sum(convert_to_int(board))
    if not board.turn:
        score = -1 * score
    newScore = score
    if score > 9:
        newScore = 9
    if score < -9:
        newScore = -9
    posVal = newScore/9
    return score, posVal

def betterFakeNN(board):
    moveList = [move for move in board.legal_moves]
    curScore = -math.inf
    moveIdx = []
    posVal = 0
    for x in range(0, len(moveList)):
        moveToTake = str(moveList[x])
        board.push_san(moveToTake)
        newScore, newPosVal = boardScore(board)
        if (newScore > curScore):
            moveIdx = [x]
            curScore = newScore
            posVal = newPosVal
        if (newScore==curScore):
            moveIdx.append(x)
        board.pop()
    num_moves = board.legal_moves.count()
    moveProbs = [0] * num_moves
    for move in moveIdx:
        moveProbs[move] = 1/len(moveIdx)
    return moveProbs, posVal



def MCTS(n, verbose):
    children = n.children
    if(children == []):
        if n.board.is_game_over():
            #We need this to be numerical TODO
            result = n.board.result()
            if result[2] == '0':
                return 1
            if result[2] == '1':
                return -1
            return 0
        else:
            
            moveList = [move for move in n.board.legal_moves]
            prior_p, state_val = fakeNN(moveList)
            #should this be -1
            for x in range(0, len(moveList)):
                moveToTake = str(moveList[x])
                temp = n.board.copy()
                n.board.push_san(moveToTake)
                try:
                    node = Node(n.board.copy(), [0, 0, 0, prior_p[x]])
                except:
                    print("PRIORP:  ")
                    print(prior_p)
                    print("MOVELIST:  ")
                    print(moveList)
                    prior_ps, state_vals = fakeNN(moveList, True)
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


"""
Function: Plays a move using the monteCarlo tresearch
Inputs
Returns
"""
def pickMove(node, maxIter= 1600):
    if node.board.is_game_over():
        result = node.board.result()
        if result[2] == '0':   # you win --output looks like 1-0, 0-1, 1/2-1/2
            return 1, 0
        if result[2] == '1':   # you lose
            return -1, 0
        return 0, 0
    #Do we just wanna reset each iter?
    #node = Node(board, [0,0,0,0])
    for x in range(maxIter):
        result = MCTS(node, False)
    probs = [0] * len(node.children)
    for x in range(len(probs)):
        probs[x] = (node.children[x].visit_ct)/node.visit_ct
    #this works and returns the max index dk why tho ask ari
    if len(probs) == 0:
        print(node.board)
        print(node.children)
    index_max = max(range(len(probs)), key=probs.__getitem__)
    bestNode = node.children[index_max]
    return probs, bestNode

"""
Function: 
Inputs
Returns
"""
def playGame(maxMoves= 1600, maxIter = 1600):
    game_states = []
    board = chess.Board()
    node = Node(board, [0,0,0,0])
    for x in range(maxMoves):
        probs, node = pickMove(node, maxIter)
        if node == 0:   # game is over
            for x in reversed(range(len(game_states))):
                game_states[x].endVal = probs
                probs *= -1
            break
        print(node.board, '\n')
        game_states.append(GameState(node, probs, 0))
    return game_states

for x in range(0, 100):
    game = playGame(500,10)
    print(game[-1].node.board)
    print(game[-1].endVal)