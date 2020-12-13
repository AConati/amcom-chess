# amnom-chess
Chess engine written in Python

Board Representation:
Approach 1(Minimum Bit approach):
There are a total of 12 unique pieces on a 64 square board. One extra bit of information is needed to determine who's turn it is at the current board. In total, this means the board can be represented by 257 bits, where the first bit is the turn bit and the rest represent pieces in groups of four:

Turn bit: 0 if white's turn to move, 1 if black's
Piece bits:

White Pawn: 0001
White Bishop: 0010
White Horse: 0011
White Rook: 0100
White Queen: 0101
White King: 0110

Black Pawn: 1001
Black Bishop: 1010
Black Horse: 1011
Black Rook: 1100
Black Queen: 1101
Black King: 1110

In this representation, the most significant bit is the color bit and the three other bits are the piece type bits.


Chess Engine approaches:

Approach 1(Fully reinforcement learning):

The general approach is to use a deep learning neural net as the heuristic value of a monte carlo tree search:
1. A neural net is chosen with randomized weights. Ex: Resnet50
2. MCTS value of each legal move is determined by playing out many games from that move. Ex 100 playouts
3. The DCNN chooses moves with the highest value within simulated playouts(Neural net has one output node)
4. DCNN backpropogation with the new MCTS value
5. Repeat 2-4 for many moves and games

