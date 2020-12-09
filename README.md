# amnom-chess
Chess engine written in Python

Board Representation:
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
