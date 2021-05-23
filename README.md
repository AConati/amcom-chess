# amnom-chess
Chess engine written in Python
This was all wroooong

TOOOOODDDDDDOOOOOOs:

1. Generate a stack from a board position - Ari
We have a board like a chess.whatever board y'kno and we need a stack being the 8*8*73 stack where look at movemask for plane order. in train.py

1. fix bugs in nuralnat - Nut/marco
- squashed a little but also having more bugs now, after conv_1 in policy head we're getting shape [2,128,6,6]

2-5 have comments in train.py with more details
2. Fix the stepping in the train loop. Should only step once per batch and not for every sample in the batch

3. add model saving and loading

4. add functionality to play against itself and see if its less dumb

5. Update the training set by playing best net against itself - treesearch also needs to take in a model

6. model initialization and optimizer and such should be in a main. Generally just make a main that imports things and organize the function calls

7. castling rights function

8. no progress plane update function

9. total move count plane update function

10. too long game game end function? ^

8. keep in mind whether we need a history? repititions? who knows paper linked in nuralnat

