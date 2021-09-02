import chess
import random
import numpy as np
import torch
import treesearch
from Nuralnat import AmnomZero, ResidualLayer, CustomLoss

#Create a training set with treesearch to start:
#Play 100,000 games - done

#1. Training loop: sample 512 positions from the last 100,000 games
#retrain on positions -mostly done
#do this 1000 times - mostly done

#2. Totally tactical test training tournament:
#Play 400 games between new network and old version
#1. mechanism to save a neural net and load one
#2. mechanism to play games against each other
# If new wins 55% it becomes old

#3. Update training set:
# Play 10,000 games with old network against itself,
# add them to the end of training set
#and cycle the first 10,000 games off the beginning

#Repeat 1-3 until you feel nice


#TODO list order means nothing:
# XX Done Make loss function(its the sum of both the mean squared  loss for game states and cross entropy loss for move probs)

#-Make training loop
#-Make function to initialize and update training set(needs to open model in eval mode)
#-Make function to evaluate latest training loop with function that can be called to make a move with the model in eval mode


def get_batch(game_list, batch_size =32):
    if len(game_list) < 100000:
        print('Maybe not the end of the world?: ' + str(len(game_list)))

    random_games = random.sample(game_list, batch_size)
    return [random.sample(random_games[x], 1) for x in range(0, len(random_games))]
    
#Neural net expects 119*8*8 stack representation of a board. Make it
def convert_for_nn(board, history=1):
    bitboards = []
    black, white = board.occupied_co
    bitboards.extend((white & board.pawns, white & board.knights, white & board.bishops, white & board.rooks, white & board.queens, white & board.kings, black & board.pawns, black & board.knights, black & board.bishops, black & board.rooks, black & board.queens, black & board.kings))
    # Should add repetitions here, but it appears like python chess doesn't keep track of
    # repetitions, so the operation is slow. Ie. maybe not worth to check for repetitions?
    # Still unsure how much difference it makes

    plane_list = bitboards_to_array(bitboards)
    print(plane_list)
    print(plane_list.shape)
    turn_arr = np.full((8,8), board.turn)
    turn_arr = np.expand_dims(turn_arr, axis=0)
    plane_list = np.append(plane_list, turn_arr, axis=0)
    print( np.full((8,8), board.turn))
    print(plane_list)
    print(plane_list.shape)
    ## Total move count? No progress count? How to represent?

    plane_list = np.append(plane_list, np.full((8,8), board.castling_rights & chess.BB_H1))
    plane_list = np.append(plane_list, np.full((8,8), board.castling_rights & chess.BB_A1))
    plane_list = np.append(plane_list, np.full((8,8), board.castling_rights & chess.BB_H8))
    plane_list = np.append(plane_list, np.full((8,8), board.castling_rights & chess.BB_A8))
    return plane_list


def bitboards_to_array(bb):
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)
    
# Training set
# Function to initialize the training loop
def make_trainSet(model, samples = 100000):
    game_list = [] 
    for x in range(0, samples):
        print(x)
        game_list.append(treesearch.playGame(model))
    return game_list

def train(game_list, num_epochs = 1000):
    #put model in training mode and iterate through epochs
    model.train()
    for x in range(num_epochs):
        #Get a batch of 512 board positions and their corresponding mcProb(pi) and endVal(z)
        print("Starting epoch: " + str(x+1))
        curBatch = get_batch(game_list)
        #loop each game state
        loss = 0
        for game_state in curBatch:
            #retrieve endVal(z), mcProb(pi), and board from game state
            endval = game_state.endval
            mcProb = game_state.mcProb
            board = game_state.node.board
            #find the move probs(p) and board value(v) from applying the nn to the gamestate board
            nn_board = convert_for_nn(board)
            legal_moves, pout, vout = model(nn_board)
            #find the loss by comparing endVal to vout and mcProb to pout
            game_loss = criterion(endVal, vout, mcProb, pout)
            loss += game_loss
            #zero out past gradients
        loss = loss/len(curBatch)
        optimizer.zero_grad()
        #find new gradients created by loss with autograd
        loss.backward()
        #use optimizer to step based on gradients
        optimizer.step()
#should have all the data and take one step based on sum of error
#not step individually
