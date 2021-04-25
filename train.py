import random
import treesearch
import Nuralnat

#Create a training set with treesearch to start:
#Play 100,000 games

#1. Training loop: sample 512 positions from the last 100,000 games
#retrain on positions
#do this 1000 times

#2. Totally tactical test training tournament:
#Play 400 games between new network and old version
#If new wins 55% it becomes old

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


def get_batch(game_list):
    if len(game_list) < 100000:
        print('Maybe not the end of the world?: ' + str(len(game_list)))

    random_games = random.sample(game_list, 512)
    return [random.sample(random_games[x], 1) for x in range(0, len(random_games))]
    
#Neural net expects 119*8*8 stack representation of a board. Make it
def convert_for_nn(board):
    pass

# Training set
# Function to initialize the training loop
def make_trainSet(samples = 100000):
    game_list = [] 
    for x in range(0, samples):
        print(x)
        game_list.append(treesearch.playGame())
    return game_list

def train(game_list, num_epochs = 1000):
    #put model in training mode and iterate through epochs
    model.train()
    for x in range(num_epochs):
        #Get a batch of 512 board positions and their corresponding mcProb(pi) and endVal(z)
        print("Starting epoch: " + str(x+1))
        curBatch = get_batch(game_list)
        #loop each game state
        for game_state in curBatch:
            #retrieve endVal(z), mcProb(pi), and board from game state
            endval = game_state.endval
            mcProb = game_state.mcProb
            board = game_state.node.board
            #find the move probs(p) and board value(v) from applying the nn to the gamestate board
            nn_board = convert_for_nn(board)
            legal_moves, pout, vout = model(nn_board)
            #find the loss by comparing endVal to vout and mcProb to pout
            loss = criterion(endVal, vout, mcProb, pout)
            #zero out past gradients
            optimizer.zero_grad()
            #find new gradients created by loss with autograd
            loss.backward()
            #use optimizer to step based on gradients
            optimizer.step()



print("wooohooo")
g_list = make_trainSet()
print(g_list)