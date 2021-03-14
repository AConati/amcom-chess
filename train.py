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
#-Make training loop
#XX Done Make loss function(its the sum of both the mean squared 
# loss for game states and cross entropy loss for move probs)
#-Make function to initialize and update training set(needs to open model in eval mode)
#-Make function to evaluate latest training loop
#with function that can be called to make a move with the model in eval mode