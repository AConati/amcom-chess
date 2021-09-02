import torch
#from Nuralnat import AmnomZero, ResidualLayer
import joblib
import os

class nnClass:
    def __init__(self, newNet, oldNet, gameList):
        self.newNet = newNet
        self.oldNet = oldNet
        self.gameList = gameList

    def saveClass(self, name, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        saveName = str(name) + '.pkl'
        model_path = os.path.join(path, saveName)
        print(self.oldNet)
        print(model_path)
        joblib.dump(nnClass(self.newNet, self.oldNet, self.gameList), model_path)
    
    def loadClass(self, path):
        nnClass = joblib.load(path)
        return nnClass

