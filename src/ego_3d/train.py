from ego_3d.model import Model
from ego_3d.data import MyDataset

def train():
    dataset = MyDataset("data/processed")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
