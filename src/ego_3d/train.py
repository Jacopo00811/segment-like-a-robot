# from ego_3d.model import Model
# from ego_3d.data import MyDataset

# def train():
#     dataset = MyDataset("data.processed")
#     model = Model()
#     # add rest of your training code here

# if __name__ == "__main__":
#     train()

# Pointcept/configs/scannet/semseg-pt-v3m1-0-base.py
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import PointTransformerV3

model = PointTransformerV3()
