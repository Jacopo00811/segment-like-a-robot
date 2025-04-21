from pointcept.datasets.scannet import ScanNetDataset
import torch

print(torch.__version__)

DATASET_ROOT = "/dtu/blackhole/0e/169006/ScanNet/preprocessed"



if __name__ == '__main__':

    dataset = ScanNetDataset(
        split='val',
        data_root=DATASET_ROOT
    )

    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )


