from pointcept.datasets.scannet import ScanNetDataset




dataset_root_path = "/dtu/blackhole/0e/169006/ScanNet/preprocessed"


dataset = ScanNetDataset(
    split="train",
    data_root=dataset_root_path
)

print(dataset.get_data_list())