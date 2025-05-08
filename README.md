# Ego_3D, A dataset for robots
This project evaluates state-of-the-art 3D semantic segmentation models in robotics-relevant scenarios. While current top models perform well on complete 360-degree point clouds, they haven't been thoroughly tested on the egocentric, partially occluded point clouds typical in real-world robotics applications. The project aims to assess how these advanced segmentation techniques perform when faced with the partial observations and occlusions that robots commonly encounter, bridging the gap between idealized test conditions and practical robotic vision challenges.

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template), a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting started with Machine Learning Operations (MLOps).

## PointTransformerV3 and Sonata

### Environment set-up

**Dependencies:**
- Cuda 12.4
- Python <= 3.12.9

1. Create or activate virtual environment
   We used uv packet manager but if you are using pip you should be able to make it work.
3. Ensure submodules are up to date by running:
```
git submodule init
git submodule update
```

3. Export CUDA_HOME environment variable
```
echo 'export CUDA_HOME=/appl/cuda/12.4.0' >> ~/.bashrc
```

4. Install torch modules
```
pip install --no-build-isolation torch==2.5.1  torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

5. Install Pointcept as a module and run other CUDA commands
```
pip install --no-build-isolation -e ./Pointcept
source ~/.bashrc
export CUDA_HOME=/appl/cuda/12.4.0
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
```

6. Install Pointops as a module
```
pip install --no-build-isolation ./Pointcept/libs/pointops
```

7. Install pointgroup_ops
```
cd externals 
git clone https://github.com/sparsehash/sparsehash.git
cd sparsehash
./configure --prefix=$HOME/.local
make
make install
export CPLUS_INCLUDE_PATH=$HOME/.local/include:$CPLUS_INCLUDE_PATH
cd ../../Pointcept/libs/pointgroup_ops/
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
pip install --no-build-isolation .
```
9. Install flash attention:
```
uv pip install flash-attn --no-build-isolation
```
10. Update torch-scatter implementation to be CUDA complied
```
pip uninstall torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

# Development (Only for developers of the project)
Blackhole location of data in DTU HPC cluster (read-only)
```
/dtu/blackhole/0e/169006/ScanNet
```

Pointcept pre-processed validation dataset
```
/dtu/blackhole/0e/169006/ScanNet/preprocessed/val/
```

Inside the path there's scenes like scene0356_01. Inside the scenes there's folders called 
```
color.npy  coord.npy  instance.npy  normal.npy  segment200.npy  segment20.npy
```

Ego-centric slices of these scenes are stored in the folder
```
/dtu/blackhole/0e/169006/ScanNet/preprocessed/val/slices/
```
The slices are stored in folders like scene0356_01_0, scene0356_01_1, etc.
The slices are stored in the format
```
color.npy  coord.npy  instance.npy  normal.npy  segment200.npy  segment20.npy
```
