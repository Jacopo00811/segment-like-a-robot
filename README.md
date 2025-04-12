# ego_3d

This project evaluates state-of-the-art 3D instance semantic segmentation models in robotics-relevant scenarios. While current top models perform well on complete 360-degree point clouds, they haven't been thoroughly tested on the egocentric, partially occluded point clouds typical in real-world robotics applications. The project aims to assess how these advanced segmentation techniques perform when faced with the partial observations and occlusions that robots commonly encounter, bridging the gap between idealized test conditions and practical robotic vision challenges.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Mask3D
### Project set-up
- Create conda environment: conda create -n ENVNAME python=3.10.9
- Downgrade pip to 23.3: python -m pip install --force-reinstall pip==23.3
- activate environment
- pip install numpy
- pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.4.1
- conda install -c conda-forge pycocotools==2.0.4
- conda env update --name ENVNAME --file PATH_TO_REPO/segment-like-a-robot/externals/Mask3D/environment.yml
- dependencies cuda 11.3, openblas default, gcc 9.5.0: module load gcc/9.5.0-binutils-2.38 openblas cuda/11.3   
- source ~/.bashrc 
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" 
export CUDA_HOME=/appl/cuda/11.3.0
- follow the rest of the project set-up (from the first pip3 command)


## PointTransformerV3

### Environment set-up

**Dependencies:**
- Cuda 12.4
- Python <= 3.12.9

1. Create or activate virtual environment

2. Ensure submodules are up to date by running:
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
cd externals 
git clone https://github.com/sparsehash/sparsehash.git
cd sparsehash
./configure --prefix=$HOME/.local
make
make install
export CPLUS_INCLUDE_PATH=$HOME/.local/include:$CPLUS_INCLUDE_PATH
cd ../../Pointcept/libs/pointgroup_ops/
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
uv pip install --no-build-isolation .


# Development (Only for developers of the project)
Blackhole location of data in DTU HPC cluster (read-only)
/dtu/blackhole/0e/169006/ScanNet
