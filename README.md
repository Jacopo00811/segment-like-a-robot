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
- Create conda environment
- Downgrade pip to 23.3
- activate environment
- pip install "cython<3.0.0" && pip install --no-build-isolation pyyaml==5.4.1
- pip install "cython<3.0.0" \
&& pip install --no-build-isolation "pycocotools==2.0.4" \
&& pip install --no-build-isolation "pyyaml==5.4.1"
- follow the rest of the project set-up.
- dependencies cuda 11.3, openblas default, gcc 9.5.0


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
pip install --no-build-isolation torch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 --index-url https://download.pytorch.org/whl/cu124
```
5. Install Pointcept as a module
```
pip install --no-build-isolation -e ./Pointcept
```

6. Install Pointops as a module
```
pip install --no-build-isolation ./Pointcept/libs/pointops
```

7. Install pointgroup_ops
TODO: Fix this and update docs

