# for python 3.7 & cuda 11.4
mamba create -n paddle python==3.7
<!-- mamba install cudatoolkit==11.8 -c conda-forge -->
mamba install cudatoolkit==11.4* -c conda-forge
mamba install -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-libraries-dev cudnn
pip install paddlepaddle-gpu==2.5.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install pgl
pip install rdkit seaborn scikit-learn==1.0.2 scipy numpy protobuf "urllib3<2.0" ml_collections tensorboardX
pip install paddlehelix --no-dependencies
mamba install openbabel

# for python 3.10 & cuda 11.8 paddle 2.5+ pgl 2.2.5
mamba create -n paddle_py310 python==3.10 rdkit==2024.3.5
mamba install cudatoolkit==11.8 -c conda-forge
mamba install -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-libraries-dev cudnn
pip install paddlepaddle-gpu==2.5.2 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install seaborn scikit-learn==1.3.1 scipy numpy==1.26.0 protobuf "urllib3<2.0" ml_collections tensorboardX
wget https://github.com/PaddlePaddle/PGL/archive/refs/tags/2.2.5.zip
    # run_command('unzip -n PGL-2.2.5.zip')
    # run_command('cd PGL-2.2.5/ && python setup.py build && python setup.py install && cd ../')