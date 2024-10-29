mamba create -n paddle python==3.7
<!-- mamba install cudatoolkit==11.8 -c conda-forge -->
mamba install cudatoolkit==11.4* -c conda-forge
mamba install -c nvidia/label/cuda-11.4.3 cuda-nvcc cuda-libraries-dev cudnn
pip install paddlepaddle-gpu==2.5.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install pgl rdkit==2022.3.4 seaborn scikit-learn==1.0.2 scipy numpy protobuf "urllib3<2.0" ml_collections tensorboardX
pip install paddlehelix --no-dependencies
mamba install openbabel