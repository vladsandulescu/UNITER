conda create -n uniter python=3.6 pip --yes
source ~/anaconda3/etc/profile.d/conda.sh
conda activate uniter

cd $HOME/playground/hmm/lib
git clone https://github.com/ChenRocks/UNITER.git
cd $HOME/playground/hmm/lib/UNITER

conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=9.0 -c pytorch --yes
conda install scikit-learn --yes
pip install pytorch-pretrained-bert==0.6.2 tensorboardX==1.7 ipdb==0.12 lz4==2.1.9 lmdb==0.97
pip install toolz cytoolz msgpack msgpack-numpy

# update OpenMPI to avoid horovod bug
mkdir tmp
cd tmp
rm -r /usr/local/mpi &&\ 
    wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.4.tar.gz &&\
    gunzip -c openmpi-3.1.4.tar.gz | tar xf - &&\
    cd openmpi-3.1.4 &&\
    ./configure --prefix=/usr/local/mpi --enable-orterun-prefix-by-default \
        --with-verbs --disable-getpwuid &&\
    sudo make -j$(nproc) all && sudo make install &&\
    sudo ldconfig &&\
    cd - && rm -r openmpi-3.1.4 && rm openmpi-3.1.4.tar.gz &&\
    cd - && rm -r tmp
    
# horovod
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod==0.16.4 &&\
    sudo ldconfig

# apex
echo "----------------- Install Apex -----------------"

# install according to this repo works
git clone https://github.com/jackroos/apex
cd ./apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./  

# install according to NVIDIA repo failes
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

#  install according to https://github.com/LuoweiZhou/VLP/blob/master/setup.sh also fails
# git clone -q https://github.com/NVIDIA/apex.git
# cd apex
# git reset --hard 1603407bf49c7fc3da74fceb6a6c7b47fece2ef8
# python setup.py install --cuda_ext --cpp_ext

cd $HOME/playground/hmm/lib/UNITER

pip install ipykernel
python -m ipykernel install --prefix=/home/ubuntu/anaconda3 --name uniter --display-name "Python (uniter)"

cd $HOME
echo "Done"
