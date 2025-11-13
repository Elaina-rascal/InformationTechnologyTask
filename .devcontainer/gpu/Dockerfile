FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

RUN pip3 install pandas openpyxl matplotlib

RUN apt-get update &&\
    apt-get install -y git cmake gcc g++ libjsoncpp-dev libxlsxwriter-dev
WORKDIR /package
RUN git clone https://github.com/hosseinmoein/DataFrame.git &&\
    cd DataFrame &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j4 &&\
    make install
#安装clangd20
RUN apt-get update && apt-get install -y wget && wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add - \
    && apt-add-repository "deb http://apt.llvm.org/bullseye/ llvm-toolchain-bullseye-20 main" \
    && apt-get update && apt-get install -y clangd-20 \
    && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-20 100