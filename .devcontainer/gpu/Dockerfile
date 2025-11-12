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