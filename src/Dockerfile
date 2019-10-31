ARG PYROOT_IMAGE
FROM $PYROOT_IMAGE

WORKDIR /masters

RUN pip install --no-cache redis

ADD data ./data/
ADD src/*.cpp ./
ADD src/*.h ./
ADD src/limit_dist.py ./
ADD src/limit_dist_data_qstar.py ./
ADD src/plot_brazil.py ./
