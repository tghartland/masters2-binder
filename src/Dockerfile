FROM gitlab-registry.cern.ch/thartlan/containers/pyroot:v5

WORKDIR /masters

RUN pip install --no-cache redis

ADD ../data ./data/
ADD *.cpp ./
ADD *.h ./
# ADD *.so ./
ADD limit_dist.py ./
ADD limit_dist_data_qstar.py ./
ADD plot_brazil.py ./
