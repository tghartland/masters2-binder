FROM centos:7

WORKDIR /masters

RUN yum install -y epel-release
RUN yum install -y root root-core python2-root
RUN yum install -y scipy

ADD data ./data/
ADD *.cpp ./
ADD *.h ./
ADD *.so ./
ADD limit_dist.py ./
ADD limit_dist_data_qstar.py ./
ADD plot_brazil.py ./
