FROM gitlab-registry.cern.ch/thartlan/containers/pyroot:v4

ARG NB_USER=jovyan
ARG NB_UID=1000

ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN yum install -y python-pip python-jupyroot
RUN pip install --upgrade pip
RUN pip install --no-cache-dir notebook==5.*
RUN pip install jupyterhub
RUN pip install jupyterlab

RUN useradd -m --shell=/bin/bash --uid=${NB_UID} ${NB_USER}
ADD . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

WORKDIR ${HOME}

ENTRYPOINT []
