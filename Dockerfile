FROM gitlab-registry.cern.ch/thartlan/containers/jupyterlab:v1

ARG NB_USER=jovyan
ARG NB_UID=1000

ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN useradd -m --shell=/bin/bash --uid=${NB_UID} ${NB_USER}

WORKDIR ${HOME}
ADD data ${HOME}/data
ADD masters ${HOME}/masters
ADD src ${HOME}/src
ADD *.yaml ${HOME}/
ADD *.ipynb ${HOME}/

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

ENTRYPOINT []
