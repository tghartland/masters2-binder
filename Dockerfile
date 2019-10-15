FROM gitlab-registry.cern.ch/thartlan/containers/jupyter3:v5

ARG NB_USER=jovyan
ARG NB_UID=1000

ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN useradd -m --shell=/bin/bash --uid=${NB_UID} ${NB_USER}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

WORKDIR ${HOME}

ADD . ${HOME}

ENTRYPOINT []
