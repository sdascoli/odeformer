FROM ic-registry.epfl.ch/mlo/base:ubuntu20.04-cuda110-cudnn8

# install some necessary tools.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        software-properties-common
RUN apt-get install -y \
        rsync \
        git \
        curl \
        wget \
        unzip \
        zsh \
        git \
        screen \
        tmux \
	emacs \
	openssh-server
	
# configure environments.
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# configure user.
ENV SHELL=/bin/bash \
    NB_USER=dascoli \
    NB_UID=136905 \
    NB_GROUP=liac-unit \
    NB_GID=11169
ENV HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER}
RUN chown -R ${NB_USER}:${NB_GROUP} ${HOME}

# The user gets passwordless sudo
RUN echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

###### switch to user and compile test example.
USER ${NB_USER}

###### switch to root
# expose port for ssh and start ssh service.
EXPOSE 22
# expose port for notebook.
EXPOSE 8888
# expose port for tensorboard.
EXPOSE 6666