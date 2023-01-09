FROM continuumio/miniconda3 AS build
RUN apt-get update
# Installing git and SSH in case required by conda
RUN apt-get install -y git openssh-client openssh-server ca-certificates
RUN git config --global http.sslVerify false
# Authorize SSH Host
RUN mkdir -p /root/.ssh
RUN chmod 0700 /root/.ssh
RUN ssh-keyscan github.com > /root/.ssh/known_hosts
# Updating conda
RUN conda update -n base -c defaults conda
WORKDIR /FastAPI
COPY . .
# Creating the environment
RUN --mount=type=ssh,id=github_ssh_key conda env create -f environment.yml