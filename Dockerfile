FROM continuumio/anaconda3:5.3.0

RUN apt update
RUN apt install -y vim
RUN pip install awscli
RUN pip install pdb

COPY environment.yaml /
RUN conda env update -f environment.yaml

RUN mkdir /src
RUN mkdir /src/data
WORKDIR /src
