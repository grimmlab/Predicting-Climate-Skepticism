FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y python3 && apt-get install -y python3-pip  &&  \
    apt-get install -y r-base && apt-get install -y vim && apt-get install -y git
RUN mkdir /configfiles
COPY docker_requirements.txt /configfiles
RUN pip install --default-timeout=400 -r /configfiles/docker_requirements.txt
RUN Rscript -e "install.packages('BGLR')"