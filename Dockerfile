FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y sudo  wget
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y git-all

RUN useradd -m efklidis

RUN chown -R efklidis:efklidis /home/efklidis

COPY --chown=efklidis . /home/efklidis/assessment

USER efklidis

RUN cd /home/efklidis/assessment && pip install -r requirements.txt

WORKDIR /home/efklidis/assessment