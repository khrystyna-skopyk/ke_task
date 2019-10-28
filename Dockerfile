FROM python:3.7-stretch
ADD ./ /opt/app
WORKDIR /opt/app
RUN pip install -r requirements.txt
RUN python nltk_download.py