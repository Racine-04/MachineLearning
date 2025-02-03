FROM ubuntu:22.04

WORKDIR /app

RUN apt update
RUN apt install python3-pip -y
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install flask
RUN pip install waitress
RUN pip install transformers
RUN pip install requests

COPY . .

EXPOSE 80

CMD [ "python3", "./ResNetModel.py" ]