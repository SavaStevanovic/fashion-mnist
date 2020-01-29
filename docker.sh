docker build -t tf15playground .
docker run --gpus all -it -p 6006:6006 tf15playground
