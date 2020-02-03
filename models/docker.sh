docker build -t tf15playground .
docker run --gpus all -it -v `pwd`/project:/app -p 6006:6006 tf15playground
