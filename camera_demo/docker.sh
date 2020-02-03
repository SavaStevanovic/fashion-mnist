docker build -t tf15playgroundcam .
xhost + 
docker run -e DISPLAY=$DISPLAY -it -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/video0 tf15playgroundcam
