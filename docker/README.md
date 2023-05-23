<h2>Docker</h2>

Build docker image
```
./build
```
Once the image is built, run the docker image, specifying as argument the path to the folder where you have downloaded the data
```
./run path-to-dataset-folder
```

Once you are inside the container, build our package
```
cd /catkin_ws/ && gpu_build
```

You are ready to go!
