# Installation and Implementation Notes


## Running `mac_sim.app`

I kept on getting errors after downloading the `mac_sim.app` files. The error message wasn't too descript. It ended up being, and this was verified over a couple of laptops, that we needed to update the permissions on the application. So, after downloading and unzipping, a directory called `mac_sim.app/` was created. Simply type

`chmod -R u+x mac_sim.app`

And things should be fixed.

## Docker Container

I decided to try to install the Docker container via the instructions from CarND-Term1-Starter-Kit this time for educational and informational reasons. This turned out to be a nontrivial effort, and things didn't work right out of the box. For anyone who's interested in learning about how to do it, here are my trials and tribulations.

The basic problem is that cuDNN v5.1 is not installed as a library. Because of that, you're likely to run into something like:

`
tensorflow/stream_executor/cuda/cuda_dnn.cc:221] Check failed: s.ok() could not find cudnnCreate in cudnn DSO; dlerror: /root/miniconda3/envs/carnd-term1/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow.so: undefined symbol: cudnnCreate Aborted
`

A quick note about what *didn't work*. I knew that I could go the driver route and NVIDIA libraries route. That seemed a bit much, and I thought that a bunch of conda installs, or inherits could have solved it. Basically, relying on other people having looked at the problem before. Scouring through online forums, I came across the note to replace the first line (where the container inherits from) that currently looks like:

```
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
```

to

```
FROM gcr.io/tensorflow/tensorflow.
```

In the end, going to NVIDIA and getting the libraries was the right answer, and signing up, providing an e-mail and number, and downloading wasn't as much of a hassle compared to the time that I had spent trying to circumnavigate with Docker and conda. I was also worried that NVIDIA wouldn't have older versions. As it turns out, it has v5.1, one of the last things it has on its download page. I didn't manage to get the debian version to work, but I did end up getting the tar file to work by copying the appropriate files (after unzipping) to the right directories, e.g., the `*.so` libraries to `/usr/local/cuda/lib64`, and the `*.h` files accordingly to the includes folder.
