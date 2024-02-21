# Deep Learning

## Prerequisits
### Install psbody.mesh Compute Canada

```console
1. Clone mesh repository
$ git clone https://github.com/MPI-IS/mesh.git

2. Activate boost on compute canada cluster
$ module load boost

3. Specify boost include dir
$ module show boost

4. Install using Makefile
$ BOOST_INCLUDE_DIRS=/path/to/boost/include make all
```

### Install psbody.mesh local

```console
1. Clone mesh repository
$ git clone https://github.com/MPI-IS/mesh.git

2. Change mod
$ chmod 755 /mesh/Makefile

3. Install using Makefile
$ BOOST_INCLUDE_DIRS=/usr/include/boost make all

4. Install OpenGL
$ conda install -c anaconda pyopengl
```