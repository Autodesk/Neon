# HDF5

## What is HDF5?

Hierarchical Data Format (HDF) is a set of file formats (HDF4, HDF5) designed to store and organize large amounts of data. Originally developed at the U.S. National Center for Supercomputing Applications, it is supported by The HDF Group, a non-profit corporation whose mission is to ensure continued development of HDF5 technologies and the continued accessibility of data stored in HDF.

*The definition was taken from Wikipedia https://en.wikipedia.org/wiki/Hierarchical_Data_Format*


## What is HighFive?

It is a modern header-only C++11 friendly interface for libhdf5. It is used in this context for HDF5 output.

You can read its documentation here: https://bluebrain.github.io/HighFive/

## How do I compile a program that uses HDF5?:

When making Neon, you must cmake it with the setting ` -NEON_USE_HDF5=ON `.

For example, you can make Neon with ` cmake -NEON_USE_HDF5=ON .. `, which will then download and install HighFive on your computer

You must also have HDF5 and Boost installed. For Ubuntu, you can do `sudo apt install libhdf5-dev` and `sudo apt install libboost-all-dev`

## Where is the tool located?:

At Neon/libNeonCore/include/Neon/core/tools/io/ioToHDF5.h

## How do I use the tool?:

For the user, you simply have to instantiate the object `Neon::ioToHDF5`. At the time of writing, its constructor has 7 arguments.

1. `filename`: The name of the file you want to outupt to. For example, if it is `coolGrid`, the output file will be written to `coolGrid.nvdb`.
2. `dim`: The dimension of the output. If you set it to `(10, 10, 10)`, there will be 1000 datapoints outputted.
3. `fun`: This is an anonymous function which takes in an index and a cardinality (of types `Neon::Integer_3d<intType_ta>` and `int`, respectively), and should output the value you want to be stored at the corresponding index in the output. This function allows this tool to access internal values for your grid/field in the way you specify.
4. `card`: The cardinality of the output. Currently, only cardinalities of `1`, `3`, or `4` are supported.
5. `scalingData`: This is a scalar which scales the voxels in the output by the amount given.
6. `origin`: This is the index where the output starts at. The indices stored will be from `origin` to `origin + dim - (1,1,1)`.
7. `chunking`: This is an integer 3d which stores the dimensions for which the HDF5 output should be chunked. You can play around with it to get different optimization results based on your use case.
8. `mask`: The anonymous function detailing which indices inside the inclusive range [`origin`, `origin + dim - (1,1,1)`] should be included in the output. This allows for sparse matrices to be stored. It should return `true` for indices that should be outputted, and `false` for those that shouldn't. You only need to consider indices inside that range.
