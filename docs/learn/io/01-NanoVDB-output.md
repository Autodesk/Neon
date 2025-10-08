# NanoVDB

## What is NanoVDB?

As the name indicates it's a mini-version of the much bigger OpenVDB library, both in terms of functionality and scope. In fact, a stand-alone C++11 implementation of NanoVDB is available in the file NanoVDB.h and the C99 equivalent in the files CNanoVDB.h, and PNanoVDB.h. However, NanoVDB offers one major advantage over OpenVDB, namely support for GPUs. In short, NanoVDB is a standalone static-topology implementation of the well-known sparse volumetric VDB data structure. In other words, while values can be modified in a NanoVDB grid its tree topology cannot.

Additionally, it also can have no external dependencies.

*The explanation is taken from their documentation*

## How do I compile a program that uses NanoVDB?:

When making Neon, you must cmake it with the setting ` -DNEON_USE_NANOVDB=ON `.

For example, you can make Neon with ` cmake -DNEON_USE_NANOVDB=ON .. `, which will then download and install NanoVDB on your computer.

## Where is the tool located?:

At Neon/libNeonCore/include/Neon/core/tools/io/ioToNanoVDB.h

## How do I use the tool?:

For the user, you simply have to instantiate the object `Neon::ioToNanoVDB`. At the time of writing, its constructor has 7 arguments.

1. `filename`: The name of the file you want to outupt to. For example, if it is `coolGrid`, the output file will be written to `coolGrid.nvdb`.
2. `dim`: The dimension of the output. If you set it to `(10, 10, 10)`, there will be 1000 datapoints outputted.
3. `fun`: This is an anonymous function which takes in an index and a cardinality (of types `Neon::Integer_3d<intType_ta>` and `int`, respectively), and should output the value you want to be stored at the corresponding index in the output. This function allows this tool to access internal values for your grid/field in the way you specify.
4. `card`: The cardinality of the output. Currently, only cardinalities of `1`, `3`, or `4` are supported.
5. `scalingData`: This is a scalar which scales the voxels in the output by the amount given.
6. `origin`: This is the index where the output starts at. The indices stored will be from `origin` to `origin + dim - (1,1,1)`.
7. `mask`: The anonymous function detailing which indices inside the inclusive range [`origin`, `origin + dim - (1,1,1)`] should be included in the output. This allows for sparse matrices to be stored. It should return `true` for indices that should be outputted, and `false` for those that shouldn't. You only need to consider indices inside that range.
