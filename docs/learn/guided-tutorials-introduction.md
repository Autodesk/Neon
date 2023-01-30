# Guided Tutorial - Introduction

Neon aims at making **multi-XPU** programming easier for computations on **volumetric data structure** by providing users
with a simple sequential programming model, while automatically applying varius optimizations under the hood.

Nowadays, accelerators are at the core of high performance computing and they   
 come in different types and configurations. Term **XPU** has been introduced as generic term to capture the diversity of accelerators, GPU, FPGA, TPU etc.
We target a collection of XPU computing accelerators that can be connected in shared memory or distributed fashion.
While this is the long-term scope of the project, at the moment, we support shared memory CPU and GPUs.
In particular, our current implementation is based on openMP (on CPU) and CUDA (on GPU).

**Volumetric data** structures are used as a way to create a digital representation of properties of three-dimensional objects. For example, we can use them to represent pressure, displacement or material properties. Volumetric data structures are used both in simulation and computer graphics tools. 

Automatic parallelization of generic sequential code is still an open challenge which could be referred as the holy grail for the HPC community. 
Neon addresses a more tractable challenge by **restricting the problem to specific domains**. Neon focus on problems based on regular spatial discretions like uniform cartesian grids (support for non-uniform cartesian is in development). 

Moreover, Neon primarily support **local operations** like **map** (the value of a cell depends only on the metadata stored in the same cell) or **stencil** (a cell value is computed using metadata associated to close cells). Reductions (like dot or norm) are the only global operations in Neon. Thanks to the information on both the structure of volumetric representation and the supported operations, Neon is able to **automatically decompose the problem domain** in different partitions, map partitions to available XPUs, handle any required communication between partitions and finally to organize the computation **to introduce optimizations like overlapping computation and communication** which are essential to achieve good performance.  

We use the following simple example of three operations (AXPY, Laplace and dot) to showcase the operation that are done by Neon under the hood.

| Simple user application in Neon and its dependency graph                   |                                                                             |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| <image src = "/learn/img/axpy-laplace-dot-code.png" width="300px"></image> | <image src = "/learn/img/axpy-laplace-dot-app.png" width="300px" ></image>  |  

As first step, Neon parses the user code and creates a data dependency graph of the application.  
The dependency graph is then extended into a multi-GPU graph, where required communication between GPUs have been added. 

| Different Multi-XPU Graph created by Neon                                   ||
|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| <image src = "/learn/img/axpy-laplace-dot-nocc.png" width="400px"></image> |<image src = "/learn/img/axpy-laplace-dot-eocc.png" width="400px"></image>  |
| <image src = "/learn/img/axpy-laplace-dot-socc.png" width="400px"></image> |<image src = "/learn/img/axpy-laplace-dot-e2occ.png" width="400px"></image> |

The multi-GPU graph, however, is just the starting point. Indeed Neon can apply a set of optimization to enable overlapping of computation and communication to improve scalability.  
All this optimization are entirely transparent to the users who focus on authoring code following a sequential programming model.

The structure of Neon is based ona set of abstraction levels, each one represented by a C++ library.

![](img/neon-layers.png){ style="width:500"}

The **[Domain](the-bases/03-domain-level.md)** and **[Skeleton](the-bases/04-skeleton-level.md)** are the most important abstraction for Neon users.
The Domain level introduces domain-specific mechanisms, currently Neon targets voxel based
computations: mechanisms are Cartesian grids, fields and stencils. The Skeleton level provides users with a
sequential programming model and in charge of transforming and optimizing user applications to be deployed into a
multi-device system. The Skeleton abstraction has its roots in the fields of parallel patterns and skeleton. 

Both the Domain and Skeleton rely on the other Neon abstraction levels: the **[System](the-bases/01-system-level.md)**, 
abstracts the specific XPU capabilities, 
and the Set **[Set](the-bases/02-the-set-level.md)** provides a simple interface to manage a set of XPUs. 


The following is the structure of the `Introduction and tutorial` section:

[//]: # ()
[//]: # ()
[//]: # (<center>)

[//]: # ()
[//]: # ()
[//]: # (| Abstraction |           Description           |                      Library                      |                                    )

[//]: # ()
[//]: # (|-------------|:-------------------------------:|:-------------------------------------------------:|)

[//]: # ()
[//]: # (| System      |        Device management        |    [libNeonSys]&#40;the-bases/01-system-level.md&#41;     |)

[//]: # ()
[//]: # (| Set         |     Multi device management     |    [libNeonSet]&#40;the-bases/02-the-set-level.md&#41;    |)

[//]: # ()
[//]: # (| Domain      | Domain mechanism  - voxel grids |   [libNeonDomain]&#40;the-bases/03-domain-level.md&#41;   |)

[//]: # ()
[//]: # (| Skeleton    |  Sequential programming model   | [libNeonSkeleton]&#40;the-bases/04-skeleton-level.md&#41; |)

[//]: # ()
[//]: # ()
[//]: # (</center>)

!!! Note

    To learn how to write an application with Neon, new users can mainly focus on the **Domain** ([link](the-bases/03-domain-level.md)) and **Skeleton** ([link](the-bases/04-skeleton-level.md)) documentation as it implicitelly covers all the nedded information from the other Neon abtraction levels. 