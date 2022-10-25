# Guided Tutorial - Introduction

Neon aims at making **multi-XPU** easier for computation on **volumetric data structure** by providing users with a
simple
sequential programming model.

Nowadays, computing accelerators come in different types and configurations. With the term **XPU**, we refer to an
abstract
computing device like CPU, GPU, FPGA, TPU etc. We target a collection of XPU computing accelerators that can be
connected in shared memory or distributed fashion. While this is the long-term scope of the project, at the moment, we
support shared memory CPU and GPUs. In particular, our current implementation is based on openMP and CUDA.

**Volumetric data** structures capture properties of a digitized representation of an object and are used in different
fields, from graphics to simulation. Neon primarily focuses on supporting local operations, usually called stencil
operations. Stencil operations are at the core of various numerical methods such as finite difference, finite element,
lattice Boltzmann and others.


| Simple user application in Neon and its dependency graph                               |                                                                                             |
|--------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| <image src = "/learn/guided-tutorials/img/axpy-laplace-dot-code.png" width="300px"></image> | <image src = "/learn/guided-tutorials/img/axpy-laplace-dot-app.png" width="300px" ></image> |

| Different Multi-XPU Graph created by Neon                                                   ||
|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| <image src = "/learn/guided-tutorials/img/axpy-laplace-dot-nocc.png" width="400px"></image> |<image src = "/learn/guided-tutorials/img/axpy-laplace-dot-eocc.png" width="400px"></image>  |
| <image src = "/learn/guided-tutorials/img/axpy-laplace-dot-socc.png" width="400px"></image> |<image src = "/learn/guided-tutorials/img/axpy-laplace-dot-e2occ.png" width="400px"></image> |

Neon is composed of a set of abstraction levels, each one represented by a C++ library.
The following picture shows the main high-level mechanisms provided by each level.

![](img/neon-layers.png){ style="width:500"}

The Domain and Skeleton are the most important abstraction for Neon users.
As expected by its name, the Domain level introduces domain-specific mechanisms, as for now we target voxel based
computations and the mechanisms are Cartesian grids, fields and stencils. The Skeleton level provides users with a
sequential programming model and in charge of transforming and optimizing user applications to be deployed into a
multi-device system.

!!! Note

    To learn how to write an application with Neon, new users can mainly focus on the Domain and Skeleton documentation as it implicitelly covers all the nedded information from the other Neon abtraction levels. 

The following is the structure of the `Introduction and tutorial` section:


<center>

| Abstraction |           Description           |     Library     |                                   Link |
|-------------|:-------------------------------:|:---------------:|---------------------------------------:|
| System      |        Device management        |   libNeonSys    |   [info](the-bases/01-system-level.md) |
| Set         |     Multi device management     |   libNeonSet    |  [info](the-bases/02-the-set-level.md) |
| Domain      | Domain mechanism  - voxel grids |  libNeonDomain  |   [info](the-bases/03-domain-level.md) |
| Skeleton    |  Sequential programming model   | libNeonSkeleton | [info](the-bases/04-skeleton-level.md) |

</center>