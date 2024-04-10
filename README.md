![Neon logo](docs/logo/neonDarkLogo.jpg "Neon")

# For information about the GPU LBM Grid Refinement work (IPDPS 2024), go to this [README](/apps/lbmMultiRes/README.md)

Neon is a research framework for programming multi-device systems maintained by [Autodesk Research](https://www.autodesk.com/research/overview). Neon's goal is to automatically transform user sequential code into, for example, a scalable multi-GPU execution.

To reach its goal, Neon takes a domain-specific approach based on the parallel skeleton philosophy (a.k.a parallel patterns). Neon provides a set of domain-specific and programmable patterns that users compose through a sequential programming model to author their applications. Then, thanks to the knowledge of the domain, the patterns and their composition, Neon automatically optimizes the sequential code into an execution optimized for multi-device systems.

Currently, Neon targets grid-based computations on multi-core CPUs or single node multi-GPU systems. 

It is important to keep in mind that Neon is a research project in continuous evolution. So, while we have successfully tested the system with different applications (Finite Difference, Finite Element, Lattice Boltzmann Method), Neon interfaces may change between versions to introduce new capabilities.

## Quick Start

Neon code is hosted on a GitHub [repository](https://github.com/Autodesk/Neon).
To clone the repo, use the command:

```
git clone https://github.com/Autodesk/Neon
```

Once cloned, you can compile Neon like any other CMake project. A C++ compiler (with C++17 standard support) and a CUDA (version 11 or later) must be present already installed on the system. You can use the following commands to compile with a default configuration:

```
mkdir build
cd build
cmake ../
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 

## User Documentation

A description of the system and its capabilities can be found in our paper [link](https://escholarship.org/uc/item/9fz7k633).

We use mkdocs to organize Neon documentation which is available online via GitHub Pages ([https://autodesk.github.io/Neon/](https://autodesk.github.io/Neon/)).
The documentation includes a tutorial, application and benchmark sessions.  

## Communicate With Us

We are working to define the best way to communicate with us. Please stay tuned. 

## Contributions From the Community

The Neon team welcome and greatly appreciate contributions from the community. The document [CONTRIBUTING.md](docs/CONTRIBUTING.md) goes more into the details on the process we follow. 

As a community, we have a responsibility to create a respectful and inclusive environment, so we kindly ask any member and contributor to respect and follow [Neon's Code of Conduct](docs/CODE_OF_CONDUCT.md)

## Authors and Maintainers 

Please check out the [CONTRIBUTORS.md](docs/CONTRIBUTORS.md), to see the full list of contributors to the project.

The current maintainers of project Neon are:
- Massimiliano Meneghin (Autodesk Research)
- Ahmed Mahmoud (Autodesk Research)

## License

Neon is licenced under the Apache License, Version 2.0. For more information please check out our licence file ([LICENSE.txt](./LICENSE.txt))

## How to cite Neon

```
@INPROCEEDINGS{Meneghin:2022:NAM,
  author = {Meneghin, Massimiliano and Mahmoud, Ahmed H. and Jayaraman, Pradeep Kumar and Morris, Nigel J. W.},
  booktitle = {Proceedings of the 36th IEEE International Parallel and Distributed Processing Symposium},
  title = {Neon: A Multi-GPU Programming Model for Grid-based Computations},
  year = {2022},
  month = {june},
  pages = {817--827},
  doi = {10.1109/IPDPS53621.2022.00084},
  url = {https://escholarship.org/uc/item/9fz7k633}
}
```
