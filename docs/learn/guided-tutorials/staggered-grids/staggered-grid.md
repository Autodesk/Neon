![](../the-bases/img/03-layers-domain.png){ align=right style="width:250px"}

# The Domain Level - Staggered Grids  [WIP]


![](img/staggered-grid.png)


```cpp linenums="113" title="Neon/tutorials/introduction/domainLevel/domainLevel.cpp"
    // ...
    
    using G = dGrid;
    // using G = bGrid;
    // using G = eGrid;
    Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<G> FEA(
        data.getBackend(),
        dims,
        [](const Neon::index_3d&) -> bool {
            return trueIfNodeActive();
        });

    auto velocityNodeField = FEA.template newNodeField<double, 1>("Velocity", 3, 0);
    auto densityVoxelField = FEA.template newVoxelField<double, 1>("Density", 1, 0);


    const std::string appName(testFilePrefix + "_" + grid.getImplementationName());

    temperature.ioToVtk(appName + "-temperature", "temperature");
```


```cpp linenums="113" title="Neon/tutorials/introduction/domainLevel/domainLevel.cpp"
    // ...
    
        Neon::set::Container myFunction =  FEA.getContainerOnNodes(
        "MAP-on-nodes",
        [&](Neon::set::Loader& loader) {
            const auto& vel = loader.load(velocityNodeField);
            auto&       rho = loader.load(densityVoxelField);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Self::NodeField::Node& node) mutable {
            // 2D example
                // for each voxel connected to the node
                for( auto direction : {1,1}, {1,-1}, {-1,1}, {-,-1}, {0,1}}){
                    rhoVox = rho.getVelueFromNode(node)
                    //....
                }
            };
        });
```