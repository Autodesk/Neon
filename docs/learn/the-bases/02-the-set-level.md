![](img/02-layers-set.png){ align=right style="width:200px"}

# The Set Level

The set Level introduces the concept of multi-devices. The provided mechanisms are an extension of the standard
functionalities exported by a single device.

In general, the capabilities of this Level are essential for those looking at extending Neon. Therefore, Neon end users
should skip this section and move to the Domain Level, where the required mechanisms of this Level are already covered.

The Neon::Banckend object represents a set of XPU devices (CPU or GPU) mapped into a 1D index space. The defined logical
topology is not related to the actual topology of the devices. The Neon::Backend abstraction also 

```cpp linenums="26"  title="Neon/tutorials/introduction/domainLevel/domainLevel.cpp"
int main(int, char**)
{
    // Step 1 -> Neon backend: choosing the hardware for the computation
    Neon::Backend backend = [] {
        Neon::init();
        // auto runtime = Neon::Runtime::openmp;
        auto runtime = Neon::Runtime::stream;
        // We are overbooking XPU 0 three times
        std::vector<int> xpuIds{0, 0, 0};
        Neon::Backend    backend(xpuIds, runtime);
        // Printing some information
        NEON_INFO(backend.toString());
        return backend;
    }();
    
    return 0;
}
```