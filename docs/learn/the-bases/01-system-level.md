![](img/01-layers-system.png){align=right style="width:200px"}

# The System Level

The **System** abstraction shields the rest of Neon from architecture and hardware-specific mechanisms. Therefore, this abstraction level should be invisible to the final Neon users.

The System defines an object-oriented interface to manage resources and requires the following backend capabilities:

**Memory Management**:
    This allows Neon to create device buffers and move data between devices or host.

**Queue-based Run-time Model**:
    Neon uses a queue-based model to abstract asynchronous kernels running on the same device. It is a generic model widely used at the hardware level and other programming models. For example, in CUDA, Streams represent command queues, while Events are the mechanism to inject dependencies between different queues.

**Lambda Functions**:
    Neon leverages the expressiveness of lambda functions to lessen the complexity of authoring multi-GPU applications.

Therefore, to port Neon to a new accelerator, only the Neon System abstraction has to be implemented; higher levels in Neon can remain unchanged.