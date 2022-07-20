## **Neon Code Convention**

## *General rules*:
- Use [doxgen](https://www.doxygen.nl/manual/docblocks.html) for documentation with Javadoc style comment block i.e., 
    ```c++
    /**
     * ... text ...
     */
    ```
- Use inline documentation of the function input variables and class members variables e.g., 
    ```c++
    void foo(int v /**< [in] docs for input parameter v */);
    ```

- ``using namespace`` is only allowed inside ``.cpp``/``.cu`` files. It's disallowed in headers.

- ``using namespace std`` is disallowed even in ``.cpp``/``.cu`` files. If you want to save some work, just typedef the type you need from the std namespace, or use ``auto``.

- For consistency reasons, use ``using`` declaration instead of ``typedef`` e.g.,
    ```c++
    using UintVector = std::vector<uint32_t>;
    ```

- Use only sized types (e.g., ``int32_t``, ``uint32_t``, ``int16_t``). Conceptually, ``bool`` has unknown size, so no size equivalent. ``char`` is special and can be used only for C strings (use ``int8_t`` otherwise).

- Don't use ``NULL`` or 0 to initialize pointers. ``nullptr`` is part of the language now.

- Preprocessor definitions are all capitals and may contain ``_`` e.g., 
    ```c++
     #define SOME_DEFINE
     ```

- Don't use long line comment separator e.g., ``/////////////`` or ``/*****************/``

- We use `NEON_TRACE`, `NEON_INFO`, `NEON_WARNING`, `NEON_ERROR`, and `NEON_CRITICAL` for logging. Using `printf` or `std::cout` is prohibited. `NEON_*` logging macro rely on [`spdlog`](https://github.com/gabime/spdlog#spdlog) with easy python-like string formatting e.g., 
    ```c++
    NEON_INFO("Welcome to spdlog!");
    NEON_ERROR("Some error message with arg: {}", 1);
    
    NEON_WARNING("Easy padding in numbers like {:08d}", 12);
    NEON_CRITICAL("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    NEON_INFO("Support for floats {:03.2f}", 1.23456);
    NEON_INFO("Positional args are {1} {0}..", "too", "supported");
    NEON_INFO("{:<30}", "left aligned");
    ```


## *Variable prefixes:*
Use the following prefixes for variable names based on the scope ``s``, ``m``, ``g``, and ``k``. These are mutually exclusive prefixes, where ``k`` takes precedence above the rest.
 - ``k`` for compile-time const variables.
 - ``g`` for global variables, including static global variables.
 - ``s`` for class static variables.
 - ``m`` for member variables in classes (not in structs).

In addition ``p`` is used for pointers e.g.,

```c++
//Global Variables:
const uint32_t kConstGlobal; // compile-time-const, so 'k' takes precedence
int32_t gSomeGlobal;         // global start with 'g'
static int gStaticGlobal;    // Static globals start with 'g'
void* gpSomePointer;         // Global variables which is a pointer is prefixed with 'gp'
const void* gpPointer2;      // Not compile-time constant.
```

## *Functions:* 
- Use auto style for function but explicitly defining the return type using trailing arrow e.g.,
```c++
auto getGirdSize() -> int;
```

- Function names should be descriptive:
  * Functions that perform an action should be named after the action it performs e.g., ``Fbo::clear()``, ``createTextureFromFile()``.
  * Getters/Setters should start with ``get`` and ``set``
  * Functions names that return a ``bool`` should be phrased as a question e.g., ``isWhite()``, ``doesFileExist()``, ``hasTexture()``

- Function names are lower-camel-case e.g., 
```c++
void someFunction()
```

## *Classes:*
- Classes should hide their internal data as much as possible.

- Class names should be Upper Camel case (``UpperCamelClass``) or Lower Camel case (``lowerCamelClass``)

```c++
class UpperCamelClass
{

    bool isValid();                // Function names are lower-camel-case
    static uint32_t sInt;          // Static variables start with 's'
    static const uint32_t kValue;  // Const static is prefixed with 'k'
    int32_t mMemberVar;           // Member variables start with 'm'
    int16_t* mpSomePointer;       // Note that with a pointer variable, "p" counts as the first word, so the next letter *is* capitalized
};
```

- Header file must be a ``.h`` containing only the class declaration i.e., class name, methods' signature and variables

- Source file must be a ``.cpp`` or ``.cu`` and it contains the definition of all methods

- Templated methods definition must be in separate ``.h`` file that is included by the corresponding ``.h``. File name should end with ``_imp.h``

- File names only contain dot (".") before the file extension suffix

- Each class has it's own files with same name e.g., ``Grid`` class goes into ``Grid.h``, ``Grid.cpp``, and ``Grid_imp.h``

- The order of public/private members and methods as they appear in the class ``.h`` file is:

    1. public members
    2. public methods
    3. private methods
    4. private members

```c++
// Grid.h
#pragma once
namespace Neon::grid {
/**
 * Grid is the blueprint of creating physical domains 
*/
class Grid
{
   private:   
    int mNumCells = 0 /**< number of cells */;

   public:
    /**
     * default constructor 
    */
    Grid() = default;

    /**
     * Grid constructor      
    */
    Grid(int& const bool padding /**< [in] Enable memory padding if true */);

    /**
     * Create new field      
     * @return the new field 
    */
    template <typename T /**< Field type */>
    auto newField() -> Grid::Field<T>;


};
}  // namespace Neon::grid
#include "Neon/domain/Grid_imp.h"
```


```c++
// Grid.cpp
#pragma once
#include "Neon/domain/Grid.h"
namspeace Neon::grid {
    Grid::Grid(int& const bool padding){
        // ....
    }
} // namspeace Neon::grid 
```

```c++
//Grid_imp.h
#pragma once
#include "Neon/domain/Grid.h"
namspeace Neon::grid {
    template<typename T>
    auto Grid::newField() -> Grid::Field<T>{        
        //...        
    }
} // namspeace Neon::grid 
```

## *Structs:*

Use struct only as a data-container. All fields must be public. No member functions are allowed. In-class initialization is allowed.

```c++
//UpperCamelStruct.h
struct UpperCamelStruct
{
    int32_t someVar;               // Struct members are lower-camel-case    
    int32_t** pDoublePointer;      // Double pointer is just a pointer
    std::smart_ptr<int> pSmartPtr; // Smart pointer is a pointer
    char charArray[];              // Array is not a pointer
    std::string myString;          // String is a string
    bool isBoolean;                // bool name implies that it's a bool. 'enable', 'is*', 'has*', etc. Don't use negative meaning (use 'enable' instead of 'disable')
    uint32_t& refVal;              // Reference is not a pointer
};
```

## *Enums:*

Use typed enums (i.e, ``enum class``) as they are type safe and they automatically define their namespace. When approperiate, each enum class should be followed by utility class with static methods. These utility method provide a ``toString()`` funtionality as well as ``toInt()`` for easy conversions.

```c++
//DataView.h
enum class DataView : char {
    standard,
    internal, 
    boundary
};
class DataViewUtils {
    static auto toString(DataView d) -> std::string;
    static auto toInt(DataView d) -> int;
}
```


