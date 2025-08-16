/**
 * @file Report.cpp
 * @brief C API wrapper for Neon::Report class to enable Python bindings
 * 
 * This file provides a C interface for the Neon::Report class, allowing
 * Python code to interact with Neon's reporting functionality through
 * ctypes bindings. The Report class is used for collecting and outputting
 * performance metrics, configuration data, and other diagnostic information.
 * 
 * The C API follows a consistent pattern:
 * - Functions return int (0 for success, non-zero for error)
 * - Objects are managed through void* handles
 * - Memory management is explicit (new/delete pairs)
 * 
 * @author Neon Development Team
 * @version 1.0
 */

#include "Neon/Neon.h"
#include "Neon/Report.h"
#include "Neon/py/macros.h"

/**
 * @brief Initialize a new Report object on the heap
 * 
 * Creates a new Neon::Report instance with the specified name and returns
 * a handle that can be used with other report functions. The caller is
 * responsible for calling report_delete() to free the memory.
 * 
 * @param handle Pointer to void* that will receive the report object handle
 * @param name   C-string name for the report (used in output files)
 * 
 * @return 0 on success, -1 on failure (memory allocation error or exception)
 * 
 * @note This function calls Neon::init() to ensure proper initialization
 * @see report_delete()
 */
extern "C" auto report_new(
    void**      handle,
    const char* name)
    -> int
{
    try {
        Neon::init();
        auto reportPtr = new (std::nothrow) Neon::Report(name);
        if (reportPtr != nullptr) {
            *handle = reportPtr;
            return 0;
        }
        return -1;
    } catch (...) {
        return -1;
    }
}

/**
 * @brief Delete a Report object and free its memory
 * 
 * Safely destroys a Neon::Report instance that was created with report_new().
 * This function handles null pointer checks and sets the handle to nullptr
 * after deletion to prevent use-after-free errors.
 * 
 * @param handle Pointer to void* containing the report object handle
 * 
 * @return 0 on success, -1 on exception
 * 
 * @note After this call, the handle will be set to nullptr
 * @note Safe to call with null or already-deleted handles
 * @see report_new()
 */
extern "C" auto report_delete(
    void** handle)
    -> int
{
    try {
        NEON_PY_PRINT_BEGIN(*handle);

        auto report = reinterpret_cast<Neon::Report*>(*handle);

        if (report != nullptr) {
            delete report;
        }
        *handle = nullptr;
        NEON_PY_PRINT_END(*handle);
        return 0;
    } catch (...) {
        return -1;
    }
}

/**
 * @brief Add a string member to the report
 * 
 * Adds a key-value pair to the report where both key and value are strings.
 * This is commonly used for configuration settings, version information,
 * and other textual data.
 * 
 * @param handle     Report object handle from report_new()
 * @param memberKey  C-string key name for the member
 * @param memberVal  C-string value to associate with the key
 * 
 * @return 0 on success, -1 on exception
 * 
 * @note The strings are copied internally, so the caller retains ownership
 * @see report_new(), Neon::Report::addMember()
 */
extern "C" auto report_add_member_string(void*       handle,
                       const char* memberKey,
                       const char*     memberVal) -> int
{
    try {
        auto report = reinterpret_cast<Neon::core::Report*>(handle);
        report->addMember(memberKey, memberVal);
        return 0;
    } catch (...) {
        return -1;
    }
}

/**
 * @brief Add a 64-bit integer member to the report
 * 
 * Adds a key-value pair to the report where the key is a string and the
 * value is a 64-bit signed integer. This is commonly used for performance
 * counters, iteration counts, memory sizes, and other numeric metrics.
 * 
 * @param handle     Report object handle from report_new()
 * @param memberKey  C-string key name for the member
 * @param memberVal  64-bit signed integer value
 * 
 * @return 0 on success, -1 on exception
 * 
 * @see report_new(), Neon::Report::addMember()
 */
extern "C" auto report_add_member_int64(void*       handle,
                       const char* memberKey,
                       const int64_t     memberVal) -> int
{
    try {
        auto report = reinterpret_cast<Neon::core::Report*>(handle);
        report->addMember(memberKey, memberVal);
        return 0;
    } catch (...) {
        return -1;
    }
}


/**
 * @brief Add a vector of 64-bit integers to the report
 * 
 * Adds a key-value pair to the report where the key is a string and the
 * value is an array/vector of 64-bit signed integers. The data is copied
 * from the provided C array into an std::vector for storage.
 * 
 * @param handle     Report object handle from report_new()
 * @param memberKey  C-string key name for the member
 * @param vec_len    Length of the integer array
 * @param vec_val    Pointer to array of int64_t values to copy
 * 
 * @return 0 on success, -1 on exception
 * 
 * @note The array data is copied, so the caller retains ownership of vec_val
 * @see report_new(), Neon::Report::addMember()
 */
extern "C" auto report_add_member_vector_int64(void*       handle,
                              const char* memberKey,
                              int         vec_len,
                              const int64_t*    vec_val) -> int
{
    try {
        auto           report = reinterpret_cast<Neon::core::Report*>(handle);
        // initialize an std::vector from the row pointer

        std::vector<int64_t> vec(vec_val, vec_val+vec_len);
        report->addMember(memberKey, vec);
        return 0;
    } catch (...) {
        return -1;
    }
}

/**
 * @brief Add a double-precision floating-point member to the report
 * 
 * Adds a key-value pair to the report where the key is a string and the
 * value is a double-precision floating-point number. This is commonly used
 * for timing measurements, performance metrics, and scientific calculations.
 * 
 * @param handle     Report object handle from report_new()
 * @param memberKey  C-string key name for the member
 * @param memberVal  Double-precision floating-point value
 * 
 * @return 0 on success, -1 on exception
 * 
 * @see report_new(), Neon::Report::addMember()
 */
extern "C" auto report_add_member_double(void*       handle,
                       const char* memberKey,
                       const double     memberVal) -> int
{
    try {
        auto report = reinterpret_cast<Neon::core::Report*>(handle);
        report->addMember(memberKey, memberVal);
        return 0;
    } catch (...) {
        return -1;
    }
}


/**
 * @brief Add a vector of double-precision floating-point values to the report
 * 
 * Adds a key-value pair to the report where the key is a string and the
 * value is an array/vector of double-precision floating-point numbers.
 * The data is copied from the provided C array into an std::vector for storage.
 * This is commonly used for time series data, performance arrays, and
 * scientific datasets.
 * 
 * @param handle     Report object handle from report_new()
 * @param memberKey  C-string key name for the member
 * @param vec_len    Length of the double array
 * @param vec_val    Pointer to array of double values to copy
 * 
 * @return 0 on success, -1 on exception
 * 
 * @note The array data is copied, so the caller retains ownership of vec_val
 * @see report_new(), Neon::Report::addMember()
 */
extern "C" auto report_add_member_vector_double(void*       handle,
                              const char* memberKey,
                              int           vec_len,
                              const double*    vec_val) -> int
{
    try {
        auto           report = reinterpret_cast<Neon::core::Report*>(handle);
        // initialize an std::vector from the row pointer

        std::vector<double> vec(vec_val, vec_val+vec_len);
        report->addMember(memberKey, vec);
        return 0;
    } catch (...) {
        return -1;
    }
}

/**
 * @brief Write the report to a file
 * 
 * Outputs all collected report data to a file in a structured format
 * (JSON). The report can optionally include a timestamp in
 * the filename to ensure uniqueness across multiple runs.
 * 
 * @param handle              Report object handle from report_new()
 * @param fname               Base filename for the output file
 * @param append_time_to_file If true, appends timestamp to filename
 * 
 * @return 0 on success, -1 on exception
 * 
 * @note The actual output format depends on the Neon::Report implementation
 * @note If append_time_to_file is true, the timestamp format is implementation-defined
 * @see report_new(), Neon::Report::write()
 */
extern "C" auto report_write(void*       handle,
                             const char* fname,
                             bool        append_time_to_file) -> int
{
    try {
        auto report = reinterpret_cast<Neon::core::Report*>(handle);
        report->write(fname, append_time_to_file);
        return 0;
    } catch (...) {
        return -1;
    }
}