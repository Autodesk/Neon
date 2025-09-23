#pragma once

#include <unordered_map>

#include "Neon/core/core.h"

namespace Neon::domain::tool {

/**
 * @brief A hash table for efficiently storing and retrieving metadata associated with 3D discrete points 
 * within a finite background grid.
 * 
 * This class provides a spatial hash table implementation that maps 3D discrete coordinates to associated
 * metadata. It uses a std::unordered_map internally, where the key is computed using the pitch (linearized
 * index) of the 3D point within the specified bounding box dimensions. This allows for O(1) average-case
 * insertion, lookup, and deletion operations.
 * 
 * The hash table enforces bounds checking to ensure all points lie within the specified bounding box,
 * throwing exceptions for out-of-bounds operations during insertion.
 * 
 * @tparam IntegerT The integer type used for point coordinates (e.g., int32_t, int64_t)
 * @tparam MetaT The type of metadata to store for each point
 * 
 * @note Points are stored using row-major order linearization (pitch calculation)
 * @note All coordinates are expected to be non-negative and within the bounding box
 * 
 * Example usage:
 * @code
 * // Create a hash table for a 100x100x100 grid storing float values
 * using Point3D = Neon::Integer_3d<int32_t>;
 * PointHashTable<int32_t, float> hashTable(Point3D(100, 100, 100));
 * 
 * // Add a point with associated data
 * hashTable.addPoint(Point3D(10, 20, 30), 42.5f);
 * 
 * // Retrieve metadata for a point
 * if (auto* data = hashTable.getMetadata(Point3D(10, 20, 30))) {
 *     std::cout << "Value: " << *data << std::endl;
 * }
 * 
 * // Iterate through all points
 * hashTable.forEach([](const Point3D& point, float& data) {
 *     std::cout << "Point " << point << " has value " << data << std::endl;
 * });
 * @endcode
 */
template <typename IntegerT,
          typename MetaT>
class PointHashTable
{
   public:
    using Meta = MetaT /**< Information stored for each point by the hash table */;
    using Integer = IntegerT /**< Type for each discrete point coordinate */;
    using Point = Neon::Integer_3d<Integer> /**< Point type */;
    
    /**
     * @brief Default constructor that creates an empty hash table with a zero bounding box.
     * 
     * Creates a PointHashTable with a bounding box of (0, 0, 0). This constructor is primarily 
     * useful when the bounding box will be set later or when creating container objects.
     * 
     * @note A hash table created with this constructor will reject all point insertions 
     *       since the bounding box is effectively empty.
     */
    PointHashTable();

    /**
     * @brief Constructs a hash table with the specified bounding box dimensions.
     * 
     * Creates a PointHashTable that can store points with coordinates in the range [0, bbox)
     * for each dimension. The bounding box defines the valid coordinate space for all points
     * that can be stored in this hash table.
     * 
     * @param bbox The dimensions of the 3D grid. Each component must be positive to allow 
     *             point storage. Points with coordinates >= bbox in any dimension will be 
     *             rejected during insertion.
     * 
     * @note The bounding box represents exclusive upper bounds (i.e., valid coordinates 
     *       are in the range [0, bbox_component))
     * 
     * Example:
     * @code
     * Point3D bounds(100, 200, 50);  // Creates a 100x200x50 grid
     * PointHashTable<int, float> table(bounds);  // Can store points (0,0,0) to (99,199,49)
     * @endcode
     */
    explicit PointHashTable(const Point& bbox);

    /**
     * @brief Retrieves read-only metadata associated with the specified point.
     * 
     * Performs a const lookup for the metadata stored at the given 3D point. This method
     * provides safe read-only access to the stored data without allowing modifications.
     * The lookup time is O(1) on average.
     * 
     * @param point The 3D point coordinates to look up. The point coordinates should be 
     *              within the valid range [0, bbox) for each dimension, although 
     *              out-of-bounds points will simply return nullptr rather than throwing.
     * 
     * @return A pointer to the const metadata if the point exists in the hash table,
     *         nullptr if the point is not found or is outside the bounding box.
     * 
     * @note This method performs bounds checking and will return nullptr for points 
     *       outside the bounding box without throwing exceptions.
     * 
     * Example:
     * @code
     * if (const auto* data = table.getMetadata(Point3D(10, 20, 30))) {
     *     std::cout << "Found data: " << *data << std::endl;
     * } else {
     *     std::cout << "Point not found" << std::endl;
     * }
     * @endcode
     */
    auto getMetadata(Point const& point) const
        -> Meta const*;

    /**
     * @brief Retrieves mutable metadata associated with the specified point.
     * 
     * Performs a lookup for the metadata stored at the given 3D point, returning
     * a mutable pointer that allows modification of the stored data. The lookup 
     * time is O(1) on average.
     * 
     * @param point The 3D point coordinates to look up. The point coordinates should be 
     *              within the valid range [0, bbox) for each dimension, although 
     *              out-of-bounds points will simply return nullptr rather than throwing.
     * 
     * @return A pointer to the mutable metadata if the point exists in the hash table,
     *         nullptr if the point is not found or is outside the bounding box.
     * 
     * @note This method performs bounds checking and will return nullptr for points 
     *       outside the bounding box without throwing exceptions.
     * 
     * Example:
     * @code
     * if (auto* data = table.getMetadata(Point3D(10, 20, 30))) {
     *     *data = newValue;  // Modify the stored data
     * }
     * @endcode
     */
     auto getMetadata(Point const& point)
        -> Meta*;

    /**
     * @brief Inserts a point with associated metadata into the hash table.
     * 
     * Adds a new point-metadata pair to the hash table. If a point with the same
     * coordinates already exists, the behavior depends on the underlying std::unordered_map
     * implementation (typically the existing entry is preserved). The insertion time 
     * is O(1) on average.
     * 
     * @param point The 3D point coordinates to insert. Must be within the valid range 
     *              [0, bbox) for each dimension.
     * @param data The metadata to associate with this point. The data is copied into 
     *             the hash table.
     * 
     * @throws NeonException if the point coordinates are outside the bounding box 
     *                      (negative or >= bbox in any dimension).
     * 
     * @note This method performs strict bounds checking and will throw an exception 
     *       for out-of-bounds points, unlike the getMetadata methods which return nullptr.
     * 
     * Example:
     * @code
     * try {
     *     table.addPoint(Point3D(10, 20, 30), 42.5f);
     *     std::cout << "Point added successfully" << std::endl;
     * } catch (const NeonException& e) {
     *     std::cout << "Failed to add point: " << e.what() << std::endl;
     * }
     * @endcode
     */
    auto addPoint(Point const& point,
                  Meta const& data)
        -> void;

    /**
     * @brief Executes a user-provided function for each point-metadata pair in the hash table.
     * 
     * Iterates through all stored points in the hash table and calls the provided lambda
     * function for each entry. The iteration order is unspecified and depends on the 
     * internal hash table implementation. The time complexity is O(n) where n is the 
     * number of stored points.
     * 
     * @tparam UserLambda A callable type (function pointer, lambda, functor) that accepts
     *                    two parameters: (const Point&, Meta&)
     * 
     * @param lambda The function to execute for each point. The lambda receives:
     *               - const Point&: The 3D coordinates of the point (read-only)
     *               - Meta&: The associated metadata (mutable reference)
     * 
     * @note The metadata parameter is passed as a mutable reference, allowing the lambda
     *       to modify the stored data during iteration.
     * @note The iteration order is not guaranteed and may vary between calls.
     * 
     * Example:
     * @code
     * // Print all points and their data
     * table.forEach([](const Point3D& point, float& data) {
     *     std::cout << "Point " << point << " has value " << data << std::endl;
     * });
     * 
     * // Modify all data values
     * table.forEach([](const Point3D& point, float& data) {
     *     data *= 2.0f;  // Double all stored values
     * });
     * @endcode
     */
    template <typename UserLambda>
    auto forEach(const UserLambda& lambda);

    /**
     * @brief Returns the number of points currently stored in the hash table.
     * 
     * Provides the count of point-metadata pairs currently stored in the hash table.
     * This operation has O(1) time complexity.
     * 
     * @return The number of points stored in the hash table as a size_t value.
     * 
     * Example:
     * @code
     * std::cout << "Hash table contains " << table.size() << " points" << std::endl;
     * @endcode
     */
    auto size() const -> size_t;

    /**
     * @brief Returns the bounding box dimensions of the hash table.
     * 
     * Retrieves the current bounding box that defines the valid coordinate space
     * for this hash table. All valid points must have coordinates in the range
     * [0, bbox) for each dimension.
     * 
     * @return A const reference to the Point object representing the bounding box
     *         dimensions. Each component represents the exclusive upper bound for
     *         that dimension.
     * 
     * Example:
     * @code
     * const auto& bounds = table.getBBox();
     * std::cout << "Grid dimensions: " << bounds.x << "x" << bounds.y << "x" << bounds.z << std::endl;
     * @endcode
     */
    auto getBBox() const -> Point const&;

   private:
    using Key = size_t; /**< Type used for internal hash table keys */

    /**
     * @brief Computes the hash key for a given 3D point.
     * 
     * Converts 3D coordinates to a linearized key using pitch calculation
     * (row-major order). This internal method is used by both addPoint
     * and getMetadata methods to maintain consistency in key generation.
     * 
     * @param point The 3D point coordinates to convert to a key
     * 
     * @return The computed hash key for the point
     * 
     * @note This method assumes the point is within valid bounds and does 
     *       not perform bounds checking.
     */
    auto helpGetKey(Point const& point)
        -> Key;

    /**
     * @brief Reconstructs 3D coordinates from a hash key.
     * 
     * Converts a linearized key back to 3D coordinates using the inverse
     * of the pitch calculation. This internal method is used by the forEach
     * method to provide point coordinates during iteration.
     * 
     * @param key The hash key to convert back to 3D coordinates
     * 
     * @return The reconstructed 3D point coordinates
     * 
     * @note The reconstructed point is guaranteed to be within the bounding box
     *       since the key was originally derived from a valid point.
     */
    auto helpGetPoint(Key const& key)
        -> Point;

    std::unordered_map<Key, Meta> mMap;  /**< Internal hash table storing key-metadata pairs */
    Point                         mBBox; /**< Bounding box defining valid coordinate space */
};

}  // namespace Neon::domain::tool

#include "Neon/domain/tools/PointHashTable_imp.h"