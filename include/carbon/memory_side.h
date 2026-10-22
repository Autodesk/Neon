#pragma once

#include <carbon/export.h>

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string_view>

namespace carbon {

/**
 * Where a field's storage lives across a multi-GPU run.
 *
 * @c Host       – allocate the host-side mirror only; no device buffers are reserved. Useful
 *                 for CPU-only staging fields or when the device side is filled by copy from
 *                 elsewhere.
 * @c Device     – allocate device storage on every GPU; no host mirror. The default for
 *                 compute kernels that never round-trip through the host.
 * @c HostDevice – allocate both sides. Pairs naturally with @ref Transfer for explicit
 *                 host/device sync.
 *
 * Mirrors the spelling/conversion conventions of @ref DataView (PascalCase enumerators,
 * snake_case free helpers). Conversions to/from @c std::string_view and @c int are exposed via
 * @ref to_string, @ref memory_side_from_string, @ref to_int, @ref memory_side_from_int — the
 * "from_*" pair uses a unique name because @c from_string / @c from_int already overload on
 * @ref DataView's return type and cannot be overloaded again purely on result type.
 *
 * @par Per-grid allocation policy (consult the grid header for the binding rules)
 *
 * The exact moment a host mirror is allocated for @c HostDevice fields is a
 * per-grid policy — it varies because each grid family settled on a different
 * "first touch" pattern during its initial port:
 *
 * - @c Dense and @c DenseDisg : @b lazy host-mirror allocation. The host buffer
 *   is allocated on the first call to @c Field::cpu_buffers() (or any helper
 *   that triggers it, e.g. @c Halo::launch_on_host ). This keeps the cost out
 *   of the construction path for users that build a @c HostDevice field but
 *   only round-trip through the host occasionally.
 * - @c Block : @b eager host-mirror allocation. The host buffer is sized and
 *   pinned at @c Block::newField time. Block's brick layout makes the size
 *   exactly known up front and the lazy path would have to duplicate the
 *   sizing logic; the eager path keeps the @c Field state immutable after
 *   construction.
 *
 * Both policies are observable through @c Field::cpu_buffers() returning a
 * non-empty span when the side carries host storage; what differs is when the
 * underlying allocation happens.
 */
enum class MemorySide : std::uint8_t {
  Host = 0,
  Device = 1,
  HostDevice = 2,
};

/** Number of valid MemorySide values. */
inline constexpr size_t memory_side_count = 3;

/**
 * Lower-case canonical name for @p v: "host" | "device" | "host_device".
 * Returned view points to a static string literal; safe to store.
 */
[[nodiscard]] CARBON_API auto
to_string(MemorySide v) -> std::string_view;

/**
 * Parse a MemorySide from its canonical name. Case-insensitive: accepts "host", "Host",
 * "device", "Device", "host_device", "HostDevice", "hostdevice", "host-device", etc.
 * @throws std::out_of_range if @p s is not one of the recognised names.
 */
[[nodiscard]] CARBON_API auto
memory_side_from_string(std::string_view s) -> MemorySide;

/** Integer encoding: 0 = Host, 1 = Device, 2 = HostDevice. */
[[nodiscard]] CARBON_API auto
to_int(MemorySide v) noexcept -> int;

/**
 * Build a MemorySide from an integer encoding.
 * @throws std::out_of_range if @p v is outside [0, memory_side_count).
 */
[[nodiscard]] CARBON_API auto
memory_side_from_int(int v) -> MemorySide;

/** All MemorySide values in enum order: {Host, Device, HostDevice}. */
[[nodiscard]] CARBON_API auto
memory_side_options() noexcept -> std::array<MemorySide,
                                             memory_side_count>;

/** Stream operator: writes the lower-case canonical name (same as to_string). */
CARBON_API auto
operator<<(std::ostream& os,
           MemorySide v) -> std::ostream&;

/** Convenience: true iff @p v reserves device storage (Device or HostDevice). */
[[nodiscard]] CARBON_API auto
has_device(MemorySide v) noexcept -> bool;

/** Convenience: true iff @p v reserves host storage (Host or HostDevice). */
[[nodiscard]] CARBON_API auto
has_host(MemorySide v) noexcept -> bool;

}  // namespace carbon
