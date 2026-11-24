#pragma once

/**
 * @file row_copy.h
 * @brief Strided row copies inside one @c DenseDisgDistributed chunk buffer.
 *
 * The pencil and brick halos refresh the face send shadows and the edge / corner send strips
 * (see @c dense_disg_distributed::Span ) right before they send. Each refresh is a handful of
 * @ref RowCopyJob s — @c rows rows of @c len elements; element @c x of row @c r is copied from
 * @c src_first + r·src_stride + x·src_step to @c dst_first + r·dst_stride + x·dst_step of the
 * same buffer — issued as one kernel launch per chunk on CUDA / HIP (kernel in
 * @c src/dense_disg_distributed_row_copy.cu , type-erased on the element width) and as plain
 * loops on the host mirror. Pencil jobs are contiguous rows (unit steps); brick jobs also gather
 * strided columns (an x-face row read out of a z plane).
 *
 * Every job reads canonical storage and writes shadow / strip storage, so the copies are
 * order-independent.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/detail/backend/select.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace carbon::detail::dense_disg_distributed {

/**
 * Maximum jobs per launch. Brick worst case: 4 y-face shadow rows, 8 x-face shadow rows /
 * columns, 4 + 4 single-job edge strips along x / y, 12 for the three-piece edge strips along z,
 * 8 corners.
 */
inline constexpr int kMaxRowCopyJobs = 40;
/** Threads per block along the row for the row-copy kernel. */
inline constexpr uint32_t kRowCopyBlockX = 256U;

/** One strided block of row copies, offsets in elements. */
struct RowCopyJob {
  int64_t src_first{};
  int64_t src_stride{};
  int64_t dst_first{};
  int64_t dst_stride{};
  int64_t rows{};
  int64_t len{};
  int64_t src_step{1};
  int64_t dst_step{1};
};

/** Fixed-size kernel argument: up to @ref kMaxRowCopyJobs jobs. */
struct RowCopyArgs {
  RowCopyJob jobs[kMaxRowCopyJobs]{};
  int32_t job_count{};
  int64_t max_len{};
  int64_t max_rows{};

  /**
   * Append a job; returns @c false when @p job is empty ( @c rows or @c len not positive).
   *
   * @throws std::length_error when the argument already holds @ref kMaxRowCopyJobs jobs.
   */
  auto
  push(const RowCopyJob& job) -> bool {
    if (job.rows <= 0 || job.len <= 0) {
      return false;
    }
    if (job_count >= kMaxRowCopyJobs) {
      throw std::length_error("dense_disg_distributed::RowCopyArgs: job capacity exceeded");
    }
    jobs[job_count++] = job;
    max_rows = job.rows > max_rows ? job.rows : max_rows;
    max_len = job.len > max_len ? job.len : max_len;
    return true;
  }

  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return job_count == 0;
  }
};

/** Copy element @p x of row @p r of job @p j ; out-of-range coordinates are a no-op. */
template <typename W>
CARBON_HD inline void
row_copy_thread(W* base,
                const RowCopyArgs& a,
                int64_t j,
                int64_t r,
                int64_t x) {
  if (j >= a.job_count) {
    return;
  }
  const RowCopyJob& job = a.jobs[j];
  if (r >= job.rows || x >= job.len) {
    return;
  }
  base[job.dst_first + r * job.dst_stride + x * job.dst_step] =
      base[job.src_first + r * job.src_stride + x * job.src_step];
}

/** Run every job of @p a on host memory @p base . */
template <typename T>
inline void
row_copy_host(T* base,
              const RowCopyArgs& a) {
  for (int32_t j = 0; j < a.job_count; ++j) {
    const RowCopyJob& job = a.jobs[j];
    for (int64_t r = 0; r < job.rows; ++r) {
      T* const dst = base + job.dst_first + r * job.dst_stride;
      const T* const src = base + job.src_first + r * job.src_stride;
      if (job.src_step == 1 && job.dst_step == 1) {
        std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src),
                    static_cast<size_t>(job.len) * sizeof(T));
        continue;
      }
      for (int64_t x = 0; x < job.len; ++x) {
        dst[x * job.dst_step] = src[x * job.src_step];
      }
    }
  }
}

/** Element widths the type-erased kernel is instantiated for. */
[[nodiscard]] constexpr auto
row_copy_element_size_supported(std::size_t bytes) noexcept -> bool {
  return bytes == 1U || bytes == 2U || bytes == 4U || bytes == 8U || bytes == 16U;
}

}  // namespace carbon::detail::dense_disg_distributed

#if CARBON_BACKEND_IS_CUDALIKE

#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/gpu.h>
#include <carbon/export.h>

namespace carbon::detail::dense_disg_distributed {

/**
 * Launch every job of @p a on @p base (device memory of @p gpu ) asynchronously on @p stream .
 * Returns the launch error; @c gpu_error_invalid_value for an unsupported @p elem_bytes .
 */
[[nodiscard]] CARBON_API auto
launch_row_copy(const Gpu& gpu,
                gpu_stream_t stream,
                std::size_t elem_bytes,
                void* base,
                const RowCopyArgs& a) -> gpu_error_t;

}  // namespace carbon::detail::dense_disg_distributed

#endif  // CARBON_BACKEND_IS_CUDALIKE
