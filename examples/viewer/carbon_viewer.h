#pragma once

#include <carbon/floating3d.h>
#include <carbon/render.h>
#include <carbon/system.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace carbon::viewer {

/** Spherical orbit camera; converts to @ref Camera via @ref to_camera . */
struct OrbitCamera {
  float32_3d target{};
  float distance{1.0F};
  float azimuth{0.0F};
  float elevation{0.0F};
  float fov_y_rad{0.785398F};
  float32_3d up{0.0F, 0.0F, 1.0F};
  float32_3d bounds_min{};
  float32_3d bounds_max{};

  [[nodiscard]] auto
  to_camera(int width,
            int height) const -> Camera;

  /**
   * Seed orbit state from the same inputs as @ref Camera::frame_bounds so the interactive
   * view starts at the static frame.
   */
  [[nodiscard]] static auto
  from_frame_bounds(float32_3d bounds_min,
                    float32_3d bounds_max,
                    float32_3d eye_offset_dir,
                    float32_3d up,
                    float fov_y_rad,
                    float padding = 1.15F) -> OrbitCamera;
};

/**
 * Frame timing filled by @ref Viewer when @c perf_stats is wired through run options.
 *
 * Atomic because the web viewer fills these from its background sim thread while the imgui-ws
 * thread reads them for the performance panel, and the two threads hold different mutexes.
 */
struct ViewerPerfStats {
  std::atomic<double> solver_steps_s{0.0};
  std::atomic<double> render_ms{0.0};
  std::atomic<double> render_ms_smooth{0.0};
  std::atomic<double> transfer_ms{0.0};
  std::atomic<double> transfer_ms_smooth{0.0};
  std::atomic<size_t> transfer_bytes{0};
  std::atomic<double> transfer_mib_s{0.0};
  std::atomic<double> transfer_mib_s_smooth{0.0};
  std::atomic<double> display_fps{0.0};
};

/** Axis-aligned box in the same world coordinates as @ref OrbitCamera::bounds_min/max . */
struct AxisAlignedBox {
  float32_3d min{};
  float32_3d max{};
};

/**
 * Interactive GLFW + ImGui viewer. Rank 0 opens the window; under MPI every rank participates
 * in the collective loop and re-renders when the orbit camera changes.
 *
 * @p render_fn fills @p rgb_out with @c width * height * 3 interleaved RGB samples in @c [0, 1] .
 */
class Viewer {
public:
  /**
   * @p mode is the render-style index the user selects at runtime (0 by default); the meaning is
   * defined by the caller via the matching @c render_modes labels.
   */
  using RenderFn = std::function<
      void(const Camera& camera, int width, int height, int mode, std::vector<float>& rgb_out)>;

  /** Advance a live simulation once per frame while @c simulate is enabled. */
  using TickFn =
      std::function<void(float dt, int64_t frame, bool reset_requested, int steps_per_frame)>;

  /** Draw app-owned ImGui controls on rank 0. Return @c true to force a redraw. */
  using UiFn = std::function<bool()>;

  /** Called when the user changes the internal render resolution (all ranks). */
  using ResolutionChangeFn = std::function<void(int width, int height)>;

  /** Per-GPU owned-slab bounds for optional overlay wireframes (rank 0 only). */
  using GpuBoundsFn = std::function<std::vector<AxisAlignedBox>()>;

  /** Carbon grid extent for the domain wireframe overlay (rank 0 only). */
  using DomainBoundsFn = std::function<AxisAlignedBox()>;

  /** Wake / scene framing for reset-camera (may differ from @ref DomainBoundsFn ). */
  using FrameBoundsFn = std::function<AxisAlignedBox()>;

  /** Options shared by the desktop and browser backends. */
  struct RunOptions {
    std::vector<std::string> render_modes{};
    bool simulate_default{false};
    int steps_per_frame_default{1};
    TickFn tick_fn{};
    UiFn ui_fn{};
    ResolutionChangeFn resolution_change_fn{};
    GpuBoundsFn gpu_bounds_fn{};
    DomainBoundsFn domain_bounds_fn{};
    FrameBoundsFn frame_bounds_fn{};
    ViewerPerfStats* perf_stats{nullptr};
  };

  [[nodiscard]] static auto
  display_available() noexcept -> bool;

  Viewer(int width,
         int height,
         const char* title);

  ~Viewer();

  Viewer(const Viewer&) = delete;
  Viewer&
  operator=(const Viewer&) = delete;

  auto
  run(const System& sys,
      OrbitCamera initial,
      RenderFn render_fn,
      std::vector<std::string> render_modes = {}) -> void;

  auto
  run(const System& sys,
      OrbitCamera initial,
      RenderFn render_fn,
      RunOptions opts) -> void;

  [[nodiscard]] auto
  frame_width() const noexcept -> int;

  [[nodiscard]] auto
  frame_height() const noexcept -> int;

  auto
  set_frame_size(int width,
                 int height) -> void;

#if defined(CARBON_HAVE_VIEWER_WEB)
  /** Options for the browser (imgui-ws / WebSocket) backend. */
  struct WebOptions {
    int port{5000};
    int tex_edge{1024};
    std::vector<std::string> render_modes{};
    bool simulate_default{false};
    int steps_per_frame_default{1};
    TickFn tick_fn{};
    UiFn ui_fn{};
    ResolutionChangeFn resolution_change_fn{};
    GpuBoundsFn gpu_bounds_fn{};
    DomainBoundsFn domain_bounds_fn{};
    FrameBoundsFn frame_bounds_fn{};
    ViewerPerfStats* perf_stats{nullptr};
  };

  auto
  run_web(const System& sys,
          OrbitCamera initial,
          RenderFn render_fn,
          WebOptions opts) -> void;
#endif

private:
  int width_{0};
  int height_{0};
};

}  // namespace carbon::viewer
