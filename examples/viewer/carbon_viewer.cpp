#include "carbon_viewer.h"

#include <carbon/detail/backend/select.h>
#include <carbon/logging.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#include <imgui.h>

#if defined(CARBON_HAVE_VIEWER_DESKTOP)
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>
#endif

#if defined(CARBON_HAVE_VIEWER_WEB)
#include <imgui-ws/imgui-ws.h>

#include <atomic>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

namespace carbon::viewer {
namespace {

struct ViewerStatePacket {
  float target_x{0.0F};
  float target_y{0.0F};
  float target_z{0.0F};
  float distance{1.0F};
  float azimuth{0.0F};
  float elevation{0.0F};
  float fov_y_rad{0.785398F};
  float up_x{0.0F};
  float up_y{0.0F};
  float up_z{1.0F};
  float bounds_min_x{0.0F};
  float bounds_min_y{0.0F};
  float bounds_min_z{0.0F};
  float bounds_max_x{0.0F};
  float bounds_max_y{0.0F};
  float bounds_max_z{0.0F};
  int32_t width{0};
  int32_t height{0};
  int32_t running{1};
  int32_t dirty{1};
  int32_t render_mode{0};
  int32_t simulate{0};
  int32_t steps_per_frame{1};
  int32_t reset_requested{0};
  int64_t frame{0};
};

struct SimPanelState {
  bool simulating{false};
  int steps_per_frame{1};
  bool reset_clicked{false};
  double steps_per_sec{0.0};
  double steps_per_sec_smooth{0.0};
  int64_t steps_accum{0};
  std::chrono::steady_clock::time_point steps_t0{std::chrono::steady_clock::now()};
};

auto
draw_simulation_panel(SimPanelState& sim,
                      bool tick_enabled) -> bool {
  bool changed = false;
  if (!tick_enabled) {
    return changed;
  }
  if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::Button(sim.simulating ? "Pause" : "Play")) {
      sim.simulating = !sim.simulating;
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset sim")) {
      sim.reset_clicked = true;
      changed = true;
    }
    changed |= ImGui::SliderInt("steps / frame", &sim.steps_per_frame, 1, 1000);
    ImGui::TextDisabled("%.0f steps/s", sim.steps_per_sec_smooth);
  }
  return changed;
}

auto
apply_simulation_tick(ViewerStatePacket& packet,
                      const Viewer::TickFn& tick_fn,
                      float dt,
                      SimPanelState& sim) -> void {
  const bool reset = packet.reset_requested != 0;
  const bool advance = packet.simulate != 0;
  if (!tick_fn) {
    return;
  }
  const int steps = advance ? packet.steps_per_frame : 0;
  tick_fn(dt, packet.frame, reset, steps);
  if (reset) {
    sim.reset_clicked = false;
    packet.reset_requested = 0;
  }
  if (advance) {
    sim.steps_accum += packet.steps_per_frame;
    packet.dirty = 1;
    ++packet.frame;
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - sim.steps_t0).count();
    if (elapsed >= 0.5) {
      sim.steps_per_sec = static_cast<double>(sim.steps_accum) / elapsed;
      sim.steps_per_sec_smooth = 0.85 * sim.steps_per_sec_smooth + 0.15 * sim.steps_per_sec;
      sim.steps_accum = 0;
      sim.steps_t0 = now;
    }
  }
}

auto
fill_simulation_packet(ViewerStatePacket& packet,
                       const SimPanelState& sim) -> void {
  packet.simulate = sim.simulating ? 1 : 0;
  packet.steps_per_frame = sim.steps_per_frame;
  if (sim.reset_clicked) {
    packet.reset_requested = 1;
  }
}

[[nodiscard]] auto
frame_height_for_width(int width) noexcept -> int {
  const int snapped_w = std::max(320, (width / 16) * 16);
  int height = (snapped_w * 9 + 8) / 16;
  height = (height / 2) * 2;
  return std::clamp(height, 360, 2160);
}

struct FrameResolutionPreset {
  const char* label;
  int width;
  int height;
};

constexpr FrameResolutionPreset k_frame_resolution_presets[] = {
    {"0.5K", 960, 540}, {"1K", 1920, 1080}, {"2K", 2560, 1440},
    {"3K", 2880, 1620}, {"4K", 3840, 2160},
};

constexpr int k_frame_resolution_preset_count =
    static_cast<int>(sizeof(k_frame_resolution_presets) / sizeof(k_frame_resolution_presets[0]));

[[nodiscard]] auto
find_frame_resolution_preset(int width,
                             int height) noexcept -> int {
  for (int i = 0; i < k_frame_resolution_preset_count; ++i) {
    if (k_frame_resolution_presets[i].width == width &&
        k_frame_resolution_presets[i].height == height) {
      return i;
    }
  }
  return -1;
}

auto
apply_frame_resolution(int& frame_w,
                       int& frame_h,
                       int width,
                       int height,
                       const Viewer::ResolutionChangeFn& on_change) -> bool {
  if (width == frame_w && height == frame_h) {
    return false;
  }
  frame_w = width;
  frame_h = height;
  if (on_change) {
    on_change(frame_w, frame_h);
  }
  return true;
}

auto
draw_frame_resolution_menu(int& frame_w,
                           int& frame_h,
                           const Viewer::ResolutionChangeFn& on_change) -> bool {
  bool changed = false;
  const int current = find_frame_resolution_preset(frame_w, frame_h);
  const char* preview = (current >= 0) ? k_frame_resolution_presets[current].label : "custom";

  if (ImGui::BeginCombo("resolution", preview)) {
    for (int i = 0; i < k_frame_resolution_preset_count; ++i) {
      const FrameResolutionPreset& preset = k_frame_resolution_presets[i];
      const bool selected = (i == current);
      const char* item = (i == 0)   ? "0.5K  (960 x 540)"
                         : (i == 1) ? "1K    (1920 x 1080)"
                         : (i == 2) ? "2K    (2560 x 1440)"
                         : (i == 3) ? "3K    (2880 x 1620)"
                                    : "4K    (3840 x 2160)";
      if (ImGui::Selectable(item, selected)) {
        changed |= apply_frame_resolution(frame_w, frame_h, preset.width, preset.height, on_change);
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }
  ImGui::TextDisabled("%d x %d px (16:9)", frame_w, frame_h);
  return changed;
}

namespace {

constexpr double k_perf_smooth = 0.85;

[[nodiscard]] auto
elapsed_ms(std::chrono::steady_clock::time_point t0) -> double {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

auto
smooth(double prev,
       double sample) noexcept -> double {
  return k_perf_smooth * prev + (1.0 - k_perf_smooth) * sample;
}

auto
record_solver_perf(ViewerPerfStats& stats,
                   double solver_steps_s) -> void {
  stats.solver_steps_s = solver_steps_s;
}

auto
record_render_perf(ViewerPerfStats& stats,
                   double render_ms) -> void {
  stats.render_ms = render_ms;
  stats.render_ms_smooth = smooth(stats.render_ms_smooth.load(), render_ms);
}

auto
record_transfer_perf(ViewerPerfStats& stats,
                     double transfer_ms,
                     size_t transfer_bytes) -> void {
  stats.transfer_ms = transfer_ms;
  stats.transfer_ms_smooth = smooth(stats.transfer_ms_smooth.load(), transfer_ms);
  stats.transfer_bytes = transfer_bytes;
  const double sec = transfer_ms * 1.0e-3;
  const double mib_s =
      (sec > 1.0e-9) ? (static_cast<double>(transfer_bytes) / (1024.0 * 1024.0)) / sec : 0.0;
  stats.transfer_mib_s = mib_s;
  stats.transfer_mib_s_smooth = smooth(stats.transfer_mib_s_smooth.load(), mib_s);
}

auto
record_display_fps(ViewerPerfStats& stats,
                   double display_fps) -> void {
  stats.display_fps = display_fps;
}

}  // namespace

[[nodiscard]] auto
world_rank(const System& sys) noexcept -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_rank();
  }
#endif
  return 0;
}

[[nodiscard]] auto
world_size(const System& sys) noexcept -> int {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_size();
  }
#endif
  (void)sys;
  return 1;
}

// Emit an ImGui block describing the compute backend and the GPU(s) this
// process drives (name, compute capability, memory, SM count). Shared by both
// the desktop and browser UIs. Under MPI only rank 0 draws the UI, and each
// rank owns a single GPU, so the panel reports the rank count and details rank
// 0's device; the single-process case lists every visible GPU.
void
draw_system_info(const System& sys) {
  const auto& devices = sys.gpu_group().devices();
  ImGui::Text("Backend: %s", CARBON_BACKEND_NAME);
#ifdef CARBON_HAVE_MPI_NCCL
  const int ranks = world_size(sys);
  if (ranks > 1) {
    ImGui::Text("Transport: MPI + NCCL (%d ranks, 1 GPU/rank)", ranks);
    ImGui::Text("GPUs: %d total  (rank 0 shown)", ranks);
  } else {
    ImGui::Text("GPUs: %zu", devices.size());
  }
#else
  ImGui::Text("GPUs: %zu", devices.size());
#endif
  for (size_t i = 0; i < devices.size(); ++i) {
    const auto& p = devices[i].properties();
    const double gib = static_cast<double>(p.total_global_mem_bytes) / (1024.0 * 1024.0 * 1024.0);
    ImGui::BulletText("GPU %d: %s", devices[i].device_id(), p.name.c_str());
    ImGui::Text("      compute %d.%d  |  %.0f GiB  |  %d SMs", p.compute_major, p.compute_minor,
                gib, p.multiprocessor_count);
  }
}

auto
broadcast_packet(const System& sys,
                 ViewerStatePacket& packet,
                 int root) -> void {
#ifdef CARBON_HAVE_MPI_NCCL
  if (const auto* d = sys.distributed(); d != nullptr) {
    d->broadcast(&packet, sizeof(packet), root);
  }
#else
  (void)sys;
  (void)root;
  (void)packet;
#endif
}

[[nodiscard]] auto
orbit_to_packet(const OrbitCamera& orbit,
                int width,
                int height,
                bool running,
                bool dirty) -> ViewerStatePacket {
  ViewerStatePacket packet{};
  packet.target_x = orbit.target.x;
  packet.target_y = orbit.target.y;
  packet.target_z = orbit.target.z;
  packet.distance = orbit.distance;
  packet.azimuth = orbit.azimuth;
  packet.elevation = orbit.elevation;
  packet.fov_y_rad = orbit.fov_y_rad;
  packet.up_x = orbit.up.x;
  packet.up_y = orbit.up.y;
  packet.up_z = orbit.up.z;
  packet.bounds_min_x = orbit.bounds_min.x;
  packet.bounds_min_y = orbit.bounds_min.y;
  packet.bounds_min_z = orbit.bounds_min.z;
  packet.bounds_max_x = orbit.bounds_max.x;
  packet.bounds_max_y = orbit.bounds_max.y;
  packet.bounds_max_z = orbit.bounds_max.z;
  packet.width = width;
  packet.height = height;
  packet.running = running ? 1 : 0;
  packet.dirty = dirty ? 1 : 0;
  return packet;
}

[[nodiscard]] auto
packet_to_orbit(const ViewerStatePacket& packet) -> OrbitCamera {
  OrbitCamera orbit{};
  orbit.target = {packet.target_x, packet.target_y, packet.target_z};
  orbit.distance = packet.distance;
  orbit.azimuth = packet.azimuth;
  orbit.elevation = packet.elevation;
  orbit.fov_y_rad = packet.fov_y_rad;
  orbit.up = {packet.up_x, packet.up_y, packet.up_z};
  orbit.bounds_min = {packet.bounds_min_x, packet.bounds_min_y, packet.bounds_min_z};
  orbit.bounds_max = {packet.bounds_max_x, packet.bounds_max_y, packet.bounds_max_z};
  return orbit;
}

[[nodiscard]] auto
normalize3(float& x,
           float& y,
           float& z) noexcept -> bool {
  const float len = std::sqrt(x * x + y * y + z * z);
  if (len <= 0.0F) {
    return false;
  }
  const float inv = 1.0F / len;
  x *= inv;
  y *= inv;
  z *= inv;
  return true;
}

auto
draw_projected_wire_box(ImDrawList* dl,
                        const Camera& cam,
                        int width,
                        int height,
                        ImVec2 p0,
                        ImVec2 sz,
                        float32_3d bmin,
                        float32_3d bmax,
                        ImU32 col,
                        float thickness = 1.5F) -> void {
  ImVec2 pts[8];
  bool ok[8];
  for (int i = 0; i < 8; ++i) {
    const float32_3d corner{
        ((i & 1) != 0) ? bmax.x : bmin.x,
        ((i & 2) != 0) ? bmax.y : bmin.y,
        ((i & 4) != 0) ? bmax.z : bmin.z,
    };
    float fx = 0.0F;
    float fy = 0.0F;
    float depth = 0.0F;
    ok[i] = cam.project_pixel(corner, fx, fy, depth);
    pts[i] = ImVec2(p0.x + (fx / static_cast<float>(width)) * sz.x,
                    p0.y + (fy / static_cast<float>(height)) * sz.y);
  }
  constexpr int box_edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                    {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
  for (const auto& e : box_edges) {
    if (ok[e[0]] && ok[e[1]]) {
      dl->AddLine(pts[e[0]], pts[e[1]], col, thickness);
    }
  }
}

[[nodiscard]] auto
gpu_overlay_color(int index,
                  int count) noexcept -> ImU32 {
  if (count <= 0) {
    return IM_COL32(255, 140, 0, 220);
  }
  const float hue = (static_cast<float>(index) + 0.5F) / static_cast<float>(count);
  const float s = 0.85F;
  const float v = 1.0F;
  const float c = v * s;
  const float x = c * (1.0F - std::fabs(std::fmod(hue * 6.0F, 2.0F) - 1.0F));
  const float m = v - c;
  float r = 0.0F;
  float g = 0.0F;
  float b = 0.0F;
  switch (static_cast<int>(hue * 6.0F) % 6) {
    case 0:
      r = c;
      g = x;
      break;
    case 1:
      r = x;
      g = c;
      break;
    case 2:
      g = c;
      b = x;
      break;
    case 3:
      g = x;
      b = c;
      break;
    case 4:
      r = x;
      b = c;
      break;
    default:
      r = c;
      b = x;
      break;
  }
  return IM_COL32(static_cast<int>((r + m) * 255.0F), static_cast<int>((g + m) * 255.0F),
                  static_cast<int>((b + m) * 255.0F), 220);
}

constexpr float k_pi = 3.14159265358979F;
constexpr float k_orbit_sensitivity = 0.005F;
// Keep the camera off the up / anti-up poles where the orbit basis degenerates.
constexpr float k_orbit_polar_min = 0.05F;
constexpr float k_orbit_polar_max = k_pi - k_orbit_polar_min;

[[nodiscard]] auto
orbit_offset_dir(const OrbitCamera& orbit) noexcept -> float32_3d {
  return {
      std::cos(orbit.elevation) * std::sin(orbit.azimuth),
      std::cos(orbit.elevation) * std::cos(orbit.azimuth),
      std::sin(orbit.elevation),
  };
}

auto
orbit_set_offset_dir(OrbitCamera& orbit,
                     float ox,
                     float oy,
                     float oz) noexcept -> void {
  if (!normalize3(ox, oy, oz)) {
    return;
  }
  orbit.elevation = std::asin(std::clamp(oz, -1.0F, 1.0F));
  orbit.azimuth = std::atan2(ox, oy);
}

[[nodiscard]] auto
rotate_vec(float32_3d v,
           float32_3d axis,
           float angle) noexcept -> float32_3d {
  float ax = axis.x;
  float ay = axis.y;
  float az = axis.z;
  if (!normalize3(ax, ay, az)) {
    return v;
  }
  const float c = std::cos(angle);
  const float s = std::sin(angle);
  const float dot = v.x * ax + v.y * ay + v.z * az;
  const float rx = ay * v.z - az * v.y;
  const float ry = az * v.x - ax * v.z;
  const float rz = ax * v.y - ay * v.x;
  return {
      v.x * c + rx * s + ax * dot * (1.0F - c),
      v.y * c + ry * s + ay * dot * (1.0F - c),
      v.z * c + rz * s + az * dot * (1.0F - c),
  };
}

auto
orbit_apply_drag(OrbitCamera& orbit,
                 float dx,
                 float dy,
                 float sensitivity = k_orbit_sensitivity) noexcept -> void {
  if (dx == 0.0F && dy == 0.0F) {
    return;
  }
  float32_3d v = orbit_offset_dir(orbit);
  float ux = orbit.up.x;
  float uy = orbit.up.y;
  float uz = orbit.up.z;
  if (!normalize3(ux, uy, uz)) {
    return;
  }

  if (dx != 0.0F) {
    v = rotate_vec(v, {ux, uy, uz}, -dx * sensitivity);
  }
  if (dy != 0.0F) {
    float rx = uy * v.z - uz * v.y;
    float ry = uz * v.x - ux * v.z;
    float rz = ux * v.y - uy * v.x;
    if (normalize3(rx, ry, rz)) {
      v = rotate_vec(v, {rx, ry, rz}, -dy * sensitivity);
    }
  }

  const float dot_up = v.x * ux + v.y * uy + v.z * uz;
  const float polar = std::acos(std::clamp(dot_up, -1.0F, 1.0F));
  if (polar < k_orbit_polar_min || polar > k_orbit_polar_max) {
    const float clamped_polar = std::clamp(polar, k_orbit_polar_min, k_orbit_polar_max);
    float qx = v.x - ux * dot_up;
    float qy = v.y - uy * dot_up;
    float qz = v.z - uz * dot_up;
    if (normalize3(qx, qy, qz)) {
      const float para = std::cos(clamped_polar);
      const float perp = std::sin(clamped_polar);
      v.x = ux * para + qx * perp;
      v.y = uy * para + qy * perp;
      v.z = uz * para + qz * perp;
    }
  }

  orbit_set_offset_dir(orbit, v.x, v.y, v.z);
}

auto
orbit_apply_yaw(OrbitCamera& orbit,
                float angle) noexcept -> void {
  float32_3d v = orbit_offset_dir(orbit);
  float ux = orbit.up.x;
  float uy = orbit.up.y;
  float uz = orbit.up.z;
  if (!normalize3(ux, uy, uz)) {
    return;
  }
  v = rotate_vec(v, {ux, uy, uz}, angle);
  orbit_set_offset_dir(orbit, v.x, v.y, v.z);
}

#if defined(CARBON_HAVE_VIEWER_DESKTOP)
struct UiState {
  bool dragging_orbit{false};
  bool dragging_pan{false};
  double last_cursor_x{0.0};
  double last_cursor_y{0.0};
};

struct WindowScrollState {
  OrbitCamera* orbit{nullptr};
  bool* dirty{nullptr};
};

void
scroll_callback(GLFWwindow* window,
                double /*xoffset*/,
                double yoffset) {
  auto* state = static_cast<WindowScrollState*>(glfwGetWindowUserPointer(window));
  if (state == nullptr || state->orbit == nullptr) {
    return;
  }
  state->orbit->distance *= (yoffset > 0.0) ? 0.9F : 1.1F;
  state->orbit->distance = std::max(state->orbit->distance, 0.05F);
  if (state->dirty != nullptr) {
    *state->dirty = true;
  }
}

[[nodiscard]] auto
handle_input(GLFWwindow* window,
             OrbitCamera& orbit,
             UiState& ui,
             bool& dirty) -> bool {
  if (glfwWindowShouldClose(window)) {
    return false;
  }

  if (ImGui::GetIO().WantCaptureMouse) {
    ui.dragging_orbit = false;
    ui.dragging_pan = false;
    return true;
  }

  double cx = 0.0;
  double cy = 0.0;
  glfwGetCursorPos(window, &cx, &cy);

  const bool left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  const bool right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

  if (left && !ui.dragging_orbit) {
    ui.dragging_orbit = true;
    ui.last_cursor_x = cx;
    ui.last_cursor_y = cy;
  }
  if (!left) {
    ui.dragging_orbit = false;
  }

  if (right && !ui.dragging_pan) {
    ui.dragging_pan = true;
    ui.last_cursor_x = cx;
    ui.last_cursor_y = cy;
  }
  if (!right) {
    ui.dragging_pan = false;
  }

  const double dx = cx - ui.last_cursor_x;
  const double dy = cy - ui.last_cursor_y;

  if (ui.dragging_orbit && (dx != 0.0 || dy != 0.0)) {
    orbit_apply_drag(orbit, static_cast<float>(dx), static_cast<float>(dy));
    dirty = true;
  }

  if (ui.dragging_pan && (dx != 0.0 || dy != 0.0)) {
    float ux = orbit.up.x;
    float uy = orbit.up.y;
    float uz = orbit.up.z;
    if (normalize3(ux, uy, uz)) {
      const float32_3d off = orbit_offset_dir(orbit);
      float rx = uy * off.z - uz * off.y;
      float ry = uz * off.x - ux * off.z;
      float rz = ux * off.y - uy * off.x;
      if (normalize3(rx, ry, rz)) {
        const float pan_scale = orbit.distance * 0.001F;
        orbit.target.x -= static_cast<float>(dx) * rx * pan_scale;
        orbit.target.y -= static_cast<float>(dx) * ry * pan_scale;
        orbit.target.z -= static_cast<float>(dx) * rz * pan_scale;
        orbit.target.x += static_cast<float>(dy) * ux * pan_scale;
        orbit.target.y += static_cast<float>(dy) * uy * pan_scale;
        orbit.target.z += static_cast<float>(dy) * uz * pan_scale;
        dirty = true;
      }
    }
  }

  ui.last_cursor_x = cx;
  ui.last_cursor_y = cy;
  return true;
}
#endif  // CARBON_HAVE_VIEWER_DESKTOP

enum class AxisView : unsigned char {
  Top,
  Front,
  Side,
};

/**
 * Canonical view directions built from @c orbit.up , not from the Z-up azimuth / elevation
 * parametrization the orbit stores. An app that frames with @c up = +y would otherwise get a
 * "front" direction equal to its own up vector, where @c look_at loses the side axis and rolls
 * the scene by an arbitrary angle. @c Top deliberately stops short of the pole for the same
 * reason.
 */
[[nodiscard]] auto
orbit_axis_view_dir(const OrbitCamera& orbit,
                    AxisView view) noexcept -> float32_3d {
  float ux = orbit.up.x;
  float uy = orbit.up.y;
  float uz = orbit.up.z;
  if (!normalize3(ux, uy, uz)) {
    ux = 0.0F;
    uy = 0.0F;
    uz = 1.0F;
  }

  const float32_3d candidates[3] = {
      {0.0F, 1.0F, 0.0F},
      {1.0F, 0.0F, 0.0F},
      {0.0F, 0.0F, 1.0F},
  };
  float32_3d fwd = candidates[0];
  float best = 2.0F;
  for (const auto& axis : candidates) {
    const float align = std::fabs(axis.x * ux + axis.y * uy + axis.z * uz);
    if (align < best) {
      best = align;
      fwd = axis;
    }
  }

  const float along = fwd.x * ux + fwd.y * uy + fwd.z * uz;
  float fx = fwd.x - ux * along;
  float fy = fwd.y - uy * along;
  float fz = fwd.z - uz * along;
  if (!normalize3(fx, fy, fz)) {
    return {ux, uy, uz};
  }

  switch (view) {
    case AxisView::Front:
      return {fx, fy, fz};
    case AxisView::Side:
      return {fy * uz - fz * uy, fz * ux - fx * uz, fx * uy - fy * ux};
    case AxisView::Top:
    default: {
      constexpr float k_polar_offset = 0.17F;
      const float c = std::cos(k_polar_offset);
      const float s = std::sin(k_polar_offset);
      return {ux * c + fx * s, uy * c + fy * s, uz * c + fz * s};
    }
  }
}

auto
orbit_set_axis_view(OrbitCamera& orbit,
                    AxisView view) noexcept -> void {
  const float32_3d dir = orbit_axis_view_dir(orbit, view);
  orbit_set_offset_dir(orbit, dir.x, dir.y, dir.z);
}

[[nodiscard]] auto
orbit_reset_frame(const OrbitCamera& view_template,
                  const Viewer::FrameBoundsFn& frame_bounds_fn) -> OrbitCamera {
  if (!frame_bounds_fn) {
    return view_template;
  }
  const AxisAlignedBox box = frame_bounds_fn();
  return OrbitCamera::from_frame_bounds(box.min, box.max, orbit_offset_dir(view_template),
                                        view_template.up, view_template.fov_y_rad);
}

auto
overlay_domain_bounds(const OrbitCamera& orbit,
                      const Viewer::DomainBoundsFn& fn) -> std::pair<float32_3d,
                                                                     float32_3d> {
  if (fn) {
    const AxisAlignedBox box = fn();
    return {box.min, box.max};
  }
  return {orbit.bounds_min, orbit.bounds_max};
}

}  // namespace

auto
OrbitCamera::to_camera(int width,
                       int height) const -> Camera {
  const float cx = std::cos(elevation) * std::sin(azimuth);
  const float cy = std::cos(elevation) * std::cos(azimuth);
  const float cz = std::sin(elevation);
  const float32_3d eye{
      target.x + distance * cx,
      target.y + distance * cy,
      target.z + distance * cz,
  };

  const float size_x = bounds_max.x - bounds_min.x;
  const float size_y = bounds_max.y - bounds_min.y;
  const float size_z = bounds_max.z - bounds_min.z;
  const float radius = 0.5F * std::sqrt(size_x * size_x + size_y * size_y + size_z * size_z);
  const float znear = std::max(0.01F, (distance - radius) * 0.1F);
  const float zfar = distance + radius * 2.0F + std::max({size_x, size_y, size_z});

  return Camera::perspective(eye, target, up, fov_y_rad, znear, zfar, width, height);
}

auto
OrbitCamera::from_frame_bounds(float32_3d bounds_min,
                               float32_3d bounds_max,
                               float32_3d eye_offset_dir,
                               float32_3d up,
                               float fov_y_rad,
                               float padding) -> OrbitCamera {
  OrbitCamera orbit{};
  orbit.bounds_min = bounds_min;
  orbit.bounds_max = bounds_max;
  orbit.up = up;
  orbit.fov_y_rad = fov_y_rad;

  orbit.target = {
      0.5F * (bounds_min.x + bounds_max.x),
      0.5F * (bounds_min.y + bounds_max.y),
      0.5F * (bounds_min.z + bounds_max.z),
  };

  float dir_x = eye_offset_dir.x;
  float dir_y = eye_offset_dir.y;
  float dir_z = eye_offset_dir.z;
  if (!normalize3(dir_x, dir_y, dir_z)) {
    dir_x = 0.0F;
    dir_y = -1.0F;
    dir_z = 0.0F;
  }

  const float size_x = bounds_max.x - bounds_min.x;
  const float size_y = bounds_max.y - bounds_min.y;
  const float size_z = bounds_max.z - bounds_min.z;
  const float radius = 0.5F * std::sqrt(size_x * size_x + size_y * size_y + size_z * size_z);
  const float tan_half = std::tan(0.5F * fov_y_rad);
  orbit.distance = (radius / tan_half) * padding;

  orbit.elevation = std::asin(std::clamp(dir_z, -1.0F, 1.0F));
  orbit.azimuth = std::atan2(dir_x, dir_y);
  return orbit;
}

auto
Viewer::display_available() noexcept -> bool {
#if defined(CARBON_HAVE_VIEWER_DESKTOP)
  if (!glfwInit()) {
    return false;
  }
  glfwTerminate();
  return true;
#else
  return false;
#endif
}

Viewer::Viewer(int width,
               int height,
               const char* /*title*/)
    : width_(width),
      height_(height) {}

Viewer::~Viewer() = default;

auto
Viewer::frame_width() const noexcept -> int {
  return width_;
}

auto
Viewer::frame_height() const noexcept -> int {
  return height_;
}

auto
Viewer::set_frame_size(int width,
                       int height) -> void {
  width_ = std::clamp((width / 16) * 16, 640, 3840);
  height_ = std::clamp((height / 2) * 2, 360, 2160);
  if (find_frame_resolution_preset(width_, height_) < 0) {
    height_ = frame_height_for_width(width_);
  }
}

#if defined(CARBON_HAVE_VIEWER_DESKTOP)
auto
Viewer::run(const System& sys,
            OrbitCamera initial,
            RenderFn render_fn,
            std::vector<std::string> render_modes) -> void {
  RunOptions opts{};
  opts.render_modes = std::move(render_modes);
  run(sys, initial, std::move(render_fn), std::move(opts));
}

auto
Viewer::run(const System& sys,
            OrbitCamera initial,
            RenderFn render_fn,
            RunOptions opts) -> void {
  const int rank = world_rank(sys);
  const bool tick_enabled = static_cast<bool>(opts.tick_fn);
  SimPanelState sim{};
  sim.simulating = opts.simulate_default;
  sim.steps_per_frame = std::max(1, opts.steps_per_frame_default);

  if (rank != 0) {
    WebViewerSharedState shared{};
    shared.orbit = initial;
    shared.sim = sim;
    shared.frame_w = width_;
    shared.frame_h = height_;
    std::mutex ws_mtx;
    std::atomic<bool> app_running{true};
    ImGuiWS dummy_ws;
    std::thread sim_thread([&]() {
      run_web_sim_thread(sys, opts, render_fn, shared, ws_mtx, dummy_ws, 0,
                         static_cast<ImGuiWS::TextureId>(0), app_running);
    });
    sim_thread.join();
    return;
  }

  if (!glfwInit()) {
    CARBON_LOG_ERROR("Viewer: no display available; skipping interactive mode");
    return;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

  GLFWwindow* window = glfwCreateWindow(width_, height_, "carbon viewer", nullptr, nullptr);
  if (window == nullptr) {
    glfwTerminate();
    CARBON_LOG_ERROR("Viewer: glfwCreateWindow failed; skipping interactive mode");
    return;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  OrbitCamera orbit = initial;
  const OrbitCamera reset_orbit = initial;
  UiState ui{};
  bool dirty = true;
  int render_mode = 0;
  WindowScrollState scroll_state{&orbit, &dirty};
  glfwSetWindowUserPointer(window, &scroll_state);
  glfwSetScrollCallback(window, scroll_callback);

  unsigned int texture = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);

  std::vector<float> rgb;
  std::vector<unsigned char> upload;
  bool running = true;
  int tex_w = width_;
  int tex_h = height_;

  auto last_frame = std::chrono::steady_clock::now();
  float fps = 0.0F;

  while (running) {
    glfwPollEvents();

    running = handle_input(window, orbit, ui, dirty);

    ViewerStatePacket packet = orbit_to_packet(orbit, width_, height_, running, dirty);
    packet.render_mode = render_mode;
    fill_simulation_packet(packet, sim);
    broadcast_packet(sys, packet, 0);
    running = packet.running != 0;
    orbit = packet_to_orbit(packet);
    render_mode = packet.render_mode;

    const auto now = std::chrono::steady_clock::now();
    const float dt = std::chrono::duration<float>(now - last_frame).count();
    last_frame = now;
    apply_simulation_tick(packet, opts.tick_fn, dt, sim);
    if (opts.perf_stats != nullptr) {
      record_solver_perf(*opts.perf_stats, sim.steps_per_sec_smooth);
    }

    if (packet.dirty != 0) {
      const Camera camera = orbit.to_camera(width_, height_);
      const auto t_render = std::chrono::steady_clock::now();
      render_fn(camera, width_, height_, render_mode, rgb);
      dirty = false;

      upload.resize(rgb.size());
      for (size_t i = 0; i < rgb.size(); ++i) {
        const float c = std::clamp(rgb[i], 0.0F, 1.0F);
        upload[i] = static_cast<unsigned char>(std::lround(c * 255.0F));
      }
      const auto t_upload = std::chrono::steady_clock::now();
      glBindTexture(GL_TEXTURE_2D, texture);
      if (width_ != tex_w || height_ != tex_h) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE,
                     upload.data());
        tex_w = width_;
        tex_h = height_;
      } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE,
                        upload.data());
      }
      if (opts.perf_stats != nullptr) {
        record_render_perf(*opts.perf_stats, elapsed_ms(t_render));
        record_transfer_perf(*opts.perf_stats, elapsed_ms(t_upload),
                             static_cast<size_t>(width_) * static_cast<size_t>(height_) * 3U);
      }
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::GetBackgroundDrawList()->AddImage(static_cast<ImTextureID>(texture), ImVec2(0.0F, 0.0F),
                                             display);

    if (dt > 0.0F) {
      fps = 0.9F * fps + 0.1F * (1.0F / dt);
    }
    if (opts.perf_stats != nullptr) {
      record_display_fps(*opts.perf_stats, static_cast<double>(fps));
    }

    ImGui::SetNextWindowBgAlpha(0.35F);
    ImGui::Begin("Controls", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoNav);
    ImGui::Text("FPS: %.1f", static_cast<double>(fps));
    ImGui::Text("L-drag: orbit  R-drag: pan  scroll: zoom");
    ImGui::Text("target: (%.2f, %.2f, %.2f)", static_cast<double>(orbit.target.x),
                static_cast<double>(orbit.target.y), static_cast<double>(orbit.target.z));
    ImGui::Text("distance: %.3f  az: %.2f  el: %.2f", static_cast<double>(orbit.distance),
                static_cast<double>(orbit.azimuth), static_cast<double>(orbit.elevation));
    ImGui::Separator();
    draw_system_info(sys);
    ImGui::Separator();
    if (!opts.render_modes.empty()) {
      const char* preview =
          opts.render_modes[static_cast<size_t>(std::clamp(
                                render_mode, 0, static_cast<int>(opts.render_modes.size()) - 1))]
              .c_str();
      if (ImGui::BeginCombo("style", preview)) {
        for (int m = 0; m < static_cast<int>(opts.render_modes.size()); ++m) {
          const bool selected = (m == render_mode);
          if (ImGui::Selectable(opts.render_modes[static_cast<size_t>(m)].c_str(), selected)) {
            render_mode = m;
            dirty = true;
          }
          if (selected) {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
    }
    if (draw_frame_resolution_menu(width_, height_, opts.resolution_change_fn)) {
      dirty = true;
    }
    if (draw_simulation_panel(sim, tick_enabled)) {
      dirty = true;
    }
    if (opts.ui_fn && opts.ui_fn()) {
      dirty = true;
    }
    if (ImGui::Button("Reset camera")) {
      orbit = orbit_reset_frame(reset_orbit, opts.frame_bounds_fn);
      dirty = true;
    }
    ImGui::End();

    ImGui::Render();
    glViewport(0, 0, width_, height_);
    glClearColor(0.05F, 0.06F, 0.09F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  glDeleteTextures(1, &texture);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}
#else   // CARBON_HAVE_VIEWER_DESKTOP
auto
Viewer::run(const System& sys,
            OrbitCamera initial,
            RenderFn render_fn,
            std::vector<std::string> render_modes) -> void {
  RunOptions opts{};
  opts.render_modes = std::move(render_modes);
  run(sys, initial, std::move(render_fn), std::move(opts));
}

auto
Viewer::run(const System& sys,
            OrbitCamera initial,
            RenderFn render_fn,
            RunOptions opts) -> void {
  (void)sys;
  (void)initial;
  (void)render_fn;
  (void)opts;
  CARBON_LOG_ERROR(
      "Viewer::run: the desktop (GLFW/OpenGL) backend was not built. Rebuild with "
      "-DCARBON_VIEWER_DESKTOP=ON, or use the browser backend via run_web() "
      "(-DCARBON_VIEWER_WEB=ON).");
}
#endif  // CARBON_HAVE_VIEWER_DESKTOP

#if defined(CARBON_HAVE_VIEWER_WEB)
namespace {

[[nodiscard]] auto
round_pot(int v) noexcept -> int {
  int p = 64;
  while (p < v && p < 4096) {
    p <<= 1;
  }
  return p;
}

// Nearest-neighbour downscale of an interleaved RGB float image into a square power-of-two
// RGB24 byte buffer. The image is packed at the top-left corner (aspect-preserving); the used
// sub-rect is reported back so the caller can crop with UVs. The web client uses a WebGL1
// context, which requires power-of-two textures with the default wrap.
auto
pack_fit(const std::vector<float>& rgb,
         int width,
         int height,
         int edge,
         std::vector<char>& out,
         int& used_w,
         int& used_h) -> void {
  out.assign(static_cast<size_t>(edge) * edge * 3, static_cast<char>(242));
  used_w = edge;
  used_h = edge;
  if (width <= 0 || height <= 0 ||
      rgb.size() < static_cast<size_t>(width) * static_cast<size_t>(height) * 3U) {
    return;
  }
  const float scale = std::min(static_cast<float>(edge) / width, static_cast<float>(edge) / height);
  used_w = std::clamp(static_cast<int>(static_cast<float>(width) * scale), 1, edge);
  used_h = std::clamp(static_cast<int>(static_cast<float>(height) * scale), 1, edge);
  for (int y = 0; y < used_h; ++y) {
    const int sy = std::min(height - 1, static_cast<int>((static_cast<float>(y) + 0.5F) / scale));
    for (int x = 0; x < used_w; ++x) {
      const int sx = std::min(width - 1, static_cast<int>((static_cast<float>(x) + 0.5F) / scale));
      const size_t si = (static_cast<size_t>(sy) * width + sx) * 3U;
      const size_t di = (static_cast<size_t>(y) * edge + x) * 3U;
      out[di + 0] = static_cast<char>(std::lround(std::clamp(rgb[si + 0], 0.0F, 1.0F) * 255.0F));
      out[di + 1] = static_cast<char>(std::lround(std::clamp(rgb[si + 1], 0.0F, 1.0F) * 255.0F));
      out[di + 2] = static_cast<char>(std::lround(std::clamp(rgb[si + 2], 0.0F, 1.0F) * 255.0F));
    }
  }
}

[[nodiscard]] auto
web_index_html(int /*edge*/) -> std::string {
  return R"HTML(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>carbon viewer</title>
<style>html,body{margin:0;padding:0;overflow:hidden;background:#f2f4f3;color:#202020;font-family:monospace}
#canvas_main{display:block;position:fixed;top:0;left:0;width:100vw;height:100vh}</style>
<script src="incppect.js"></script>
<script src="imgui-ws.js"></script>
</head>
<body>
<canvas id="canvas_main"></canvas>
<script>
function apply_size(){
  var w=Math.max(64,window.innerWidth), h=Math.max(64,window.innerHeight);
  var c=imgui_ws.canvas, gl=imgui_ws.gl;
  c.width=w; c.height=h;
  gl.viewport(0,0,w,h);
  var L=0,R=w,T=0,B=h;
  var proj=new Float32Array([2/(R-L),0,0,0, 0,2/(T-B),0,0, 0,0,-1,0, (R+L)/(L-R),(T+B)/(B-T),0,1]);
  gl.useProgram(imgui_ws.shader_program);
  gl.uniformMatrix4fv(imgui_ws.attribute_location_proj_mtx,false,proj);
}
function init(){
  var c=document.getElementById('canvas_main');
  c.width=Math.max(64,window.innerWidth); c.height=Math.max(64,window.innerHeight);
  imgui_ws._sent_w=0; imgui_ws._sent_h=0;
  incppect.render=function(){
    var c=imgui_ws.canvas;
    // Report the live canvas size to the server every time it changes. Done from the render tick
    // (which only runs once the socket is open) so the very first size is never lost to a send
    // that races the WebSocket handshake.
    if(imgui_ws._sent_w!==c.width || imgui_ws._sent_h!==c.height){
      imgui_ws._sent_w=c.width; imgui_ws._sent_h=c.height;
      this.send('7 '+c.width+' '+c.height);
    }
    imgui_ws.gl.clearColor(0.949,0.957,0.953,1.0);
    imgui_ws.gl.clear(imgui_ws.gl.COLOR_BUFFER_BIT);
    imgui_ws.incppect_textures(this);
    imgui_ws.incppect_draw_lists(this);
    imgui_ws.render();
  };
  incppect.k_requests_update_freq_ms=16;
  incppect.init();
  imgui_ws.set_incppect_handlers(incppect);
  imgui_ws.init('canvas_main');
  apply_size();
  window.addEventListener('resize',apply_size);
}
window.addEventListener('load',init);
</script>
</body>
</html>)HTML";
}

[[nodiscard]] auto
dom_button_to_imgui(int32_t dom_button) noexcept -> int {
  if (dom_button == 2) {
    return 1;  // DOM right -> ImGui right
  }
  if (dom_button == 1) {
    return 2;  // DOM middle -> ImGui middle
  }
  return 0;  // DOM left -> ImGui left
}

// A compact, rounded dark theme with a single green accent to match the render palette.
auto
apply_viewer_style() -> void {
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 9.0F;
  style.ChildRounding = 9.0F;
  style.FrameRounding = 6.0F;
  style.PopupRounding = 6.0F;
  style.GrabRounding = 6.0F;
  style.ScrollbarRounding = 9.0F;
  style.TabRounding = 6.0F;
  style.WindowBorderSize = 0.0F;
  style.FrameBorderSize = 0.0F;
  style.WindowPadding = ImVec2(13.0F, 13.0F);
  style.FramePadding = ImVec2(9.0F, 5.0F);
  style.ItemSpacing = ImVec2(9.0F, 8.0F);
  style.ItemInnerSpacing = ImVec2(7.0F, 6.0F);
  style.GrabMinSize = 11.0F;
  style.ScrollbarSize = 12.0F;

  const ImVec4 accent(0.24F, 0.80F, 0.44F, 1.0F);
  const ImVec4 accent_dim(0.24F, 0.80F, 0.44F, 0.35F);
  ImVec4* c = style.Colors;
  c[ImGuiCol_Text] = ImVec4(0.90F, 0.93F, 0.91F, 1.0F);
  c[ImGuiCol_TextDisabled] = ImVec4(0.48F, 0.54F, 0.50F, 1.0F);
  c[ImGuiCol_WindowBg] = ImVec4(0.070F, 0.082F, 0.094F, 1.0F);
  c[ImGuiCol_PopupBg] = ImVec4(0.070F, 0.082F, 0.094F, 1.0F);
  c[ImGuiCol_Border] = ImVec4(1.0F, 1.0F, 1.0F, 0.06F);
  c[ImGuiCol_FrameBg] = ImVec4(0.14F, 0.16F, 0.18F, 1.0F);
  c[ImGuiCol_FrameBgHovered] = ImVec4(0.18F, 0.21F, 0.23F, 1.0F);
  c[ImGuiCol_FrameBgActive] = ImVec4(0.20F, 0.24F, 0.26F, 1.0F);
  c[ImGuiCol_TitleBg] = ImVec4(0.055F, 0.065F, 0.075F, 1.0F);
  c[ImGuiCol_TitleBgActive] = ImVec4(0.055F, 0.065F, 0.075F, 1.0F);
  c[ImGuiCol_Separator] = ImVec4(1.0F, 1.0F, 1.0F, 0.07F);
  c[ImGuiCol_SliderGrab] = accent;
  c[ImGuiCol_SliderGrabActive] = ImVec4(0.32F, 0.88F, 0.52F, 1.0F);
  c[ImGuiCol_CheckMark] = accent;
  c[ImGuiCol_Button] = ImVec4(0.16F, 0.19F, 0.21F, 1.0F);
  c[ImGuiCol_ButtonHovered] = accent_dim;
  c[ImGuiCol_ButtonActive] = accent;
  c[ImGuiCol_Header] = ImVec4(0.16F, 0.19F, 0.21F, 1.0F);
  c[ImGuiCol_HeaderHovered] = accent_dim;
  c[ImGuiCol_HeaderActive] = accent;
}

struct WebViewerSharedState {
  std::mutex mtx;
  OrbitCamera orbit{};
  SimPanelState sim{};
  int render_mode{0};
  bool dirty{true};
  bool running{true};
  int frame_w{0};
  int frame_h{0};
  int used_w{0};
  int used_h{0};
};

auto
run_web_sim_thread(const System& sys,
                   const Viewer::WebOptions& opts,
                   const Viewer::RenderFn& render_fn,
                   WebViewerSharedState& shared,
                   std::mutex& ws_mtx,
                   ImGuiWS& imgui_ws,
                   int edge,
                   ImGuiWS::TextureId frame_tex,
                   std::atomic<bool>& app_running) -> void {
  std::vector<float> rgb;
  std::vector<char> tex;
  auto sim_last = std::chrono::steady_clock::now();

  while (app_running.load(std::memory_order_relaxed)) {
    ViewerStatePacket packet{};
    bool sim_active = false;
    {
      std::lock_guard<std::mutex> lock(shared.mtx);
      packet = orbit_to_packet(shared.orbit, shared.frame_w, shared.frame_h, shared.running,
                               shared.dirty);
      packet.render_mode = shared.render_mode;
      fill_simulation_packet(packet, shared.sim);
      sim_active = shared.sim.simulating;
    }
    broadcast_packet(sys, packet, 0);

    const auto sim_now = std::chrono::steady_clock::now();
    const float sim_dt = std::chrono::duration<float>(sim_now - sim_last).count();
    sim_last = sim_now;

    SimPanelState sim_work{};
    {
      std::lock_guard<std::mutex> lock(shared.mtx);
      shared.running = packet.running != 0;
      sim_work = shared.sim;
    }
    const bool reset_consumed = packet.reset_requested != 0;
    apply_simulation_tick(packet, opts.tick_fn, sim_dt, sim_work);
    {
      std::lock_guard<std::mutex> lock(shared.mtx);
      // Write back only the throughput counters this thread owns. The imgui-ws thread owns
      // simulating / steps_per_frame and may have written them while the tick above ran
      // unlocked, so assigning the whole struct would restore the pre-tick snapshot over the
      // top of them and make Play/Pause and the steps slider bounce back.
      shared.sim.steps_per_sec = sim_work.steps_per_sec;
      shared.sim.steps_per_sec_smooth = sim_work.steps_per_sec_smooth;
      shared.sim.steps_accum = sim_work.steps_accum;
      shared.sim.steps_t0 = sim_work.steps_t0;
      if (reset_consumed) {
        shared.sim.reset_clicked = false;
      }
      if (packet.dirty != 0) {
        shared.dirty = true;
      }
      if (opts.perf_stats != nullptr) {
        record_solver_perf(*opts.perf_stats, shared.sim.steps_per_sec_smooth);
      }
    }

    if (packet.dirty != 0) {
      OrbitCamera orbit_snap{};
      int fw = 0;
      int fh = 0;
      int mode = 0;
      {
        std::lock_guard<std::mutex> lock(shared.mtx);
        orbit_snap = shared.orbit;
        fw = shared.frame_w;
        fh = shared.frame_h;
        mode = shared.render_mode;
      }
      const Camera camera = orbit_snap.to_camera(fw, fh);
      const auto t_render = std::chrono::steady_clock::now();
      render_fn(camera, fw, fh, mode, rgb);
      const auto t_pack = std::chrono::steady_clock::now();
      int uw = 0;
      int uh = 0;
      pack_fit(rgb, fw, fh, edge, tex, uw, uh);
      if (frame_tex != static_cast<ImGuiWS::TextureId>(0)) {
        std::lock_guard<std::mutex> lock(ws_mtx);
        imgui_ws.setTexture(frame_tex, ImGuiWS::Texture::Type::RGB24, edge, edge, tex.data());
      }
      {
        std::lock_guard<std::mutex> lock(shared.mtx);
        shared.used_w = uw;
        shared.used_h = uh;
        shared.dirty = false;
      }
      if (opts.perf_stats != nullptr) {
        record_render_perf(*opts.perf_stats, elapsed_ms(t_render));
        record_transfer_perf(*opts.perf_stats, elapsed_ms(t_pack),
                             static_cast<size_t>(edge) * static_cast<size_t>(edge) * 3U);
      }
    } else if (!sim_active) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

}  // namespace

auto
Viewer::run_web(const System& sys,
                OrbitCamera initial,
                RenderFn render_fn,
                WebOptions opts) -> void {
  const int rank = world_rank(sys);
  const bool tick_enabled = static_cast<bool>(opts.tick_fn);
  SimPanelState sim{};
  sim.simulating = opts.simulate_default;
  sim.steps_per_frame = std::max(1, opts.steps_per_frame_default);

  if (rank != 0) {
    WebViewerSharedState shared{};
    shared.orbit = initial;
    shared.sim = sim;
    shared.frame_w = width_;
    shared.frame_h = height_;
    std::mutex ws_mtx;
    std::atomic<bool> app_running{true};
    ImGuiWS dummy_ws;
    std::thread sim_thread([&]() {
      run_web_sim_thread(sys, opts, render_fn, shared, ws_mtx, dummy_ws, 0,
                         static_cast<ImGuiWS::TextureId>(0), app_running);
    });
    sim_thread.join();
    return;
  }

  const int edge = round_pot(opts.tex_edge);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  apply_viewer_style();
  ImGuiIO& io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.DisplaySize = ImVec2(static_cast<float>(edge), static_cast<float>(edge));

  ImGuiWS imgui_ws;
  if (!imgui_ws.init(opts.port, std::string(), {std::string(), std::string("index.html")})) {
    CARBON_LOG_ERROR("Viewer: imgui-ws failed to start on port {}", opts.port);
    ImGui::DestroyContext();
    return;
  }
  imgui_ws.addResource("/index.html", web_index_html(edge));

  {
    const char* font_paths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    };
    ImFontConfig cfg;
    cfg.OversampleH = 2;
    cfg.OversampleV = 2;
    cfg.PixelSnapH = true;
    bool loaded = false;
    for (const char* path : font_paths) {
      if (std::ifstream(path).good()) {
        if (io.Fonts->AddFontFromFileTTF(path, 17.0F, &cfg) != nullptr) {
          loaded = true;
          break;
        }
      }
    }
    if (!loaded) {
      io.Fonts->AddFontDefault();
      io.FontGlobalScale = 1.2F;
    }
  }

  {
    unsigned char* pixels = nullptr;
    int fw = 0;
    int fh = 0;
    io.Fonts->GetTexDataAsAlpha8(&pixels, &fw, &fh);
    imgui_ws.setTexture(0, ImGuiWS::Texture::Type::Alpha8, fw, fh,
                        reinterpret_cast<const char*>(pixels));
    io.Fonts->SetTexID(static_cast<ImTextureID>(0));
  }
  constexpr ImGuiWS::TextureId k_frame_tex = 1;

  CARBON_LOG_INFO("Viewer: browser viewer on http://<host>:{}/  ({}x{} stream)", opts.port, edge,
                  edge);

  WebViewerSharedState shared{};
  shared.orbit = initial;
  shared.sim = sim;
  shared.frame_w = width_;
  shared.frame_h = height_;
  shared.used_w = edge;
  shared.used_h = edge;
  std::mutex ws_mtx;
  std::atomic<bool> app_running{true};
  std::thread sim_thread([&]() {
    run_web_sim_thread(sys, opts, render_fn, shared, ws_mtx, imgui_ws, edge, k_frame_tex,
                       app_running);
  });

  OrbitCamera orbit = initial;
  const OrbitCamera reset_orbit = initial;
  bool dirty = true;
  bool running = true;

  float disp_w = static_cast<float>(edge);
  float disp_h = static_cast<float>(edge);

  bool auto_rotate = false;
  bool show_bbox = false;
  bool show_gpu_bbox = false;
  int render_mode = 0;
  float rotate_speed = 0.6F;
  float last_dt = 1.0F / 60.0F;
  const float dist_max = std::max(reset_orbit.distance * 4.0F, 1.0F);

  int used_w = edge;
  int used_h = edge;
  auto last_frame = std::chrono::steady_clock::now();
  float fps = 0.0F;

  constexpr int k_fps_hist = 90;
  std::vector<float> fps_hist(k_fps_hist, 0.0F);
  int fps_head = 0;

  int64_t diag_events = 0;
  int diag_prev_clients = 0;
  auto diag_last = std::chrono::steady_clock::now();

  while (running) {
    if (auto_rotate) {
      orbit_apply_yaw(orbit, rotate_speed * last_dt);
      dirty = true;
    }

    for (const auto& event : imgui_ws.takeEvents()) {
      ++diag_events;
      switch (event.type) {
        case ImGuiWS::Event::MouseMove:
          io.AddMousePosEvent(event.mouse_x, event.mouse_y);
          break;
        case ImGuiWS::Event::MouseDown:
          io.AddMousePosEvent(event.mouse_x, event.mouse_y);
          io.AddMouseButtonEvent(dom_button_to_imgui(event.mouse_but), true);
          break;
        case ImGuiWS::Event::MouseUp:
          io.AddMouseButtonEvent(dom_button_to_imgui(event.mouse_but), false);
          break;
        case ImGuiWS::Event::MouseWheel:
          io.AddMouseWheelEvent(event.wheel_x, event.wheel_y);
          break;
        case ImGuiWS::Event::Resize:
          disp_w = static_cast<float>(std::max(64, event.client_width));
          disp_h = static_cast<float>(std::max(64, event.client_height));
          break;
        default:
          break;
      }
    }

    {
      const auto diag_now = std::chrono::steady_clock::now();
      const int clients = imgui_ws.nConnected();
      if (clients != diag_prev_clients) {
        CARBON_LOG_INFO("Viewer: clients now {}", clients);
        diag_prev_clients = clients;
      }
      if (std::chrono::duration<float>(diag_now - diag_last).count() >= 1.0F) {
        if (diag_events > 0) {
          CARBON_LOG_INFO("Viewer: {} input event(s) in last ~1s (clients={})", diag_events,
                          clients);
        }
        diag_events = 0;
        diag_last = diag_now;
      }
    }

    {
      std::lock_guard<std::mutex> lock(shared.mtx);
      running = shared.running;
      orbit = shared.orbit;
      render_mode = shared.render_mode;
      width_ = shared.frame_w;
      height_ = shared.frame_h;
      used_w = shared.used_w;
      used_h = shared.used_h;
      sim.simulating = shared.sim.simulating;
      sim.steps_per_frame = shared.sim.steps_per_frame;
      sim.steps_per_sec_smooth = shared.sim.steps_per_sec_smooth;
    }

    const auto now = std::chrono::steady_clock::now();
    const float dt = std::chrono::duration<float>(now - last_frame).count();
    last_frame = now;
    if (dt > 0.0F) {
      fps = 0.9F * fps + 0.1F * (1.0F / dt);
    }
    fps_hist[static_cast<size_t>(fps_head)] = fps;
    fps_head = (fps_head + 1) % k_fps_hist;
    last_dt = (dt > 0.0F && dt < 0.5F) ? dt : (1.0F / 60.0F);
    if (opts.perf_stats != nullptr) {
      record_display_fps(*opts.perf_stats, static_cast<double>(fps));
    }
    io.DeltaTime = (dt > 0.0F) ? dt : (1.0F / 60.0F);
    io.DisplaySize = ImVec2(disp_w, disp_h);

    ImGui::NewFrame();

    // The rendered frame lives inside a full-viewport ImGui window (fills the browser canvas).
    // An invisible button over the fitted image captures orbit / pan / zoom.
    {
      ImGui::SetNextWindowPos(ImVec2(0.0F, 0.0F));
      ImGui::SetNextWindowSize(io.DisplaySize);
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(242, 244, 243, 255));
      constexpr ImGuiWindowFlags host_flags =
          ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
          ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
          ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoSavedSettings;
      ImGui::Begin("Render", nullptr, host_flags);

      const ImVec2 avail = ImGui::GetContentRegionAvail();
      const float ar =
          (used_h > 0) ? static_cast<float>(used_w) / static_cast<float>(used_h) : 1.0F;
      ImVec2 sz = avail;
      if (avail.x / std::max(1.0F, avail.y) > ar) {
        sz = ImVec2(avail.y * ar, avail.y);
      } else {
        sz = ImVec2(avail.x, avail.x / ar);
      }
      const ImVec2 cur = ImGui::GetCursorScreenPos();
      const ImVec2 p0(cur.x + (avail.x - sz.x) * 0.5F, cur.y + (avail.y - sz.y) * 0.5F);
      ImGui::SetCursorScreenPos(p0);
      ImGui::InvisibleButton("##render_input", sz,
                             ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
      const bool hovered = ImGui::IsItemHovered();
      const bool active = ImGui::IsItemActive();

      const ImVec2 uv1(static_cast<float>(used_w) / static_cast<float>(edge),
                       static_cast<float>(used_h) / static_cast<float>(edge));
      ImDrawList* dl = ImGui::GetWindowDrawList();
      dl->AddImage(static_cast<ImTextureID>(k_frame_tex), p0, ImVec2(p0.x + sz.x, p0.y + sz.y),
                   ImVec2(0.0F, 0.0F), uv1);

      if (active) {
        const ImVec2 d = io.MouseDelta;
        if (d.x != 0.0F || d.y != 0.0F) {
          if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            orbit_apply_drag(orbit, d.x, d.y);
            dirty = true;
          } else if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            float ux = orbit.up.x;
            float uy = orbit.up.y;
            float uz = orbit.up.z;
            if (normalize3(ux, uy, uz)) {
              const float32_3d off = orbit_offset_dir(orbit);
              float rx = uy * off.z - uz * off.y;
              float ry = uz * off.x - ux * off.z;
              float rz = ux * off.y - uy * off.x;
              if (normalize3(rx, ry, rz)) {
                const float pan = orbit.distance * 0.001F;
                orbit.target.x += (-d.x * rx + d.y * ux) * pan;
                orbit.target.y += (-d.x * ry + d.y * uy) * pan;
                orbit.target.z += (-d.x * rz + d.y * uz) * pan;
                dirty = true;
              }
            }
          }
        }
      }
      if (hovered && io.MouseWheel != 0.0F) {
        orbit.distance *= (io.MouseWheel > 0.0F) ? 0.9F : 1.1F;
        orbit.distance = std::max(orbit.distance, 0.05F);
        dirty = true;
      }

      if (show_bbox || show_gpu_bbox) {
        const Camera cam = orbit.to_camera(width_, height_);
        if (show_bbox) {
          const auto domain = overlay_domain_bounds(orbit, opts.domain_bounds_fn);
          draw_projected_wire_box(dl, cam, width_, height_, p0, sz, domain.first, domain.second,
                                  IM_COL32(255, 140, 0, 220));
        }
        if (show_gpu_bbox && opts.gpu_bounds_fn) {
          const std::vector<AxisAlignedBox> gpu_boxes = opts.gpu_bounds_fn();
          const int n = static_cast<int>(gpu_boxes.size());
          for (int i = 0; i < n; ++i) {
            draw_projected_wire_box(dl, cam, width_, height_, p0, sz,
                                    gpu_boxes[static_cast<size_t>(i)].min,
                                    gpu_boxes[static_cast<size_t>(i)].max, gpu_overlay_color(i, n));
          }
        }
      }

      ImGui::End();
      ImGui::PopStyleColor();
      ImGui::PopStyleVar();
    }

    constexpr float k_panel_item_w = 168.0F;
    constexpr float k_panel_w = 300.0F;

    ImGui::SetNextWindowPos(ImVec2(16.0F, 16.0F), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowBgAlpha(1.0F);
    ImGui::Begin("Camera controls", nullptr,
                 ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);
    ImGui::PushItemWidth(k_panel_item_w);

    ImGui::Text("%d client%s connected", imgui_ws.nConnected(),
                imgui_ws.nConnected() == 1 ? "" : "s");
    ImGui::SameLine();
    ImGui::TextDisabled("%.0f FPS", static_cast<double>(fps));
    ImGui::PlotLines("##fps", fps_hist.data(), k_fps_hist, fps_head, nullptr, 0.0F, 120.0F,
                     ImVec2(k_panel_w, 28.0F));
    ImGui::TextDisabled("L-drag orbit   R-drag pan   wheel zoom");

    bool ui_changed = false;

    constexpr auto k_open = ImGuiTreeNodeFlags_DefaultOpen;

    if (ImGui::CollapsingHeader("System", k_open)) {
      draw_system_info(sys);
    }

    if (ImGui::CollapsingHeader("Camera", k_open)) {
      ui_changed |= ImGui::SliderFloat("distance", &orbit.distance, 0.05F, dist_max, "%.3f");

      float az_deg = orbit.azimuth * 180.0F / k_pi;
      if (ImGui::SliderFloat("azimuth", &az_deg, -180.0F, 180.0F, "%.1f deg")) {
        orbit.azimuth = az_deg * k_pi / 180.0F;
        ui_changed = true;
      }

      float el_deg = orbit.elevation * 180.0F / k_pi;
      if (ImGui::SliderFloat("elevation", &el_deg, -80.0F, 80.0F, "%.1f deg")) {
        orbit.elevation = std::clamp(el_deg * k_pi / 180.0F, -1.4F, 1.4F);
        ui_changed = true;
      }

      float fov_deg = orbit.fov_y_rad * 180.0F / k_pi;
      if (ImGui::SliderFloat("fov", &fov_deg, 15.0F, 110.0F, "%.1f deg")) {
        orbit.fov_y_rad = fov_deg * k_pi / 180.0F;
        ui_changed = true;
      }

      float target[3] = {orbit.target.x, orbit.target.y, orbit.target.z};
      if (ImGui::DragFloat3("target", target, 0.5F)) {
        orbit.target.x = target[0];
        orbit.target.y = target[1];
        orbit.target.z = target[2];
        ui_changed = true;
      }

      if (ImGui::Button("Reset camera")) {
        orbit = orbit_reset_frame(reset_orbit, opts.frame_bounds_fn);
        auto_rotate = false;
        ui_changed = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Top")) {
        orbit_set_axis_view(orbit, AxisView::Top);
        ui_changed = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Front")) {
        orbit_set_axis_view(orbit, AxisView::Front);
        ui_changed = true;
      }
      ImGui::SameLine();
      if (ImGui::Button("Side")) {
        orbit_set_axis_view(orbit, AxisView::Side);
        ui_changed = true;
      }
    }

    if (!opts.render_modes.empty() && ImGui::CollapsingHeader("Rendering", k_open)) {
      const int last = static_cast<int>(opts.render_modes.size()) - 1;
      const char* preview =
          opts.render_modes[static_cast<size_t>(std::clamp(render_mode, 0, last))].c_str();
      if (ImGui::BeginCombo("style", preview)) {
        for (int m = 0; m <= last; ++m) {
          const bool selected = (m == render_mode);
          if (ImGui::Selectable(opts.render_modes[static_cast<size_t>(m)].c_str(), selected)) {
            render_mode = m;
            ui_changed = true;
          }
          if (selected) {
            ImGui::SetItemDefaultFocus();
          }
        }
        ImGui::EndCombo();
      }
      if (draw_frame_resolution_menu(width_, height_, opts.resolution_change_fn)) {
        ui_changed = true;
      }
    }

    if (ImGui::CollapsingHeader("Animation", k_open)) {
      ImGui::Checkbox("auto-rotate", &auto_rotate);
      ImGui::SameLine();
      ImGui::SetNextItemWidth(120.0F);
      ImGui::SliderFloat("speed", &rotate_speed, 0.05F, 3.0F, "%.2f rad/s");
    }

    if (draw_simulation_panel(sim, tick_enabled)) {
      ui_changed = true;
    }
    if (opts.ui_fn && opts.ui_fn()) {
      ui_changed = true;
    }

    if (ImGui::CollapsingHeader("Overlays", k_open)) {
      ImGui::Checkbox("show Carbon grid", &show_bbox);
      ImGui::TextDisabled("wireframe matches dense extent");
      if (opts.gpu_bounds_fn) {
        ImGui::Checkbox("show GPU slabs", &show_gpu_bbox);
      }
    }

    if (ImGui::CollapsingHeader("Orientation", k_open)) {
      constexpr float k_giz = 108.0F;
      ImGui::Dummy(ImVec2(k_giz, k_giz));
      const ImVec2 rmin = ImGui::GetItemRectMin();
      const ImVec2 gc(rmin.x + k_giz * 0.5F, rmin.y + k_giz * 0.5F);

      const float ex = std::cos(orbit.elevation) * std::sin(orbit.azimuth);
      const float ey = std::cos(orbit.elevation) * std::cos(orbit.azimuth);
      const float ez = std::sin(orbit.elevation);
      float fx = -ex;
      float fy = -ey;
      float fz = -ez;
      float ux = orbit.up.x;
      float uy = orbit.up.y;
      float uz = orbit.up.z;
      if (normalize3(fx, fy, fz) && normalize3(ux, uy, uz)) {
        float rx = fy * uz - fz * uy;
        float ry = fz * ux - fx * uz;
        float rz = fx * uy - fy * ux;
        if (normalize3(rx, ry, rz)) {
          const float cux = ry * fz - rz * fy;
          const float cuy = rz * fx - rx * fz;
          const float cuz = rx * fy - ry * fx;

          constexpr float k_len = 44.0F;
          ImDrawList* dl = ImGui::GetWindowDrawList();
          dl->AddCircleFilled(gc, k_len + 7.0F, IM_COL32(255, 255, 255, 255), 40);
          dl->AddCircle(gc, k_len + 7.0F, IM_COL32(0, 0, 0, 45), 40, 1.5F);

          struct Axis {
            float wx;
            float wy;
            float wz;
            ImU32 col;
            const char* label;
          };
          const Axis axes[3] = {
              {1.0F, 0.0F, 0.0F, IM_COL32(228, 66, 66, 255), "X"},
              {0.0F, 1.0F, 0.0F, IM_COL32(70, 190, 90, 255), "Y"},
              {0.0F, 0.0F, 1.0F, IM_COL32(80, 130, 240, 255), "Z"},
          };
          float sxa[3];
          float sya[3];
          for (int i = 0; i < 3; ++i) {
            sxa[i] = axes[i].wx * rx + axes[i].wy * ry + axes[i].wz * rz;
            sya[i] = axes[i].wx * cux + axes[i].wy * cuy + axes[i].wz * cuz;
          }

          // Negative-axis stubs first (hollow dots), so positive axes overlay them.
          for (int i = 0; i < 3; ++i) {
            const ImVec2 ntip(gc.x - sxa[i] * k_len * 0.5F, gc.y + sya[i] * k_len * 0.5F);
            const ImU32 dim = (axes[i].col & 0x00FFFFFFU) | 0x66000000U;
            dl->AddLine(gc, ntip, dim, 1.5F);
            dl->AddCircle(ntip, 3.0F, dim, 10, 1.5F);
          }

          int order[3] = {0, 1, 2};
          float depth[3];
          for (int i = 0; i < 3; ++i) {
            depth[i] = axes[i].wx * fx + axes[i].wy * fy + axes[i].wz * fz;
          }
          for (int a = 0; a < 3; ++a) {
            for (int b = a + 1; b < 3; ++b) {
              if (depth[order[b]] > depth[order[a]]) {
                const int tmp = order[a];
                order[a] = order[b];
                order[b] = tmp;
              }
            }
          }
          for (int k = 0; k < 3; ++k) {
            const int a = order[k];
            const ImVec2 tip(gc.x + sxa[a] * k_len, gc.y - sya[a] * k_len);
            dl->AddLine(gc, tip, axes[a].col, 2.5F);
            dl->AddCircleFilled(tip, 4.0F, axes[a].col, 12);
            dl->AddText(ImVec2(tip.x + 4.0F, tip.y - 7.0F), axes[a].col, axes[a].label);
          }
        }
      }
    }

    if (ui_changed) {
      dirty = true;
    }

    {
      std::lock_guard<std::mutex> lock(shared.mtx);
      shared.orbit = orbit;
      shared.render_mode = render_mode;
      shared.frame_w = width_;
      shared.frame_h = height_;
      shared.running = running;
      shared.sim.simulating = sim.simulating;
      shared.sim.steps_per_frame = sim.steps_per_frame;
      // One-shot handoff: raise the flag for the sim thread, then drop it here. The pull-back
      // at the top of this loop deliberately does not carry reset_clicked, so re-publishing a
      // local copy every frame would re-raise it the instant the sim thread cleared it and
      // rebuild the simulation on every tick instead of once.
      if (sim.reset_clicked) {
        shared.sim.reset_clicked = true;
        sim.reset_clicked = false;
      }
      if (dirty) {
        shared.dirty = true;
      }
    }

    ImGui::PopItemWidth();
    ImGui::End();

    ImGui::Render();
    {
      std::lock_guard<std::mutex> lock(ws_mtx);
      imgui_ws.setDrawData(ImGui::GetDrawData());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(imgui_ws.nConnected() > 0 ? 16 : 100));
  }

  app_running.store(false, std::memory_order_relaxed);
  sim_thread.join();

  ImGui::DestroyContext();
}
#endif  // CARBON_HAVE_VIEWER_WEB

}  // namespace carbon::viewer
