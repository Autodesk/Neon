// Live LBM wind-tunnel viewer: steps the D3Q27+BGK solve continuously while ray-marching
// the wake. Presets (small / medium / large / xlarge), runtime Re and inlet-speed control, and the
// same eight render styles as render_wind_tunnel.
//
// Build & run
// -----------
//   cmake --build build --target carbon_example_render_wind_tunnel_live
//   ./build/examples/carbon_example_render_wind_tunnel_live --web --port 5000
//   ./build/examples/carbon_example_render_wind_tunnel_live --preset medium --multi-gpu

#include "wind_tunnel_common.h"

#include <carbon/logging.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/distributed.h>
#endif

#ifdef CARBON_HAVE_VIEWER
#include <imgui.h>
#include "viewer/carbon_viewer.h"
#include "viewer/render_rgb_utils.h"
#endif

#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

namespace wt = carbon::examples::wind_tunnel;

namespace {

constexpr int k_mode_cp = 4;
constexpr int k_rho_refresh_period = 8;

#ifdef CARBON_HAVE_MPI_NCCL
auto
broadcast_params(const carbon::System& sys,
                 wt::ParamPacket& packet,
                 int root) -> void {
  if (const auto* d = sys.distributed(); d != nullptr) {
    d->broadcast(&packet, sizeof(packet), root);
  }
}
#else
auto
broadcast_params(const carbon::System& /*sys*/,
                 wt::ParamPacket& /*packet*/,
                 int /*root*/) -> void {}
#endif

[[nodiscard]] auto
parse_preset(const std::string& name) -> wt::Preset {
  if (name == "small") {
    return wt::Preset::Small;
  }
  if (name == "medium") {
    return wt::Preset::Medium;
  }
  if (name == "xlarge" || name == "extra-large" || name == "xl") {
    return wt::Preset::XLarge;
  }
  return wt::Preset::Large;
}

[[nodiscard]] auto
preset_label(wt::Preset p) -> const char* {
  switch (p) {
    case wt::Preset::Small:
      return "small";
    case wt::Preset::Medium:
      return "medium";
    case wt::Preset::Large:
      return "large";
    default:
      return "xlarge";
  }
}

[[nodiscard]] auto
shape_label(wt::ObstacleShape s) -> const char* {
  switch (s) {
    case wt::ObstacleShape::Sphere:
      return "sphere";
    case wt::ObstacleShape::Box:
      return "box";
    default:
      return "cylinder";
  }
}

[[nodiscard]] auto
needs_q_refresh(int mode) -> bool {
  return mode == 1 || mode == 3;
}

[[nodiscard]] auto
needs_rho_refresh(int mode) -> bool {
  return mode == k_mode_cp;
}

struct LiveAppState {
  carbon::System* sys{nullptr};
  std::unique_ptr<wt::LiveSim> sim;
  wt::WindTunnelRenderContext render;
  wt::ParamPacket params{};
  wt::Preset pending_preset{wt::Preset::Large};
  bool pending_restart{false};
  int current_mode{0};
  int64_t rho_refresh_counter{0};
  int frame_w{1920};
  int frame_h{1080};
  size_t visible_gpu_count{0};
  bool use_all_gpus{false};
  bool dark_background{false};

  struct RenderBuffers {
    std::unique_ptr<carbon::RayMarchImage> vel;
    std::unique_ptr<carbon::RayMarchImage> q;
    std::unique_ptr<carbon::RayMarchImage> sphere;
    std::unique_ptr<carbon::SurfelImage> cp;

    auto
    ensure(carbon::System& sys,
           int w,
           int h) -> void {
      const auto rebuild = [&](auto& img) {
        using Image = std::decay_t<decltype(*img)>;
        if (img == nullptr || img->width() != w || img->height() != h) {
          img = std::make_unique<Image>(sys, w, h);
        }
      };
      rebuild(vel);
      rebuild(q);
      rebuild(sphere);
      if (cp == nullptr || cp->width() != w || cp->height() != h) {
        cp = std::make_unique<carbon::SurfelImage>(sys, w, h, /*with_attribute=*/true);
      }
    }
  };

  RenderBuffers buffers{};

  auto
  apply_background() -> void {
    render.background = dark_background ? carbon::float32_3d{0.18F, 0.18F, 0.20F}
                                        : carbon::float32_3d{1.0F, 1.0F, 1.0F};
  }

  auto
  rebuild_sim(wt::LiveSimConfig cfg) -> void {
    sim = std::make_unique<wt::LiveSim>(*sys, cfg);
    render = wt::WindTunnelRenderContext::build(*sim);
    apply_background();
    params.re = cfg.re;
    params.ulb = cfg.ulb;
    params.preset = static_cast<int32_t>(cfg.preset);
    params.shape = static_cast<int32_t>(cfg.shape);
    params.restart = 0;
    pending_preset = cfg.preset;
    rho_refresh_counter = 0;
    CARBON_LOG_INFO("render_wind_tunnel_live: preset={} N={}x{}x{} Re={:.1f} u={:.4g} omega={:.4f}",
                    preset_label(cfg.preset), cfg.nx, cfg.ny, cfg.nz, sim->re(), sim->params().ulb,
                    sim->params().omega);
  }

  auto
  apply_params() -> void {
    if (sim == nullptr) {
      return;
    }
    const bool preset_changed = static_cast<wt::Preset>(params.preset) != sim->config().preset;
    const bool shape_changed = static_cast<wt::ObstacleShape>(params.shape) != sim->config().shape;
    const bool physics_changed = std::abs(sim->config().re - params.re) > 1.0e-3F ||
                                 std::abs(sim->config().ulb - params.ulb) > 1.0e-5F;
    if (params.restart != 0 || preset_changed || shape_changed) {
      wt::LiveSimConfig cfg =
          wt::LiveSimConfig::from_preset(static_cast<wt::Preset>(params.preset));
      cfg.re = params.re;
      cfg.ulb = params.ulb;
      cfg.shape = static_cast<wt::ObstacleShape>(params.shape);
      rebuild_sim(cfg);
      params.restart = 0;
      return;
    }
    if (physics_changed) {
      sim->set_physics(params.re, params.ulb);
      render = wt::WindTunnelRenderContext::build(*sim);
      apply_background();
    }
  }

  /**
   * Runs the per-mode derived-field passes. Called without @c g_app_mtx held: a Q refresh is a
   * kernel plus three halo exchanges plus a device reduce, and holding the lock across it stalls
   * the UI thread for the whole frame.
   */
  auto
  refresh_for_mode(wt::LiveSim& live,
                   int mode) -> void {
    if (needs_q_refresh(mode)) {
      live.refresh_q();
    } else if (needs_rho_refresh(mode) && (rho_refresh_counter++ % k_rho_refresh_period == 0)) {
      live.refresh_rho_inf();
    }
  }
};

std::mutex g_app_mtx;

LiveAppState g_app;
carbon::viewer::ViewerPerfStats g_perf{};

auto
draw_domain_info() -> bool {
  bool changed = false;
  if (g_app.sys == nullptr || g_app.sim == nullptr) {
    return changed;
  }
  const wt::LiveSimConfig& cfg = g_app.sim->config();
  const size_t active_gpus = g_app.sys->gpu_group().size();
  const int64_t cell_count = cfg.nx * cfg.ny * cfg.nz;
  const int64_t z_per_gpu =
      (active_gpus > 0U) ? (cfg.nz / static_cast<int64_t>(active_gpus)) : cfg.nz;
  const int frame_w = g_app.frame_w;
  const int frame_h = g_app.frame_h;

  if (ImGui::CollapsingHeader("Domain & display", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Text("Carbon grid (nx x ny x nz): %lld x %lld x %lld", static_cast<long long>(cfg.nx),
                static_cast<long long>(cfg.ny), static_cast<long long>(cfg.nz));
    ImGui::Text("Cells: %.2f M", static_cast<double>(cell_count) / 1.0e6);
    if (active_gpus > 1U) {
      ImGui::Text("Z-slab (flow): %lld cells / GPU (%zu GPUs)", static_cast<long long>(z_per_gpu),
                  active_gpus);
    }
    ImGui::TextDisabled("x = cross-flow, y = span (+y cylinder axis), z = flow / GPU split");
    ImGui::Text("LBM iteration: %lld", static_cast<long long>(g_app.sim->iteration()));
    ImGui::Separator();
    ImGui::Text("Frame: %d x %d px", frame_w, frame_h);
    ImGui::TextDisabled("Presets: 0.5K–4K (16:9)");
    ImGui::Separator();
    ImGui::Text("GPUs active: %zu / %zu visible", active_gpus, g_app.visible_gpu_count);
    if (g_app.use_all_gpus) {
      ImGui::TextDisabled("All visible GPUs (--multi-gpu)");
    } else {
      ImGui::TextDisabled("Single GPU (add --multi-gpu to use all)");
    }
    if (g_app.visible_gpu_count > 1U) {
      ImGui::TextWrapped(
          "GPU count is fixed at launch; stop and restart with or without --multi-gpu to change.");
    }
    ImGui::Separator();
    if (ImGui::Checkbox("dark gray background", &g_app.dark_background)) {
      g_app.apply_background();
      changed = true;
    }
  }
  return changed;
}

auto
draw_performance_stats() -> void {
  if (g_app.sys == nullptr || g_app.sim == nullptr) {
    return;
  }
  if (!ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }
  const wt::LiveSimConfig& cfg = g_app.sim->config();
  const int64_t cell_count = cfg.nx * cfg.ny * cfg.nz;
  const size_t active_gpus = std::max<size_t>(1U, g_app.sys->gpu_group().size());
  const double mlups = g_perf.solver_steps_s.load() * static_cast<double>(cell_count) / 1.0e6;
  const double mlups_per_gpu = mlups / static_cast<double>(active_gpus);

  ImGui::Text("Solver: %.1f MLUPS", mlups);
  ImGui::TextDisabled("  %.1f MLUPS / GPU", mlups_per_gpu);
  ImGui::Separator();
  ImGui::Text("Render: %.1f ms", g_perf.render_ms_smooth.load());
  ImGui::Text("Upload: %.1f ms", g_perf.transfer_ms_smooth.load());
  ImGui::Text("Payload: %.2f MiB / frame",
              static_cast<double>(g_perf.transfer_bytes.load()) / (1024.0 * 1024.0));
  ImGui::Text("Upload BW: %.1f MiB/s", g_perf.transfer_mib_s_smooth.load());
  ImGui::Text("Display: %.0f FPS", g_perf.display_fps.load());
  ImGui::TextDisabled("Upload = RGB pack + imgui-ws texture push");
}

auto
draw_physics_ui() -> bool {
  bool changed = false;
  changed |= draw_domain_info();
  draw_performance_stats();
  if (ImGui::CollapsingHeader("Wind tunnel", ImGuiTreeNodeFlags_DefaultOpen)) {
    const char* preview = preset_label(static_cast<wt::Preset>(g_app.params.preset));
    if (ImGui::BeginCombo("preset", preview)) {
      for (int p = 0; p < 4; ++p) {
        const auto preset = static_cast<wt::Preset>(p);
        const bool selected = (g_app.params.preset == p);
        if (ImGui::Selectable(preset_label(preset), selected)) {
          g_app.params.preset = p;
          g_app.pending_restart = true;
          changed = true;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    const char* shape_preview = shape_label(static_cast<wt::ObstacleShape>(g_app.params.shape));
    if (ImGui::BeginCombo("obstacle", shape_preview)) {
      for (int s = 0; s < 3; ++s) {
        const auto shape = static_cast<wt::ObstacleShape>(s);
        const bool selected = (g_app.params.shape == s);
        if (ImGui::Selectable(shape_label(shape), selected)) {
          g_app.params.shape = s;
          g_app.pending_restart = true;
          changed = true;
        }
        if (selected) {
          ImGui::SetItemDefaultFocus();
        }
      }
      ImGui::EndCombo();
    }

    float re = g_app.params.re;
    float ulb = g_app.params.ulb;
    changed |= ImGui::SliderFloat("Re", &re, 50.0F, 800.0F, "%.0f");
    changed |= ImGui::SliderFloat("u / lb", &ulb, 0.03F, 0.20F, "%.3f");
    if (changed) {
      g_app.params.re = re;
      g_app.params.ulb = ulb;
    }

    if (g_app.sim != nullptr) {
      const float d = g_app.sim->config().diameter();
      const float nu = wt::compute_nu(ulb, d, wt::clamp_re(ulb, d, re));
      const float omega = wt::compute_omega(ulb, d, wt::clamp_re(ulb, d, re));
      const ImVec4 warn = (omega > 1.95F) ? ImVec4(1.0F, 0.45F, 0.2F, 1.0F) : ImVec4(1, 1, 1, 1);
      ImGui::TextColored(warn, "nu = %.5g   omega = %.4f", static_cast<double>(nu),
                         static_cast<double>(omega));
    }

    if (ImGui::Button("Reset simulation")) {
      g_app.pending_restart = true;
      changed = true;
    }
  }
  return changed;
}

[[noreturn]] auto
print_usage(const char* prog) -> void {
  std::fprintf(stderr,
               "Usage: %s [OPTIONS]\n"
               "\n"
               "Options:\n"
               "  --preset small|medium|large|xlarge   Domain preset (default large)\n"
               "                                xlarge is 537M cells and needs several GPUs\n"
               "  --warmup N                    LBM iterations before viewer (default 3000)\n"
               "  --bench N                     Run N iterations and print steps/s (no viewer)\n"
               "  --multi-gpu, --all-gpus       Use every visible GPU\n"
               "  --interactive                 Desktop viewer (needs display)\n"
#ifdef CARBON_HAVE_VIEWER_WEB
               "  --web                         Browser viewer (default when built)\n"
               "  --port PORT                   Web viewer port (default 5000)\n"
#endif
               "  -h, --help                    Print help\n",
               prog);
  std::exit(EXIT_SUCCESS);
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  carbon::init();

#ifndef CARBON_HAVE_VIEWER
  CARBON_LOG_ERROR("render_wind_tunnel_live: rebuild with CARBON_BUILD_VIEWER=ON");
  return EXIT_FAILURE;
#endif

  wt::Preset preset = wt::Preset::Large;
  int64_t warmup = 3000;
  int64_t bench_iters = 0;
  bool want_multi_gpu = false;
  bool interactive = false;
#ifdef CARBON_HAVE_VIEWER_WEB
  bool web = true;
  int web_port = 5000;
#else
  bool web = false;
#endif

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
    } else if (arg == "--multi-gpu" || arg == "--all-gpus") {
      want_multi_gpu = true;
    } else if (arg == "--interactive") {
      interactive = true;
      web = false;
#ifdef CARBON_HAVE_VIEWER_WEB
    } else if (arg == "--web") {
      web = true;
    } else if (arg == "--port" && i + 1 < argc) {
      web_port = std::atoi(argv[++i]);
#endif
    } else if (arg == "--preset" && i + 1 < argc) {
      preset = parse_preset(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::atoll(argv[++i]);
    } else if (arg == "--bench" && i + 1 < argc) {
      bench_iters = std::atoll(argv[++i]);
    }
  }

  if (!want_multi_gpu) {
    const char* multi = std::getenv("CARBON_WT_MULTI_GPU");
    want_multi_gpu = (multi != nullptr && std::atoi(multi) != 0);
  }

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int ranks = distributed->world_size();
  carbon::System sys = (ranks > 1) ? carbon::System(distributed)
                                   : carbon::System(wt::pick_render_devices(want_multi_gpu));
#else
  carbon::System sys(wt::pick_render_devices(want_multi_gpu));
#endif

  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("render_wind_tunnel_live: no devices visible");
    return EXIT_FAILURE;
  }

  CARBON_LOG_INFO("render_wind_tunnel_live: {} GPU(s) selected ({} visible)",
                  sys.gpu_group().size(), carbon::get_all_devices().size());

  g_app.sys = &sys;
  g_app.visible_gpu_count = carbon::get_all_devices().size();
  g_app.use_all_gpus = want_multi_gpu;
  wt::LiveSimConfig cfg = wt::LiveSimConfig::from_preset(preset);
  g_app.rebuild_sim(cfg);

  g_app.params.preset = static_cast<int32_t>(preset);
  g_app.params.shape = static_cast<int32_t>(wt::ObstacleShape::Cylinder);
  g_app.params.re = cfg.re;
  g_app.params.ulb = cfg.ulb;

  if (bench_iters > 0) {
    const auto t0 = std::chrono::steady_clock::now();
    g_app.sim->step(static_cast<int>(bench_iters));
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    const double steps_per_sec = static_cast<double>(bench_iters) / sec;
    if (wt::world_rank(sys) == 0) {
      std::printf("preset=%s gpus=%zu iters=%lld time=%.3fs steps/s=%.1f\n", preset_label(preset),
                  sys.gpu_group().size(), static_cast<long long>(bench_iters), sec, steps_per_sec);
    }
    return EXIT_SUCCESS;
  }

  if (warmup > 0) {
    CARBON_LOG_INFO("render_wind_tunnel_live: warm-up {} iterations", warmup);
    g_app.sim->step(static_cast<int>(warmup));
    g_app.sim->refresh_macroscopic();
    g_app.sim->refresh_q();
    g_app.sim->refresh_rho_inf();
    g_app.render = wt::WindTunnelRenderContext::build(*g_app.sim);
  }

  g_app.buffers.ensure(sys, g_app.frame_w, g_app.frame_h);

  const auto on_resolution_change = [&](int w, int h) {
    std::lock_guard<std::mutex> lock(g_app_mtx);
    g_app.frame_w = w;
    g_app.frame_h = h;
  };

  const auto render_cb = [&](const carbon::Camera& camera, int w, int h, int mode,
                             std::vector<float>& rgb_out) {
    carbon::float32_3d background{};
    {
      std::lock_guard<std::mutex> lock(g_app_mtx);
      g_app.buffers.ensure(sys, w, h);
      g_app.current_mode = mode;
      // The UI thread flips this from the dark-background checkbox, so snapshot it here rather
      // than reading g_app.render.background unsynchronized below.
      background = g_app.render.background;
    }
    if (mode == k_mode_cp) {
      g_app.render.render_cp_for_camera(camera, *g_app.buffers.cp);
      carbon::viewer::rgb_from_surfel_attribute(*g_app.buffers.cp, camera,
                                                carbon::Colormap::CoolWarm, wt::detail::k_cp_min,
                                                wt::detail::k_cp_max, rgb_out);
      return;
    }
    const wt::Layers layers = wt::detail::style_layers(mode);
    const bool on_q = (layers == wt::Layers::Q || layers == wt::Layers::QTinted);
    g_app.render.render_for_camera(camera, *g_app.buffers.vel, *g_app.buffers.q,
                                   *g_app.buffers.sphere, layers);
    carbon::viewer::rgb_from_composited_layers(on_q ? *g_app.buffers.q : *g_app.buffers.vel,
                                               *g_app.buffers.sphere, background, rgb_out);
  };

  const auto tick_fn = [&](float /*dt*/, int64_t /*frame*/, bool reset_requested,
                           int steps_per_frame) {
    wt::ParamPacket local{};
    {
      std::lock_guard<std::mutex> lock(g_app_mtx);
      local = g_app.params;
      if (wt::world_rank(sys) == 0) {
        if (reset_requested || g_app.pending_restart) {
          local.restart = 1;
          g_app.pending_restart = false;
        }
      }
    }
    broadcast_params(sys, local, 0);
    {
      std::lock_guard<std::mutex> lock(g_app_mtx);
      g_app.params = local;
      g_app.apply_params();
    }
    wt::LiveSim* sim_ptr = nullptr;
    int mode = 0;
    {
      std::lock_guard<std::mutex> lock(g_app_mtx);
      sim_ptr = g_app.sim.get();
      mode = g_app.current_mode;
    }
    if (sim_ptr != nullptr && steps_per_frame > 0) {
      sim_ptr->step(steps_per_frame);
      sim_ptr->refresh_macroscopic();
      g_app.refresh_for_mode(*sim_ptr, mode);
    }
  };

  const auto ui_fn = [&]() -> bool {
    if (wt::world_rank(sys) != 0) {
      return false;
    }
    std::lock_guard<std::mutex> lock(g_app_mtx);
    return draw_physics_ui();
  };

  const auto domain_bounds_fn = [&]() -> carbon::viewer::AxisAlignedBox {
    std::lock_guard<std::mutex> lock(g_app_mtx);
    if (g_app.sim == nullptr) {
      return {};
    }
    const auto bounds = wt::dense_world_bounds(g_app.sim->dense());
    return {bounds.first, bounds.second};
  };

  const auto frame_bounds_fn = [&]() -> carbon::viewer::AxisAlignedBox {
    std::lock_guard<std::mutex> lock(g_app_mtx);
    if (g_app.sim == nullptr) {
      return {};
    }
    const auto bounds = wt::view_frame_bounds(g_app.sim->config());
    return {bounds.first, bounds.second};
  };

  const auto gpu_bounds_fn = [&]() -> std::vector<carbon::viewer::AxisAlignedBox> {
    std::vector<carbon::viewer::AxisAlignedBox> out;
    std::lock_guard<std::mutex> lock(g_app_mtx);
    if (g_app.sim == nullptr) {
      return out;
    }
    for (const auto& box : wt::gpu_slab_boxes(g_app.sim->dense())) {
      out.push_back({box.first, box.second});
    }
    return out;
  };

  const auto view_bounds = wt::view_frame_bounds(cfg);
  const auto orbit = carbon::viewer::OrbitCamera::from_frame_bounds(
      view_bounds.first, view_bounds.second, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}, 0.785398F);

#ifdef CARBON_HAVE_VIEWER_WEB
  if (web) {
    carbon::viewer::Viewer viewer(g_app.frame_w, g_app.frame_h, "render_wind_tunnel_live");
    carbon::viewer::Viewer::WebOptions wopts;
    wopts.port = web_port;
    wopts.render_modes = g_app.render.render_mode_names;
    wopts.simulate_default = true;
    wopts.steps_per_frame_default = 4;
    wopts.tick_fn = tick_fn;
    wopts.ui_fn = ui_fn;
    wopts.resolution_change_fn = on_resolution_change;
    wopts.gpu_bounds_fn = gpu_bounds_fn;
    wopts.domain_bounds_fn = domain_bounds_fn;
    wopts.frame_bounds_fn = frame_bounds_fn;
    wopts.perf_stats = &g_perf;
    viewer.run_web(sys, orbit, render_cb, wopts);
    return EXIT_SUCCESS;
  }
#endif

  if (interactive && wt::world_rank(sys) == 0 && !carbon::viewer::Viewer::display_available()) {
    CARBON_LOG_WARN("render_wind_tunnel_live: no display; use --web");
    return EXIT_FAILURE;
  }

  carbon::viewer::Viewer viewer(g_app.frame_w, g_app.frame_h, "render_wind_tunnel_live");
  carbon::viewer::Viewer::RunOptions ropts;
  ropts.render_modes = g_app.render.render_mode_names;
  ropts.simulate_default = true;
  ropts.steps_per_frame_default = 4;
  ropts.tick_fn = tick_fn;
  ropts.ui_fn = ui_fn;
  ropts.resolution_change_fn = on_resolution_change;
  ropts.gpu_bounds_fn = gpu_bounds_fn;
  ropts.domain_bounds_fn = domain_bounds_fn;
  ropts.frame_bounds_fn = frame_bounds_fn;
  ropts.perf_stats = &g_perf;
  viewer.run(sys, orbit, render_cb, ropts);
  return EXIT_SUCCESS;
}
