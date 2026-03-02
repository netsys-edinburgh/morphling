#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "scheduler/green_context_runtime.h"
#include "scheduler/green_trace_parser.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// StreamRole string ↔ enum helpers
// ---------------------------------------------------------------------------

static StreamRole RoleFromString(const std::string& s) {
  if (s == "compute" || s == "comp") return StreamRole::kCompute;
  if (s == "recv") return StreamRole::kRecv;
  if (s == "send") return StreamRole::kSend;
  if (s == "dp") return StreamRole::kDp;
  throw py::value_error("Unknown StreamRole: '" + s +
                        "'. Expected: compute|comp|recv|send|dp");
}

static SwitchSyncMode SyncModeFromString(const std::string& s) {
  if (s == "none") return SwitchSyncMode::kNone;
  if (s == "event" || s == "event_chain") return SwitchSyncMode::kEventChain;
  if (s == "stream_sync") return SwitchSyncMode::kStreamSync;
  throw py::value_error("Unknown SwitchSyncMode: '" + s +
                        "'. Expected: none|event|event_chain|stream_sync");
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Green context runtime for dynamic SM partitioning";

  // ── StreamRole enum ────────────────────────────────────
  py::enum_<StreamRole>(m, "StreamRole")
      .value("compute", StreamRole::kCompute)
      .value("recv", StreamRole::kRecv)
      .value("send", StreamRole::kSend)
      .value("dp", StreamRole::kDp);

  // ── SwitchSyncMode enum ────────────────────────────────
  py::enum_<SwitchSyncMode>(m, "SwitchSyncMode")
      .value("none", SwitchSyncMode::kNone)
      .value("event_chain", SwitchSyncMode::kEventChain)
      .value("stream_sync", SwitchSyncMode::kStreamSync);

  // ── GreenContextRuntime::Options ───────────────────────
  py::class_<GreenContextRuntime::Options>(m, "GreenCtxOptions")
      .def(py::init<>())
      .def_readwrite("gpu_id", &GreenContextRuntime::Options::gpu_id)
      .def_readwrite("num_partitions",
                     &GreenContextRuntime::Options::num_partitions)
      .def_readwrite("partition_idx",
                     &GreenContextRuntime::Options::partition_idx)
      .def_readwrite("stream_priority",
                     &GreenContextRuntime::Options::stream_priority)
      .def_readwrite("strict", &GreenContextRuntime::Options::strict)
      .def_readwrite("switch_sync", &GreenContextRuntime::Options::switch_sync)
      .def_readwrite("roles", &GreenContextRuntime::Options::roles);

  // ── GreenContextRuntime ────────────────────────────────
  py::class_<GreenContextRuntime, std::shared_ptr<GreenContextRuntime>>(
      m, "GreenContextRuntime")
      .def(py::init<const GreenContextRuntime::Options&>(), py::arg("opts"))

      // Capability / introspection
      .def("is_supported", &GreenContextRuntime::IsSupported)
      .def("unsupported_reason", &GreenContextRuntime::UnsupportedReason)
      .def("sm_step", &GreenContextRuntime::SmStep)
      .def("partition_sm_count", &GreenContextRuntime::PartitionSmCount)
      .def("active_sm_count", &GreenContextRuntime::ActiveSmCount)
      .def("available_sm_counts", &GreenContextRuntime::AvailableSmCounts)
      .def("gpu_id", &GreenContextRuntime::GpuId)

      // Stream table
      .def(
          "get_stream_ptr",
          [](const GreenContextRuntime& self, int sm_count,
             const std::string& role) -> uintptr_t {
            return self.GetStreamPtr(sm_count, RoleFromString(role));
          },
          py::arg("sm_count"), py::arg("role"),
          "Get CUstream handle as int for torch.cuda.ExternalStream")
      .def(
          "get_stream_ptr",
          [](const GreenContextRuntime& self, int sm_count, StreamRole role)
              -> uintptr_t { return self.GetStreamPtr(sm_count, role); },
          py::arg("sm_count"), py::arg("role"),
          "Get CUstream handle as int (enum role)")

      // Control
      .def("set_active_sm_count", &GreenContextRuntime::SetActiveSmCount,
           py::arg("num_sms"))

      // Trace
      .def("load_trace", &GreenContextRuntime::LoadTrace, py::arg("path"))
      .def("sm_count_at_time", &GreenContextRuntime::SmCountAtTime,
           py::arg("elapsed_us"))
      .def("sm_count_at_step", &GreenContextRuntime::SmCountAtStep,
           py::arg("step"))

      // Activation (thread-level stream swap)
      .def(
          "activate_for_thread",
          [](GreenContextRuntime& self) -> int {
            py::gil_scoped_release release;
            return self.ActivateForThread();
          },
          "Activate current SM partition streams on calling thread")
      .def(
          "activate_sm_for_thread",
          [](GreenContextRuntime& self, int num_sms) -> int {
            py::gil_scoped_release release;
            return self.ActivateSmForThread(num_sms);
          },
          py::arg("num_sms"), "Activate specific SM count on calling thread")
      .def(
          "deactivate_for_thread",
          [](GreenContextRuntime& self, int prev_sm_count) {
            py::gil_scoped_release release;
            self.DeactivateForThread(prev_sm_count);
          },
          py::arg("prev_sm_count"),
          "Restore previous streams from activate call")

      // Lifecycle
      .def("close", &GreenContextRuntime::Close)

      // Stats
      .def("switch_count", &GreenContextRuntime::SwitchCount)
      .def("generation", &GreenContextRuntime::Generation);

  // ── Convenience factory ────────────────────────────────
  m.def(
      "create_runtime",
      [](int gpu_id, int num_partitions, int partition_idx,
         std::vector<std::string> role_names, int stream_priority, bool strict,
         const std::string& switch_sync)
          -> std::shared_ptr<GreenContextRuntime> {
        GreenContextRuntime::Options opts;
        opts.gpu_id = gpu_id;
        opts.num_partitions = num_partitions;
        opts.partition_idx = partition_idx;
        opts.stream_priority = stream_priority;
        opts.strict = strict;
        opts.switch_sync = SyncModeFromString(switch_sync);

        if (!role_names.empty()) {
          opts.roles.clear();
          for (const auto& name : role_names) {
            opts.roles.push_back(RoleFromString(name));
          }
        }

        return std::make_shared<GreenContextRuntime>(opts);
      },
      py::arg("gpu_id") = 0, py::arg("num_partitions") = 1,
      py::arg("partition_idx") = 0,
      py::arg("roles") =
          std::vector<std::string>{"compute", "recv", "send", "dp"},
      py::arg("stream_priority") = -1, py::arg("strict") = false,
      py::arg("switch_sync") = "event_chain",
      "Create a GreenContextRuntime with string-based configuration");
}
