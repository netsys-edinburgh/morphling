#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <filesystem>

#include "checkpoint/checkpoint_handle.h"
#include "common/types_and_defs.h"
#include "morphling.grpc.pb.h"
#include "morphling.pb.h"
#include "utils/json_reader.h"
#include "utils/logger.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using morphling::LoadParamRequest;
using morphling::LoadParamResponse;

struct ParamMeta {
  uint32_t id;
  size_t size;
  size_t shm_offset;
  size_t file_offset;
};

// Register the struct with RTTR
RTTR_REGISTRATION {
  rttr::registration::class_<ParamMeta>("ParamMeta")
      .property("id", &ParamMeta::id)
      .property("size", &ParamMeta::size)
      .property("shm_offset", &ParamMeta::shm_offset)
      .property("file_offset", &ParamMeta::file_offset);
}

std::tuple<uint64_t, std::unordered_map<size_t, size_t>> compute_shm_offsets(
    const std::unordered_map<std::string, ParamMeta>& param_meta_map) {
  size_t shm_mem_size = 0;
  std::unordered_map<size_t, size_t> shm_mem_offsets;
  std::unordered_map<size_t, size_t> unique_sizes_counter;

  for (const auto& [param_name, param_meta] : param_meta_map) {
    if (unique_sizes_counter.find(param_meta.size) ==
        unique_sizes_counter.end()) {
      unique_sizes_counter[param_meta.size] = 0;
    }
    unique_sizes_counter[param_meta.size]++;
  }

  for (const auto& [size, count] : unique_sizes_counter) {
    shm_mem_size += size + 4 * count;
    shm_mem_offsets[size] = 0;
  }

  // map ordering is only guaranteed after keys are fixed
  size_t shm_offset = 0;
  for (const auto& [size, offset] : shm_mem_offsets) {
    shm_mem_offsets[size] = shm_offset;
    shm_offset += size + 4 * unique_sizes_counter[size];
  }

  return {shm_mem_size, shm_mem_offsets};
}

std::tuple<uint64_t, std::unordered_map<std::string, size_t>>
compute_pin_offsets(
    const std::unordered_map<std::string, ParamMeta>& param_meta_map) {
  std::unordered_map<std::string, size_t> pin_mem_offsets;
  size_t pin_mem_size = 0;
  size_t current_offset = 0;

  for (const auto& [param_name, param_meta] : param_meta_map) {
    pin_mem_offsets[param_name] = current_offset;
    current_offset += param_meta.size;
    pin_mem_size += param_meta.size;
  }

  return {pin_mem_size, pin_mem_offsets};
}

std::vector<uint32_t> find_ids_same_size(
    const std::unordered_map<std::string, ParamMeta>& param_meta_map,
    size_t size) {
  std::vector<uint32_t> ids;
  for (const auto& [param_name, param_meta] : param_meta_map) {
    if (param_meta.size == size) {
      ids.push_back(param_meta.id);
    }
  }
  return ids;
}

class MemoryManagerServer final : public morphling::MemoryManager::Service {
 public:
  MemoryManagerServer(const std::string& storage_path) {
    // manager_ = std::make_unique<MemoryManager>(storage_path);
  }
  Status LoadParamAsync(ServerContext* context, const LoadParamRequest* request,
                        LoadParamResponse* response) override {
    return Status::OK;
  }

  Status ConfirmParamLoaded(ServerContext* context,
                            const LoadParamRequest* request,
                            LoadParamResponse* response) override {
    return Status::OK;
  }

 private:
  // std::unique_ptr<MemoryManager> manager_;
};

void RunServer(const std::string& server_address,
               const std::string& storage_path) {
  MemoryManagerServer service(storage_path);

  grpc::EnableDefaultHealthCheckService(true);
  grpc::reflection::InitProtoReflectionServerBuilderPlugin();

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  LOG_INFO("Server listening on {}", server_address);
  server->Wait();
}

int main(int argc, char** argv) {
  // parse command line arguments using standard c fashion
  int opt;
  char* path = NULL;

  // getopt() is used to parse command-line options
  while ((opt = getopt(argc, argv, "n:h")) != -1) {
    switch (opt) {
      case 'p':
        path = optarg;  // Get the argument for the -n option
        break;
      case 'h':
        printf("Usage: %s [-n name] [-h]\n", argv[0]);
        exit(EXIT_SUCCESS);
      default:
        fprintf(stderr, "Usage: %s [-p checkpoint path] [-h]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }

  std::filesystem::path ckpt_path(path);
  LOG_FATAL_IF(std::filesystem::exists(ckpt_path),
               "Checkpoint path does not exist {}", ckpt_path.string());

  auto param_meta_map_file = ckpt_path / PARAM_META_FILE;
  auto json_reader = JsonReader<ParamMeta>(ckpt_path.string());
  auto param_meta_map = json_reader.ParseIntoMap();

  auto [shm_mem_size, shm_mem_offsets] = compute_shm_offsets(param_meta_map);

  // create shared memory
  int shm_fd = shm_open(PARAM_SHM_NAME, O_CREAT | O_RDWR, 0666);
  LOG_FATAL_IF(shm_fd >= 0, "Failed to create shared memory");

  void* shm_mem =
      mmap(NULL, shm_mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  LOG_FATAL_IF(shm_mem != MAP_FAILED, "Failed to map shared memory");

  std::unordered_map<size_t, size_t> unique_sizes_counter;
  for (const auto& [param_name, param_meta] : param_meta_map) {
    unique_sizes_counter[param_meta.size]++;
  }

  for (const auto& [size, count] : unique_sizes_counter) {
    auto ids_of_size = find_ids_same_size(param_meta_map, size);
    memcpy((char*)shm_mem + shm_mem_offsets[size], ids_of_size.data(),
           sizeof(uint32_t) * count);
    memcpy((char*)shm_mem + shm_mem_offsets[size] + size -
               count * sizeof(uint32_t),
           ids_of_size.data(), sizeof(uint32_t) * count);
  }

  auto [pin_mem_size, pin_mem_offsets] = compute_pin_offsets(param_meta_map);

  std::unordered_map<std::string, size_t> name_id_map;
  for (const auto& [param_name, param_meta] : param_meta_map) {
    name_id_map[param_name] = param_meta.id;
  }
  auto checkpoint_handle = CheckpointHandle(path);
  checkpoint_handle.ReadCheckpoint(pin_mem_offsets, name_id_map);

  return 0;
}