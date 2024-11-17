#include "backend/server_base.h"

int main() {
  std::vector<int> list = {2, 3, 4};
  auto result = CartesianProduct(list);
  for (auto& res : result) {
    for (auto& elem : res) {
      std::cout << elem << " ";
    }
    std::cout << std::endl;
  }

  torch::Tensor mat_a = torch::randn({2, 3, 4});
  torch::Tensor mat_b = torch::randn({2, 3, 4});

  std::cout << mat_a << std::endl;
  std::cout << mat_b << std::endl;

  auto partitions = PartitionMatrices(mat_a, mat_b, 2);
  assert(partitions.size() == 6);
  for (auto& partition : partitions) {
    fprintf(stderr, "row: %ld, col: %ld, ld: %ld\n", partition.row,
            partition.col, partition.ld);
    for (auto& mat : partition.mat) {
      auto [ptr, size] = mat;
      for (int i = 0; i < size / sizeof(float); ++i) {
        if (i % 4 == 0) {
          std::cout << std::endl;
        }
        std::cout << ((float*)ptr)[i] << " ";
      }
      std::cout << "size: " << size << std::endl;
    }
  }

  return 0;
}