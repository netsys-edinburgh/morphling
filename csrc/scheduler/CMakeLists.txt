
make_minimum_required(VERSION 3.23 FATAL_ERROR)

project(scheduler)

# get github project
include(FetchContent)
FetchContent_Declare(
  umpire
  GIT_REPOSITORY https://github.com/LLNL/Umpire.git
  GIT_TAG      v2024.07.0
)

FetchContent_MakeAvailable(umpire)
