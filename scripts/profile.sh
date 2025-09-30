sudo perf record -e cycles -e sched:sched_switch --switch-events \
    --sample-cpu -g \
    -m 8M --aio \
    --call-graph dwarf \
    ./build/temp.linux-x86_64-cpython-39/bin/tests/echo_server_test
