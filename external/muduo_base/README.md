# Muduo Base Library (Vendored)

This directory contains a vendored subset of the muduo networking library
by Chen Shuo.

- Upstream: <https://github.com/chenshuo/muduo>
- License: BSD-3-Clause (see [`LICENSE`](LICENSE) in this directory)
- Local modifications: none beyond namespace re-rooting to keep the
  subset self-contained.

These files provide foundational C++ utilities (logging, threading,
file I/O, timestamps) used by the Morphling backend.

**Do not modify these files directly.** Bug fixes should be reported
upstream; if a local workaround is unavoidable, document the diff at
the top of the affected file.
