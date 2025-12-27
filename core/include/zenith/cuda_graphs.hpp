// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// CUDA Graphs Implementation for Zenith
//
// Technical Foundation:
// - NVIDIA CUDA Graphs capture kernel execution sequences into replay-able
// graphs
// - Reduces CPU launch overhead from ~25us/kernel to ~5us/graph (5-10x
// reduction)
// - Referenced in: BLUEPRINT_ZENITH_3X.md, CetakBiru.md Section 4.2
//
// Mathematical Justification:
// Traditional: T = N * (T_cpu + T_kernel)
// With Graphs: T = T_graph_launch + N * T_kernel
// Speedup Factor: (N * T_cpu) / T_graph_launch for CPU-bound scenarios
//
// Architecture:
// - CudaGraph: RAII wrapper for cudaGraph_t
// - CudaGraphExec: RAII wrapper for cudaGraphExec_t (executable graph)
// - GraphCapture: Stream-capture mechanism for automatic graph creation
// - GraphCaptureSession: High-level session manager for capture workflow
//
// References:
// - NVIDIA CUDA Programming Guide: Graph Management APIs
// - TensorRT: Automatic CUDA Graph integration
// - PyTorch: torch.cuda.make_graphed_callables

#ifndef ZENITH_CUDA_GRAPHS_HPP
#define ZENITH_CUDA_GRAPHS_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ZENITH_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace zenith {
namespace cuda {

// ============================================================================
// Error Handling
// ============================================================================

class CudaGraphError : public std::runtime_error {
public:
  explicit CudaGraphError(const std::string &msg) : std::runtime_error(msg) {}
};

#ifdef ZENITH_HAS_CUDA
inline void check_cuda_graph_error(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw CudaGraphError(std::string(context) + ": " + cudaGetErrorString(err));
  }
}
#endif

// ============================================================================
// CudaGraph: RAII wrapper for cudaGraph_t
// ============================================================================

#ifdef ZENITH_HAS_CUDA
class CudaGraph {
public:
  CudaGraph() : graph_(nullptr), owns_graph_(false) {}

  explicit CudaGraph(cudaGraph_t graph, bool owns = true)
      : graph_(graph), owns_graph_(owns) {}

  ~CudaGraph() {
    if (graph_ && owns_graph_) {
      cudaGraphDestroy(graph_);
    }
  }

  // Non-copyable
  CudaGraph(const CudaGraph &) = delete;
  CudaGraph &operator=(const CudaGraph &) = delete;

  // Movable
  CudaGraph(CudaGraph &&other) noexcept
      : graph_(other.graph_), owns_graph_(other.owns_graph_),
        node_count_(other.node_count_) {
    other.graph_ = nullptr;
    other.owns_graph_ = false;
    other.node_count_ = 0;
  }

  CudaGraph &operator=(CudaGraph &&other) noexcept {
    if (this != &other) {
      if (graph_ && owns_graph_) {
        cudaGraphDestroy(graph_);
      }
      graph_ = other.graph_;
      owns_graph_ = other.owns_graph_;
      node_count_ = other.node_count_;
      other.graph_ = nullptr;
      other.owns_graph_ = false;
      other.node_count_ = 0;
    }
    return *this;
  }

  // Access native handle
  [[nodiscard]] cudaGraph_t get() const { return graph_; }
  [[nodiscard]] bool valid() const { return graph_ != nullptr; }
  [[nodiscard]] size_t node_count() const { return node_count_; }

  // Get graph statistics
  void update_node_count() {
    if (graph_) {
      size_t count;
      cudaGraphGetNodes(graph_, nullptr, &count);
      node_count_ = count;
    }
  }

private:
  cudaGraph_t graph_ = nullptr;
  bool owns_graph_ = false;
  size_t node_count_ = 0;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// CudaGraphExec: RAII wrapper for cudaGraphExec_t (executable graph)
// ============================================================================

#ifdef ZENITH_HAS_CUDA
class CudaGraphExec {
public:
  CudaGraphExec() : graph_exec_(nullptr), instantiated_(false) {}

  // Create executable graph from CudaGraph
  explicit CudaGraphExec(const CudaGraph &graph) : graph_exec_(nullptr) {
    if (!graph.valid()) {
      throw CudaGraphError("Cannot instantiate from invalid graph");
    }
    instantiate(graph);
  }

  ~CudaGraphExec() {
    if (graph_exec_) {
      cudaGraphExecDestroy(graph_exec_);
    }
  }

  // Non-copyable
  CudaGraphExec(const CudaGraphExec &) = delete;
  CudaGraphExec &operator=(const CudaGraphExec &) = delete;

  // Movable
  CudaGraphExec(CudaGraphExec &&other) noexcept
      : graph_exec_(other.graph_exec_), instantiated_(other.instantiated_) {
    other.graph_exec_ = nullptr;
    other.instantiated_ = false;
  }

  CudaGraphExec &operator=(CudaGraphExec &&other) noexcept {
    if (this != &other) {
      if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
      }
      graph_exec_ = other.graph_exec_;
      instantiated_ = other.instantiated_;
      other.graph_exec_ = nullptr;
      other.instantiated_ = false;
    }
    return *this;
  }

  // Instantiate from graph
  void instantiate(const CudaGraph &graph) {
    if (graph_exec_) {
      cudaGraphExecDestroy(graph_exec_);
      graph_exec_ = nullptr;
    }

    cudaGraphNode_t error_node;
    char log_buffer[1024];
    cudaError_t err = cudaGraphInstantiate(&graph_exec_, graph.get(),
                                           &error_node, log_buffer, 1024);
    if (err != cudaSuccess) {
      throw CudaGraphError(std::string("Graph instantiation failed: ") +
                           log_buffer);
    }
    instantiated_ = true;
  }

  // Launch graph on stream
  void launch(cudaStream_t stream = 0) {
    if (!instantiated_ || !graph_exec_) {
      throw CudaGraphError("Cannot launch uninstantiated graph");
    }
    cudaError_t err = cudaGraphLaunch(graph_exec_, stream);
    check_cuda_graph_error(err, "cudaGraphLaunch");
    ++launch_count_;
  }

  // Update graph with new graph (for dynamic updates without reinstantiation)
  bool try_update(const CudaGraph &new_graph) {
    if (!graph_exec_) {
      return false;
    }
    cudaGraphExecUpdateResult update_result;
    cudaGraphNode_t error_node;
    cudaError_t err = cudaGraphExecUpdate(graph_exec_, new_graph.get(),
                                          &error_node, &update_result);
    return (err == cudaSuccess && update_result == cudaGraphExecUpdateSuccess);
  }

  // Access
  [[nodiscard]] cudaGraphExec_t get() const { return graph_exec_; }
  [[nodiscard]] bool valid() const { return instantiated_ && graph_exec_; }
  [[nodiscard]] uint64_t launch_count() const { return launch_count_; }

private:
  cudaGraphExec_t graph_exec_ = nullptr;
  bool instantiated_ = false;
  uint64_t launch_count_ = 0;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// GraphCapture: Stream capture mechanism
// ============================================================================

#ifdef ZENITH_HAS_CUDA
enum class CaptureMode {
  // Global mode: All CUDA work on any stream is captured
  Global = cudaStreamCaptureModeGlobal,
  // Thread-local mode: Only work issued by capturing thread is captured
  ThreadLocal = cudaStreamCaptureModeThreadLocal,
  // Relaxed mode: Less strict ordering guarantees
  Relaxed = cudaStreamCaptureModeRelaxed
};

enum class CaptureStatus {
  None = cudaStreamCaptureStatusNone,
  Active = cudaStreamCaptureStatusActive,
  Invalidated = cudaStreamCaptureStatusInvalidated
};

class GraphCapture {
public:
  GraphCapture() : stream_(nullptr), capturing_(false) {}

  explicit GraphCapture(cudaStream_t stream)
      : stream_(stream), capturing_(false) {}

  ~GraphCapture() {
    if (capturing_) {
      // Attempt to end capture on destruction to prevent resource leaks
      try {
        end_capture();
      } catch (...) {
        // Suppress exceptions in destructor
      }
    }
  }

  // Non-copyable, non-movable during capture
  GraphCapture(const GraphCapture &) = delete;
  GraphCapture &operator=(const GraphCapture &) = delete;

  // Begin stream capture
  void begin_capture(CaptureMode mode = CaptureMode::Global) {
    if (capturing_) {
      throw CudaGraphError("Capture already in progress");
    }
    cudaError_t err = cudaStreamBeginCapture(
        stream_, static_cast<cudaStreamCaptureMode>(mode));
    check_cuda_graph_error(err, "cudaStreamBeginCapture");
    capturing_ = true;
  }

  // End capture and return graph
  [[nodiscard]] CudaGraph end_capture() {
    if (!capturing_) {
      throw CudaGraphError("No capture in progress");
    }
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(stream_, &graph);
    capturing_ = false;
    check_cuda_graph_error(err, "cudaStreamEndCapture");
    CudaGraph result(graph, true);
    result.update_node_count();
    return result;
  }

  // Check capture status
  [[nodiscard]] CaptureStatus status() const {
    cudaStreamCaptureStatus cuda_status;
    cudaStreamIsCapturing(stream_, &cuda_status);
    return static_cast<CaptureStatus>(cuda_status);
  }

  [[nodiscard]] bool is_capturing() const { return capturing_; }
  [[nodiscard]] cudaStream_t stream() const { return stream_; }

private:
  cudaStream_t stream_;
  bool capturing_;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// GraphCaptureSession: High-level session manager
// ============================================================================

#ifdef ZENITH_HAS_CUDA
class GraphCaptureSession {
public:
  // Create session with owned stream
  GraphCaptureSession() : owns_stream_(true) {
    cudaStreamCreate(&stream_);
    capture_ = std::make_unique<GraphCapture>(stream_);
  }

  // Create session with external stream
  explicit GraphCaptureSession(cudaStream_t stream)
      : stream_(stream), owns_stream_(false) {
    capture_ = std::make_unique<GraphCapture>(stream_);
  }

  ~GraphCaptureSession() {
    if (owns_stream_ && stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  // Non-copyable
  GraphCaptureSession(const GraphCaptureSession &) = delete;
  GraphCaptureSession &operator=(const GraphCaptureSession &) = delete;

  // Capture a function's CUDA operations
  template <typename Func>
  CudaGraph capture(Func &&func, CaptureMode mode = CaptureMode::Global) {
    // Begin capture
    capture_->begin_capture(mode);

    try {
      // Execute function - all CUDA calls are captured
      std::forward<Func>(func)();

      // Synchronize to ensure all work is captured
      cudaStreamSynchronize(stream_);

      // End capture and return graph
      return capture_->end_capture();
    } catch (...) {
      // Attempt to end capture cleanly
      if (capture_->is_capturing()) {
        try {
          capture_->end_capture();
        } catch (...) {
        }
      }
      throw;
    }
  }

  // Capture and immediately create executable graph
  template <typename Func>
  CudaGraphExec
  capture_and_instantiate(Func &&func, CaptureMode mode = CaptureMode::Global) {
    CudaGraph graph = capture(std::forward<Func>(func), mode);
    return CudaGraphExec(graph);
  }

  [[nodiscard]] cudaStream_t stream() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
  bool owns_stream_ = false;
  std::unique_ptr<GraphCapture> capture_;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// GraphCache: Cache for compiled graphs
// ============================================================================

#ifdef ZENITH_HAS_CUDA
class GraphCache {
public:
  using GraphKey = std::string;

  GraphCache() = default;

  // Insert graph into cache
  void insert(const GraphKey &key, CudaGraphExec &&graph_exec) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_[key] = std::move(graph_exec);
  }

  // Get graph from cache (returns nullptr if not found)
  CudaGraphExec *get(const GraphKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return &it->second;
    }
    return nullptr;
  }

  // Check if key exists
  bool contains(const GraphKey &key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.find(key) != cache_.end();
  }

  // Remove from cache
  void erase(const GraphKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.erase(key);
  }

  // Clear all cached graphs
  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
  }

  // Get cache statistics
  [[nodiscard]] size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
  }

  // Get total launch count across all cached graphs
  [[nodiscard]] uint64_t total_launches() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t total = 0;
    for (const auto &pair : cache_) {
      total += pair.second.launch_count();
    }
    return total;
  }

private:
  mutable std::mutex mutex_;
  std::unordered_map<GraphKey, CudaGraphExec> cache_;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// GraphRunner: High-level graph execution manager
// ============================================================================

#ifdef ZENITH_HAS_CUDA
class GraphRunner {
public:
  GraphRunner() : cache_(std::make_shared<GraphCache>()) {}

  explicit GraphRunner(std::shared_ptr<GraphCache> cache) : cache_(cache) {}

  // Execute function with graph caching
  // If graph not in cache, captures and caches it
  // If graph in cache, launches cached graph
  template <typename Func>
  void run_with_graph(const std::string &key, cudaStream_t stream, Func &&func,
                      CaptureMode mode = CaptureMode::Global) {
    // Check if graph is cached
    CudaGraphExec *cached = cache_->get(key);
    if (cached && cached->valid()) {
      // Launch cached graph
      cached->launch(stream);
      ++cache_hits_;
      return;
    }

    // Capture and cache
    GraphCaptureSession session(stream);
    CudaGraphExec graph_exec =
        session.capture_and_instantiate(std::forward<Func>(func), mode);
    cache_->insert(key, std::move(graph_exec));

    // Launch the newly cached graph
    CudaGraphExec *newly_cached = cache_->get(key);
    if (newly_cached) {
      newly_cached->launch(stream);
    }
    ++cache_misses_;
  }

  // Force recapture of a graph
  template <typename Func>
  void recapture(const std::string &key, cudaStream_t stream, Func &&func,
                 CaptureMode mode = CaptureMode::Global) {
    // Remove existing cached graph
    cache_->erase(key);
    // Recapture
    run_with_graph(key, stream, std::forward<Func>(func), mode);
  }

  // Get statistics
  [[nodiscard]] uint64_t cache_hits() const { return cache_hits_; }
  [[nodiscard]] uint64_t cache_misses() const { return cache_misses_; }
  [[nodiscard]] double hit_rate() const {
    uint64_t total = cache_hits_ + cache_misses_;
    return total > 0 ? static_cast<double>(cache_hits_) / total : 0.0;
  }

  // Access cache
  [[nodiscard]] std::shared_ptr<GraphCache> cache() const { return cache_; }

private:
  std::shared_ptr<GraphCache> cache_;
  uint64_t cache_hits_ = 0;
  uint64_t cache_misses_ = 0;
};
#endif // ZENITH_HAS_CUDA

// ============================================================================
// Global Graph Runner Instance
// ============================================================================

#ifdef ZENITH_HAS_CUDA
inline GraphRunner &global_graph_runner() {
  static GraphRunner runner;
  return runner;
}
#endif

// ============================================================================
// Helper Macros for Graph Capture
// ============================================================================

#ifdef ZENITH_HAS_CUDA
// Macro to simplify graph capture with lambda
#define ZENITH_CAPTURE_GRAPH(stream, key, ...)                                 \
  zenith::cuda::global_graph_runner().run_with_graph(key, stream,              \
                                                     [&]() { __VA_ARGS__; })

// Macro for static graph (captured once, never updated)
#define ZENITH_STATIC_GRAPH(stream, ...)                                       \
  do {                                                                         \
    static bool captured = false;                                              \
    static zenith::cuda::CudaGraphExec graph_exec;                             \
    if (!captured) {                                                           \
      zenith::cuda::GraphCaptureSession session(stream);                       \
      zenith::cuda::CudaGraph graph = session.capture([&]() { __VA_ARGS__; }); \
      graph_exec = zenith::cuda::CudaGraphExec(graph);                         \
      captured = true;                                                         \
    }                                                                          \
    graph_exec.launch(stream);                                                 \
  } while (0)
#endif // ZENITH_HAS_CUDA

} // namespace cuda
} // namespace zenith

#endif // ZENITH_CUDA_GRAPHS_HPP
