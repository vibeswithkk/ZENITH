// Copyright 2025 Wahyu Ardiansyah
// Licensed under the Apache License, Version 2.0
//
// Full Model Benchmark Framework
// Berdasarkan CetakBiru.md: Pengujian Kinerja dan Regresi
// Referensi: MLPerf Inference, AI Benchmark, PyTorch Benchmark

#ifndef ZENITH_BENCHMARK_FRAMEWORK_HPP
#define ZENITH_BENCHMARK_FRAMEWORK_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef ZENITH_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace zenith {
namespace benchmark {

// ============================================================================
// Timer Utilities
// ============================================================================

/// High-resolution timer
class Timer {
public:
  void start() { start_ = std::chrono::high_resolution_clock::now(); }

  void stop() { end_ = std::chrono::high_resolution_clock::now(); }

  /// Get elapsed time in milliseconds
  [[nodiscard]] double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(end_ - start_).count();
  }

  /// Get elapsed time in microseconds
  [[nodiscard]] double elapsed_us() const {
    return std::chrono::duration<double, std::micro>(end_ - start_).count();
  }

  /// Get elapsed time in seconds
  [[nodiscard]] double elapsed_s() const {
    return std::chrono::duration<double>(end_ - start_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_;
  std::chrono::high_resolution_clock::time_point end_;
};

#ifdef ZENITH_HAS_CUDA
/// CUDA event-based timer for accurate GPU timing
class CudaTimer {
public:
  CudaTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&end_);
  }

  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(end_);
  }

  void start(cudaStream_t stream = 0) { cudaEventRecord(start_, stream); }

  void stop(cudaStream_t stream = 0) {
    cudaEventRecord(end_, stream);
    cudaEventSynchronize(end_);
  }

  /// Get elapsed time in milliseconds
  [[nodiscard]] float elapsed_ms() const {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, end_);
    return ms;
  }

private:
  cudaEvent_t start_;
  cudaEvent_t end_;
};
#endif

// ============================================================================
// Latency Statistics
// ============================================================================

/// Statistik latensi
struct LatencyStats {
  double mean_ms = 0.0;
  double std_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double p50_ms = 0.0; // Median
  double p90_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  size_t num_samples = 0;

  /// Compute stats from samples
  static LatencyStats compute(std::vector<double> samples) {
    LatencyStats stats;
    if (samples.empty())
      return stats;

    stats.num_samples = samples.size();

    // Sort for percentiles
    std::sort(samples.begin(), samples.end());

    stats.min_ms = samples.front();
    stats.max_ms = samples.back();

    // Mean
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean_ms = sum / samples.size();

    // Std
    double sq_sum = 0.0;
    for (double s : samples) {
      sq_sum += (s - stats.mean_ms) * (s - stats.mean_ms);
    }
    stats.std_ms = std::sqrt(sq_sum / samples.size());

    // Percentiles
    auto percentile = [&](double p) {
      size_t idx = static_cast<size_t>(p * (samples.size() - 1));
      return samples[idx];
    };

    stats.p50_ms = percentile(0.50);
    stats.p90_ms = percentile(0.90);
    stats.p95_ms = percentile(0.95);
    stats.p99_ms = percentile(0.99);

    return stats;
  }

  /// Get summary string
  [[nodiscard]] std::string summary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "Latency: mean=" << mean_ms << "ms, std=" << std_ms << "ms, ";
    ss << "min=" << min_ms << "ms, max=" << max_ms << "ms, ";
    ss << "p50=" << p50_ms << "ms, p90=" << p90_ms << "ms, ";
    ss << "p95=" << p95_ms << "ms, p99=" << p99_ms << "ms ";
    ss << "(n=" << num_samples << ")";
    return ss.str();
  }
};

/// Statistik throughput
struct ThroughputStats {
  double samples_per_sec = 0.0;
  double batches_per_sec = 0.0;
  int batch_size = 1;

  [[nodiscard]] std::string summary() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Throughput: " << samples_per_sec << " samples/sec, ";
    ss << batches_per_sec << " batches/sec (batch_size=" << batch_size << ")";
    return ss.str();
  }
};

// ============================================================================
// Benchmark Result
// ============================================================================

/// Hasil benchmark lengkap
struct BenchmarkResult {
  std::string model_name;
  std::string framework;
  std::string device;
  std::string precision;

  LatencyStats latency;
  ThroughputStats throughput;

  size_t memory_bytes = 0;
  size_t model_params = 0;
  size_t flops = 0;

  bool success = false;
  std::string error_message;

  /// Get full report
  [[nodiscard]] std::string report() const {
    std::ostringstream ss;
    ss << "========================================\n";
    ss << "Benchmark Result: " << model_name << "\n";
    ss << "========================================\n";
    ss << "Framework: " << framework << "\n";
    ss << "Device: " << device << "\n";
    ss << "Precision: " << precision << "\n";
    ss << "Status: " << (success ? "SUCCESS" : "FAILED") << "\n";

    if (!success) {
      ss << "Error: " << error_message << "\n";
      return ss.str();
    }

    ss << "\n";
    ss << latency.summary() << "\n";
    ss << throughput.summary() << "\n";

    if (memory_bytes > 0) {
      ss << "Memory: " << (memory_bytes / (1024.0 * 1024.0)) << " MB\n";
    }
    if (model_params > 0) {
      ss << "Parameters: " << (model_params / 1e6) << "M\n";
    }
    if (flops > 0) {
      ss << "FLOPs: " << (flops / 1e9) << "G\n";
    }

    return ss.str();
  }
};

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Konfigurasi untuk benchmark
struct BenchmarkConfig {
  /// Jumlah warmup iterations
  int warmup_iterations = 10;

  /// Jumlah benchmark iterations
  int benchmark_iterations = 100;

  /// Batch size
  int batch_size = 1;

  /// Sequence length (untuk NLP models)
  int seq_length = 512;

  /// Image size (untuk vision models)
  int image_size = 224;

  /// Input channels
  int input_channels = 3;

  /// Number of classes
  int num_classes = 1000;

  /// Hidden size (untuk transformer)
  int hidden_size = 768;

  /// Number of layers
  int num_layers = 12;

  /// Number of attention heads
  int num_heads = 12;

  /// Precision mode
  std::string precision = "fp32";

  /// Device
  std::string device = "cuda";
};

// ============================================================================
// Model Specification
// ============================================================================

/// Spesifikasi model untuk benchmark
struct ModelSpec {
  std::string name;
  std::string category; // "vision", "nlp", "audio"

  // Input shape
  std::vector<int> input_shape;

  // Model properties
  size_t num_params = 0;
  size_t flops = 0;

  /// Common model specs
  static ModelSpec ResNet50() {
    ModelSpec spec;
    spec.name = "ResNet-50";
    spec.category = "vision";
    spec.input_shape = {1, 3, 224, 224};
    spec.num_params = 25600000; // ~25.6M
    spec.flops = 4100000000;    // ~4.1 GFLOPs
    return spec;
  }

  static ModelSpec ResNet101() {
    ModelSpec spec;
    spec.name = "ResNet-101";
    spec.category = "vision";
    spec.input_shape = {1, 3, 224, 224};
    spec.num_params = 44500000; // ~44.5M
    spec.flops = 7800000000;    // ~7.8 GFLOPs
    return spec;
  }

  static ModelSpec BERT_Base() {
    ModelSpec spec;
    spec.name = "BERT-Base";
    spec.category = "nlp";
    spec.input_shape = {1, 512};
    spec.num_params = 110000000; // ~110M
    spec.flops = 22000000000;    // ~22 GFLOPs (per token)
    return spec;
  }

  static ModelSpec BERT_Large() {
    ModelSpec spec;
    spec.name = "BERT-Large";
    spec.category = "nlp";
    spec.input_shape = {1, 512};
    spec.num_params = 340000000; // ~340M
    spec.flops = 65000000000;    // ~65 GFLOPs
    return spec;
  }

  static ModelSpec GPT2() {
    ModelSpec spec;
    spec.name = "GPT-2";
    spec.category = "nlp";
    spec.input_shape = {1, 1024};
    spec.num_params = 124000000; // ~124M
    spec.flops = 35000000000;    // ~35 GFLOPs
    return spec;
  }

  static ModelSpec VisionTransformer_Base() {
    ModelSpec spec;
    spec.name = "ViT-Base";
    spec.category = "vision";
    spec.input_shape = {1, 3, 224, 224};
    spec.num_params = 86000000; // ~86M
    spec.flops = 17000000000;   // ~17 GFLOPs
    return spec;
  }
};

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Runner untuk menjalankan benchmarks
class BenchmarkRunner {
public:
  using BenchmarkFn = std::function<void()>;

  explicit BenchmarkRunner(BenchmarkConfig config = {})
      : config_(std::move(config)) {}

  /// Run benchmark dengan fungsi yang diberikan
  BenchmarkResult run(const std::string &name, BenchmarkFn fn) {
    BenchmarkResult result;
    result.model_name = name;
    result.framework = "Zenith";
    result.device = config_.device;
    result.precision = config_.precision;

    try {
      // Warmup
      for (int i = 0; i < config_.warmup_iterations; ++i) {
        fn();
      }

#ifdef ZENITH_HAS_CUDA
      cudaDeviceSynchronize();
#endif

      // Benchmark
      std::vector<double> latencies;
      latencies.reserve(config_.benchmark_iterations);

      for (int i = 0; i < config_.benchmark_iterations; ++i) {
        Timer timer;
        timer.start();
        fn();
        timer.stop();

#ifdef ZENITH_HAS_CUDA
        cudaDeviceSynchronize();
#endif

        latencies.push_back(timer.elapsed_ms());
      }

      // Compute stats
      result.latency = LatencyStats::compute(latencies);

      // Throughput
      result.throughput.batch_size = config_.batch_size;
      result.throughput.batches_per_sec = 1000.0 / result.latency.mean_ms;
      result.throughput.samples_per_sec =
          result.throughput.batches_per_sec * config_.batch_size;

      result.success = true;

    } catch (const std::exception &e) {
      result.success = false;
      result.error_message = e.what();
    }

    return result;
  }

#ifdef ZENITH_HAS_CUDA
  /// Run benchmark dengan CUDA events untuk timing akurat
  BenchmarkResult run_cuda(const std::string &name, BenchmarkFn fn,
                           cudaStream_t stream = 0) {
    BenchmarkResult result;
    result.model_name = name;
    result.framework = "Zenith";
    result.device = config_.device;
    result.precision = config_.precision;

    try {
      // Warmup
      for (int i = 0; i < config_.warmup_iterations; ++i) {
        fn();
      }
      cudaDeviceSynchronize();

      // Benchmark with CUDA events
      std::vector<double> latencies;
      latencies.reserve(config_.benchmark_iterations);

      CudaTimer timer;
      for (int i = 0; i < config_.benchmark_iterations; ++i) {
        timer.start(stream);
        fn();
        timer.stop(stream);
        latencies.push_back(timer.elapsed_ms());
      }

      result.latency = LatencyStats::compute(latencies);

      result.throughput.batch_size = config_.batch_size;
      result.throughput.batches_per_sec = 1000.0 / result.latency.mean_ms;
      result.throughput.samples_per_sec =
          result.throughput.batches_per_sec * config_.batch_size;

      result.success = true;

    } catch (const std::exception &e) {
      result.success = false;
      result.error_message = e.what();
    }

    return result;
  }
#endif

  /// Set model metadata
  void set_model_info(BenchmarkResult &result, const ModelSpec &spec) {
    result.model_params = spec.num_params;
    result.flops = spec.flops;
  }

private:
  BenchmarkConfig config_;
};

// ============================================================================
// Benchmark Suite
// ============================================================================

/// Suite untuk menjalankan multiple benchmarks
class BenchmarkSuite {
public:
  void add_result(BenchmarkResult result) {
    results_.push_back(std::move(result));
  }

  /// Get full report
  [[nodiscard]] std::string report() const {
    std::ostringstream ss;
    ss << "╔══════════════════════════════════════════════════════════════════╗"
          "\n";
    ss << "║                    ZENITH BENCHMARK SUITE                        "
          "║\n";
    ss << "╠══════════════════════════════════════════════════════════════════╣"
          "\n";

    for (const auto &result : results_) {
      ss << result.report();
      ss << "\n";
    }

    ss << "╚══════════════════════════════════════════════════════════════════╝"
          "\n";
    return ss.str();
  }

  /// Get comparison table
  [[nodiscard]] std::string comparison_table() const {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);

    // Header
    ss << "\n";
    ss << "+----------------------+----------+----------+----------+----------+"
          "\n";
    ss << "| Model                | Mean(ms) | P50(ms)  | P99(ms)  | Thrpt/s  "
          "|\n";
    ss << "+----------------------+----------+----------+----------+----------+"
          "\n";

    for (const auto &r : results_) {
      ss << "| " << std::left << std::setw(20) << r.model_name.substr(0, 20)
         << " | ";
      ss << std::right << std::setw(8) << r.latency.mean_ms << " | ";
      ss << std::setw(8) << r.latency.p50_ms << " | ";
      ss << std::setw(8) << r.latency.p99_ms << " | ";
      ss << std::setw(8) << r.throughput.samples_per_sec << " |\n";
    }

    ss << "+----------------------+----------+----------+----------+----------+"
          "\n";
    return ss.str();
  }

  /// Get results
  [[nodiscard]] const std::vector<BenchmarkResult> &results() const {
    return results_;
  }

  /// Clear results
  void clear() { results_.clear(); }

private:
  std::vector<BenchmarkResult> results_;
};

} // namespace benchmark
} // namespace zenith

#endif // ZENITH_BENCHMARK_FRAMEWORK_HPP
