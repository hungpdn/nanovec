# nanovec

![Go Version](https://img.shields.io/badge/go-1.23-blue)
![Build Status](https://github.com/hungpdn/nanovec/actions/workflows/go.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
[![Go Report Card](https://goreportcard.com/badge/github.com/hungpdn/nanovec)](https://goreportcard.com/report/github.com/hungpdn/nanovec)

**nanovec** is an embedded, serverless vector database for Go, built with the philosophy of SQLite. It runs in-process, requires zero configuration, and persists data to a single file.

## Features

* **Embedded:** No external server required. Links directly into your Go binary.
* **Fast:** Uses HNSW for approximate nearest neighbor search and SIMD-optimized math.
* **Memory Efficient:** Supports **SQ8 Quantization** to reduce RAM usage by 4x.
* **Reliable:** ACID-compliant storage (via bbolt) ensures data safety on crash.
* **Simple:** Pure Go, no CGO required (unless standard library dependencies change).

## Installation

```bash
go get github.com/hungpdn/nanovec
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    "github.com/hungpdn/nanovec"
)

func main() {
    // Open database
    cfg := nanovec.Config{
        Dimension: 3,
        IndexType: nanovec.IndexTypeHNSW,
    }
    db, _ := nanovec.Open("mydata.db", &cfg)
    defer db.Close()

    // Insert
    _ = db.Insert("vec1", []float32{1.0, 0.0, 0.0}, map[string]any{"tag": "A"})

    // Search
    results, _ := db.Search([]float32{1.0, 0.0, 0.0}, 1, nil)
    fmt.Printf("Found: %s\n", results[0].ID)
}

```

### 🚀 Instant Startup with Read-Only Mode (Mmap)

For large datasets (e.g., 10M+ vectors), loading the index into RAM can take minutes. Nanovec supports **Memory Mapping (mmap)** for `FlatIndex`, allowing **instant startup** (0ms load time) and OS-managed memory paging.

```go
cfg := nanovec.Config{
    Dimension: 128,
    IndexType: nanovec.IndexTypeFlat,
    ReadOnly:  true, // <--- Enable Zero-Copy Load
}

// Opens instantly, even with 100GB data!
db, _ := nanovec.Open("large_data.db", &cfg)
```

## Benchmarks

Benchmarks run on an Intel Core i7-8850H CPU @ 2.60GHz using a dataset of 128-dimensional vectors.

### 🚀 Search Performance (HNSW)

Latencies are measured on a dataset of 10,000 vectors.

| Operation | Latency (p99) | Throughput | Note |
| --- | --- | --- | --- |
| **HNSW Search** (k=10) | **0.65 ms** | ~1,900 QPS | Sub-millisecond search |

### 💾 Ingestion Speed (Write Throughput)

*Nanovec uses ACID storage (BoltDB). Using `InsertBatch` significantly reduces fsync overhead.*

| Method | Time per Item | Operations/Sec | Recommendation |
| --- | --- | --- | --- |
| **Batch Insert** (1k items) | **0.06 ms** | **~15,400 ops/s** | ✅ Highly Recommended |
| Sequential Insert | 45.9 ms | ~22 ops/s | ❌ Only for small updates |

### 🧠 Memory vs. Speed (Float32 vs. SQ8)

*Micro-benchmarks on Dot Product (1536 dimensions).*

| Index Type | SIMD Speed (ns/op) | RAM Usage | Best For |
| --- | --- | --- | --- |
| **Float32** | **575 ns** | 100% (6KB/vec) | Maximum Speed |
| **SQ8** | 1199 ns | **25% (1.5KB/vec)** | Large Datasets (Saves 75% RAM) |

## License

MIT License

## References

* [bbolt](https://github.com/etcd-io/bbolt)
* [HNSW Paper](https://arxiv.org/abs/1603.09320)
* [HNSW in Pinecone](https://www.pinecone.io/learn/series/faiss/hnsw/)
* [Faiss](https://github.com/facebookresearch/faiss)
