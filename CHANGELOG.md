# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial implementation of Flat Index (Float32 & SQ8).
- Initial implementation of HNSW Index (Graph-based search).
- Persistence layer using bbolt (WAL support).
- Thread-safe `AddBatch` with fine-grained locking.
- SIMD-optimized Dot Product (via loop unrolling).
- `Vacuum` method for storage optimization.

### Fixed
- Rounding error in SQ8 quantization logic.