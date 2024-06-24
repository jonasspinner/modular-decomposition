# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

+ `MDTree::strong_module_count()` to replace `MDTree::node_count()`
+ Tests for `make_index` macro
+ `MDTree::nodes(module)` function

### Changed

+ Docs

### Removed

+ `MDTree::node_count()` is replaced by `MDTree::strong_module_count()`

### Fixed

+ Deque push_back behavior after multiple grow/shrink cycles

## [0.2.0] - 2024-06-10

### Added

+ CI: check, test, fmt and clippy
+ CI: coverage reporting with Coveralls
+ `md_tree::ModuleIndex` is now `pub`
+ `md_tree::ModuleIndex` now has `new` and `index` methods to convert to and from `usize`
+ Changelog.md

### Changed

+ Docs
+ `ModuleKind` and `MDTree` are now generic over `NodeId`. The output type of `modular_decomposition` is now dependent
  on the `GraphBase::NodeId` type of the input graph.

### Removed

+ `md_tree::NodeIndex` is now `pub(crate)` instead of `pub`

## [0.1.0] - 2024-03-12

### Added

+ Initial release

[unreleased]: https://github.com/jonasspinner/modular-decomposition/compare/v0.2.0...HEAD

[0.2.0]: https://github.com/jonasspinner/modular-decomposition/compare/v0.1.0...v0.2.0

[0.1.0]: https://github.com/jonasspinner/modular-decomposition/releases/tag/v0.1.0
