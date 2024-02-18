# Linear algorithm

A $O(n + m)$ algorithm based on [[TCHP08]](https://doi.org/10.1007/978-3-540-70575-8_52).

This is a port of [[Miz23]](https://github.com/mogproject/modular-decomposition) from C++ to Rust.
The original code is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) and
linked as a submodule in the `miz23-sys` crate.

The port adds iterators for a pre-order traversal of the node indices of a subtree and the leaf indices of a subtree.
Additionally, there are methods that iterate over a subtree or its leaves and call a user provided callback for each
node and provide mutable access. This avoids building a vector of node indices first, iterating over the indices and
then dropping the vector.
