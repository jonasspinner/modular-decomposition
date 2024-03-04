# Modular Decomposition

This project implements several algorithm to compute the [modular decomposition](https://en.wikipedia.org/wiki/Modular_decomposition) of a graph.

+ `fracture`
  + A $O(n + m \log n)$ algorithm based on [[HPV99]](https://doi.org/10.1142/S0129054199000125) and [[CHM02]](https://doi.org/10.46298/dmtcs.298).
  + First sketch based on a Julia implementation [[Kar19]](https://github.com/StefanKarpinski/GraphModularDecomposition.jl), but now heavily modified.
+ `skeleton`
  + A $O(n + m \log n)$ algorithm based on [[MS00]](https://doi.org/10.46298/dmtcs.274).
+ `linear (ref)`
  + A $O(n + m)$ algorithm based on [[TCHP08]](https://doi.org/10.1007/978-3-540-70575-8_52).
  + A wrapper for the C++ implementation of [[Miz23]](https://github.com/mogproject/modular-decomposition).
+ `linear`
  + A $O(n + m)$ algorithm based on [[TCHP08]](https://doi.org/10.1007/978-3-540-70575-8_52).
  + A port of the C++ implementation of [[Miz23]](https://github.com/mogproject/modular-decomposition) to Rust.
  + Some additional modifications.

## Evaluation

![](evaluation.png)

This figure shows the algorithm performance for some datasets.
The time for multiple runs is averaged for each instance and algorithm. The time for each algorithm is divided by the best time and the distribution is plotted.
The `fracture` algorithm performs best for most instances.

## As a library

The crates implementing the algorithms provide a API for the modular decompostion.
```
pub fn modular_decomposition<N, E>(graph: &Graph<N, E, Undirected>) -> DiGraph<MDNodeKind, ()>
{
    prepare(graph).compute().finalize()
}
```

## References

+ [HPV99] Michel Habib, Christophe Paul, and Laurent Viennot. “Partition Refinement Techniques: An Interesting Algorithmic Tool Kit”. https://doi.org/10.1142/S0129054199000125.
+ [CHM02] Christian Capelle, Michel Habib, and Fabien Montgolfier. “Graph Decompositions and Factorizing Permutations”. https://doi.org/10.46298/dmtcs.298
+ [MS00] Ross M. Mcconnell and Jeremy P. Spinrad. “Ordered Vertex Partitioning”. https://doi.org/10.46298/dmtcs.274
+ [TCHP08] Marc Tedder, Derek Corneil, Michel Habib, and Christophe Paul. “Simpler Linear-Time Modular Decomposition Via Recursive Factorizing Permutations”. https://doi.org/10.1007/978-3-540-70575-8_52.
+ [Miz23] https://github.com/mogproject/modular-decomposition
+ [Kar19] https://github.com/StefanKarpinski/GraphModularDecomposition.jl
