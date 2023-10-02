#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum MDNodeKind {
    Prime,
    Series,
    Parallel,
    Vertex(usize),
}
