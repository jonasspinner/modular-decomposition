use std::fmt::{Debug, Formatter};

#[derive(Copy, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum MDNodeKind {
    Prime,
    Series,
    Parallel,
    Vertex(usize),
}

impl Debug for MDNodeKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            MDNodeKind::Prime => {
                write!(f, "Prime")
            }
            MDNodeKind::Series => {
                write!(f, "Series")
            }
            MDNodeKind::Parallel => {
                write!(f, "Parallel")
            }
            MDNodeKind::Vertex(v) => {
                write!(f, "{v}")
            }
        }
    }
}
