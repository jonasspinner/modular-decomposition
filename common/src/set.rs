pub trait Generation: Sized + Copy + Eq {
    const ZERO: Self;
    fn increment(self) -> Self;
}
macro_rules! generation {
    ($t:ty) => {
        impl Generation for $t {
            const ZERO: Self = 0;
            fn increment(self) -> Self {
                self.wrapping_add(1)
            }
        }
    };
}
generation!(u16);
generation!(u32);
generation!(u64);
generation!(usize);


type DefaultGeneration = u32;

pub struct FastResetBitSet<G = DefaultGeneration> {
    data: Vec<G>,
    generation: G,
}

impl<G: Generation> FastResetBitSet<G> {
    pub fn new(size: usize) -> Self {
        let generation = G::increment(G::ZERO);
        Self { data: vec![G::ZERO; size], generation }
    }
    pub fn capacity(&self) -> usize { self.data.len() }
    pub fn clear(&mut self) {
        self.generation = self.generation.increment();
        if self.generation == G::ZERO {
            self.generation = self.generation.increment();
            self.data.fill(G::ZERO);
        }
    }
    pub fn resize(&mut self, size: usize) {
        self.generation = G::increment(G::ZERO);
        self.data.clear();
        self.data.resize(size, G::ZERO);
    }
    pub fn set(&mut self, x: impl Into<usize>) { self.data[x.into()] = self.generation; }
    pub fn reset(&mut self, x: impl Into<usize>) { self.data[x.into()] = G::ZERO }
    pub fn get(&self, x: impl Into<usize>) -> bool { self.data[x.into()] == self.generation }
}
