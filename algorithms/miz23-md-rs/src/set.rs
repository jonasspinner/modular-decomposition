use std::iter::zip;

pub(crate) struct FastSet {
    data: Vec<u32>,
    generation: u32,
}

#[allow(unused)]
impl FastSet {
    pub fn new(size: usize) -> Self {
        Self { data: vec![u32::MAX; size], generation: 0 }
    }
    pub fn capacity(&self) -> usize { self.data.len() }
    pub fn clear(&mut self) {
        let (gen, overflow) = self.generation.overflowing_add(1);
        self.generation = gen;
        if overflow {
            self.data.fill(u32::MAX);
        }
    }
    pub fn resize(&mut self, size: usize) {
        self.generation = 0;
        self.data.clear();
        self.data.resize(size, u32::MAX);
    }
    pub fn set(&mut self, x: impl Into<usize>) { self.data[x.into()] = self.generation; }
    pub fn reset(&mut self, x: impl Into<usize>) { self.data[x.into()] = u32::MAX }
    pub fn get(&self, x: impl Into<usize>) -> bool { self.data[x.into()] == self.generation }

    pub fn items(&self) -> impl Iterator<Item=usize> + '_ {
        self.data.iter().enumerate().flat_map(|(i, &gen)| if gen == self.generation { Some(i) } else { None })
    }
}

impl PartialEq<Self> for FastSet {
    fn eq(&self, other: &Self) -> bool {
        if self.data.len() == other.data.len() {
            let lhs = self.data.iter().map(|&x| x == self.generation);
            let rhs = other.data.iter().map(|&x| x == other.generation);
            zip(lhs, rhs).all(|(a, b)| a == b)
        } else {
            false
        }
    }
}

impl Eq for FastSet {}