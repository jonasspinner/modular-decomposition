pub(crate) struct SegmentedStack<T> {
    values: Vec<T>,
    starts: Vec<(u32, u32)>,
    len: usize,
}

impl<T> SegmentedStack<T> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        let starts = Vec::with_capacity(128);
        Self { values: Vec::with_capacity(capacity), starts, len: 0 }
    }
    pub(crate) fn extend(&mut self, n: usize) {
        let start = self.values.len() as u32;
        if n > 0 {
            self.starts.push((start, n as u32));
        }
    }
    pub(crate) fn add(&mut self, value: T) {
        self.values.truncate(self.len);
        self.values.push(value);
        self.len += 1;
    }
    pub(crate) fn pop(&mut self) -> &[T] {
        let (i, last_count) = self.starts.last_mut().unwrap();
        self.len = *i as usize;
        *last_count -= 1;
        if *last_count == 0 {
            self.starts.pop().unwrap();
        }
        &self.values[self.len..]
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}
