macro_rules! make_index {
    ($vis:vis $name:ident) => {
        /// Index type.
        #[derive(
            Copy,
            Clone,
            Debug,
            Hash,
            Eq,
            PartialEq,
            Ord,
            PartialOrd,
        )]
        $vis struct $name(u32);

        impl $name {
            /// Create new index from `usize`.
            #[inline(always)]
            $vis fn new(x: usize) -> Self {
                debug_assert!(x < u32::MAX as usize);
                Self(x as u32)
            }

            /// Returns the index as `usize`.
            #[inline(always)]
            $vis fn index(&self) -> usize { self.0 as usize }

            /// Create the maximum possible index.
            #[inline(always)]
            $vis fn end() -> Self { Self(u32::MAX) }
        }

        impl ::std::default::Default for $name {
            #[inline(always)]
            fn default() -> Self {
                Self::end()
            }
        }

        impl ::std::convert::From<usize> for $name {
            #[inline(always)]
            fn from(x: usize) -> Self {
                Self::new(x)
            }
        }

        impl ::std::convert::From<u32> for $name {
            #[inline(always)]
            fn from(x: u32) -> Self {
                Self(x)
            }
        }

        impl ::std::convert::From<$name> for usize {
            #[inline(always)]
            fn from(x: $name) -> Self {
                x.index()
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

pub(crate) use make_index;

#[cfg(test)]
mod test {
    #[test]
    fn make_index() {
        make_index!(TestIndex);

        let idx = TestIndex::new(42);

        assert_eq!(idx.index(), 42);
        assert_eq!(TestIndex::end().index(), u32::MAX as usize);
        assert_eq!(TestIndex::default(), TestIndex::end());
        assert_eq!(TestIndex::from(42_u32), idx);
        assert_eq!(TestIndex::from(42_usize), idx);
        assert_eq!(usize::from(idx), 42_usize);
        assert_eq!(format!("{:?}", idx), "TestIndex(42)".to_string());
        assert_eq!(format!("{}", idx), "42".to_string());
    }
}
