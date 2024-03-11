use std::fmt::Debug;
use std::hash::Hash;

pub trait IndexType: From<u32> + From<usize> + Into<u32> + Into<usize> + Copy + Eq + Hash + Debug + Ord {
    fn new(x: usize) -> Self;
    fn index(&self) -> usize;
    fn invalid() -> Self;
    fn is_valid(&self) -> bool {
        self != &Self::invalid()
    }
}

#[macro_export]
macro_rules! make_index {
    ($vis:vis $name:ident) => {
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
            #[inline(always)]
            $vis fn new(x: usize) -> Self {
                debug_assert!(x < u32::MAX as usize);
                Self(x as u32)
            }

            #[inline(always)]
            $vis fn index(&self) -> usize { self.0 as usize }

            #[inline(always)]
            $vis fn invalid() -> Self { Self(u32::MAX) }

            #[inline(always)]
            $vis fn is_valid(&self) -> bool { self.0 < u32::MAX }
        }

        impl common::index::IndexType for $name {
            #[inline(always)]
            fn new(x: usize) -> Self { Self::new(x) }

            #[inline(always)]
            fn index(&self) -> usize { self.index() }

            #[inline(always)]
            fn invalid() -> Self { Self::invalid() }

            #[inline(always)]
            fn is_valid(&self) -> bool { self.is_valid() }
        }

        impl ::std::default::Default for $name {
            fn default() -> Self {
                Self::invalid()
            }
        }

        impl ::std::convert::From<usize> for $name {
            fn from(x: usize) -> Self {
                Self::new(x)
            }
        }

        impl ::std::convert::From<u32> for $name {
            fn from(x: u32) -> Self {
                Self(x)
            }
        }

        impl ::std::convert::From<$name> for usize {
            fn from(x: $name) -> Self {
                x.index()
            }
        }

        impl ::std::convert::From<$name> for u32 {
            fn from(x: $name) -> Self {
                x.0
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}
