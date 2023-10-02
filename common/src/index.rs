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
        )]
        $vis struct $name(u32);

        impl $name {
            #[inline(always)]
            fn new(x: usize) -> Self {
                debug_assert!(x < u32::MAX as usize);
                Self(x as u32)
            }

            #[inline(always)]
            fn index(&self) -> usize { self.0 as usize }

            #[inline(always)]
            fn invalid() -> Self { Self(u32::MAX) }

            #[inline(always)]
            fn is_valid(&self) -> bool { self.0 < u32::MAX }
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