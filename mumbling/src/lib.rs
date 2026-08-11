//! Mumbling v1 compressed bitmap format.
//!
//! Mumbling is a Roaring-derived bitmap format for small, bounded bitmaps such as
//! Iceberg deletion vectors. See `format/mumbling-spec.md` in apache/iceberg.

pub mod bit_packing;
pub mod mumbling;
pub mod pfor;

pub use mumbling::{MumblingBitmap, MumblingReader};

/// A tiny deterministic PRNG (SplitMix64) used only by unit tests, so the
/// library carries no dependencies — not even dev-dependencies.
#[cfg(test)]
pub(crate) struct TestRng {
    state: u64,
}

#[cfg(test)]
impl TestRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns the next 64-bit value (SplitMix64).
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    /// Uniform value in `[0, bound)`; `bound` must be non-zero.
    pub(crate) fn below(&mut self, bound: u32) -> u32 {
        self.next_u32() % bound
    }

    /// Uniform value in `[low, high)`; `high` must exceed `low`.
    pub(crate) fn range(&mut self, low: u32, high: u32) -> u32 {
        low + self.below(high - low)
    }

    /// Returns `true` with probability `1/n`.
    pub(crate) fn one_in(&mut self, n: u32) -> bool {
        self.below(n) == 0
    }
}
