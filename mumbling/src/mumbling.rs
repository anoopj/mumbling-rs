//! Mumbling v1 compressed bitmap.
//!
//! The serialized layout follows the Mumbling spec (all integers unsigned,
//! little-endian):
//!
//! * Header (6 bytes): version (1), cardinality (3), container count (2)
//! * Descriptor array: PFOR-encoded, one byte per container
//! * Containers: concatenated sparse (0-31 bytes) or dense (32 bytes)
//!
//! A container index (the Roaring "key") is `pos >> 8`; the position within a
//! container is the low 8 bits. Containers with fewer than 32 set positions are
//! sparse (a sorted list of position bytes); containers with 32 or more set
//! positions are dense (a 32-byte bitset, the MSB of byte 0 being position 0).
//!
//! The in-memory representation follows Roaring's design (see `~/src/roaring-rs`):
//! a vector of non-empty containers sorted by key, each an enum over a sparse
//! array or a dense bitset. [`MumblingReader`] is the zero-copy read view: it
//! decodes the descriptor array into an offset table once and then answers
//! `is_set`/iteration directly against the borrowed buffer, iterating dense
//! containers 64 bits at a time with `leading_zeros` (MSB-first).

use crate::pfor;

const VERSION: u8 = 1;
const HEADER_SIZE: usize = 6;
/// Descriptor MSB pattern `001` marks a dense (32-byte) container.
const DENSE_CONTAINER_DESCRIPTOR: u8 = 0b0010_0000;
/// A container with this many or more set bits is stored dense.
const DENSE_THRESHOLD: usize = 32;
/// Dense containers always occupy 32 bytes.
const DENSE_CONTAINER_SIZE: usize = 32;
const MAX_CONTAINERS: usize = 8192;
const MAX_CARDINALITY: usize = 2_097_152;

/// The payload of a single 256-bit container.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Store {
    /// Sorted set positions (0-255), fewer than 32 of them.
    Sparse(Vec<u8>),
    /// A 32-byte bitset in the spec's MSB-first layout: the MSB of byte 0 is
    /// position 0, the LSB of byte 31 is position 255. This is identical to the
    /// on-disk form, so decode and encode are plain copies.
    Dense(Box<[u8; DENSE_CONTAINER_SIZE]>),
}

impl Store {
    fn len(&self) -> usize {
        match self {
            Store::Sparse(positions) => positions.len(),
            Store::Dense(bytes) => bytes.iter().map(|b| b.count_ones() as usize).sum(),
        }
    }

    fn contains(&self, pos: u8) -> bool {
        match self {
            Store::Sparse(positions) => positions.binary_search(&pos).is_ok(),
            Store::Dense(bytes) => {
                let byte = bytes[(pos >> 3) as usize];
                let shift = 7 - (pos & 0b111);
                (byte >> shift) & 1 == 1
            }
        }
    }

    /// Inserts a position, promoting a sparse store to dense once it would hold
    /// 32 positions. Returns `true` if the position was newly set.
    fn insert(&mut self, pos: u8) -> bool {
        match self {
            Store::Sparse(positions) => match positions.binary_search(&pos) {
                Ok(_) => false,
                Err(index) => {
                    if positions.len() + 1 >= DENSE_THRESHOLD {
                        let mut dense = Box::new([0u8; DENSE_CONTAINER_SIZE]);
                        for &existing in positions.iter() {
                            set_dense_bit(&mut dense, existing);
                        }

                        set_dense_bit(&mut dense, pos);
                        *self = Store::Dense(dense);
                    } else {
                        positions.insert(index, pos);
                    }

                    true
                }
            },
            Store::Dense(bytes) => {
                let newly_set = !self_dense_contains(bytes, pos);
                set_dense_bit(bytes, pos);
                newly_set
            }
        }
    }
}

fn set_dense_bit(bytes: &mut [u8; DENSE_CONTAINER_SIZE], pos: u8) {
    let byte_index = (pos >> 3) as usize;
    let bit_shift = 7 - (pos & 0b111);
    bytes[byte_index] |= 1 << bit_shift;
}

fn self_dense_contains(bytes: &[u8; DENSE_CONTAINER_SIZE], pos: u8) -> bool {
    let byte = bytes[(pos >> 3) as usize];
    let shift = 7 - (pos & 0b111);
    (byte >> shift) & 1 == 1
}

/// A non-empty container and its key (the high bits of its positions).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Container {
    key: u16,
    store: Store,
}

/// A Mumbling bitmap over `u32` positions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MumblingBitmap {
    /// Non-empty containers sorted by key.
    containers: Vec<Container>,
}

impl MumblingBitmap {
    /// Creates an empty bitmap.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a position to the bitmap. Returns `true` if it was newly set.
    pub fn insert(&mut self, pos: u32) -> bool {
        let key = (pos >> 8) as u16;
        let index = self.containers.binary_search_by_key(&key, |c| c.key);
        let pos_in_container = (pos & 0xFF) as u8;
        match index {
            Ok(i) => self.containers[i].store.insert(pos_in_container),
            Err(i) => {
                self.containers.insert(
                    i,
                    Container {
                        key,
                        store: Store::Sparse(vec![pos_in_container]),
                    },
                );
                true
            }
        }
    }

    /// Returns `true` if `pos` is set.
    pub fn is_set(&self, pos: u32) -> bool {
        let key = (pos >> 8) as u16;
        match self.containers.binary_search_by_key(&key, |c| c.key) {
            Ok(i) => self.containers[i].store.contains((pos & 0xFF) as u8),
            Err(_) => false,
        }
    }

    /// Returns the number of bits set.
    pub fn cardinality(&self) -> usize {
        self.containers.iter().map(|c| c.store.len()).sum()
    }

    /// Returns `true` if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.containers.is_empty()
    }

    /// Iterates set positions in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.containers.iter().flat_map(|container| {
            let base = (container.key as u32) << 8;
            ContainerIter::new(&container.store).map(move |offset| base + offset as u32)
        })
    }

    /// Serializes the bitmap to the Mumbling v1 binary format.
    pub fn serialize(&self) -> Vec<u8> {
        let cardinality = self.cardinality();
        assert!(
            cardinality <= MAX_CARDINALITY,
            "Invalid cardinality (max {MAX_CARDINALITY}): {cardinality}"
        );

        let container_count = match self.containers.last() {
            Some(container) => container.key as usize + 1,
            None => 0,
        };
        assert!(
            container_count <= MAX_CONTAINERS,
            "Invalid container count (max {MAX_CONTAINERS}): {container_count}"
        );

        // Descriptor per container index, including 0 for empty gaps.
        let mut descriptors = vec![0u32; container_count];
        for container in &self.containers {
            descriptors[container.key as usize] = match &container.store {
                Store::Dense(_) => DENSE_CONTAINER_DESCRIPTOR as u32,
                Store::Sparse(positions) => positions.len() as u32,
            };
        }

        let mut out = Vec::with_capacity(HEADER_SIZE + container_count);

        // Header: version, cardinality (3 bytes LE), container count (2 bytes LE).
        out.push(VERSION);
        out.push((cardinality & 0xFF) as u8);
        out.push(((cardinality >> 8) & 0xFF) as u8);
        out.push(((cardinality >> 16) & 0xFF) as u8);
        out.push((container_count & 0xFF) as u8);
        out.push(((container_count >> 8) & 0xFF) as u8);

        // PFOR-encoded descriptor array.
        out.extend(pfor::encode(&descriptors));

        // Container payloads, in key order.
        for container in &self.containers {
            match &container.store {
                Store::Sparse(positions) => out.extend_from_slice(positions),
                Store::Dense(bytes) => out.extend_from_slice(bytes.as_slice()),
            }
        }

        out
    }

    /// Deserializes a bitmap from the Mumbling v1 binary format.
    pub fn deserialize(bytes: &[u8]) -> Self {
        let version = bytes[0];
        assert_eq!(
            version, VERSION,
            "Unsupported Mumbling bitmap version: {version}"
        );

        let container_count = (bytes[4] as usize) | ((bytes[5] as usize) << 8);

        let (descriptors, descriptor_bytes) =
            pfor::decode_len(&bytes[HEADER_SIZE..], container_count);
        let mut cursor = HEADER_SIZE + descriptor_bytes;

        let mut containers = Vec::new();
        for (index, &descriptor) in descriptors.iter().enumerate() {
            let descriptor = descriptor as u8;
            if is_dense(descriptor) {
                let mut dense = Box::new([0u8; DENSE_CONTAINER_SIZE]);
                dense.copy_from_slice(&bytes[cursor..cursor + DENSE_CONTAINER_SIZE]);
                cursor += DENSE_CONTAINER_SIZE;
                containers.push(Container {
                    key: index as u16,
                    store: Store::Dense(dense),
                });
            } else if descriptor > 0 {
                let len = descriptor as usize;
                let positions = bytes[cursor..cursor + len].to_vec();
                cursor += len;
                containers.push(Container {
                    key: index as u16,
                    store: Store::Sparse(positions),
                });
            }
            // descriptor == 0 is an empty container; nothing is stored for it.
        }

        Self { containers }
    }

    /// Estimated decoded in-memory footprint in bytes.
    ///
    /// Models the payload each container holds plus its 2-byte key: sparse
    /// containers cost one byte per set position, dense containers cost 32
    /// bytes. This mirrors the way the benchmark models Roaring's array/bitmap
    /// containers, without counting `Vec`/`Box` allocator overhead on either
    /// side.
    pub fn in_memory_bytes(&self) -> usize {
        self.containers
            .iter()
            .map(|c| {
                let payload = match &c.store {
                    Store::Sparse(positions) => positions.len(),
                    Store::Dense(_) => DENSE_CONTAINER_SIZE,
                };
                2 + payload
            })
            .sum()
    }
}

impl FromIterator<u32> for MumblingBitmap {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        // Sort and dedup once, then build containers in a single pass so no
        // container is repeatedly re-searched or promoted incrementally.
        let mut positions: Vec<u32> = iter.into_iter().collect();
        positions.sort_unstable();
        positions.dedup();

        let mut containers: Vec<Container> = Vec::new();
        let mut i = 0;
        while i < positions.len() {
            let key = (positions[i] >> 8) as u16;
            let mut j = i;
            while j < positions.len() && (positions[j] >> 8) as u16 == key {
                j += 1;
            }

            let group = &positions[i..j];
            let store = if group.len() >= DENSE_THRESHOLD {
                let mut dense = Box::new([0u8; DENSE_CONTAINER_SIZE]);
                for &pos in group {
                    set_dense_bit(&mut dense, (pos & 0xFF) as u8);
                }

                Store::Dense(dense)
            } else {
                Store::Sparse(group.iter().map(|&pos| (pos & 0xFF) as u8).collect())
            };

            containers.push(Container { key, store });
            i = j;
        }

        Self { containers }
    }
}

fn is_dense(descriptor: u8) -> bool {
    descriptor & DENSE_CONTAINER_DESCRIPTOR == DENSE_CONTAINER_DESCRIPTOR
}

/// Iterates the set positions (0-255) within a single container.
enum ContainerIter<'a> {
    Sparse(std::slice::Iter<'a, u8>),
    Dense {
        bytes: &'a [u8; DENSE_CONTAINER_SIZE],
        byte_index: usize,
        current: u8,
    },
}

impl<'a> ContainerIter<'a> {
    fn new(store: &'a Store) -> Self {
        match store {
            Store::Sparse(positions) => ContainerIter::Sparse(positions.iter()),
            Store::Dense(bytes) => ContainerIter::Dense {
                bytes,
                byte_index: 0,
                current: bytes[0],
            },
        }
    }
}

impl Iterator for ContainerIter<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        match self {
            ContainerIter::Sparse(iter) => iter.next().copied(),
            ContainerIter::Dense {
                bytes,
                byte_index,
                current,
            } => {
                loop {
                    if *current != 0 {
                        // The MSB is the lowest position, so scan from the top.
                        let bit = current.leading_zeros() as u8;
                        *current &= !(0x80 >> bit);
                        return Some((*byte_index as u8) * 8 + bit);
                    }

                    *byte_index += 1;
                    if *byte_index >= DENSE_CONTAINER_SIZE {
                        return None;
                    }

                    *current = bytes[*byte_index];
                }
            }
        }
    }
}

/// A read-only, zero-copy view over a serialized Mumbling bitmap.
///
/// The header is parsed eagerly, the PFOR descriptor array is decoded once into
/// an offset table, and `cardinality`/`is_set`/iteration then answer directly
/// against the borrowed buffer with no per-container allocation. Cardinality is
/// taken from the header rather than recomputed.
///
/// This is the form a reader on the manifest path would use: it borrows the
/// bytes it was handed and never copies container payloads.
#[derive(Debug, Clone)]
pub struct MumblingReader<'a> {
    /// The full serialized bitmap, positioned at the header.
    data: &'a [u8],
    cardinality: usize,
    /// Decoded descriptor byte per container index.
    descriptors: Vec<u8>,
    /// Absolute byte offset of each container's payload in `data`. Length is
    /// `descriptors.len() + 1`; the final entry is the end of the last
    /// container.
    offsets: Vec<usize>,
}

impl<'a> MumblingReader<'a> {
    /// Parses the header and descriptor array of a serialized Mumbling bitmap,
    /// borrowing `data` without copying container payloads.
    pub fn new(data: &'a [u8]) -> Self {
        let version = data[0];
        assert_eq!(
            version, VERSION,
            "Unsupported Mumbling bitmap version: {version}"
        );

        let cardinality =
            (data[1] as usize) | ((data[2] as usize) << 8) | ((data[3] as usize) << 16);
        let container_count = (data[4] as usize) | ((data[5] as usize) << 8);

        let (descriptor_ints, descriptor_bytes) =
            pfor::decode_len(&data[HEADER_SIZE..], container_count);
        let descriptors: Vec<u8> = descriptor_ints.iter().map(|&d| d as u8).collect();

        // Build the offset table: each container's payload starts after the
        // descriptor array, at the running sum of prior container lengths.
        let mut offsets = Vec::with_capacity(container_count + 1);
        let mut offset = HEADER_SIZE + descriptor_bytes;
        for &descriptor in &descriptors {
            offsets.push(offset);
            offset += container_len(descriptor);
        }
        offsets.push(offset);

        Self {
            data,
            cardinality,
            descriptors,
            offsets,
        }
    }

    /// Returns the number of bits set, read from the header.
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Returns `true` if no bits are set.
    pub fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Returns `true` if `pos` is set, without materializing any container.
    pub fn is_set(&self, pos: u32) -> bool {
        let container_index = (pos >> 8) as usize;
        if container_index >= self.descriptors.len() {
            return false;
        }

        let descriptor = self.descriptors[container_index];
        let start = self.offsets[container_index];
        let pos_in_container = (pos & 0xFF) as u8;

        if is_dense(descriptor) {
            let byte = self.data[start + (pos_in_container >> 3) as usize];
            let shift = 7 - (pos_in_container & 0b111);
            (byte >> shift) & 1 == 1
        } else {
            // Sparse: sorted position bytes; binary search the borrowed slice.
            let len = descriptor as usize;
            self.data[start..start + len]
                .binary_search(&pos_in_container)
                .is_ok()
        }
    }

    /// Iterates set positions in ascending order, borrowing the buffer.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.descriptors.len()).flat_map(move |index| {
            let base = (index as u32) << 8;
            let descriptor = self.descriptors[index];
            let start = self.offsets[index];
            let payload = &self.data[start..self.offsets[index + 1]];
            ReaderContainerIter::new(descriptor, payload).map(move |offset| base + offset as u32)
        })
    }

    /// Materializes an owning [`MumblingBitmap`] from this view.
    pub fn to_bitmap(&self) -> MumblingBitmap {
        self.iter().collect()
    }
}

/// Serialized byte length of a container with the given descriptor.
fn container_len(descriptor: u8) -> usize {
    if is_dense(descriptor) {
        DENSE_CONTAINER_SIZE
    } else {
        descriptor as usize
    }
}

/// Number of 64-bit words in a dense container (32 bytes).
const DENSE_WORDS: usize = DENSE_CONTAINER_SIZE / 8;

/// Iterates the set positions (0-255) within a container's raw payload slice.
///
/// The dense path processes 64 bits per step using big-endian words and
/// `leading_zeros`, mirroring Roaring's word-at-a-time iteration. Big-endian is
/// what makes the spec's MSB-first bit order (position 0 = MSB of byte 0) land
/// as the most significant bit of the word.
enum ReaderContainerIter<'a> {
    Sparse(std::slice::Iter<'a, u8>),
    Dense {
        payload: &'a [u8],
        word_index: usize,
        current: u64,
    },
}

impl<'a> ReaderContainerIter<'a> {
    fn new(descriptor: u8, payload: &'a [u8]) -> Self {
        if is_dense(descriptor) {
            ReaderContainerIter::Dense {
                payload,
                word_index: 0,
                current: u64::from_be_bytes(payload[0..8].try_into().unwrap()),
            }
        } else {
            ReaderContainerIter::Sparse(payload.iter())
        }
    }
}

impl Iterator for ReaderContainerIter<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        match self {
            ReaderContainerIter::Sparse(iter) => iter.next().copied(),
            ReaderContainerIter::Dense {
                payload,
                word_index,
                current,
            } => loop {
                if *current != 0 {
                    // MSB is the lowest position within the word.
                    let bit = current.leading_zeros();
                    *current &= !(1u64 << (63 - bit));
                    return Some((*word_index * 64 + bit as usize) as u8);
                }

                *word_index += 1;
                if *word_index >= DENSE_WORDS {
                    return None;
                }

                let start = *word_index * 8;
                *current = u64::from_be_bytes(payload[start..start + 8].try_into().unwrap());
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitmap(positions: &[u32]) -> MumblingBitmap {
        positions.iter().copied().collect()
    }

    #[test]
    fn empty_bitmap_round_trips() {
        let bitmap = MumblingBitmap::new();
        let bytes = bitmap.serialize();
        // Version + zero cardinality + zero container count.
        assert_eq!(bytes, vec![VERSION, 0, 0, 0, 0, 0]);
        let decoded = MumblingBitmap::deserialize(&bytes);
        assert_eq!(decoded, bitmap);
        assert_eq!(decoded.cardinality(), 0);
    }

    #[test]
    fn sparse_container_positions() {
        let bitmap = bitmap(&[0, 5, 100, 255]);
        assert_eq!(bitmap.cardinality(), 4);
        for pos in [0, 5, 100, 255] {
            assert!(bitmap.is_set(pos));
        }
        for pos in [1, 4, 6, 99, 101, 254, 256] {
            assert!(!bitmap.is_set(pos));
        }

        let decoded = MumblingBitmap::deserialize(&bitmap.serialize());
        assert_eq!(decoded, bitmap);
    }

    #[test]
    fn dense_container_when_threshold_reached() {
        // 32 positions in one container must serialize as dense (32 bytes).
        let positions: Vec<u32> = (0..32).map(|i| i * 8).collect();
        let bitmap = bitmap(&positions);
        let bytes = bitmap.serialize();

        let descriptor_len = pfor::encode(&[DENSE_CONTAINER_DESCRIPTOR as u32]).len();
        assert_eq!(
            bytes.len(),
            HEADER_SIZE + descriptor_len + DENSE_CONTAINER_SIZE
        );

        let decoded = MumblingBitmap::deserialize(&bytes);
        assert_eq!(decoded, bitmap);
        assert_eq!(decoded.cardinality(), 32);
    }

    #[test]
    fn sparse_stays_sparse_at_31() {
        let positions: Vec<u32> = (0..31).map(|i| i * 8).collect();
        let bitmap = bitmap(&positions);
        let bytes = bitmap.serialize();
        let descriptor_len = pfor::encode(&[31]).len();
        // 31 sparse payload bytes, not a 32-byte dense container.
        assert_eq!(bytes.len(), HEADER_SIZE + descriptor_len + 31);
        assert_eq!(MumblingBitmap::deserialize(&bytes), bitmap);
    }

    #[test]
    fn dense_full_container_iterates_in_order() {
        let positions: Vec<u32> = (0..256).collect();
        let bitmap = bitmap(&positions);
        assert_eq!(bitmap.cardinality(), 256);
        let decoded = MumblingBitmap::deserialize(&bitmap.serialize());
        assert_eq!(decoded, bitmap);
        assert_eq!(decoded.iter().collect::<Vec<_>>(), positions);
    }

    #[test]
    fn dense_iteration_matches_set_positions() {
        // Even positions in container 0: exercises the leading_zeros scan.
        let positions: Vec<u32> = (0..256).step_by(2).collect();
        let bitmap = bitmap(&positions);
        let decoded = MumblingBitmap::deserialize(&bitmap.serialize());
        assert_eq!(decoded.iter().collect::<Vec<_>>(), positions);
    }

    #[test]
    fn multiple_containers_with_empty_gap() {
        // Container 0 (pos 5), container 1 empty, container 2 (pos 10 -> 522).
        let bitmap = bitmap(&[5, 522]);
        assert_eq!(bitmap.cardinality(), 2);
        assert!(bitmap.is_set(5));
        assert!(bitmap.is_set(522));
        assert!(!bitmap.is_set(256));

        let bytes = bitmap.serialize();
        // Container count spans the gap: max key 2 -> count 3.
        assert_eq!((bytes[4] as usize) | ((bytes[5] as usize) << 8), 3);
        let decoded = MumblingBitmap::deserialize(&bytes);
        assert_eq!(decoded, bitmap);
        assert_eq!(decoded.iter().collect::<Vec<_>>(), vec![5, 522]);
    }

    #[test]
    fn mixed_sparse_and_dense() {
        let mut positions: Vec<u32> = (0..32).collect(); // dense container 0
        positions.push(257); // sparse container 1, pos 1
        let bitmap = bitmap(&positions);
        let decoded = MumblingBitmap::deserialize(&bitmap.serialize());
        assert_eq!(decoded, bitmap);
    }

    #[test]
    fn insert_builds_same_bitmap_as_from_iter() {
        let positions = [1u32, 5, 5, 300, 257, 70000, 70001, 42];
        let mut incremental = MumblingBitmap::new();
        for &pos in &positions {
            incremental.insert(pos);
        }

        let bulk: MumblingBitmap = positions.iter().copied().collect();
        assert_eq!(incremental, bulk);
    }

    #[test]
    fn insert_promotes_to_dense_at_32() {
        let mut bitmap = MumblingBitmap::new();
        for pos in 0..32 {
            assert!(bitmap.insert(pos));
        }

        // Re-inserting an existing position reports no change.
        assert!(!bitmap.insert(0));
        assert_eq!(bitmap.cardinality(), 32);
        let decoded = MumblingBitmap::deserialize(&bitmap.serialize());
        assert_eq!(decoded, bitmap);
    }

    #[test]
    fn round_trips_random_densities() {
        let mut rng = crate::TestRng::new(99);

        for _ in 0..300 {
            let universe = rng.range(1, 50_000);
            let count = rng.below(universe + 1);
            let positions: Vec<u32> = (0..count).map(|_| rng.below(universe)).collect();
            let bitmap: MumblingBitmap = positions.iter().copied().collect();

            let decoded = MumblingBitmap::deserialize(&bitmap.serialize());
            assert_eq!(
                decoded, bitmap,
                "round trip mismatch at universe {universe}"
            );

            // Iteration yields the exact set, sorted and deduped.
            let mut expected = positions;
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(decoded.iter().collect::<Vec<_>>(), expected);
        }
    }

    #[test]
    fn reader_matches_owning_bitmap() {
        let mut rng = crate::TestRng::new(1234);

        for _ in 0..300 {
            let universe = rng.range(1, 50_000);
            let count = rng.below(universe + 1);
            let bitmap: MumblingBitmap = (0..count).map(|_| rng.below(universe)).collect();
            let bytes = bitmap.serialize();

            let reader = MumblingReader::new(&bytes);
            assert_eq!(reader.cardinality(), bitmap.cardinality());
            assert_eq!(
                reader.iter().collect::<Vec<_>>(),
                bitmap.iter().collect::<Vec<_>>()
            );
            assert_eq!(reader.to_bitmap(), bitmap);
        }
    }

    #[test]
    fn reader_is_set_matches_owning_bitmap() {
        // Spans sparse, dense, and empty-gap containers.
        let mut positions: Vec<u32> = (0..40).collect(); // dense container 0
        positions.extend([300, 305, 511]); // sparse container 1
        positions.push(2000); // sparse container 7, after empty gaps
        let bitmap: MumblingBitmap = positions.iter().copied().collect();
        let bytes = bitmap.serialize();
        let reader = MumblingReader::new(&bytes);

        for pos in 0..2100u32 {
            assert_eq!(
                reader.is_set(pos),
                bitmap.is_set(pos),
                "is_set mismatch at {pos}"
            );
        }
    }

    #[test]
    fn reader_reads_empty_bitmap() {
        let bytes = MumblingBitmap::new().serialize();
        let reader = MumblingReader::new(&bytes);
        assert_eq!(reader.cardinality(), 0);
        assert!(reader.is_empty());
        assert!(!reader.is_set(0));
        assert_eq!(reader.iter().count(), 0);
    }
}
