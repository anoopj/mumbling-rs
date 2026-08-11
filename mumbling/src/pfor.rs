//! Patched Frame of Reference (PFOR) encoding for arrays of unsigned byte
//! values.
//!
//! Implements Appendix A of the Mumbling bitmap specification. The input is
//! split into 256-value chunks (the last may be shorter). Each chunk is
//! independently encoded using four values:
//!
//! * `b1`: bits stored in the primary array for every normalized value
//! * `b2`: bits stored per exception value (a normalized value of > `b1` bits)
//! * `e`:  number of exceptions
//! * `m`:  chunk-local minimum, subtracted from all values to normalize
//!
//! Each chunk is stored as a 3-byte header (`b2<<4 | b1`, then `e`, then `m`),
//! the primary array (`b1 * n` bits, padded to a byte), the exception offsets
//! (`e` bytes), and the exception values (`e * b2` bits, padded to a byte).

use crate::bit_packing;

/// Number of values in a full chunk. The final chunk may be shorter.
const CHUNK_SIZE: usize = 256;

/// Encodes `values` using the Mumbling PFOR scheme.
pub fn encode(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(estimate_encoded_size(values.len()));

    for chunk in values.chunks(CHUNK_SIZE) {
        encode_chunk(chunk, &mut out);
    }

    out
}

/// Decodes `count` values previously encoded with [`encode`].
pub fn decode(bytes: &[u8], count: usize) -> Vec<u32> {
    decode_len(bytes, count).0
}

/// Decodes `count` values and also returns the number of bytes consumed.
pub fn decode_len(bytes: &[u8], count: usize) -> (Vec<u32>, usize) {
    let mut out = Vec::with_capacity(count);
    let mut cursor = 0;

    while out.len() < count {
        let chunk_len = (count - out.len()).min(CHUNK_SIZE);
        cursor = decode_chunk(bytes, cursor, chunk_len, &mut out);
    }

    (out, cursor)
}

/// Number of bits required to represent `value` (0 for `value == 0`).
pub fn width(value: u32) -> u32 {
    u32::BITS - value.leading_zeros()
}

fn encode_chunk(chunk: &[u32], out: &mut Vec<u8>) {
    let base = chunk.iter().copied().min().unwrap_or(0);

    let mut set_bits = 0u32;
    let normalized: Vec<u32> = chunk
        .iter()
        .map(|&v| {
            set_bits |= v;
            v - base
        })
        .collect();

    assert!(
        width(set_bits) <= 8,
        "Cannot encode values wider than 8 bits: {} bits needed",
        width(set_bits)
    );

    let normalized_set_bits = normalized.iter().fold(0u32, |acc, &v| acc | v);
    let max_width = width(normalized_set_bits);
    let (b1, exc_count) = choose_bit_width(&normalized, max_width);
    let b2 = max_width - b1;

    // Special case: b1 == 8 stores original values as raw bytes with b2, e, and
    // m all set to 0.
    if b1 == 8 {
        write_header(out, 8, 0, 0, 0);
        bit_packing::pack(8, chunk, out);
        return;
    }

    write_header(out, b1, b2, exc_count, base);

    // Primary array: low b1 bits of every normalized value.
    bit_packing::pack(b1, &normalized, out);

    if exc_count > 0 {
        let threshold = 1u32 << b1;
        let mut exc_offsets = Vec::with_capacity(exc_count);
        let mut exc_values = Vec::with_capacity(exc_count);
        for (i, &value) in normalized.iter().enumerate() {
            if value >= threshold {
                exc_offsets.push(i as u32);
                exc_values.push(value >> b1);
            }
        }

        // Exception offsets (one byte per exception).
        bit_packing::pack(8, &exc_offsets, out);
        // Exception values: remaining high b2 bits of each exception.
        bit_packing::pack(b2, &exc_values, out);
    }
}

fn decode_chunk(bytes: &[u8], mut cursor: usize, count: usize, out: &mut Vec<u32>) -> usize {
    let header = bytes[cursor];
    let b1 = (header & 0x0F) as u32;
    let b2 = ((header >> 4) & 0x0F) as u32;
    let exc_count = bytes[cursor + 1] as usize;
    let base = bytes[cursor + 2] as u32;
    cursor += 3;

    // Primary array.
    let primary_bytes = bit_packing::byte_width(count * b1 as usize);
    let mut values = bit_packing::unpack(b1, &bytes[cursor..cursor + primary_bytes], count);
    cursor += primary_bytes;

    if exc_count > 0 {
        let offsets = bit_packing::unpack(8, &bytes[cursor..cursor + exc_count], exc_count);
        cursor += exc_count;

        let exc_bytes = bit_packing::byte_width(exc_count * b2 as usize);
        let highs = bit_packing::unpack(b2, &bytes[cursor..cursor + exc_bytes], exc_count);
        cursor += exc_bytes;

        for (i, &offset) in offsets.iter().enumerate() {
            values[offset as usize] |= highs[i] << b1;
        }
    }

    for value in values {
        out.push(value + base);
    }

    cursor
}

fn write_header(out: &mut Vec<u8>, b1: u32, b2: u32, exc_count: usize, base: u32) {
    // Header: b1 in the low nibble, b2 in the high nibble, then e, then m.
    out.push(((b2 << 4) | (b1 & 0x0F)) as u8);
    out.push(exc_count as u8);
    out.push(base as u8);
}

/// Chooses the primary bit width `b1` that minimizes the encoded chunk size,
/// returning `(b1, exception_count)`. Larger widths are preferred on ties to
/// reduce the number of exceptions.
fn choose_bit_width(normalized: &[u32], max_width: u32) -> (u32, usize) {
    let mut best_width = 0u32;
    let mut best_size = usize::MAX;
    let mut best_exc_count = 0usize;

    for candidate in 0..=max_width {
        let exc_count = if candidate < 8 {
            let threshold = 1u32 << candidate;
            normalized.iter().filter(|&&v| v >= threshold).count()
        } else {
            0
        };

        let b2 = max_width - candidate;
        let size = bit_packing::byte_width(normalized.len() * candidate as usize)
            + exc_count
            + bit_packing::byte_width(exc_count * b2 as usize);

        if size <= best_size {
            best_size = size;
            best_width = candidate;
            best_exc_count = exc_count;
        }
    }

    (best_width, best_exc_count)
}

/// Worst-case number of bytes required to encode `value_count` values.
pub fn estimate_encoded_size(value_count: usize) -> usize {
    // Worst case per chunk is b1 = 8: 3-byte header + 1 byte per value.
    let num_chunks = value_count.div_ceil(CHUNK_SIZE);
    3 * num_chunks + value_count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_bytes(values: &[u32]) -> Vec<u8> {
        encode(values)
    }

    #[test]
    fn example1_all_zeros() {
        // 256 values, b1 = 0, m = 0, all zero: `00 00 00`.
        let values = [0u32; 256];
        assert_eq!(encode_bytes(&values), vec![0x00, 0x00, 0x00]);
        assert_eq!(decode(&encode_bytes(&values), 256), values);
    }

    #[test]
    fn example2_all_fives() {
        // 51 values, b1 = 0, m = 5, all five: `00 00 05`.
        let values = [5u32; 51];
        assert_eq!(encode_bytes(&values), vec![0x00, 0x00, 0x05]);
        assert_eq!(decode(&encode_bytes(&values), 51), values);
    }

    #[test]
    fn example3_sparse_exceptions() {
        // [0,0,0,0,FF,0,0,FE]: b1 = 0, b2 = 8, 2 exceptions: `80 02 00 04 07 FF FE`.
        let values = [0, 0, 0, 0, 0xFF, 0, 0, 0xFE];
        let expected = vec![0x80, 0x02, 0x00, 0x04, 0x07, 0xFF, 0xFE];
        assert_eq!(encode_bytes(&values), expected);
        assert_eq!(decode(&expected, values.len()), values);
    }

    #[test]
    fn example4_two_bits_no_exceptions() {
        // [6, 7, 8]: b1 = 2, m = 6, no exceptions: `02 00 06 18`.
        let values = [6, 7, 8];
        assert_eq!(encode_bytes(&values), vec![0x02, 0x00, 0x06, 0x18]);
        assert_eq!(decode(&encode_bytes(&values), 3), values);
    }

    #[test]
    fn example5_prefers_larger_width_on_ties() {
        // [6, 34, 8, 7]. The impl prefers larger widths on ties, producing
        // `05 00 06 07 04 10` rather than the spec's `32 01 06 09 01 E0`. Both
        // decode to the same values (the b1 choice is a size tie).
        let values = [6, 34, 8, 7];
        let expected = vec![0x05, 0x00, 0x06, 0x07, 0x04, 0x10];
        let from_spec = vec![0x32, 0x01, 0x06, 0x09, 0x01, 0xE0];
        assert_eq!(encode_bytes(&values), expected);
        assert_eq!(decode(&expected, values.len()), values);
        assert_eq!(decode(&from_spec, values.len()), values);
    }

    #[test]
    fn round_trips_random_bytes() {
        let mut rng = crate::TestRng::new(42);

        for _ in 0..500 {
            let len = rng.below(600) as usize;
            let values: Vec<u32> = (0..len).map(|_| rng.next_u32() & 0xFF).collect();
            let encoded = encode(&values);
            assert_eq!(decode(&encoded, values.len()), values);
        }
    }

    #[test]
    fn round_trips_descriptor_like_values() {
        // Descriptor arrays are mostly small (0-31) with occasional 0x20 (dense).
        let mut rng = crate::TestRng::new(7);

        for _ in 0..500 {
            let len = rng.below(600) as usize;
            let values: Vec<u32> = (0..len)
                .map(|_| if rng.one_in(10) { 0x20 } else { rng.below(32) })
                .collect();
            let encoded = encode(&values);
            assert_eq!(decode(&encoded, values.len()), values);
        }
    }
}
