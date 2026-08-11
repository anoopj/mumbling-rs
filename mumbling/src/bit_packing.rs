//! Bit packing and unpacking for values of 0-8 bits.
//!
//! The least-significant bits of each value are packed with the first value
//! occupying the most significant bits of the output; output is padded to the
//! next byte boundary with zeros. For example, packing `[0b11, 0b10, 0b01]` at
//! width 2 produces `0b11100100`.
//!
//! Each width has a specialized routine that processes eight values per group
//! (eight values of `width` bits occupy exactly `width` bytes, so groups never
//! straddle a byte boundary). Rather than hand-writing one routine per width,
//! the group logic is generic over a `const W` so the compiler specializes it —
//! every shift and mask becomes a compile-time constant. Output is
//! byte-identical to a straightforward streaming packer.

/// Packs the low `width` bits of each value, MSB-first, padding the final byte
/// with zeros. A width of 0 emits nothing; a width of 8 copies raw bytes.
pub fn pack(width: u32, values: &[u32], out: &mut Vec<u8>) {
    match width {
        0 => {}
        1 => pack_width::<1>(values, out),
        2 => pack_width::<2>(values, out),
        3 => pack_width::<3>(values, out),
        4 => pack_width::<4>(values, out),
        5 => pack_width::<5>(values, out),
        6 => pack_width::<6>(values, out),
        7 => pack_width::<7>(values, out),
        8 => out.extend(values.iter().map(|&v| v as u8)),
        _ => panic!("Invalid bit width: {width}"),
    }
}

/// Unpacks `count` values of `width` bits each from `bytes`. Unused bits in the
/// final byte are ignored.
pub fn unpack(width: u32, bytes: &[u8], count: usize) -> Vec<u32> {
    let mut values = Vec::with_capacity(count);
    match width {
        0 => values.resize(count, 0),
        1 => unpack_width::<1>(bytes, count, &mut values),
        2 => unpack_width::<2>(bytes, count, &mut values),
        3 => unpack_width::<3>(bytes, count, &mut values),
        4 => unpack_width::<4>(bytes, count, &mut values),
        5 => unpack_width::<5>(bytes, count, &mut values),
        6 => unpack_width::<6>(bytes, count, &mut values),
        7 => unpack_width::<7>(bytes, count, &mut values),
        8 => values.extend(bytes.iter().take(count).map(|&b| b as u32)),
        _ => panic!("Invalid bit width: {width}"),
    }

    values
}

/// Packs values at a compile-time-known width `W` (1-7), eight values per group
/// into exactly `W` bytes. The first value of each group occupies the most
/// significant bits.
fn pack_width<const W: usize>(values: &[u32], out: &mut Vec<u8>) {
    const GROUP_BITS_PER_BYTE: usize = 8;
    let total_bits = GROUP_BITS_PER_BYTE * W; // bits held by a full 8-value group
    let mask = (1u64 << W) - 1;

    let mut chunks = values.chunks_exact(8);
    for group in &mut chunks {
        let mut word: u64 = 0;
        for (i, &value) in group.iter().enumerate() {
            word |= (value as u64 & mask) << (total_bits - (i + 1) * W);
        }

        for k in 0..W {
            out.push((word >> (total_bits - 8 - 8 * k)) as u8);
        }
    }

    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut word: u64 = 0;
        for (i, &value) in rem.iter().enumerate() {
            word |= (value as u64 & mask) << (total_bits - (i + 1) * W);
        }

        // The remaining values occupy the top bits; emit only the bytes they need.
        let byte_count = (rem.len() * W).div_ceil(8);
        for k in 0..byte_count {
            out.push((word >> (total_bits - 8 - 8 * k)) as u8);
        }
    }
}

/// Unpacks `count` values at a compile-time-known width `W` (1-7). The inverse
/// of [`pack_width`].
fn unpack_width<const W: usize>(bytes: &[u8], count: usize, out: &mut Vec<u32>) {
    let total_bits = 8 * W;
    let mask = (1u64 << W) - 1;

    let full_groups = count / 8;
    let mut pos = 0;
    for _ in 0..full_groups {
        let mut word: u64 = 0;
        for k in 0..W {
            word |= (bytes[pos + k] as u64) << (total_bits - 8 - 8 * k);
        }

        pos += W;
        for i in 0..8 {
            out.push(((word >> (total_bits - (i + 1) * W)) & mask) as u32);
        }
    }

    let rem = count - full_groups * 8;
    if rem > 0 {
        let byte_count = (rem * W).div_ceil(8);
        let mut word: u64 = 0;
        for k in 0..byte_count {
            word |= (bytes[pos + k] as u64) << (total_bits - 8 - 8 * k);
        }

        for i in 0..rem {
            out.push(((word >> (total_bits - (i + 1) * W)) & mask) as u32);
        }
    }
}

/// Number of bytes needed to hold `bits` bits.
pub fn byte_width(bits: usize) -> usize {
    bits.div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_two_bit_example() {
        // Spec: [3, 2, 1, 2, 3] at width 2 packs to E6 C0.
        let mut out = Vec::new();
        pack(2, &[3, 2, 1, 2, 3], &mut out);
        assert_eq!(out, vec![0xE6, 0xC0]);
    }

    #[test]
    fn packs_doc_example() {
        // [0b11, 0b10, 0b01] at width 2 -> 0b11100100.
        let mut out = Vec::new();
        pack(2, &[0b11, 0b10, 0b01], &mut out);
        assert_eq!(out, vec![0b1110_0100]);
    }

    #[test]
    fn width_zero_emits_nothing_and_unpacks_zeros() {
        let mut out = Vec::new();
        pack(0, &[1, 2, 3], &mut out);
        assert!(out.is_empty());
        assert_eq!(unpack(0, &out, 3), vec![0, 0, 0]);
    }

    #[test]
    fn width_eight_copies_raw() {
        let mut out = Vec::new();
        pack(8, &[0x00, 0xFF, 0x7F], &mut out);
        assert_eq!(out, vec![0x00, 0xFF, 0x7F]);
        assert_eq!(unpack(8, &out, 3), vec![0x00, 0xFF, 0x7F]);
    }

    #[test]
    fn round_trips_all_widths() {
        let mut rng = crate::TestRng::new(1);

        for width in 0..=8u32 {
            let mask = if width == 0 { 0 } else { (1u32 << width) - 1 };
            // Lengths span many 8-value group boundaries plus every remainder.
            for len in 0..300usize {
                let values: Vec<u32> = (0..len).map(|_| rng.next_u32() & mask).collect();
                let mut out = Vec::new();
                pack(width, &values, &mut out);
                assert_eq!(byte_width(len * width as usize), out.len());
                assert_eq!(unpack(width, &out, len), values);
            }
        }
    }

    /// Independent streaming packer used only to pin the specialized packer's
    /// exact bytes (the on-disk guarantee).
    fn reference_pack(width: u32, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        if width == 0 {
            return out;
        }

        let mut acc: u64 = 0;
        let mut bits: u32 = 0;
        let mask = (1u64 << width) - 1;
        for &v in values {
            acc = (acc << width) | (v as u64 & mask);
            bits += width;
            while bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
            }
        }

        if bits > 0 {
            out.push((acc << (8 - bits)) as u8);
        }

        out
    }

    #[test]
    fn specialized_packer_is_byte_identical_to_streaming() {
        let mut rng = crate::TestRng::new(2);

        for width in 1..=8u32 {
            let mask = (1u64 << width) - 1;
            for len in 0..300usize {
                let values: Vec<u32> = (0..len).map(|_| (rng.next_u64() & mask) as u32).collect();
                let mut out = Vec::new();
                pack(width, &values, &mut out);
                assert_eq!(
                    out,
                    reference_pack(width, &values),
                    "byte mismatch at width {width}, len {len}"
                );
            }
        }
    }
}
