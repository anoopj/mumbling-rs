//! Compares the Mumbling bitmap format against Roaring (with and without zstd)
//! for small, manifest-DV-sized bitmaps.
//!
//! Methodology mirrors the "Bitmap size tests" template:
//! * Universe of 50,000 positions; a fixed number of random bits per bitmap.
//! * Bitmaps are packed as `(u32 LE length, bytes)` pairs into a buffer until it
//!   reaches 1 MiB, simulating a Parquet binary page. The bitmap count is how
//!   many fit.
//! * The 1 MiB page is compressed with zstd level 9 for the compressed size.
//! * Average bitmap size is the (compressed or uncompressed) page size divided
//!   by the bitmap count.
//!
//! For each format we also report the decoded in-memory footprint and the
//! per-bitmap decode time, both raw and via a zstd-compressed page.

use std::fmt::Write as _;
use std::fs;
use std::time::Instant;

use mumbling::{MumblingBitmap, MumblingReader};
use rand::seq::index;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use roaring::RoaringBitmap;

const UNIVERSE: u32 = 50_000;
const PAGE_TARGET: usize = 1024 * 1024;
const ZSTD_LEVEL: i32 = 9;
const DECODE_REPEATS: usize = 9;

/// Percentages of the universe that are set, matching the template rows.
const PERCENTS: &[f64] = &[
    0.002, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
];

/// One row of results for a single (format, percent) pair.
struct Row {
    format: &'static str,
    percent: f64,
    bits_per_bitmap: usize,
    bitmap_count: usize,
    total_bits_set: usize,
    uncompressed_size: usize,
    compressed_size: usize,
    avg_uncompressed: f64,
    avg_compressed: f64,
    bytes_per_set_bit: f64,
    avg_in_memory: f64,
    decode_ns_raw: f64,
    decode_ns_zstd: f64,
}

fn main() {
    let mut rows = Vec::new();

    for &percent in PERCENTS {
        let bits_per_bitmap = (percent * UNIVERSE as f64).round() as usize;
        rows.push(run_format("roaring", percent, bits_per_bitmap));
        rows.push(run_format("mumbling", percent, bits_per_bitmap));
    }

    let table = render_table(&rows);
    println!("{table}");

    let report = render_report(&rows);
    fs::write("results.md", &report).expect("write results.md");
    println!("\nWrote results.md ({} bytes)", report.len());
}

/// Generates bitmaps for one format at one density, fills a 1 MiB page, and
/// measures sizes and decode speed.
fn run_format(format: &'static str, percent: f64, bits_per_bitmap: usize) -> Row {
    let mut rng = ChaCha8Rng::seed_from_u64(0x6D_75_6D_62_6C_69_6E_67 ^ percent.to_bits());

    // Individual serialized bitmaps and the concatenated page buffer.
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut page: Vec<u8> = Vec::with_capacity(PAGE_TARGET + 4096);
    let mut in_memory_total: usize = 0;

    while page.len() < PAGE_TARGET {
        let positions = sample_positions(&mut rng, bits_per_bitmap);

        let (blob, in_memory) = match format {
            "roaring" => {
                let bitmap: RoaringBitmap = positions.iter().copied().collect();
                let mut bytes = Vec::with_capacity(bitmap.serialized_size());
                bitmap
                    .serialize_into(&mut bytes)
                    .expect("serialize roaring");
                (bytes, roaring_in_memory_bytes(&positions))
            }
            "mumbling" => {
                let bitmap: MumblingBitmap = positions.iter().copied().collect();
                let bytes = bitmap.serialize();
                (bytes, bitmap.in_memory_bytes())
            }
            other => panic!("unknown format: {other}"),
        };

        page.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        page.extend_from_slice(&blob);
        in_memory_total += in_memory;
        blobs.push(blob);
    }

    let bitmap_count = blobs.len();
    let uncompressed_size = page.len();
    let compressed = zstd::encode_all(page.as_slice(), ZSTD_LEVEL).expect("zstd encode");
    let compressed_size = compressed.len();
    let total_bits_set = bitmap_count * bits_per_bitmap;

    let decode_ns_raw = median_decode_ns_raw(format, &blobs) / bitmap_count as f64;
    let decode_ns_zstd =
        median_decode_ns_zstd(format, &compressed, bitmap_count) / bitmap_count as f64;

    Row {
        format,
        percent,
        bits_per_bitmap,
        bitmap_count,
        total_bits_set,
        uncompressed_size,
        compressed_size,
        avg_uncompressed: uncompressed_size as f64 / bitmap_count as f64,
        avg_compressed: compressed_size as f64 / bitmap_count as f64,
        bytes_per_set_bit: compressed_size as f64 / total_bits_set as f64,
        avg_in_memory: in_memory_total as f64 / bitmap_count as f64,
        decode_ns_raw,
        decode_ns_zstd,
    }
}

/// Samples `k` distinct positions in `[0, UNIVERSE)`.
fn sample_positions(rng: &mut ChaCha8Rng, k: usize) -> Vec<u32> {
    let mut positions: Vec<u32> = index::sample(rng, UNIVERSE as usize, k)
        .into_iter()
        .map(|i| i as u32)
        .collect();
    positions.sort_unstable();
    positions
}

/// Models the decoded footprint of a Roaring bitmap. The universe fits in a
/// single 64K block, so this is one array container (2 bytes per value) or one
/// bitmap container (8 KiB) once the block holds >= 4096 values.
fn roaring_in_memory_bytes(positions: &[u32]) -> usize {
    const ARRAY_TO_BITMAP_THRESHOLD: usize = 4096;
    const BITMAP_CONTAINER_BYTES: usize = 8 * 1024;

    if positions.len() >= ARRAY_TO_BITMAP_THRESHOLD {
        BITMAP_CONTAINER_BYTES
    } else {
        positions.len() * 2
    }
}

/// Median wall-clock nanoseconds to decode every raw blob once (parse + iterate
/// all set positions).
fn median_decode_ns_raw(format: &'static str, blobs: &[Vec<u8>]) -> f64 {
    let mut samples = Vec::with_capacity(DECODE_REPEATS);

    for _ in 0..DECODE_REPEATS {
        let start = Instant::now();
        let mut guard = 0u64;
        for blob in blobs {
            guard ^= decode_one(format, blob);
        }
        std::hint::black_box(guard);
        samples.push(start.elapsed().as_nanos() as f64);
    }

    median(&mut samples)
}

/// Median wall-clock nanoseconds to zstd-decompress the page and deserialize
/// every bitmap in it.
fn median_decode_ns_zstd(format: &'static str, compressed: &[u8], count: usize) -> f64 {
    let mut samples = Vec::with_capacity(DECODE_REPEATS);

    for _ in 0..DECODE_REPEATS {
        let start = Instant::now();
        let page = zstd::decode_all(compressed).expect("zstd decode");
        let mut guard = 0u64;
        let mut cursor = 0;
        let mut decoded = 0;
        while cursor + 4 <= page.len() && decoded < count {
            let len = u32::from_le_bytes(page[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            guard ^= decode_one(format, &page[cursor..cursor + len]);
            cursor += len;
            decoded += 1;
        }
        std::hint::black_box(guard);
        samples.push(start.elapsed().as_nanos() as f64);
    }

    median(&mut samples)
}

/// Decodes one blob into its set positions and returns a checksum so the work
/// is not optimized away.
///
/// Both formats do the same work: parse the serialized bytes and enumerate
/// every set position. Roaring builds its containers eagerly on
/// `deserialize_from`; Mumbling uses the zero-copy [`MumblingReader`], which
/// parses the header and descriptor array and then iterates directly over the
/// borrowed buffer.
fn decode_one(format: &'static str, blob: &[u8]) -> u64 {
    match format {
        "roaring" => {
            let bitmap = RoaringBitmap::deserialize_from(blob).expect("deserialize roaring");
            bitmap.iter().fold(0u64, |acc, pos| acc ^ pos as u64)
        }
        "mumbling" => {
            let reader = MumblingReader::new(blob);
            reader.iter().fold(0u64, |acc, pos| acc ^ pos as u64)
        }
        other => panic!("unknown format: {other}"),
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn render_table(rows: &[Row]) -> String {
    let mut out = String::new();
    writeln!(
        out,
        "{:<9} {:>7} {:>8} {:>7} {:>11} {:>12} {:>10} {:>12} {:>10} {:>9} {:>11} {:>11} {:>11}",
        "format",
        "pct",
        "bits/bm",
        "count",
        "totalbits",
        "uncompr",
        "avg_unc",
        "compressed",
        "avg_comp",
        "b/setbit",
        "in_mem/bm",
        "dec_ns_raw",
        "dec_ns_zst",
    )
    .unwrap();

    for row in rows {
        writeln!(
            out,
            "{:<9} {:>6.2}% {:>8} {:>7} {:>11} {:>12} {:>10.2} {:>12} {:>10.3} {:>9.3} {:>11.1} {:>11.1} {:>11.1}",
            row.format,
            row.percent * 100.0,
            row.bits_per_bitmap,
            row.bitmap_count,
            row.total_bits_set,
            row.uncompressed_size,
            row.avg_uncompressed,
            row.compressed_size,
            row.avg_compressed,
            row.bytes_per_set_bit,
            row.avg_in_memory,
            row.decode_ns_raw,
            row.decode_ns_zstd,
        )
        .unwrap();
    }

    out
}

fn render_report(rows: &[Row]) -> String {
    let mut out = String::new();
    writeln!(out, "# Mumbling vs Roaring — bitmap comparison\n").unwrap();
    writeln!(
        out,
        "Universe of {UNIVERSE} positions, fixed random bits per bitmap. Bitmaps are \
         packed as `(u32 LE length, bytes)` pairs into a 1 MiB page (simulating a Parquet \
         binary page); the page is compressed with zstd level {ZSTD_LEVEL}. Averages are \
         per bitmap. In-memory is the modeled decoded footprint. Decode times are median \
         wall-clock ns per bitmap over {DECODE_REPEATS} repeats.\n"
    )
    .unwrap();

    writeln!(
        out,
        "| Format | Percent set | Bits/bitmap | Bitmap count | Uncompressed size | Avg (uncompressed) | \
         Compressed size | Avg (compressed) | Bytes/set bit | In-memory/bitmap | Decode ns (raw) | \
         Decode ns (zstd) |"
    )
    .unwrap();
    writeln!(
        out,
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();

    for row in rows {
        writeln!(
            out,
            "| {} | {:.2}% | {} | {} | {} | {:.2} | {} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} |",
            row.format,
            row.percent * 100.0,
            row.bits_per_bitmap,
            row.bitmap_count,
            row.uncompressed_size,
            row.avg_uncompressed,
            row.compressed_size,
            row.avg_compressed,
            row.bytes_per_set_bit,
            row.avg_in_memory,
            row.decode_ns_raw,
            row.decode_ns_zstd,
        )
        .unwrap();
    }

    out
}
