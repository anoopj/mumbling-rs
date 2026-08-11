# mumbling-rs

A Rust implementation of the [Mumbling v1 bitmap format][spec] with a benchmark
comparing it to [Roaring bitmaps][roaring] (with and without zstd) for small,
bounded bitmaps.

The target use case is **Iceberg V4 leaf-manifest deletion vectors**: small
bitmaps inlined into the root manifest that mark which positions in a leaf
manifest are deleted or replaced. These are copied on every commit, so their
serialized size, decoded footprint, and decode speed matter directly.

[spec]: https://github.com/apache/iceberg/blob/main/format/mumbling-spec.md
[roaring]: https://roaringbitmap.org/

## Layout

A Cargo workspace with two crates:

- **`mumbling/`** — the format library, with **no dependencies** (not even
  dev-dependencies; tests use a small inline PRNG):
  - `src/bit_packing.rs` — MSB-first bit packing/unpacking for widths 0–8.
  - `src/pfor.rs` — patched frame-of-reference codec for the descriptor array
    (Appendix A of the spec).
  - `src/mumbling.rs` — `MumblingBitmap` (owning; encode, `insert`, `iter`) and
    `MumblingReader` (zero-copy read view over serialized bytes).
- **`bench/`** — the `mumbling-bench` crate; depends on `mumbling`, `roaring`,
  `zstd`, and `rand`. Isolating the deps here keeps the library clean.

On encoding, the PFOR codec prefers the larger bit width on ties (so
`[6, 34, 8, 7]` encodes to `05 00 06 07 04 10`, not the spec example's
`32 01 06 09 01 E0` — both decode identically) and stores raw bytes when
`b1 == 8`. The owning `MumblingBitmap` is a vector of non-empty containers
sorted by key, each an enum over a sparse position array or a dense 32-byte
bitset.

`MumblingReader` is the zero-copy read path: it parses the header, decodes the
descriptor array once into an offset table, and then answers `cardinality` (from
the header), `is_set`, and iteration directly against the borrowed buffer — no
per-container allocation, and dense containers are iterated 64 bits at a time
with `leading_zeros`. This is what the benchmark's decode numbers use.

## Running

```
cargo test -p mumbling                 # unit tests, incl. all spec examples
cargo run --release -p mumbling-bench  # prints the table and writes results.md
```
