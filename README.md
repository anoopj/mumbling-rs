# mumbling-rs

A Rust implementation of the [Mumbling v1 bitmap format][spec].

[spec]: https://github.com/apache/iceberg/blob/main/format/mumbling-spec.md

## Layout

A Cargo workspace with two crates:

- **`mumbling/`**: the format library, with **no dependencies** (not even
  dev-dependencies; tests use a small inline PRNG).
  - `src/bit_packing.rs`: MSB-first bit packing/unpacking for widths 0-8.
  - `src/pfor.rs`: patched frame-of-reference codec for the descriptor array
    (Appendix A of the spec).
  - `src/mumbling.rs`: `MumblingBitmap` (owning; encode, `insert`, `iter`) and
    `MumblingReader` (zero-copy read view over serialized bytes).
- **`bench/`**: the `mumbling-bench` crate; depends on `mumbling`, `roaring`,
  `zstd`, and `rand`. Isolating the deps here keeps the library clean.

On encoding, the PFOR codec prefers the larger bit width on ties (so
`[6, 34, 8, 7]` encodes to `05 00 06 07 04 10`, not the spec example's
`32 01 06 09 01 E0`; both decode identically) and stores raw bytes when
`b1 == 8`. The owning `MumblingBitmap` is a vector of non-empty containers
sorted by key, each an enum over a sparse position array or a dense 32-byte
bitset.

`MumblingReader` is the zero-copy read path: it parses the header, decodes the
descriptor array once into an offset table, and then answers `cardinality` (from
the header), `is_set`, and iteration directly against the borrowed buffer with
no per-container allocation. Dense containers are iterated 64 bits at a time
with `leading_zeros`. This is what the benchmark's decode numbers use.

## Running

```
cargo test -p mumbling                 # unit tests, incl. all spec examples
cargo run --release -p mumbling-bench  # prints the table and writes results.md
```
