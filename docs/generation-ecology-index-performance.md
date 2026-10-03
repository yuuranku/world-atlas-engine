# Continuous ecology source indexing

The ecological quantity and its scientific superlevels retain their original
model. River distance now queries an STRtree of the river's original segments.
Distance to a complete polyline equals the minimum distance to those segments.
Lake polygons remain polygons, so points inside a lake still have zero source
distance.

Each indexed segment records its original source part. Supply buffers and
adaptive end-cap refinement still use that complete part; the index only finds
which original source owns a witness. Neither source coordinates nor buffer
topology changed. The source fingerprint remains based on the complete geometry.

One bounded comparison on the development machine produced these results:

| Workload | Original index | Segment index | Speedup |
|---|---:|---:|---:|
| 524,288 queries; 32 rivers with 300 vertices each | 19.661 s | 1.334 s | 14.73× |
| 256×128 exact ecology drawing, 5 cuts | 0.689 s | 0.682 s | 1.01× |
| 256×128 exact ecology drawing, 9 cuts | 1.083 s | 1.064 s | 1.02× |
| 256×128 exact ecology drawing, 5 cuts | 0.679 s | 0.660 s | 1.03× |

Supply values in the dense long-river query case had a maximum absolute
difference of zero. All four cases kept native float32 arrays byte-identical;
the three exact drawing cases also kept band WKB byte-identical. Seven focused
ecology tests pass, including lake interiors and original buffer equality.

The query speedup is not a whole-world speedup. Simple drawing cases spend most
of their time on scalar contours and geometry operations, so the complete band
pipeline improved only slightly. A full-world estimate requires the actual
source length distribution and phase timing; this comparison does not establish
a global generation time.

Run the bounded comparison with `tools/benchmark_ecology_index.py --output ...`.
The benchmark selects the original index solely as a reference inside its
measurement process. Production code has one segment index and no alternate
drawing path.
