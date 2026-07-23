/// A piece of a source text with its exact position in that source.
///
/// The offsets refer to the original string passed to `Chunker.chunk`, so
/// `source.substring(chunk.start, chunk.end)` always equals [text]. This
/// makes it possible to highlight retrieved passages in the source document.
final class Chunk {
  /// Creates a chunk covering `[start, end)` in the source text.
  ///
  /// [metadata] is copied into an unmodifiable map, so a chunk's hash cannot
  /// change under it: mutating the map you passed in afterwards does not reach
  /// the chunk, and mutating the map you get back throws. Without that, a
  /// chunk placed in a `Set` would be lost from it the moment the caller
  /// touched the original map, since [hashCode] reads the metadata.
  Chunk({
    required this.text,
    required this.start,
    required this.end,
    Map<String, Object?> metadata = const {},
  }) : metadata = metadata.isEmpty
           ? const {}
           : Map<String, Object?>.unmodifiable(metadata);

  /// The chunk content.
  ///
  /// Always equal to `source.substring(start, end)` for the source string
  /// the chunker was given.
  final String text;

  /// Offset in the source text where this chunk starts, inclusive.
  final int start;

  /// Offset in the source text where this chunk ends, exclusive.
  final int end;

  /// Extra information attached by the chunker. Empty for the built-in
  /// chunkers.
  ///
  /// This is an unmodifiable copy of what was passed to the constructor;
  /// writing to it throws.
  final Map<String, Object?> metadata;

  /// Two chunks are equal when they carry the same text, cover the same
  /// range, and have equal metadata.
  ///
  /// This is what makes `chunks.toSet()` collapse the duplicates that
  /// overlapping windows produce, and what lets a test compare a chunker's
  /// output against expected chunks directly.
  ///
  /// Metadata is compared entry by entry, and each value with its own `==`.
  /// A `List` or `Map` stored as a metadata value therefore compares by
  /// identity, not by content.
  @override
  bool operator ==(Object other) =>
      other is Chunk &&
      text == other.text &&
      start == other.start &&
      end == other.end &&
      _metadataEquals(metadata, other.metadata);

  @override
  int get hashCode => Object.hash(text, start, end, _metadataHash(metadata));

  static bool _metadataEquals(Map<String, Object?> a, Map<String, Object?> b) {
    if (identical(a, b)) return true;
    if (a.length != b.length) return false;
    for (final entry in a.entries) {
      if (!b.containsKey(entry.key) || b[entry.key] != entry.value) {
        return false;
      }
    }
    return true;
  }

  // Combined with xor so the result does not depend on iteration order:
  // two maps that are equal must hash the same however they were built.
  static int _metadataHash(Map<String, Object?> map) {
    var hash = 0;
    for (final entry in map.entries) {
      hash ^= Object.hash(entry.key, entry.value);
    }
    return hash;
  }

  @override
  String toString() => 'Chunk($start..$end, ${text.length} chars)';
}
