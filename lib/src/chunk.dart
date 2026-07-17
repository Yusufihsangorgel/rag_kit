/// A piece of a source text with its exact position in that source.
///
/// The offsets refer to the original string passed to `Chunker.chunk`, so
/// `source.substring(chunk.start, chunk.end)` always equals [text]. This
/// makes it possible to highlight retrieved passages in the source document.
class Chunk {
  /// Creates a chunk covering `[start, end)` in the source text.
  Chunk({
    required this.text,
    required this.start,
    required this.end,
    this.metadata = const {},
  });

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
  final Map<String, Object?> metadata;

  @override
  String toString() => 'Chunk($start..$end, ${text.length} chars)';
}
