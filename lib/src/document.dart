/// A text passage stored in a vector store together with its embedding.
///
/// A document is identified by its [id], not by value: it deliberately does
/// not define `==`, so two instances are equal only when they are the same
/// object. Value equality would be misleading here, because a store keeps
/// embeddings as float32 and a document read back therefore carries slightly
/// different components than the one that was written, while being the same
/// document. Compare [id] when you need to tell documents apart, and let the
/// store deduplicate: upserting the same id replaces the stored document.
class Document {
  /// Creates a document.
  ///
  /// [embedding] must not be modified after the document is handed to a
  /// store. The same applies to any mutable value nested inside [metadata]
  /// (a `List` or `Map` used as a metadata value): a store may copy the top
  /// level of [metadata], but is not expected to deep-copy its values. To
  /// persist the store with `toBytes` or `save`, [metadata] must contain
  /// only JSON-encodable values.
  Document({
    required this.id,
    required this.text,
    required this.embedding,
    this.metadata = const {},
  });

  /// Unique identifier. Upserting another document with the same id
  /// replaces this one.
  final String id;

  /// The stored text.
  final String text;

  /// The embedding vector for [text].
  ///
  /// Documents returned from `InMemoryVectorStore` expose the stored
  /// float32 values, so components may differ from the original doubles by
  /// the float32 rounding error.
  final List<double> embedding;

  /// Application-defined metadata, usable for filtering during search.
  ///
  /// Documents returned from `InMemoryVectorStore` expose an unmodifiable
  /// copy of the map that was stored: a fresh top-level map, so mutating the
  /// map you passed in afterwards does not reach the store, and mutating the
  /// map you get back throws instead of silently rewriting the stored
  /// document.
  final Map<String, Object?> metadata;

  @override
  String toString() => 'Document($id, ${embedding.length} dims)';
}
