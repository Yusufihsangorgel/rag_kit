/// A text passage stored in a vector store together with its embedding.
class Document {
  /// Creates a document.
  ///
  /// [embedding] must not be modified after the document is handed to a
  /// store. To persist the store with `toBytes` or `save`, [metadata] must
  /// contain only JSON-encodable values.
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
  final Map<String, Object?> metadata;

  @override
  String toString() => 'Document($id, ${embedding.length} dims)';
}
