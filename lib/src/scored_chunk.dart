import 'document.dart';

/// A search result: a stored document and its similarity to the query.
class ScoredChunk {
  /// Creates a scored result.
  ScoredChunk({required this.document, required this.score});

  /// The matched document.
  final Document document;

  /// Similarity between the query and [document].
  ///
  /// For `InMemoryVectorStore` this is cosine similarity in the range
  /// -1.0 to 1.0, where higher means more similar.
  final double score;

  @override
  String toString() =>
      'ScoredChunk(${document.id}, ${score.toStringAsFixed(4)})';
}
