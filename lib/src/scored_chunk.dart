import 'document.dart';

/// A search result: a stored document and its similarity to the query.
///
/// Like [Document], this has no value equality: it wraps a document, so it
/// can only be as comparable as the document inside it. Results come back
/// ranked, so compare `document.id` when you need to check which document a
/// result points at.
final class ScoredChunk {
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
