import 'document.dart';
import 'scored_chunk.dart';

/// Stores embedded documents and finds the ones most similar to a query
/// vector.
///
/// rag_kit ships with [InMemoryVectorStore]. Implement this interface to
/// back retrieval with something else, for example a database with a vector
/// extension. All methods are asynchronous so that remote implementations
/// fit the same interface.
abstract class VectorStore {
  /// Allows subclasses to have const constructors.
  const VectorStore();

  /// Inserts [documents], replacing any existing document with the same id.
  ///
  /// All embeddings must have the same length as the documents already in
  /// the store; otherwise an [ArgumentError] is thrown and the store is
  /// left unchanged.
  Future<void> upsert(List<Document> documents);

  /// Returns up to [topK] documents most similar to [query], best first.
  ///
  /// [minScore] drops results whose score is below the given value.
  /// [where] restricts the search to documents for which it returns true;
  /// it runs before scoring, so filtered documents cost no similarity
  /// computation.
  ///
  /// Returns an empty list when the store is empty. Throws an
  /// [ArgumentError] when the store is not empty and [query] does not have
  /// the store's embedding dimension, or when [topK] is less than 1.
  Future<List<ScoredChunk>> search(
    List<double> query, {
    int topK = 5,
    double? minScore,
    bool Function(Document document)? where,
  });

  /// Removes every document for which [test] returns true.
  ///
  /// Returns the number of removed documents.
  Future<int> removeWhere(bool Function(Document document) test);

  /// The number of stored documents.
  Future<int> count();

  /// Removes all documents.
  Future<void> clear();
}
