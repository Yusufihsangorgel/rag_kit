// Backing rag_kit's retrieval with vector_kit's SIMD search.
//
//   dart run example/with_vector_kit.dart
//
// `InMemoryVectorStore` scores every row with a plain Dart loop, which is the
// right default for a package with no dependencies. `vector_kit` packs the
// rows into one aligned `Float32List`, caches each row's norm, and reads the
// whole matrix through a `Float32x4List` on the VM, so a search costs one SIMD
// dot product per row.
//
// `VectorStore` is an interface for exactly this: nothing below touches
// rag_kit's internals, and `Retriever` cannot tell the difference.
//
// Both packages are pure Dart with no runtime dependencies, so this composition
// works everywhere rag_kit does, web included.
import 'dart:typed_data';

import 'package:rag_kit/rag_kit.dart';
import 'package:vector_kit/vector_kit.dart';

/// A [VectorStore] backed by [VectorMatrix].
///
/// The matrix is append-only, so a replace or a removal rebuilds it. That
/// suits a corpus you index once and query many times, which is what most
/// retrieval looks like; a write-heavy store wants a different structure.
final class VectorKitStore extends VectorStore {
  final List<Document> _documents = [];
  VectorMatrix? _matrix;

  @override
  Future<void> upsert(List<Document> documents) async {
    if (documents.isEmpty) return;

    // Match InMemoryVectorStore: reject a mismatched dimension before writing
    // anything, so a bad batch cannot leave the store half-updated.
    final dimension = _documents.isNotEmpty
        ? _documents.first.embedding.length
        : documents.first.embedding.length;
    for (final document in documents) {
      if (document.embedding.length != dimension) {
        throw ArgumentError.value(
          document.embedding.length,
          'embedding.length',
          'every embedding must have $dimension components',
        );
      }
    }

    final byId = {for (final d in _documents) d.id: d};
    final replaced = documents.any((d) => byId.containsKey(d.id));
    for (final document in documents) {
      byId[document.id] = document;
    }
    if (replaced) {
      _documents
        ..clear()
        ..addAll(byId.values);
      _matrix = null; // rebuilt lazily on the next search
    } else {
      _documents.addAll(documents);
      final matrix = _matrix;
      if (matrix != null) {
        for (final document in documents) {
          matrix.add(_asFloat32(document.embedding));
        }
      }
    }
  }

  @override
  Future<List<ScoredChunk>> search(
    List<double> query, {
    int topK = 5,
    double? minScore,
    bool Function(Document document)? where,
  }) async {
    if (topK < 1) {
      throw ArgumentError.value(topK, 'topK', 'must be at least 1');
    }
    if (_documents.isEmpty) return const [];

    final matrix = _matrix ??= _build();

    // With a filter the top rows overall are not the top rows that pass, so
    // score everything and filter after. Without one, ask for exactly topK and
    // let the matrix keep the heap small.
    final wanted = where == null ? topK : _documents.length;
    final ranked = matrix.topKCosine(query, wanted);

    final results = <ScoredChunk>[];
    for (final (index, score) in ranked) {
      final document = _documents[index];
      if (where != null && !where(document)) continue;
      if (minScore != null && score < minScore) continue;
      results.add(ScoredChunk(document: document, score: score));
      if (results.length == topK) break;
    }
    return results;
  }

  @override
  Future<int> removeWhere(bool Function(Document document) test) async {
    final before = _documents.length;
    _documents.removeWhere(test);
    final removed = before - _documents.length;
    if (removed > 0) _matrix = null;
    return removed;
  }

  @override
  Future<int> count() async => _documents.length;

  @override
  Future<void> clear() async {
    _documents.clear();
    _matrix = null;
  }

  VectorMatrix _build() {
    final matrix = VectorMatrix(_documents.first.embedding.length);
    for (final document in _documents) {
      matrix.add(_asFloat32(document.embedding));
    }
    return matrix;
  }

  static Float32List _asFloat32(List<double> values) =>
      values is Float32List ? values : Float32List.fromList(values);
}

Future<void> main() async {
  // Stand-in embeddings. A real one comes from a model; the point here is the
  // wiring, and three dimensions are enough to see the ranking.
  final store = VectorKitStore();
  await store.upsert([
    Document(
      id: 'a',
      text: 'a cat sleeping on a keyboard',
      embedding: [0.9, 0.1, 0.0],
    ),
    Document(
      id: 'b',
      text: 'a dog running on a beach',
      embedding: [0.1, 0.9, 0.0],
    ),
    Document(
      id: 'c',
      text: 'a build server compiling Dart',
      embedding: [0.0, 0.1, 0.9],
    ),
  ]);

  print('stored: ${await store.count()}');

  for (final hit in await store.search([0.85, 0.2, 0.0], topK: 2)) {
    print('  ${hit.score.toStringAsFixed(3)}  ${hit.document.text}');
  }

  // The filter runs against the document, so metadata-scoped retrieval works
  // the same way it does with InMemoryVectorStore.
  final filtered = await store.search(
    [0.0, 0.1, 0.9],
    topK: 2,
    where: (d) => d.id != 'c',
  );
  print('excluding c: ${filtered.map((h) => h.document.id).toList()}');
}
