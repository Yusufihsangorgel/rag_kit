import 'package:rag_kit/rag_kit.dart';

/// A deterministic embedder for the demo: hashes each word into one of 64
/// buckets and counts occurrences, so texts sharing words end up with
/// similar vectors. It needs no network and gives repeatable output.
///
/// In a real application, replace this with a call to an embedding model.
/// The README shows Ollama and OpenAI bindings.
Future<List<List<double>>> hashEmbedder(List<String> texts) async {
  const dimensions = 64;
  return [
    for (final text in texts)
      () {
        final vector = List<double>.filled(dimensions, 0);
        for (final word in text.toLowerCase().split(RegExp(r'[^a-z0-9]+'))) {
          if (word.isEmpty) continue;
          vector[_fnv1a(word) % dimensions] += 1;
        }
        return vector;
      }(),
  ];
}

/// 32-bit FNV-1a. Stable across runs and platforms, unlike String.hashCode.
int _fnv1a(String input) {
  var hash = 0x811c9dc5;
  for (var i = 0; i < input.length; i++) {
    hash ^= input.codeUnitAt(i);
    hash = (hash * 0x01000193) & 0xffffffff;
  }
  return hash;
}

Future<void> main() async {
  final store = InMemoryVectorStore();
  final retriever = Retriever(
    embedder: hashEmbedder,
    store: store,
    chunker: Chunker.paragraphs(),
  );

  // Two sources. Each addText is one document, chunked into paragraphs; give
  // each document its own sourceId, or a later addText with a reused id would
  // evict the earlier one's chunks.
  await retriever.addText('''
Dart isolates do not share mutable memory. Each isolate has its own heap,
and isolates exchange data by passing messages over ports.

Dart records are immutable and let a function return several values without
a class. Pattern matching destructures them by position.
''', sourceId: 'dart-guide');
  await retriever.addText(
    'Sourdough bread needs a starter culture of wild yeast and bacteria. '
    'The dough ferments for hours, which develops flavor and structure.',
    sourceId: 'cookbook',
  );

  const query = 'How does Dart pass data between isolates?';
  print('Query: $query\n');

  // Retrieval across every source.
  print('All sources:');
  for (final r in await retriever.retrieve(query, topK: 3)) {
    print('  ${r.score.toStringAsFixed(3)}  ${r.document.id}');
  }

  // Scope the same query to one source with the metadata `where` filter, so
  // documents from other sources never enter scoring. Use it for per-tenant,
  // per-language or per-collection retrieval.
  print('\nScoped to source "dart-guide":');
  final scoped = await retriever.retrieve(
    query,
    topK: 3,
    where: (doc) => doc.metadata['sourceId'] == 'dart-guide',
  );
  for (final r in scoped) {
    print('  ${r.score.toStringAsFixed(3)}  ${r.document.id}');
  }

  // buildContext takes the same filter, returning a prompt-ready string.
  final context = await retriever.buildContext(
    query,
    topK: 2,
    where: (doc) => doc.metadata['sourceId'] == 'dart-guide',
  );
  print('\nContext for the LLM prompt (dart-guide only):\n');
  print(context);
}
