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

  await retriever.addText('''
Dart isolates do not share mutable memory. Each isolate has its own heap,
and isolates exchange data by passing messages over ports.

Sourdough bread needs a starter culture of wild yeast and bacteria. The
dough ferments for hours, which develops both flavor and structure.

Coral reefs are built by colonies of tiny animals called polyps. The
polyps secrete calcium carbonate, which forms the reef skeleton.
''', sourceId: 'notes');

  const query = 'How do Dart isolates exchange data?';
  print('Query: $query\n');

  final results = await retriever.retrieve(query, topK: 3);
  for (final result in results) {
    final preview = result.document.text.split('\n').first;
    print(
      '${result.score.toStringAsFixed(3)}  ${result.document.id}  '
      '$preview',
    );
  }

  final context = await retriever.buildContext(query, topK: 2);
  print('\nContext for the LLM prompt:\n');
  print(context);
}
