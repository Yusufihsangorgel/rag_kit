// End-to-end test against a local Ollama server.
//
// Skips itself when Ollama is not running on localhost:11434 or when the
// embedding model below is not pulled. To run it for real:
//
//   ollama pull nomic-embed-text
//   dart test test/ollama_e2e_test.dart
@TestOn('vm')
@Timeout(Duration(minutes: 2))
library;

import 'dart:convert';
import 'dart:io';

import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

const _host = 'localhost';
const _port = 11434;
const _model = 'nomic-embed-text';

/// GETs [path] and returns the body, or null when the server is
/// unreachable or does not answer with 200.
Future<String?> _tryGet(String path) async {
  final client = HttpClient()..connectionTimeout = const Duration(seconds: 2);
  try {
    final request = await client.get(_host, _port, path);
    final response = await request.close();
    final body = await response.transform(utf8.decoder).join();
    return response.statusCode == 200 ? body : null;
  } on SocketException {
    return null;
  } on HttpException {
    return null;
  } finally {
    client.close(force: true);
  }
}

/// An [Embedder] backed by Ollama's batch embedding endpoint.
Future<List<List<double>>> _ollamaEmbedder(List<String> texts) async {
  final client = HttpClient();
  try {
    final request = await client.post(_host, _port, '/api/embed');
    request.headers.contentType = ContentType.json;
    request.write(jsonEncode({'model': _model, 'input': texts}));
    final response = await request.close();
    final body = await response.transform(utf8.decoder).join();
    if (response.statusCode != 200) {
      throw HttpException('Ollama returned ${response.statusCode}: $body');
    }
    final decoded = jsonDecode(body) as Map<String, dynamic>;
    final embeddings = decoded['embeddings'] as List<dynamic>;
    return [
      for (final embedding in embeddings)
        [
          for (final value in embedding as List<dynamic>)
            (value as num).toDouble(),
        ],
    ];
  } finally {
    client.close(force: true);
  }
}

void main() {
  test('retrieves the relevant chunk with real Ollama embeddings', () async {
    final tagsBody = await _tryGet('/api/tags');
    if (tagsBody == null) {
      markTestSkipped('Ollama is not running on $_host:$_port.');
      return;
    }
    final tags = jsonDecode(tagsBody) as Map<String, dynamic>;
    final names = [
      for (final model in tags['models'] as List<dynamic>)
        (model as Map<String, dynamic>)['name'] as String,
    ];
    final available = names.any(
      (name) => name == _model || name.startsWith('$_model:'),
    );
    if (!available) {
      markTestSkipped('Model $_model is not pulled.');
      return;
    }

    final store = InMemoryVectorStore();
    final retriever = Retriever(
      embedder: _ollamaEmbedder,
      store: store,
      chunker: Chunker.paragraphs(),
    );
    await retriever.addText('''
Honey bees communicate the location of food through the waggle dance.
The angle of the dance encodes direction relative to the sun and its
duration encodes distance.

The Great Barrier Reef is the largest coral reef system in the world,
stretching over 2300 kilometers off the coast of Australia.

Sourdough bread rises through fermentation by wild yeast and lactic
acid bacteria captured in a starter culture.

The Rust compiler enforces memory safety at compile time through its
ownership and borrowing rules, without a garbage collector.
''', sourceId: 'corpus');

    expect(await store.count(), 4);

    final results = await retriever.retrieve(
      'How do bees tell each other where to find food?',
      topK: 3,
    );
    expect(results, isNotEmpty);
    expect(
      results.first.document.text,
      contains('waggle'),
      reason: 'the bee paragraph must rank first for a bee question',
    );
  });
}
