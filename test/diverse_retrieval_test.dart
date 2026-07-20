import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

/// An embedder with a hand-written vector per text, so a test can state
/// exactly how similar any two chunks are.
class FixedEmbedder {
  FixedEmbedder(this.vectors);

  final Map<String, List<double>> vectors;

  Future<List<List<double>>> call(List<String> texts) async => [
        for (final text in texts)
          vectors[text] ??
              (throw StateError('no vector for "$text"')),
      ];
}

// All unit length, so cosine similarity is just the dot product.
//
//   query      [1, 0]
//   dupA       cos(query) = 0.800   cos(dupB) = 0.9997
//   dupB       cos(query) = 0.810   cos(dupA) = 0.9997
//   distinct   cos(query) = 0.750   cos(dupA) = 0.204
//
// So the two most similar chunks to the query are near-copies of each other,
// and the third is slightly less similar but says something else. That is the
// shape a repetitive document produces, and the case MMR exists for.
const _query = 'what does it say';
const _dupA = 'the first near duplicate';
const _dupB = 'the second near duplicate';
const _distinct = 'a different point entirely';

Retriever _retriever() {
  final embedder = FixedEmbedder({
    _query: const [1.0, 0.0],
    _dupA: const [0.8, 0.6],
    _dupB: const [0.81, 0.5864],
    _distinct: const [0.75, -0.6614],
  });
  return Retriever(
    embedder: embedder.call,
    store: InMemoryVectorStore(),
    chunker: Chunker.fixed(maxChars: 1000, overlap: 0),
  );
}

Future<Retriever> _populated() async {
  final retriever = _retriever();
  for (final text in [_dupA, _dupB, _distinct]) {
    await retriever.addText(text, sourceId: text);
  }
  return retriever;
}

void main() {
  test('plain retrieve returns the near-duplicates, diverse does not',
      () async {
    final retriever = await _populated();

    final plain = await retriever.retrieve(_query, topK: 2);
    expect(
      plain.map((c) => c.document.text),
      [_dupB, _dupA],
      reason: 'similarity alone picks both copies',
    );

    final diverse = await retriever.retrieveDiverse(_query, topK: 2);
    expect(
      diverse.map((c) => c.document.text),
      [_dupB, _distinct],
      reason: 'the second slot goes to something that adds information',
    );
  });

  test('lambda 1.0 is pure relevance and matches retrieve', () async {
    final retriever = await _populated();
    final plain = await retriever.retrieve(_query, topK: 2);
    final diverse =
        await retriever.retrieveDiverse(_query, topK: 2, lambda: 1.0);
    expect(
      diverse.map((c) => c.document.text),
      plain.map((c) => c.document.text),
    );
  });

  test('lambda 0.0 is pure diversity', () async {
    final retriever = await _populated();
    final diverse =
        await retriever.retrieveDiverse(_query, topK: 2, lambda: 0.0);
    expect(diverse.last.document.text, _distinct);
  });

  test('scores stay the query similarity, not the selection score', () async {
    final retriever = await _populated();
    final diverse = await retriever.retrieveDiverse(_query, topK: 2);
    // dupB against [1, 0] is its first component.
    expect(diverse.first.score, closeTo(0.81, 0.001));
    expect(diverse.last.score, closeTo(0.75, 0.001));
  });

  test('asking for at least as many as exist changes nothing', () async {
    final retriever = await _populated();
    final all = await retriever.retrieveDiverse(_query, topK: 10);
    final plain = await retriever.retrieve(_query, topK: 10);
    expect(all.map((c) => c.document.text), plain.map((c) => c.document.text));
  });

  test('a fetchK below topK is raised rather than starving the selection',
      () async {
    final retriever = await _populated();
    final diverse =
        await retriever.retrieveDiverse(_query, topK: 3, fetchK: 1);
    expect(diverse, hasLength(3));
  });

  test('an empty store returns nothing', () async {
    final retriever = _retriever();
    expect(await retriever.retrieveDiverse(_query), isEmpty);
  });

  test('bad arguments are rejected', () async {
    final retriever = await _populated();
    expect(
      () => retriever.retrieveDiverse(_query, topK: 0),
      throwsArgumentError,
    );
    expect(
      () => retriever.retrieveDiverse(_query, lambda: 1.5),
      throwsArgumentError,
    );
    expect(
      () => retriever.retrieveDiverse(_query, lambda: -0.1),
      throwsArgumentError,
    );
  });

  test('buildContext can select the same way', () async {
    final retriever = await _populated();

    final plain = await retriever.buildContext(_query, topK: 2);
    expect(plain, contains(_dupA));
    expect(plain, isNot(contains(_distinct)));

    final diverse =
        await retriever.buildContext(_query, topK: 2, diverse: true);
    expect(diverse, contains(_distinct));
    expect(diverse, isNot(contains(_dupA)));
  });
}
