import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

Chunk chunkOf(
  String text, {
  int start = 0,
  int? end,
  Map<String, Object?> metadata = const {},
}) => Chunk(
  text: text,
  start: start,
  end: end ?? start + text.length,
  metadata: metadata,
);

void main() {
  group('Chunk equality', () {
    test('same text, range and metadata are equal and hash alike', () {
      final a = chunkOf('hello', start: 4);
      final b = chunkOf('hello', start: 4);

      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('a different field breaks equality', () {
      final base = chunkOf('hello', start: 4);

      expect(base, isNot(chunkOf('HELLO', start: 4)));
      expect(base, isNot(chunkOf('hello', start: 5)));
      expect(base, isNot(Chunk(text: 'hello', start: 4, end: 99)));
      expect(base, isNot(chunkOf('hello', start: 4, metadata: {'page': 1})));
    });

    test('metadata is compared by content, not by map identity', () {
      final a = chunkOf('x', metadata: {'page': 1, 'source': 'a.md'});
      final b = chunkOf('x', metadata: {'page': 1, 'source': 'a.md'});

      expect(identical(a.metadata, b.metadata), isFalse);
      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test('metadata insertion order does not affect equality or hash', () {
      final a = chunkOf('x', metadata: {'page': 1, 'source': 'a.md'});
      final b = chunkOf('x', metadata: {'source': 'a.md', 'page': 1});

      expect(a, equals(b));
      expect(a.hashCode, b.hashCode);
    });

    test(
      'a metadata value that is itself a collection compares by identity',
      () {
        // Documented limitation: values are compared with their own ==, so two
        // equal-looking lists in metadata are different values.
        final shared = ['a', 'b'];

        expect(
          chunkOf('x', metadata: {'tags': shared}),
          equals(chunkOf('x', metadata: {'tags': shared})),
        );
        expect(
          chunkOf(
            'x',
            metadata: {
              'tags': ['a', 'b'],
            },
          ),
          isNot(
            chunkOf(
              'x',
              metadata: {
                'tags': ['a', 'b'],
              },
            ),
          ),
        );
      },
    );

    test('a set collapses chunks that cover the same range', () {
      final chunks = [
        chunkOf('hello', start: 0),
        chunkOf('hello', start: 0),
        chunkOf('world', start: 6),
      ];

      expect(chunks.toSet(), hasLength(2));
    });

    test('chunking the same source twice produces equal chunks', () {
      const source =
          'The first paragraph.\n\nThe second one.\n\nAnd a third for luck.';

      expect(Chunker.fixed().chunk(source), Chunker.fixed().chunk(source));
      expect(
        Chunker.paragraphs().chunk(source),
        Chunker.paragraphs().chunk(source),
      );
    });

    test('a chunk is not equal to a non-chunk', () {
      expect(chunkOf('x'), isNot(equals('x')));
    });
  });

  group('Document identity', () {
    // Documents are deliberately identified by id rather than by value: a
    // store keeps embeddings as float32, so a document read back is not
    // component-wise identical to the one that was written.
    test('two documents with the same content are distinct objects', () {
      final a = Document(id: 'a', text: 'hello', embedding: [1, 0]);
      final b = Document(id: 'a', text: 'hello', embedding: [1, 0]);

      expect(a, isNot(equals(b)));
      expect(a, equals(a));
    });

    test(
      'the store deduplicates by id, so value equality is not needed',
      () async {
        final store = InMemoryVectorStore();
        await store.upsert([
          Document(id: 'a', text: 'first', embedding: [1, 0]),
        ]);
        await store.upsert([
          Document(id: 'a', text: 'replacement', embedding: [0, 1]),
        ]);

        expect(await store.count(), 1);
        final results = await store.search([0, 1]);
        expect(results.single.document.text, 'replacement');
      },
    );

    test(
      'a document read back is the same document but not the same values',
      () async {
        // The float32 round trip is exactly why == by value would mislead.
        final original = Document(
          id: 'a',
          text: 'hello',
          embedding: [0.1, 0.2],
        );
        final store = InMemoryVectorStore();
        await store.upsert([original]);

        final readBack = (await store.search([0.1, 0.2])).single.document;

        expect(readBack.id, original.id);
        expect(readBack.text, original.text);
        expect(readBack.embedding, isNot(original.embedding));
        expect(readBack.embedding.first, closeTo(0.1, 1e-7));
      },
    );
  });
}
