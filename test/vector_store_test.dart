import 'dart:typed_data';

import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

Document doc(String id, List<double> embedding, {Map<String, Object?>? meta}) =>
    Document(
      id: id,
      text: 'text of $id',
      embedding: embedding,
      metadata: meta ?? const {},
    );

void main() {
  group('InMemoryVectorStore cosine similarity', () {
    test('matches a hand-computed value', () async {
      // dot([1,2,3],[4,5,6]) = 32, |a| = sqrt(14), |b| = sqrt(77)
      // 32 / (sqrt(14) * sqrt(77)) = 0.9746318461970762
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [4, 5, 6]),
      ]);
      final results = await store.search([1, 2, 3], topK: 1);
      expect(results.single.score, closeTo(0.9746318461970762, 1e-6));
    });

    test('orthogonal vectors score zero', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [0, 1]),
      ]);
      final results = await store.search([1, 0]);
      expect(results.single.score, closeTo(0, 1e-9));
    });

    test('opposite vectors score minus one', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [-1, 0]),
      ]);
      final results = await store.search([1, 0]);
      expect(results.single.score, closeTo(-1, 1e-6));
    });

    test('same direction scores one regardless of magnitude', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [2, 0]),
      ]);
      final results = await store.search([0.5, 0]);
      expect(results.single.score, closeTo(1, 1e-6));
    });

    test('a zero-vector document scores zero, not NaN', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('zero', [0, 0]),
      ]);
      final results = await store.search([1, 0]);
      expect(results.single.score, 0);
      expect(results.single.score.isNaN, isFalse);
    });

    test('a zero-vector query scores zero for all documents', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
        doc('b', [0, 1]),
      ]);
      final results = await store.search([0, 0]);
      expect(results, hasLength(2));
      expect(results.every((r) => r.score == 0), isTrue);
    });
  });

  group('InMemoryVectorStore search', () {
    test('returns top-k in descending score order', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('orthogonal', [0, 1]), // 0.0
        doc('exact', [1, 0]), // 1.0
        doc('diagonal', [1, 1]), // 0.7071
        doc('close', [2, 1]), // 0.8944
        doc('opposite', [-1, 0]), // -1.0
      ]);
      final results = await store.search([1, 0], topK: 3);
      expect(results.map((r) => r.document.id).toList(), [
        'exact',
        'close',
        'diagonal',
      ]);
      expect(results[0].score, closeTo(1.0, 1e-6));
      expect(results[1].score, closeTo(0.8944271909999159, 1e-6));
      expect(results[2].score, closeTo(0.7071067811865475, 1e-6));
    });

    test('topK larger than the store returns everything', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
        doc('b', [0, 1]),
      ]);
      final results = await store.search([1, 0], topK: 50);
      expect(results, hasLength(2));
    });

    test('minScore drops results below the threshold', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('exact', [1, 0]), // 1.0
        doc('mid', [1, 1]), // 0.7071
        doc('zero', [0, 1]), // 0.0
      ]);
      final results = await store.search([1, 0], minScore: 0.5);
      expect(results.map((r) => r.document.id).toList(), ['exact', 'mid']);
    });

    test('minScore keeps results exactly at the threshold', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('exact', [2, 0]), // exactly 1.0
        doc('zero', [0, 1]), // exactly 0.0
      ]);
      final results = await store.search([1, 0], minScore: 1.0);
      expect(results.map((r) => r.document.id).toList(), ['exact']);
    });

    test('where filters by metadata before ranking', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('en-1', [1, 0], meta: {'lang': 'en'}),
        doc('tr-1', [0.99, 0.1], meta: {'lang': 'tr'}),
        doc('en-2', [0, 1], meta: {'lang': 'en'}),
      ]);
      final results = await store.search([
        1,
        0,
      ], where: (d) => d.metadata['lang'] == 'en');
      expect(results.map((r) => r.document.id).toList(), ['en-1', 'en-2']);
    });

    test('where combines with topK', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        for (var i = 0; i < 10; i++)
          doc('d$i', [1, i / 10], meta: {'even': i.isEven}),
      ]);
      final results = await store.search(
        [1, 0],
        topK: 2,
        where: (d) => d.metadata['even'] == true,
      );
      expect(results, hasLength(2));
      expect(results.map((r) => r.document.id).toList(), ['d0', 'd2']);
    });

    test('equal scores keep insertion order', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('first', [1, 0]),
        doc('second', [2, 0]),
        doc('third', [3, 0]),
      ]);
      final results = await store.search([1, 0], topK: 2);
      expect(results.map((r) => r.document.id).toList(), ['first', 'second']);
    });

    test('empty store returns an empty list for any query', () async {
      final store = InMemoryVectorStore();
      expect(await store.search([1, 2, 3]), isEmpty);
    });

    test('topK below one throws', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.search([1, 0], topK: 0),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('query dimension mismatch throws with a clear message', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.search([1, 0, 0]),
        throwsA(
          isA<ArgumentError>().having(
            (e) => e.message,
            'message',
            allOf(contains('3 dimensions'), contains('2-dimensional')),
          ),
        ),
      );
    });
  });

  group('InMemoryVectorStore upsert and lifecycle', () {
    test('upsert with the same id replaces the document', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await store.upsert([
        Document(id: 'a', text: 'updated', embedding: [0, 1]),
      ]);
      expect(await store.count(), 1);
      final results = await store.search([0, 1]);
      expect(results.single.document.text, 'updated');
      expect(results.single.score, closeTo(1, 1e-6));
    });

    test('dimension mismatch on upsert throws', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.upsert([
          doc('b', [1, 0, 0]),
        ]),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('a mismatched batch is rejected without partial writes', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await expectLater(
        store.upsert([
          doc('b', [0, 1]),
          doc('c', [1, 2, 3]),
        ]),
        throwsA(isA<ArgumentError>()),
      );
      expect(await store.count(), 1);
    });

    test('an empty embedding throws', () async {
      final store = InMemoryVectorStore();
      await expectLater(
        store.upsert([doc('a', [])]),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('embeddings are stored as float32', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [0.1, 0.2]),
      ]);
      final results = await store.search([1, 1]);
      final stored = results.single.document.embedding;
      final expected = Float32List.fromList([0.1, 0.2]);
      expect(stored[0], expected[0]);
      expect(stored[1], expected[1]);
    });

    test('metadata is copied at upsert time, not aliased', () async {
      final store = InMemoryVectorStore();
      final original = <String, Object?>{'status': 'draft'};
      await store.upsert([
        doc('a', [1, 0], meta: original),
      ]);

      // Mutating the caller's map after upsert must not reach the store.
      original['status'] = 'published';
      original['secret'] = 'leaked';

      final results = await store.search([1, 0]);
      expect(results.single.document.metadata, {'status': 'draft'});
    });

    test('metadata returned from search is unmodifiable', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0], meta: {'status': 'draft'}),
      ]);

      final results = await store.search([1, 0]);
      expect(
        () => results.single.document.metadata['status'] = 'hacked',
        throwsUnsupportedError,
      );

      // The rejected mutation must not have reached the store either.
      final again = await store.search([1, 0]);
      expect(again.single.document.metadata, {'status': 'draft'});
    });

    test('count, clear, and dimension lifecycle', () async {
      final store = InMemoryVectorStore();
      expect(store.dimension, isNull);
      await store.upsert([
        doc('a', [1, 0]),
        doc('b', [0, 1]),
      ]);
      expect(await store.count(), 2);
      expect(store.dimension, 2);
      await store.clear();
      expect(await store.count(), 0);
      expect(store.dimension, isNull);
      // After clearing, a different dimension is accepted.
      await store.upsert([
        doc('c', [1, 2, 3]),
      ]);
      expect(store.dimension, 3);
    });

    test('removeWhere removes matches and reports the count', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0], meta: {'kind': 'old'}),
        doc('b', [0, 1], meta: {'kind': 'new'}),
        doc('c', [1, 1], meta: {'kind': 'old'}),
      ]);
      final removed = await store.removeWhere(
        (d) => d.metadata['kind'] == 'old',
      );
      expect(removed, 2);
      expect(await store.count(), 1);
      final results = await store.search([0, 1]);
      expect(results.single.document.id, 'b');
    });

    test('removing every document resets the dimension', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      await store.removeWhere((_) => true);
      expect(store.dimension, isNull);
      await store.upsert([
        doc('b', [1, 2, 3]),
      ]);
      expect(store.dimension, 3);
    });
  });

  group('non-finite embedding guards', () {
    test('rejects NaN, infinity, and float32-overflowing components', () async {
      final store = InMemoryVectorStore();
      for (final bad in [double.nan, double.infinity, -double.infinity, 1e39]) {
        await expectLater(
          store.upsert([
            doc('bad', [bad, 0]),
          ]),
          throwsArgumentError,
          reason: 'component $bad must be rejected',
        );
      }
      expect(await store.count(), 0);
    });

    test('rejects non-finite query components', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        doc('a', [1, 0]),
      ]);
      for (final bad in [double.nan, double.infinity, 1e39]) {
        await expectLater(store.search([bad, 0]), throwsArgumentError);
      }
    });

    test('a rejected batch writes nothing', () async {
      final store = InMemoryVectorStore();
      await expectLater(
        store.upsert([
          doc('ok', [1, 0]),
          doc('bad', [double.nan, 0]),
        ]),
        throwsArgumentError,
      );
      expect(await store.count(), 0);
    });
  });
}
