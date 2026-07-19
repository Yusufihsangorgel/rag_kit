import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

/// A deterministic embedder for tests: one dimension per keyword, holding
/// the number of occurrences of that keyword in the text.
///
/// Counts calls and records every batch, so tests can prove that a whole
/// document is embedded with a single call.
class KeywordEmbedder {
  KeywordEmbedder(this.keywords);

  final List<String> keywords;
  int calls = 0;
  final List<List<String>> batches = [];

  Future<List<List<double>>> call(List<String> texts) async {
    calls++;
    batches.add(texts);
    return [for (final text in texts) embedOne(text)];
  }

  List<double> embedOne(String text) {
    final lower = text.toLowerCase();
    return [
      for (final keyword in keywords)
        RegExp(RegExp.escape(keyword)).allMatches(lower).length.toDouble(),
    ];
  }
}

void main() {
  group('Retriever.addText', () {
    test('chunks, embeds in one batch, and upserts with derived ids', () async {
      final embedder = KeywordEmbedder(['cat', 'dog', 'fish']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(
        embedder: embedder.call,
        store: store,
        chunker: Chunker.paragraphs(),
      );
      const text =
          'The cat sleeps all day.\n\n'
          'The dog plays outside.\n\n'
          'The fish swims in circles.';
      await retriever.addText(text, sourceId: 'pets');

      expect(embedder.calls, 1, reason: 'all chunks must go in one batch');
      expect(embedder.batches.single, hasLength(3));
      expect(await store.count(), 3);

      final results = await store.search(embedder.embedOne('dog'), topK: 1);
      expect(results.single.document.id, 'pets#1');
    });

    test('document metadata carries source and chunk positions', () async {
      final embedder = KeywordEmbedder(['cat', 'dog']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(
        embedder: embedder.call,
        store: store,
        chunker: Chunker.paragraphs(),
      );
      const text = 'about the cat\n\nabout the dog';
      await retriever.addText(text, sourceId: 'notes', metadata: {'v': 7});

      final results = await store.search(embedder.embedOne('dog'), topK: 1);
      final metadata = results.single.document.metadata;
      expect(metadata['v'], 7);
      expect(metadata['sourceId'], 'notes');
      expect(metadata['chunkIndex'], 1);
      expect(metadata['chunkStart'], text.indexOf('about the dog'));
      expect(metadata['chunkEnd'], text.length);
    });

    test('generates incrementing source ids when none is given', () async {
      final embedder = KeywordEmbedder(['cat', 'dog']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(embedder: embedder.call, store: store);
      await retriever.addText('cat');
      await retriever.addText('dog');
      final catHit = await store.search(embedder.embedOne('cat'), topK: 1);
      final dogHit = await store.search(embedder.embedOne('dog'), topK: 1);
      expect(catHit.single.document.id, 'text-0#0');
      expect(dogHit.single.document.id, 'text-1#0');
    });

    test('does nothing for empty or whitespace text', () async {
      final embedder = KeywordEmbedder(['cat']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(embedder: embedder.call, store: store);
      await retriever.addText('');
      await retriever.addText('   \n\n  ');
      expect(embedder.calls, 0);
      expect(await store.count(), 0);
    });

    test('uses the fixed chunker by default', () async {
      final embedder = KeywordEmbedder(['word']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(embedder: embedder.call, store: store);
      final text = List.generate(400, (i) => 'word$i').join(' ');
      await retriever.addText(text, sourceId: 'long');
      // Default Chunker.fixed() has maxChars 1000, so this text must split.
      expect(await store.count(), greaterThan(1));
      expect(embedder.calls, 1);
    });

    test(
      'throws StateError when the embedder returns the wrong count',
      () async {
        Future<List<List<double>>> broken(List<String> texts) async => [];
        final retriever = Retriever(
          embedder: broken,
          store: InMemoryVectorStore(),
        );
        await expectLater(
          retriever.addText('some text'),
          throwsA(isA<StateError>()),
        );
      },
    );
  });

  group('Retriever.retrieve', () {
    test('finds the relevant chunk', () async {
      final embedder = KeywordEmbedder(['cat', 'dog', 'fish']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(
        embedder: embedder.call,
        store: store,
        chunker: Chunker.paragraphs(),
      );
      await retriever.addText(
        'The cat sleeps.\n\nThe dog barks.\n\nThe fish swims.',
        sourceId: 'pets',
      );
      final results = await retriever.retrieve('dog', topK: 1);
      expect(results.single.document.text, 'The dog barks.');
      expect(results.single.score, closeTo(1, 1e-6));
    });

    test('embeds the query as a single one-element batch', () async {
      final embedder = KeywordEmbedder(['cat']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(embedder: embedder.call, store: store);
      await retriever.addText('cat things', sourceId: 's');
      await retriever.retrieve('cat');
      expect(embedder.calls, 2);
      expect(embedder.batches.last, ['cat']);
    });

    test('respects topK and minScore', () async {
      final embedder = KeywordEmbedder(['cat', 'dog']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(
        embedder: embedder.call,
        store: store,
        chunker: Chunker.paragraphs(),
      );
      await retriever.addText(
        'cat only\n\ncat and dog together\n\ndog only',
        sourceId: 'mix',
      );
      final top = await retriever.retrieve('cat', topK: 2);
      expect(top, hasLength(2));
      expect(top.first.document.text, 'cat only');

      final filtered = await retriever.retrieve('cat', minScore: 0.9);
      expect(filtered, hasLength(1));
      expect(filtered.single.document.text, 'cat only');
    });

    test(
      'returns empty from an empty store without calling the embedder',
      () async {
        final embedder = KeywordEmbedder(['cat']);
        final retriever = Retriever(
          embedder: embedder.call,
          store: InMemoryVectorStore(),
        );
        expect(await retriever.retrieve('anything'), isEmpty);
        expect(embedder.calls, 0);
      },
    );

    test('where restricts retrieval to matching document metadata', () async {
      final embedder = KeywordEmbedder(['dog']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(embedder: embedder.call, store: store);
      await retriever.addText('the dog barks', sourceId: 'a');
      await retriever.addText('the dog runs', sourceId: 'b');

      // Without a filter the best match could come from either source.
      final all = await retriever.retrieve('dog', topK: 2);
      expect(all, hasLength(2));

      // Scoped to source a, only a's chunk can come back.
      final scoped = await retriever.retrieve(
        'dog',
        where: (doc) => doc.metadata['sourceId'] == 'a',
      );
      expect(scoped, hasLength(1));
      expect(scoped.single.document.metadata['sourceId'], 'a');
      expect(scoped.single.document.text, 'the dog barks');
    });
  });

  group('Retriever.buildContext', () {
    /// Adds three one-chunk documents whose retrieval order for the query
    /// 'cat' is: 'cat' (1.0), 'cat dog' (0.7071), 'dog' (0.0).
    Future<Retriever> pipeline(KeywordEmbedder embedder) async {
      final retriever = Retriever(
        embedder: embedder.call,
        store: InMemoryVectorStore(),
      );
      await retriever.addText('cat', sourceId: 'a');
      await retriever.addText('cat dog', sourceId: 'b');
      await retriever.addText('dog', sourceId: 'c');
      return retriever;
    }

    test('joins chunks with the separator in relevance order', () async {
      final retriever = await pipeline(KeywordEmbedder(['cat', 'dog']));
      final context = await retriever.buildContext('cat', topK: 3);
      expect(context, 'cat\n\n---\n\ncat dog\n\n---\n\ndog');
    });

    test('supports a custom separator', () async {
      final retriever = await pipeline(KeywordEmbedder(['cat', 'dog']));
      final context = await retriever.buildContext(
        'cat',
        topK: 3,
        separator: ' | ',
      );
      expect(context, 'cat | cat dog | dog');
    });

    test('stops before a chunk that would exceed maxChars', () async {
      final retriever = await pipeline(KeywordEmbedder(['cat', 'dog']));
      // 'cat' (3) + separator (7) + 'cat dog' (7) = 17; adding the next
      // separator and 'dog' would need 27.
      final context = await retriever.buildContext(
        'cat',
        topK: 3,
        maxChars: 20,
      );
      expect(context, 'cat\n\n---\n\ncat dog');
      expect(context.length, 17);
    });

    test('a chunk fitting maxChars exactly is included', () async {
      final retriever = await pipeline(KeywordEmbedder(['cat', 'dog']));
      final context = await retriever.buildContext(
        'cat',
        topK: 3,
        maxChars: 17,
      );
      expect(context, 'cat\n\n---\n\ncat dog');
    });

    test('truncates the first chunk when it alone exceeds maxChars', () async {
      final retriever = await pipeline(KeywordEmbedder(['cat', 'dog']));
      final context = await retriever.buildContext('cat', topK: 3, maxChars: 2);
      expect(context, 'ca');
    });

    test('returns an empty string when the store is empty', () async {
      final retriever = Retriever(
        embedder: KeywordEmbedder(['cat']).call,
        store: InMemoryVectorStore(),
      );
      expect(await retriever.buildContext('cat'), isEmpty);
    });

    test('where filters the context to matching documents', () async {
      final embedder = KeywordEmbedder(['dog']);
      final retriever = Retriever(
        embedder: embedder.call,
        store: InMemoryVectorStore(),
      );
      await retriever.addText('the dog barks', sourceId: 'a');
      await retriever.addText('the dog runs', sourceId: 'b');

      final context = await retriever.buildContext(
        'dog',
        where: (doc) => doc.metadata['sourceId'] == 'b',
      );
      expect(context, 'the dog runs');
    });
  });

  group('re-adding a source', () {
    test('removes chunks left over from the previous version', () async {
      final embedder = KeywordEmbedder(['one', 'two', 'three']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(
        embedder: embedder.call,
        store: store,
        chunker: Chunker.fixed(maxChars: 12, overlap: 0),
      );
      await retriever.addText('one and two and three', sourceId: 'doc');
      final before = await store.count();
      expect(before, greaterThan(1));

      await retriever.addText('one', sourceId: 'doc');
      expect(await store.count(), 1);

      final results = await retriever.retrieve('three', topK: 5);
      for (final result in results) {
        expect(result.document.text, isNot(contains('three')));
      }
    });

    test('does not touch other sources', () async {
      final embedder = KeywordEmbedder(['alpha', 'beta']);
      final store = InMemoryVectorStore();
      final retriever = Retriever(embedder: embedder.call, store: store);
      await retriever.addText('alpha', sourceId: 'a');
      await retriever.addText('beta', sourceId: 'b');
      await retriever.addText('alpha again', sourceId: 'a');
      final results = await retriever.retrieve('beta', topK: 5);
      expect(results.first.document.metadata['sourceId'], 'b');
    });
  });

  group('buildContext validation', () {
    test('rejects a non-positive maxChars', () async {
      final embedder = KeywordEmbedder(['x']);
      final retriever = Retriever(
        embedder: embedder.call,
        store: InMemoryVectorStore(),
      );
      await expectLater(
        retriever.buildContext('x', maxChars: 0),
        throwsArgumentError,
      );
      await expectLater(
        retriever.buildContext('x', maxChars: -5),
        throwsArgumentError,
      );
    });
  });
}
