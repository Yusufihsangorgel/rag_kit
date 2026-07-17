import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

/// Asserts that every chunk's text is exactly the source slice its offsets
/// point at.
void expectOffsetsMatch(String source, List<Chunk> chunks) {
  for (final chunk in chunks) {
    expect(
      source.substring(chunk.start, chunk.end),
      chunk.text,
      reason: 'chunk [${chunk.start}, ${chunk.end}) must match its source',
    );
  }
}

void main() {
  group('Chunker.fixed', () {
    test('returns no chunks for empty input', () {
      expect(Chunker.fixed().chunk(''), isEmpty);
    });

    test('returns no chunks for whitespace-only input', () {
      expect(Chunker.fixed().chunk('  \n\t  \n'), isEmpty);
    });

    test('returns a single chunk for short input', () {
      final chunks = Chunker.fixed(
        maxChars: 100,
        overlap: 10,
      ).chunk('hello world');
      expect(chunks, hasLength(1));
      expect(chunks.first.text, 'hello world');
      expect(chunks.first.start, 0);
      expect(chunks.first.end, 11);
    });

    test('trims surrounding whitespace and keeps offsets exact', () {
      const source = '  hello  ';
      final chunks = Chunker.fixed().chunk(source);
      expect(chunks, hasLength(1));
      expect(chunks.first.text, 'hello');
      expect(chunks.first.start, 2);
      expect(chunks.first.end, 7);
      expectOffsetsMatch(source, chunks);
    });

    test('every chunk text matches its source offsets', () {
      final source = List.generate(120, (i) => 'word$i').join(' ');
      final chunks = Chunker.fixed(maxChars: 80, overlap: 20).chunk(source);
      expect(chunks.length, greaterThan(5));
      expectOffsetsMatch(source, chunks);
    });

    test('chunks never exceed maxChars', () {
      final source = List.generate(200, (i) => 'w$i').join(' ');
      final chunks = Chunker.fixed(maxChars: 50, overlap: 10).chunk(source);
      for (final chunk in chunks) {
        expect(chunk.text.length, lessThanOrEqualTo(50));
      }
    });

    test('cut points snap to word boundaries', () {
      final source = List.generate(100, (i) => 'alpha beta gamma').join(' ');
      final chunks = Chunker.fixed(maxChars: 70, overlap: 15).chunk(source);
      for (final chunk in chunks) {
        // No chunk starts or ends in the middle of a word.
        expect(
          chunk.start == 0 || source[chunk.start - 1] == ' ',
          isTrue,
          reason: 'chunk at ${chunk.start} starts mid-word',
        );
        expect(
          chunk.end == source.length || source[chunk.end] == ' ',
          isTrue,
          reason: 'chunk ending at ${chunk.end} ends mid-word',
        );
      }
    });

    test('consecutive chunks overlap', () {
      final source = List.generate(80, (i) => 'token$i').join(' ');
      final chunks = Chunker.fixed(maxChars: 60, overlap: 20).chunk(source);
      expect(chunks.length, greaterThan(2));
      for (var i = 1; i < chunks.length; i++) {
        final shared = chunks[i - 1].end - chunks[i].start;
        expect(
          shared,
          greaterThanOrEqualTo(1),
          reason: 'chunks $i-1 and $i do not overlap',
        );
        // Snapping to a word start can extend the overlap by at most one
        // word ('tokenNN' is at most 7 characters).
        expect(shared, lessThanOrEqualTo(20 + 7));
      }
    });

    test('a word longer than maxChars is hard-cut', () {
      final source = 'a' * 25;
      final chunks = Chunker.fixed(maxChars: 10, overlap: 0).chunk(source);
      expect(chunks.map((c) => c.text).toList(), ['a' * 10, 'a' * 10, 'a' * 5]);
      expect(chunks.map((c) => c.start).toList(), [0, 10, 20]);
      expectOffsetsMatch(source, chunks);
    });

    test('hard-cut chunks overlap when overlap is set', () {
      final source = 'b' * 30;
      final chunks = Chunker.fixed(maxChars: 10, overlap: 4).chunk(source);
      for (var i = 1; i < chunks.length; i++) {
        expect(chunks[i].start, chunks[i - 1].end - 4);
      }
      expectOffsetsMatch(source, chunks);
    });

    test('overlap larger than a snapped chunk still makes progress', () {
      const source = 'abc def ghi jkl mno pqr stu vwx yz';
      final chunks = Chunker.fixed(maxChars: 10, overlap: 8).chunk(source);
      expect(chunks, isNotEmpty);
      // Terminates, covers the tail, and never starts mid-word.
      expect(chunks.last.end, source.length);
      for (final chunk in chunks) {
        expect(chunk.start == 0 || source[chunk.start - 1] == ' ', isTrue);
      }
      expectOffsetsMatch(source, chunks);
    });

    test('single word input produces one exact chunk', () {
      final chunks = Chunker.fixed(maxChars: 10, overlap: 2).chunk('hello');
      expect(chunks, hasLength(1));
      expect(chunks.first.text, 'hello');
      expect(chunks.first.start, 0);
      expect(chunks.first.end, 5);
    });

    test('throws when maxChars is less than 1', () {
      expect(() => Chunker.fixed(maxChars: 0), throwsArgumentError);
    });

    test('throws when overlap is negative', () {
      expect(() => Chunker.fixed(overlap: -1), throwsArgumentError);
    });

    test('throws when overlap is not smaller than maxChars', () {
      expect(
        () => Chunker.fixed(maxChars: 10, overlap: 10),
        throwsArgumentError,
      );
    });
  });

  group('Chunker.paragraphs', () {
    test('returns no chunks for empty input', () {
      expect(Chunker.paragraphs().chunk(''), isEmpty);
      expect(Chunker.paragraphs().chunk('\n\n\n'), isEmpty);
    });

    test('splits on blank lines with exact offsets', () {
      const source = 'First paragraph.\n\nSecond one here.\n\nThird.';
      final chunks = Chunker.paragraphs().chunk(source);
      expect(chunks.map((c) => c.text).toList(), [
        'First paragraph.',
        'Second one here.',
        'Third.',
      ]);
      expect(chunks[1].start, source.indexOf('Second'));
      expect(chunks[2].start, source.indexOf('Third'));
      expectOffsetsMatch(source, chunks);
    });

    test('handles repeated blank lines and surrounding whitespace', () {
      const source = '\n\n  one  \n\n\n\n  two  \n\n';
      final chunks = Chunker.paragraphs().chunk(source);
      expect(chunks.map((c) => c.text).toList(), ['one', 'two']);
      expectOffsetsMatch(source, chunks);
    });

    test('handles CRLF blank lines', () {
      const source = 'alpha\r\n\r\nbeta';
      final chunks = Chunker.paragraphs().chunk(source);
      expect(chunks.map((c) => c.text).toList(), ['alpha', 'beta']);
      expectOffsetsMatch(source, chunks);
    });

    test('keeps single newlines inside a paragraph', () {
      const source = 'line one\nline two\n\nnext paragraph';
      final chunks = Chunker.paragraphs().chunk(source);
      expect(chunks, hasLength(2));
      expect(chunks.first.text, 'line one\nline two');
      expectOffsetsMatch(source, chunks);
    });

    test('splits a long paragraph with the fixed fallback', () {
      final long = List.generate(60, (i) => 'word$i').join(' ');
      final source = 'short intro\n\n$long\n\nshort outro';
      final chunks = Chunker.paragraphs(maxChars: 100).chunk(source);
      expect(chunks.length, greaterThan(3));
      expect(chunks.first.text, 'short intro');
      expect(chunks.last.text, 'short outro');
      final longStart = source.indexOf(long);
      final longEnd = longStart + long.length;
      for (final chunk in chunks.sublist(1, chunks.length - 1)) {
        expect(chunk.text.length, lessThanOrEqualTo(100));
        expect(chunk.start, greaterThanOrEqualTo(longStart));
        expect(chunk.end, lessThanOrEqualTo(longEnd));
      }
      expectOffsetsMatch(source, chunks);
    });

    test('single paragraph without blank lines is one chunk', () {
      const source = 'just one paragraph here';
      final chunks = Chunker.paragraphs().chunk(source);
      expect(chunks, hasLength(1));
      expect(chunks.first.text, source);
    });

    test('throws when maxChars is less than 1', () {
      expect(() => Chunker.paragraphs(maxChars: 0), throwsArgumentError);
    });
  });

  group('Chunker.sentences', () {
    test('returns no chunks for empty input', () {
      expect(Chunker.sentences().chunk(''), isEmpty);
      expect(Chunker.sentences().chunk('   '), isEmpty);
    });

    test('splits on sentence terminators with exact offsets', () {
      const source = 'One two. Three four! Five six?';
      final chunks = Chunker.sentences(maxChars: 12, overlap: 0).chunk(source);
      expect(chunks.map((c) => c.text).toList(), [
        'One two.',
        'Three four!',
        'Five six?',
      ]);
      expect(chunks[1].start, source.indexOf('Three'));
      expect(chunks[2].start, source.indexOf('Five'));
      expectOffsetsMatch(source, chunks);
    });

    test('packs multiple sentences into one chunk up to maxChars', () {
      const source = 'One two. Three four! Five six?';
      final chunks = Chunker.sentences(maxChars: 1000).chunk(source);
      expect(chunks, hasLength(1));
      expect(chunks.first.text, source);
    });

    test('overlap repeats the trailing sentence of the previous chunk', () {
      const source = 'Aaaa bbbb. Cccc dddd. Eeee ffff. Gggg hhhh.';
      // Each sentence is 10 characters; two sentences plus the space
      // between them are 21 characters.
      final chunks = Chunker.sentences(maxChars: 21, overlap: 1).chunk(source);
      expect(chunks.map((c) => c.text).toList(), [
        'Aaaa bbbb. Cccc dddd.',
        'Cccc dddd. Eeee ffff.',
        'Eeee ffff. Gggg hhhh.',
      ]);
      expectOffsetsMatch(source, chunks);
    });

    test('text without a terminator is a single chunk', () {
      const source = 'no terminator at all';
      final chunks = Chunker.sentences(maxChars: 1000).chunk(source);
      expect(chunks, hasLength(1));
      expect(chunks.first.text, source);
    });

    test('consecutive terminators stay in one sentence', () {
      const source = 'Really?! Yes. ';
      final chunks = Chunker.sentences(maxChars: 9, overlap: 0).chunk(source);
      expect(chunks.map((c) => c.text).toList(), ['Really?!', 'Yes.']);
      expectOffsetsMatch(source, chunks);
    });

    test('splitting is naive about abbreviations, as documented', () {
      const source = 'See e.g. the docs.';
      final chunks = Chunker.sentences(maxChars: 9, overlap: 0).chunk(source);
      expect(chunks.first.text, 'See e.g.');
      expectOffsetsMatch(source, chunks);
    });

    test('a sentence longer than maxChars becomes its own chunk', () {
      const source =
          'Tiny. This single sentence is much too long to fit. '
          'End.';
      final chunks = Chunker.sentences(maxChars: 10, overlap: 0).chunk(source);
      expect(
        chunks.map((c) => c.text),
        contains('This single sentence is much too long to fit.'),
      );
      expectOffsetsMatch(source, chunks);
    });

    test('throws when maxChars is less than 1', () {
      expect(() => Chunker.sentences(maxChars: 0), throwsArgumentError);
    });

    test('throws when overlap is negative', () {
      expect(() => Chunker.sentences(overlap: -1), throwsArgumentError);
    });
  });

  group('surrogate pairs at cut points', () {
    test('hard cuts never split an emoji', () {
      // 't' + emoji repeated: no whitespace, so every window is a hard cut.
      final source = List.filled(300, 't\u{1F600}').join();
      for (final chunker in [
        Chunker.fixed(maxChars: 7, overlap: 0),
        Chunker.fixed(maxChars: 8, overlap: 3),
        Chunker.paragraphs(maxChars: 9),
      ]) {
        final chunks = chunker.chunk(source);
        expectOffsetsMatch(source, chunks);
        for (final chunk in chunks) {
          final units = chunk.text.codeUnits;
          expect(
            units.first & 0xFC00,
            isNot(0xDC00),
            reason: 'chunk starts with an orphan low surrogate',
          );
          expect(
            units.last & 0xFC00,
            isNot(0xD800),
            reason: 'chunk ends with an orphan high surrogate',
          );
        }
      }
    });

    test('CJK text with an emoji survives a store roundtrip', () async {
      final cjk = '\u6c49\u5b57' * 40;
      final source = '$cjk\u{1F680}$cjk';
      final chunks = Chunker.fixed(maxChars: 50, overlap: 10).chunk(source);
      final store = InMemoryVectorStore();
      await store.upsert([
        for (var i = 0; i < chunks.length; i++)
          Document(
            id: 'c$i',
            text: chunks[i].text,
            embedding: [i.toDouble(), 1],
            metadata: const {},
          ),
      ]);
      final restored = InMemoryVectorStore.fromBytes(store.toBytes());
      final results = await restored.search([0, 1], topK: chunks.length);
      final byId = {for (final r in results) r.document.id: r.document.text};
      for (var i = 0; i < chunks.length; i++) {
        expect(
          byId['c$i'],
          chunks[i].text,
          reason: 'persisted text changed for chunk $i',
        );
      }
    });
  });
}
