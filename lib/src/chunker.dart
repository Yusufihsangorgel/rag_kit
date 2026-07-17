import 'chunk.dart';

const int _tab = 0x09;
const int _carriageReturn = 0x0d;
const int _space = 0x20;
const int _dot = 0x2e;
const int _bang = 0x21;
const int _question = 0x3f;

/// ASCII whitespace: space, tab, line feed, vertical tab, form feed, and
/// carriage return (0x09 through 0x0d).
bool _isSpace(int codeUnit) =>
    codeUnit == _space || (codeUnit >= _tab && codeUnit <= _carriageReturn);

bool _isTerminator(int codeUnit) =>
    codeUnit == _dot || codeUnit == _bang || codeUnit == _question;

/// Splits a source text into [Chunk]s.
///
/// Implement this to plug a custom splitting strategy into [Retriever], or
/// use one of the built-in strategies: [Chunker.fixed], [Chunker.paragraphs],
/// or [Chunker.sentences].
abstract class Chunker {
  /// Allows subclasses to have const constructors.
  const Chunker();

  /// A character-window chunker that snaps cut points to word boundaries.
  ///
  /// Produces chunks of at most [maxChars] characters. Consecutive chunks
  /// share roughly [overlap] characters so that a sentence cut in half by a
  /// window edge is still fully present in one of the two chunks. Cut points
  /// move back to the nearest whitespace, so words are not split unless a
  /// single word is longer than [maxChars], in which case it is cut mid-word.
  ///
  /// Throws an [ArgumentError] if [maxChars] is less than 1, [overlap] is
  /// negative, or [overlap] is not smaller than [maxChars].
  factory Chunker.fixed({int maxChars = 1000, int overlap = 200}) =>
      _FixedChunker(maxChars: maxChars, overlap: overlap);

  /// A chunker that splits on blank lines.
  ///
  /// Each paragraph becomes one chunk. A paragraph longer than [maxChars]
  /// is split further with [Chunker.fixed], using the same [maxChars] and an
  /// overlap of one tenth of it.
  ///
  /// Throws an [ArgumentError] if [maxChars] is less than 1.
  factory Chunker.paragraphs({int maxChars = 2000}) =>
      _ParagraphChunker(maxChars: maxChars);

  /// A chunker that packs whole sentences into chunks of at most [maxChars]
  /// characters.
  ///
  /// Sentences end at `.`, `!`, or `?` followed by whitespace or the end of
  /// the text. This is intentionally simple and has no abbreviation
  /// handling: "e.g. this" splits after "e.g.". For prose without dense
  /// abbreviations it works well; if you need better boundaries, implement
  /// your own [Chunker].
  ///
  /// [overlap] is measured in sentences: each chunk repeats the last
  /// [overlap] sentences of the previous chunk. A single sentence longer
  /// than [maxChars] becomes its own oversized chunk.
  ///
  /// Throws an [ArgumentError] if [maxChars] is less than 1 or [overlap] is
  /// negative.
  factory Chunker.sentences({int maxChars = 1000, int overlap = 1}) =>
      _SentenceChunker(maxChars: maxChars, overlap: overlap);

  /// Splits [text] into chunks.
  ///
  /// Returns an empty list when [text] is empty or contains only whitespace.
  /// Every returned chunk satisfies
  /// `text.substring(chunk.start, chunk.end) == chunk.text`.
  List<Chunk> chunk(String text);
}

class _FixedChunker extends Chunker {
  _FixedChunker({required this.maxChars, required this.overlap}) {
    if (maxChars < 1) {
      throw ArgumentError.value(maxChars, 'maxChars', 'must be at least 1');
    }
    if (overlap < 0) {
      throw ArgumentError.value(overlap, 'overlap', 'must not be negative');
    }
    if (overlap >= maxChars) {
      throw ArgumentError.value(
        overlap,
        'overlap',
        'must be smaller than maxChars ($maxChars)',
      );
    }
  }

  final int maxChars;
  final int overlap;

  @override
  List<Chunk> chunk(String text) {
    final chunks = <Chunk>[];
    final length = text.length;
    var start = 0;
    while (start < length) {
      while (start < length && _isSpace(text.codeUnitAt(start))) {
        start++;
      }
      if (start >= length) break;
      var end = start + maxChars;
      var hardCut = false;
      if (end >= length) {
        end = length;
      } else if (!_isSpace(text.codeUnitAt(end))) {
        // A word straddles the cut point. Move the cut back to the last
        // whitespace inside the window, unless the window is a single word.
        var i = end - 1;
        while (i > start && !_isSpace(text.codeUnitAt(i))) {
          i--;
        }
        if (i > start) {
          end = i;
        } else {
          hardCut = true;
        }
      }
      var chunkEnd = end;
      while (chunkEnd > start && _isSpace(text.codeUnitAt(chunkEnd - 1))) {
        chunkEnd--;
      }
      if (chunkEnd > start) {
        chunks.add(
          Chunk(
            text: text.substring(start, chunkEnd),
            start: start,
            end: chunkEnd,
          ),
        );
      }
      if (end >= length) break;
      start = _nextStart(text, start, end, hardCut);
    }
    return chunks;
  }

  int _nextStart(String text, int start, int end, bool hardCut) {
    var next = end - overlap;
    if (next <= start) {
      next = start + 1;
    }
    if (hardCut) {
      // The current window is a single oversized word; continue cutting it
      // mid-word.
      return next;
    }
    if (!_isSpace(text.codeUnitAt(next)) &&
        !_isSpace(text.codeUnitAt(next - 1))) {
      // The overlap position lands inside a word. Start at that word's
      // beginning so no chunk starts mid-word.
      var wordStart = next;
      while (wordStart > 0 && !_isSpace(text.codeUnitAt(wordStart - 1))) {
        wordStart--;
      }
      if (wordStart > start) {
        next = wordStart;
      } else {
        // The word begins at or before the current chunk start, so moving
        // back would not make progress. Skip ahead to the next word instead.
        while (next < text.length && !_isSpace(text.codeUnitAt(next))) {
          next++;
        }
      }
    }
    return next;
  }
}

class _ParagraphChunker extends Chunker {
  _ParagraphChunker({required this.maxChars})
    : _fallback = _FixedChunker(maxChars: maxChars, overlap: maxChars ~/ 10);

  final int maxChars;
  final _FixedChunker _fallback;

  static final _blankLine = RegExp(r'\n[ \t\r]*\n');

  @override
  List<Chunk> chunk(String text) {
    final chunks = <Chunk>[];
    var cursor = 0;
    for (final match in _blankLine.allMatches(text)) {
      _addParagraph(text, cursor, match.start, chunks);
      cursor = match.end;
    }
    _addParagraph(text, cursor, text.length, chunks);
    return chunks;
  }

  void _addParagraph(String text, int from, int to, List<Chunk> chunks) {
    var start = from;
    while (start < to && _isSpace(text.codeUnitAt(start))) {
      start++;
    }
    var end = to;
    while (end > start && _isSpace(text.codeUnitAt(end - 1))) {
      end--;
    }
    if (start >= end) return;
    if (end - start <= maxChars) {
      chunks.add(
        Chunk(text: text.substring(start, end), start: start, end: end),
      );
      return;
    }
    for (final piece in _fallback.chunk(text.substring(start, end))) {
      chunks.add(
        Chunk(
          text: piece.text,
          start: start + piece.start,
          end: start + piece.end,
        ),
      );
    }
  }
}

class _SentenceChunker extends Chunker {
  _SentenceChunker({required this.maxChars, required this.overlap}) {
    if (maxChars < 1) {
      throw ArgumentError.value(maxChars, 'maxChars', 'must be at least 1');
    }
    if (overlap < 0) {
      throw ArgumentError.value(overlap, 'overlap', 'must not be negative');
    }
  }

  final int maxChars;
  final int overlap;

  @override
  List<Chunk> chunk(String text) {
    final sentences = _split(text);
    if (sentences.isEmpty) return [];
    final chunks = <Chunk>[];
    var first = 0;
    while (first < sentences.length) {
      final start = sentences[first].start;
      var last = first;
      while (last + 1 < sentences.length &&
          sentences[last + 1].end - start <= maxChars) {
        last++;
      }
      final end = sentences[last].end;
      chunks.add(
        Chunk(text: text.substring(start, end), start: start, end: end),
      );
      if (last + 1 >= sentences.length) break;
      var next = last + 1 - overlap;
      if (next <= first) {
        next = first + 1;
      }
      first = next;
    }
    return chunks;
  }

  List<({int start, int end})> _split(String text) {
    final spans = <({int start, int end})>[];
    final length = text.length;
    var cursor = 0;
    var i = 0;
    while (i < length) {
      if (_isTerminator(text.codeUnitAt(i))) {
        var j = i + 1;
        while (j < length && _isTerminator(text.codeUnitAt(j))) {
          j++;
        }
        if (j >= length || _isSpace(text.codeUnitAt(j))) {
          _addSpan(text, cursor, j, spans);
          cursor = j;
        }
        i = j;
      } else {
        i++;
      }
    }
    _addSpan(text, cursor, length, spans);
    return spans;
  }

  void _addSpan(
    String text,
    int from,
    int to,
    List<({int start, int end})> spans,
  ) {
    var start = from;
    while (start < to && _isSpace(text.codeUnitAt(start))) {
      start++;
    }
    var end = to;
    while (end > start && _isSpace(text.codeUnitAt(end - 1))) {
      end--;
    }
    if (start < end) {
      spans.add((start: start, end: end));
    }
  }
}
