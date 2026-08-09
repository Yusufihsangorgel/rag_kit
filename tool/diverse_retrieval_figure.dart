// Draws doc/diverse-retrieval.svg from a retrieval run measured at run time.
//
//   dart run tool/diverse_retrieval_figure.dart
//   rsvg-convert -w 1600 doc/diverse-retrieval.svg -o doc/diverse-retrieval.png
//
// One query, one store, one topK, two selections. Similarity alone spends the
// whole context budget on three paraphrases of the same sentence, and the
// paragraph carrying the rest of the answer stays below the cut. Maximal
// marginal relevance keeps one paraphrase and brings that paragraph up.
//
// The handbook is written for the demonstration. Everything drawn on top of it
// is measured: the two selections come from [Retriever.retrieve] and
// [Retriever.retrieveDiverse] at their default settings, the scores are the
// store's cosine similarities, and the repetition numbers are cosines between
// the selected embeddings, the quantity the selection minimises. The embedder
// is a fixed-vocabulary character-4-gram model, which keeps the run
// deterministic and offline. [main] re-checks every claim the figure makes
// against the run and exits non-zero without touching the file if one of them
// no longer holds.

import 'dart:io';
import 'dart:math' as math;

import 'package:rag_kit/rag_kit.dart';

/// The handbook, in the order `Chunker.paragraphs` yields it. The first three
/// paragraphs say one thing three ways, which is what a real handbook does
/// across its policy page, its FAQ and its onboarding summary.
const paragraphs = [
  'Paid leave is requested in the HR portal, and your manager approves '
      'the request.',
  'Paid leave is requested through the HR portal, where your manager '
      'approves the request.',
  'You request paid leave in the HR portal and your manager approves the '
      'request.',
  'A leave request has to reach your manager two weeks before the first '
      'day off, and more than ten days needs director sign-off.',
  'Unused paid leave carries over until March 31 of the next year, and '
      'whatever is left on that date expires.',
  'Sick days work differently. Tell your team the same morning, and '
      'nothing is filed in advance.',
  'New hires pick an operating system in the onboarding form, and IT '
      'ships the laptop before the start date.',
  'Expenses go through the reimbursement tool. Upload the receipt, choose '
      'a category, and finance pays on the next cycle.',
];

/// A short name for each paragraph, in the same order, for the figure.
const names = [
  'policy page',
  'FAQ entry',
  'onboarding summary',
  'notice window',
  'carry-over rule',
  'sick days',
  'laptops',
  'expenses',
];

/// The paragraph the question needs and plain similarity does not reach.
const withheld = 3;

/// Cosine at which two chosen chunks count as repeating each other.
const repetitionMark = 0.7;

const query = 'how do I request leave?';
const topK = 3;

Future<void> main() async {
  final embedder = _GramEmbedder(paragraphs);
  final retriever = Retriever(
    embedder: embedder.call,
    store: InMemoryVectorStore(),
    chunker: Chunker.paragraphs(),
  );
  await retriever.addText(paragraphs.join('\n\n'), sourceId: 'handbook');

  final plain = await retriever.retrieve(query, topK: topK);
  final diverse = await retriever.retrieveDiverse(query, topK: topK);
  final ranking = await retriever.retrieve(query, topK: paragraphs.length);

  final plainRepetition = _repetition(plain);
  final diverseRepetition = _repetition(diverse);
  final withheldRank = ranking.indexWhere((c) => _index(c) == withheld) + 1;
  final gained = [
    for (final chunk in diverse)
      if (!plain.any((other) => _index(other) == _index(chunk)))
        'the ${names[_index(chunk)]}',
  ];

  final complaints = [
    if (plain.length != topK || diverse.length != topK)
      'both selections have to hold $topK chunks for the columns to line up',
    if (plainRepetition.reduce(math.min) < repetitionMark)
      'similarity alone no longer returns three near-copies '
          '(closest pair per row '
          '${plainRepetition.map(_two).join(', ')})',
    if (diverseRepetition.reduce(math.max) >= repetitionMark)
      'the diverse selection now repeats itself too '
          '(closest pair per row '
          '${diverseRepetition.map(_two).join(', ')})',
    if (withheldRank <= topK)
      'the ${names[withheld]} is inside the top $topK by similarity, so '
          'there is nothing for the second column to recover',
    if (plain.any((c) => _index(c) == withheld))
      'similarity alone now reaches the ${names[withheld]}',
    if (!diverse.any((c) => _index(c) == withheld))
      'the diverse selection no longer reaches the ${names[withheld]}',
    if (gained.isEmpty) 'the two selections agree, so there is nothing to draw',
  ];
  if (complaints.isNotEmpty) {
    stderr.writeln('the run stopped supporting the figure:');
    for (final complaint in complaints) {
      stderr.writeln('  $complaint');
    }
    exitCode = 1;
    return;
  }

  File('doc/diverse-retrieval.svg').writeAsStringSync(
    _svg(plain, diverse, plainRepetition, diverseRepetition, gained),
  );

  stdout.writeln('query                  "$query"');
  stdout.writeln(
    'ranked by similarity   '
    '${ranking.map((c) => names[_index(c)]).join(', ')}',
  );
  stdout.writeln(
    'retrieve               '
    '${plain.map((c) => names[_index(c)]).join(', ')}',
  );
  stdout.writeln(
    'retrieveDiverse        '
    '${diverse.map((c) => names[_index(c)]).join(', ')}',
  );
  stdout.writeln(
    'repetition, plain      ${plainRepetition.map(_two).join(', ')}',
  );
  stdout.writeln(
    'repetition, diverse    ${diverseRepetition.map(_two).join(', ')}',
  );
  stdout.writeln(
    '${names[withheld]}          rank $withheldRank of ${paragraphs.length} '
    'by similarity',
  );
  stdout.writeln('wrote doc/diverse-retrieval.svg');
}

int _index(ScoredChunk chunk) => chunk.document.metadata['chunkIndex']! as int;

String _two(double value) => value.toStringAsFixed(2);

/// For each chunk in [selection], the highest cosine between it and any other
/// chunk in the same selection: how much that pick repeats the rest of the
/// picks, measured the way the selection measures it.
List<double> _repetition(List<ScoredChunk> selection) => [
  for (var i = 0; i < selection.length; i++)
    [
      for (var j = 0; j < selection.length; j++)
        if (i != j)
          _cosine(
            selection[i].document.embedding,
            selection[j].document.embedding,
          ),
    ].fold(0.0, math.max),
];

double _cosine(List<double> a, List<double> b) {
  var dot = 0.0, normA = 0.0, normB = 0.0;
  for (var i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA == 0 || normB == 0) return 0;
  return dot / (math.sqrt(normA) * math.sqrt(normB));
}

/// A lexical embedder over character 4-grams whose vocabulary is fixed at
/// construction. Enough to put paraphrases near each other and to keep the
/// figure reproducible without a network call. A real model reaches further;
/// nothing the figure shows depends on it reaching further.
class _GramEmbedder {
  _GramEmbedder(List<String> corpus) {
    for (final text in corpus) {
      for (final gram in _grams(text)) {
        _vocabulary.putIfAbsent(gram, () => _vocabulary.length);
      }
    }
  }

  final _vocabulary = <String, int>{};

  Future<List<List<double>>> call(List<String> texts) async => [
    for (final text in texts) _vector(text),
  ];

  List<double> _vector(String text) {
    final vector = List<double>.filled(_vocabulary.length, 0);
    for (final gram in _grams(text)) {
      final slot = _vocabulary[gram];
      if (slot != null) vector[slot] += 1;
    }
    var norm = 0.0;
    for (final value in vector) {
      norm += value * value;
    }
    if (norm == 0) return vector;
    norm = math.sqrt(norm);
    for (var i = 0; i < vector.length; i++) {
      vector[i] /= norm;
    }
    return vector;
  }

  static Iterable<String> _grams(String text) sync* {
    final normalised = text
        .toLowerCase()
        .replaceAll(RegExp('[^a-z0-9]+'), ' ')
        .trim();
    for (var i = 0; i + 4 <= normalised.length; i++) {
      yield normalised.substring(i, i + 4);
    }
  }
}

String _svg(
  List<ScoredChunk> plain,
  List<ScoredChunk> diverse,
  List<double> plainRepetition,
  List<double> diverseRepetition,
  List<String> gained,
) {
  const w = 940.0, h = 430.0;
  const colW = 410.0, rowH = 58.0, firstRow = 132.0;

  String column(
    double x,
    String title,
    List<ScoredChunk> selection,
    List<double> repetition,
  ) {
    final rows = <String>[];
    for (var i = 0; i < selection.length; i++) {
      final chunk = selection[i];
      final repeats = repetition[i] >= repetitionMark;
      final y = firstRow + i * rowH;
      rows.add(
        '''
<rect x="$x" y="$y" width="$colW" height="48" rx="6"
      fill="${repeats ? '#fef2f2' : '#f0fdf4'}"
      stroke="${repeats ? '#fca5a5' : '#a7f3d0'}"/>
<text x="${x + 14}" y="${y + 19}" font-size="13" font-weight="600"
      fill="#111827">${_esc(names[_index(chunk)])}</text>
<text x="${x + colW - 14}" y="${y + 19}" font-size="12.5" text-anchor="end"
      font-family="monospace"
      fill="#6b7280">${chunk.score.toStringAsFixed(3)}</text>
<text x="${x + 14}" y="${y + 38}" font-size="12" font-family="monospace"
      fill="${repeats ? '#b91c1c' : '#047857'}">${_esc(_excerpt(chunk))}</text>''',
      );
    }
    final closest = repetition.reduce(math.max);
    return '''
<text x="$x" y="116" font-size="14" font-weight="600" font-family="monospace"
      fill="#111827">${_esc(title)}</text>
${rows.join('\n')}
<text x="$x" y="${firstRow + 3 * rowH + 12}" font-size="13" fill="#6b7280"
  >closest pair among the three <tspan font-family="monospace"
   font-weight="600"
   fill="${closest >= repetitionMark ? '#b91c1c' : '#047857'}"
  >${_two(closest)}</tspan></text>''';
  }

  return '''
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w.toInt()} ${h.toInt()}"
     font-family="-apple-system, Segoe UI, Roboto, sans-serif">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="40" y="36" font-size="17" font-weight="600" fill="#111827"
  >Three chunks of context, and similarity alone spends all three on one
   sentence</text>
<text x="40" y="60" font-size="13" fill="#6b7280"
  >Query <tspan font-family="monospace"
   fill="#111827">${_esc(query)}</tspan> against an eight-paragraph handbook
   whose leave policy is written out three times</text>
<text x="40" y="80" font-size="12.5" fill="#9ca3af"
  >A row turns red when it repeats another chosen chunk at
   ${_two(repetitionMark)} cosine or above. The number on the right of each row
   is its similarity to the query.</text>
${column(40, 'retrieve(topK: $topK)', plain, plainRepetition)}
${column(490, 'retrieveDiverse(topK: $topK)', diverse, diverseRepetition)}
<rect x="40" y="${h - 76}" width="860" height="58" rx="8" fill="#f9fafb"
      stroke="#e5e7eb"/>
<text x="58" y="${h - 52}" font-size="13" fill="#b91c1c" font-weight="600"
  >Left: the model is told the same thing three times, and the
   ${_esc(names[withheld])} it needs never arrives.</text>
<text x="58" y="${h - 30}" font-size="13" fill="#047857" font-weight="600"
  >Right: one of the three is kept, and ${_esc(_list(gained))} take the seats
   the copies were using.</text>
</svg>
''';
}

String _list(List<String> items) => items.length < 2
    ? items.join()
    : '${items.take(items.length - 1).join(', ')} and ${items.last}';

String _excerpt(ScoredChunk chunk) {
  final text = chunk.document.text;
  return text.length <= 52 ? text : '${text.substring(0, 51)}…';
}

String _esc(String text) => text
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
