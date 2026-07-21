import 'dart:math' as math;

import 'chunker.dart';
import 'document.dart';
import 'embedder.dart';
import 'scored_chunk.dart';
import 'vector_store.dart';

/// Wires a [Chunker], an [Embedder], and a [VectorStore] into a retrieval
/// pipeline: add texts once, then retrieve the passages most relevant to a
/// query.
class Retriever {
  /// Creates a retriever.
  ///
  /// [chunker] defaults to `Chunker.fixed()`.
  Retriever({required this.embedder, required this.store, Chunker? chunker})
    : chunker = chunker ?? Chunker.fixed();

  /// Embeds batches of chunk texts and single queries.
  final Embedder embedder;

  /// Holds the embedded chunks.
  final VectorStore store;

  /// Splits added texts into chunks.
  final Chunker chunker;

  int _autoSourceId = 0;

  /// Chunks [text], embeds all chunks in a single [embedder] call, and
  /// upserts them into [store].
  ///
  /// Documents get the id `'$sourceId#<chunkIndex>'`. When [sourceId] is
  /// omitted, an id of the form `text-<n>` is generated (avoid handing out
  /// your own ids of that form, or they can collide with generated ones);
  /// pass an explicit [sourceId] when you need stable ids. Re-adding a text
  /// under the same [sourceId] replaces it completely: chunks left over
  /// from the previous version are removed before the new ones are stored.
  ///
  /// [metadata] is copied onto every produced document. The keys
  /// `sourceId`, `chunkIndex`, `chunkStart`, and `chunkEnd` are reserved
  /// and always overwritten with the chunk's position information; keys
  /// set by a custom [Chunker] on its chunks take precedence over
  /// [metadata] entries with the same name.
  ///
  /// Does nothing (and does not call the embedder) when [text] chunks to
  /// nothing, for example when it is empty or whitespace. Throws a
  /// [StateError] if the embedder returns the wrong number of vectors.
  Future<void> addText(
    String text, {
    String? sourceId,
    Map<String, Object?> metadata = const {},
  }) async {
    final chunks = chunker.chunk(text);
    if (chunks.isEmpty) return;
    final source = sourceId ?? 'text-${_autoSourceId++}';
    final embeddings = await embedder([for (final c in chunks) c.text]);
    if (embeddings.length != chunks.length) {
      throw StateError(
        'Embedder returned ${embeddings.length} embeddings for '
        '${chunks.length} texts.',
      );
    }
    // Remove chunks of a previous version of this source only after the
    // embedder succeeded, so a failed re-add never destroys existing data.
    await store.removeWhere(
      (document) => document.metadata['sourceId'] == source,
    );
    final documents = <Document>[
      for (var i = 0; i < chunks.length; i++)
        Document(
          id: '$source#$i',
          text: chunks[i].text,
          embedding: embeddings[i],
          metadata: {
            ...metadata,
            ...chunks[i].metadata,
            'sourceId': source,
            'chunkIndex': i,
            'chunkStart': chunks[i].start,
            'chunkEnd': chunks[i].end,
          },
        ),
    ];
    await store.upsert(documents);
  }

  /// Embeds [query] and returns the most similar stored chunks, best first.
  ///
  /// Returns an empty list without calling the embedder when the store is
  /// empty. See [VectorStore.search] for [topK], [minScore] and [where].
  ///
  /// [where] restricts the search to documents for which it returns true, using
  /// each document's [Document.metadata]. It runs before scoring, so filtered
  /// documents cost no similarity computation. Use it to scope a query to one
  /// source, language, tenant, or any metadata field you set when adding text.
  Future<List<ScoredChunk>> retrieve(
    String query, {
    int topK = 5,
    double? minScore,
    bool Function(Document document)? where,
  }) async {
    if (await store.count() == 0) return const [];
    final embeddings = await embedder([query]);
    if (embeddings.length != 1) {
      throw StateError(
        'Embedder returned ${embeddings.length} embeddings for 1 text.',
      );
    }
    return store.search(
      embeddings.first,
      topK: topK,
      minScore: minScore,
      where: where,
    );
  }

  /// Retrieves chunks that are relevant to [query] and unlike each other.
  ///
  /// Plain [retrieve] returns the [topK] most similar chunks, and when a
  /// source repeats itself those are often near-duplicates: the context window
  /// fills with the same sentence three times while the fact that would have
  /// answered the question sits just below the cut. This runs maximal marginal
  /// relevance over a larger candidate pool, choosing each next chunk for how
  /// relevant it is minus how much it repeats what is already chosen.
  ///
  /// [fetchK] is the pool pulled by similarity before the selection runs. It
  /// defaults to four times [topK] and is raised to [topK] if smaller, since
  /// the selection can only choose among what it is handed. [lambda] balances
  /// the two halves: 1.0 is pure relevance and gives the same result as
  /// [retrieve], 0.0 is pure diversity, and the default 0.5 splits it.
  /// [minScore] and [where] are forwarded to the store exactly as in
  /// [retrieve].
  ///
  /// Results keep their query-similarity score, not the internal selection
  /// score, so they read the same as [retrieve]'s; they come back in selection
  /// order, most relevant first.
  ///
  /// Throws [ArgumentError] if [topK] is below 1 or [lambda] is outside 0..1.
  Future<List<ScoredChunk>> retrieveDiverse(
    String query, {
    int topK = 5,
    int? fetchK,
    double lambda = 0.5,
    double? minScore,
    bool Function(Document document)? where,
  }) async {
    if (topK < 1) {
      throw ArgumentError.value(topK, 'topK', 'must be at least 1');
    }
    if (lambda < 0 || lambda > 1) {
      throw ArgumentError.value(lambda, 'lambda', 'must be between 0 and 1');
    }

    final pool = fetchK == null ? topK * 4 : math.max(fetchK, topK);
    final candidates = await retrieve(
      query,
      topK: pool,
      minScore: minScore,
      where: where,
    );
    if (candidates.length <= topK) return candidates;

    final selected = <ScoredChunk>[];
    final remaining = [...candidates];
    while (selected.length < topK && remaining.isNotEmpty) {
      var bestIndex = 0;
      var bestScore = double.negativeInfinity;
      for (var i = 0; i < remaining.length; i++) {
        final candidate = remaining[i];
        // How much this candidate repeats the closest chunk already chosen.
        // Nothing is chosen on the first pass, so the term drops out and the
        // most relevant candidate wins, as it should.
        var repetition = 0.0;
        if (selected.isNotEmpty) {
          repetition = double.negativeInfinity;
          for (final chosen in selected) {
            final similarity = _cosine(
              candidate.document.embedding,
              chosen.document.embedding,
            );
            if (similarity > repetition) repetition = similarity;
          }
        }
        final score = lambda * candidate.score - (1 - lambda) * repetition;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = i;
        }
      }
      selected.add(remaining.removeAt(bestIndex));
    }
    return selected;
  }

  /// Cosine similarity between two embeddings, 0.0 when either is degenerate.
  static double _cosine(List<double> a, List<double> b) {
    if (a.length != b.length) return 0;
    var dot = 0.0;
    var normA = 0.0;
    var normB = 0.0;
    for (var i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA == 0 || normB == 0) return 0;
    return dot / (math.sqrt(normA) * math.sqrt(normB));
  }

  /// Retrieves the chunks most relevant to [query] and joins their texts
  /// with [separator] into a single string, ready to paste into an LLM
  /// prompt.
  ///
  /// When [maxChars] is given, chunks are added whole until the next chunk
  /// would push the result past the limit. As an exception, if the very
  /// first chunk is already longer than [maxChars] it is cut to [maxChars]
  /// so the result is never empty when there are results. Chunks arrive in
  /// relevance order, so what gets dropped is always the least relevant
  /// tail.
  ///
  /// [minScore] and [where] are forwarded to [retrieve], so the context can be
  /// scored-thresholded and metadata-filtered the same way.
  ///
  /// Set [diverse] to select the chunks with [retrieveDiverse] instead, which
  /// keeps near-duplicates from eating the budget; [lambda] tunes that the
  /// same way it does there and is ignored when [diverse] is false.
  ///
  /// [label] attaches a source marker to each chunk so the model can cite where
  /// a passage came from. When given, each chunk is prefixed with the label it
  /// returns and a newline, so `label: (c) => '[${c.document.metadata['sourceId']}]'`
  /// turns a chunk into `[handbook]\n<text>`. The label counts against
  /// [maxChars] like the rest of the chunk. Left null, the output is exactly
  /// the joined chunk texts.
  Future<String> buildContext(
    String query, {
    int topK = 5,
    int? maxChars,
    String separator = '\n\n---\n\n',
    double? minScore,
    bool Function(Document document)? where,
    bool diverse = false,
    double lambda = 0.5,
    String Function(ScoredChunk chunk)? label,
  }) async {
    if (maxChars != null && maxChars < 1) {
      throw ArgumentError.value(maxChars, 'maxChars', 'must be at least 1');
    }
    final results = diverse
        ? await retrieveDiverse(
            query,
            topK: topK,
            lambda: lambda,
            minScore: minScore,
            where: where,
          )
        : await retrieve(query, topK: topK, minScore: minScore, where: where);
    final buffer = StringBuffer();
    for (final result in results) {
      final text = result.document.text;
      final entry = label == null ? text : '${label(result)}\n$text';
      if (buffer.isEmpty) {
        if (maxChars != null && entry.length > maxChars) {
          buffer.write(entry.substring(0, maxChars));
          break;
        }
        buffer.write(entry);
      } else {
        if (maxChars != null &&
            buffer.length + separator.length + entry.length > maxChars) {
          break;
        }
        buffer
          ..write(separator)
          ..write(entry);
      }
    }
    return buffer.toString();
  }
}
