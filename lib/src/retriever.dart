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
  /// empty. See [VectorStore.search] for [topK] and [minScore].
  Future<List<ScoredChunk>> retrieve(
    String query, {
    int topK = 5,
    double? minScore,
  }) async {
    if (await store.count() == 0) return const [];
    final embeddings = await embedder([query]);
    if (embeddings.length != 1) {
      throw StateError(
        'Embedder returned ${embeddings.length} embeddings for 1 text.',
      );
    }
    return store.search(embeddings.first, topK: topK, minScore: minScore);
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
  Future<String> buildContext(
    String query, {
    int topK = 5,
    int? maxChars,
    String separator = '\n\n---\n\n',
  }) async {
    if (maxChars != null && maxChars < 1) {
      throw ArgumentError.value(maxChars, 'maxChars', 'must be at least 1');
    }
    final results = await retrieve(query, topK: topK);
    final buffer = StringBuffer();
    for (final result in results) {
      final text = result.document.text;
      if (buffer.isEmpty) {
        if (maxChars != null && text.length > maxChars) {
          buffer.write(text.substring(0, maxChars));
          break;
        }
        buffer.write(text);
      } else {
        if (maxChars != null &&
            buffer.length + separator.length + text.length > maxChars) {
          break;
        }
        buffer
          ..write(separator)
          ..write(text);
      }
    }
    return buffer.toString();
  }
}
