## 0.4.0

Settles how the public types compare, which is the last thing that has to be
decided before a 1.0.0: adding or removing value equality afterwards silently
changes how sets and maps behave for anyone already using them.

- `Chunk` now has `==` and `hashCode`, over its text, its range, and its
  metadata. Until now two chunks covering exactly the same span of the same
  source were different objects, so `chunks.toSet()` never collapsed the
  duplicates that overlapping windows produce, and a test could not compare a
  chunker's output against the chunks it expected. Metadata is compared entry
  by entry and each value with its own `==`, so a `List` or `Map` stored as a
  metadata value compares by identity; the hash is order-independent, so two
  equal maps built in a different order still land in the same bucket.
- `Document` and `ScoredChunk` deliberately keep identity equality, and now
  say so in their documentation. A store keeps embeddings as float32, so a
  document read back has slightly different components than the one that was
  written while being the same document: value equality would report those two
  as different and would be wrong more often than it was useful. A document is
  identified by its `id`, which is what the store already deduplicates on.
- Name every export explicitly. The library re-exported whole source files,
  so anything that became public inside one would have joined the API by
  accident, which matters much more once the API is frozen. The exported set
  is unchanged: `Chunk`, `Chunker`, `Document`, `Embedder`,
  `InMemoryVectorStore`, `Retriever`, `ScoredChunk`, `VectorStore`.

## 0.3.1

- Fix `InMemoryVectorStore` aliasing a document's `metadata` map instead of
  copying it. Unlike `embedding`, which was already defensively copied,
  `metadata` was stored as the exact map object handed to `upsert`, so
  mutating that map afterwards, or mutating a document handed back by
  `search` or `retrieve`, silently rewrote data already in the index. This
  was reachable through `Retriever.addText` too, since it builds each
  document's metadata before handing it to the store. `metadata` is now
  copied into an unmodifiable map at insert time, so mutating a returned
  document's metadata throws instead of silently corrupting the store.

## 0.3.0

- Add a `label` callback to `buildContext`. Until now it joined only the raw
  chunk texts, so the model got the passages with no way to say which document
  each came from. Pass `label` to prefix every chunk with a source marker,
  for example `label: (c) => '[${c.document.metadata['sourceId']}]'`; it counts
  against `maxChars` like the rest of the chunk. Left off, the output is
  byte-for-byte what it was before.

## 0.2.2

- Shorten the screenshot description. pub.dev accepts up to 200 characters but
  scores only those under 160, so the previous release published cleanly and
  quietly gave up the documentation points it was meant to earn.

## 0.2.1

- Declare the diagram in `pubspec.yaml` so pub.dev renders it on the package
  page. It was already in the repository and the README, but pub.dev shows only
  what the `screenshots:` field points at, so the page opened with prose where
  the picture should have been.

## 0.2.0

- Add `Retriever.retrieveDiverse`, maximal marginal relevance over a larger
  candidate pool. Similarity alone returns near-duplicates when a source
  repeats itself, so the context window pays for one fact several times while
  the one that answers the question falls below the cut; this picks each next
  chunk for its relevance minus how much it repeats what is already picked.
  `lambda` runs from pure relevance (1.0, identical to `retrieve`) to pure
  diversity (0.0), `fetchK` sets the candidate pool, and results keep their
  query-similarity score. `buildContext` takes `diverse: true` to select the
  same way.

## 0.1.4

- Docs: sharpen the pub.dev description to lead with the value and the terms people search.

## 0.1.3

- `Retriever.retrieve` now takes a `where` predicate, and `buildContext` now
  takes both `minScore` and `where`, forwarded to the store. The metadata
  filter and score threshold were already implemented in the store but could
  not be reached through the retriever's public API.

## 0.1.2

- Docs: tightened the README wording and visuals.

## 0.1.1

- Expand the package description to name what the package does in the
  words people search for. No code changes.

# Changelog

## 0.1.0

Initial release.

- `Chunker.fixed`, `Chunker.paragraphs`, and `Chunker.sentences`, all
  reporting exact source offsets.
- `VectorStore` interface and `InMemoryVectorStore`: cosine similarity over
  float32 vectors with precomputed norms, top-k via a bounded min-heap,
  `minScore` and metadata `where` filters.
- Binary serialization (`toBytes`/`fromBytes`) and, on the VM, file
  persistence via `package:rag_kit/io.dart`.
- `Retriever`: chunk, batch-embed, upsert, `retrieve`, and `buildContext`
  for assembling LLM prompt context.
