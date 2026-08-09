## 1.0.3

- **Show how to swap the two parts that are worth swapping.** `Chunker` and
  `VectorStore` have been interfaces since the start, but nothing demonstrated
  what to put behind them, and the built-ins read as the only option.
  `example/with_vector_kit.dart` is a complete `VectorStore` over
  `vector_kit`'s `VectorMatrix.topKCosine` (upsert, filtered search, minScore,
  removeWhere), and the README now points at it along with the token chunker
  that lives in `hf_tokenizers`' examples.
- No runtime dependency was added and none will be: `vector_kit` is a dev
  dependency used by that one example file. rag_kit still resolves with
  nothing behind it.
- The README says what the token chunker buys, with a number rather than an
  argument. On one mixed English-and-Japanese paragraph under a 24-token
  budget, the chunks ran between 2.2 and 4.8 characters per token; no single
  `maxChars` is right across that spread, which is what the note on
  `Chunker.fixed` has always claimed without showing.

## 1.0.2

- Say plainly that `Chunker.fixed` counts characters while an embedding model
  limits tokens, and that nothing converts between them. The ratio is not a
  constant (English prose runs near four characters per token, CJK can
  approach one), and a chunk over the limit is truncated by the model rather
  than rejected: the embedding still arrives, computed from part of the text,
  and retrieval quality drops with nothing raised to explain it. Anyone who
  copied their model's token limit into `maxChars` was reading the parameter
  as something it is not. Documentation only.

## 1.0.1

- **Fix a rejected re-index deleting the source it was replacing.**
  `Retriever.addText` removed a source's stored chunks before writing the new
  ones. The removal was placed after the embedder call so that an embedder
  failure could not destroy anything, but the write itself can also fail:
  swap in an embedder of a different dimension and `VectorStore.upsert`
  throws, leaving the old chunks deleted and the new ones never stored. That
  is precisely the case where the loss costs most: a user part-way through
  re-indexing a corpus under a new model, told by the code comment that
  "a failed re-add never destroys existing data".

  The write now comes first. Chunk ids are positional, so upserting replaces
  the chunks the new version still has, and a second pass removes only the
  tail a shorter version leaves behind. A re-add that the store rejects now
  changes nothing.

  Two regression tests cover it: a rejected re-add keeps the stored text
  retrievable, and a source re-added with fewer chunks drops its tail. The
  first fails against the old ordering.

## 1.0.0

The API is stable. This release makes no code changes; it freezes the surface
after an adversarial pass that ran the package rather than reading it, and
everything held.

Verified by execution: re-adding text under the same `sourceId` removes the old
chunks completely before storing the new ones; an embedder that returns the
wrong number of vectors throws a `StateError`; retrieving from an empty store
returns nothing rather than crashing; the `InMemoryVectorStore` round-trips
through `toBytes`/`fromBytes` with metadata intact; a query whose dimension does
not match the store, and a non-positive `topK`, both throw `ArgumentError`. A
`Document` deliberately holds the embedding list you give it (documented: do not
mutate it after handing it to a store), and the store copies on `upsert` and
returns unmodifiable views, which keeps the stored data isolated.

Every public type is `final` (`Chunker`, `Embedder`, `VectorStore` stay
implementable, since that is how you extend it), the barrel names what it
exports, and there are no runtime dependencies; it is pure Dart.

## 0.5.1

- Add `example/README.md`, which pub.dev renders on the package's Example tab
  (it was empty before). It walks through `rag_kit_example.dart` end to end
  (index two sources, retrieve, scope a query to one source with a metadata
  filter, and build a prompt-ready context) with the real output, and points
  at `semantic_demo.dart` for the model-backed version. Docs only.

## 0.5.0

- Fix a hazard that 0.4.0 introduced. Giving `Chunk` value equality made
  `hashCode` read the metadata map, but the constructor still stored the
  caller's map by reference, so a chunk could fall out of a `Set` that already
  held it: put a chunk in a set, mutate the map you passed to its constructor,
  and `set.contains(chunk)` is false for that very instance, with no error.
  This is the same aliasing class that 0.3.1 fixed inside the store, arriving
  through a different door. `Chunk` now copies metadata into an unmodifiable
  map: mutating the map you passed in cannot reach the chunk, and mutating
  the map you get back throws instead of corrupting a hash. An empty map still
  costs nothing.
- Mark the five leaf classes `final`: `Chunk`, `Document`, `ScoredChunk`,
  `InMemoryVectorStore` and `Retriever`. `Chunker` and `VectorStore` stay open,
  because those are the documented extension points. Adding `final` cannot be
  done after 1.0.0 without a major version, while removing it later is free,
  and leaving a type with `==` open lets a subclass compare equal to its base
  asymmetrically.
- Stop promising, on the `VectorStore` interface, what only the in-memory
  implementation can deliver. `search`'s documentation said it returns the
  documents "most similar" to the query and that `where` "runs before scoring,
  so filtered documents cost no similarity computation". Both are true of
  `InMemoryVectorStore` and neither can be guaranteed by an approximate index,
  which is the first item on this package's own roadmap. The interface now
  describes what an implementation must do, and names `InMemoryVectorStore`
  where the stronger guarantee actually holds.

## 0.4.0

Settles how the public types compare, which is the last thing that has to be
decided before a 1.0.0: adding or removing value equality afterwards silently
changes how sets and maps behave for anyone already using them.

- `Chunk` now has `==` and `hashCode`, over its text, its range, and its
  metadata. Until now two chunks covering exactly the same span of the same
  source were different objects. `chunks.toSet()` never collapsed the
  duplicates that overlapping windows produce, and a test could not compare a
  chunker's output against the chunks it expected. Metadata is compared entry
  by entry and each value with its own `==`, which makes a `List` or `Map`
  stored as a metadata value compare by identity; the hash is
  order-independent, and two equal maps built in a different order still land
  in the same bucket.
- `Document` and `ScoredChunk` deliberately keep identity equality, and now
  say so in their documentation. A store keeps embeddings as float32, so a
  document read back has slightly different components than the one that was
  written while being the same document: value equality would report those two
  as different and would be wrong more often than it was useful. A document is
  identified by its `id`, which is what the store already deduplicates on.
- Name every export explicitly. The library re-exported whole source files;
  anything that became public inside one would have joined the API by
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
  copied into an unmodifiable map at insert time; mutating a returned
  document's metadata throws instead of silently corrupting the store.

## 0.3.0

- Add a `label` callback to `buildContext`. Until now it joined only the raw
  chunk texts. The model got the passages with no way to say which document
  each came from. Pass `label` to prefix every chunk with a source marker,
  for example `label: (c) => '[${c.document.metadata['sourceId']}]'`; it counts
  against `maxChars` like the rest of the chunk. Left off, the output is
  byte-for-byte what it was before.

## 0.2.2

- Shorten the screenshot description. pub.dev accepts up to 200 characters but
  scores only those under 160. The previous release published cleanly and
  quietly gave up the documentation points it was meant to earn.

## 0.2.1

- Declare the diagram in `pubspec.yaml` so pub.dev renders it on the package
  page. It was already in the repository and the README, but pub.dev shows only
  what the `screenshots:` field points at. The page opened with prose where
  the picture should have been.

## 0.2.0

- Add `Retriever.retrieveDiverse`, maximal marginal relevance over a larger
  candidate pool. Similarity alone returns near-duplicates when a source
  repeats itself: the context window pays for one fact several times while
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
