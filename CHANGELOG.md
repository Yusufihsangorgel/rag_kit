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
