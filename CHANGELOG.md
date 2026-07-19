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
