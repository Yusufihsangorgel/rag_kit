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
