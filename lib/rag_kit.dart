/// Retrieval-augmented generation building blocks for Dart: chunking,
/// vector storage, similarity search, and context building.
///
/// This library is platform-neutral and works on the web. For file-based
/// persistence of [InMemoryVectorStore] on the Dart VM and Flutter, import
/// `package:rag_kit/io.dart` instead.
library;

export 'src/chunk.dart' show Chunk;
export 'src/chunker.dart' show Chunker;
export 'src/document.dart' show Document;
export 'src/embedder.dart' show Embedder;
export 'src/in_memory_vector_store.dart' show InMemoryVectorStore;
export 'src/retriever.dart' show Retriever;
export 'src/scored_chunk.dart' show ScoredChunk;
export 'src/vector_store.dart' show VectorStore;
