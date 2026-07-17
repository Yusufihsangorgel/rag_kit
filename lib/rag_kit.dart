/// Retrieval-augmented generation building blocks for Dart: chunking,
/// vector storage, similarity search, and context building.
///
/// This library is platform-neutral and works on the web. For file-based
/// persistence of [InMemoryVectorStore] on the Dart VM and Flutter, import
/// `package:rag_kit/io.dart` instead.
library;

export 'src/chunk.dart';
export 'src/chunker.dart';
export 'src/document.dart';
export 'src/embedder.dart';
export 'src/in_memory_vector_store.dart';
export 'src/retriever.dart';
export 'src/scored_chunk.dart';
export 'src/vector_store.dart';
