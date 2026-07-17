/// Embeds a batch of texts into vectors.
///
/// rag_kit does not talk to any embedding model itself. You supply this
/// function and bind it to whatever produces vectors: an HTTP call to
/// OpenAI or Ollama, a local ONNX model, or a deterministic fake in tests.
///
/// Implementations receive the whole batch in one call, so an HTTP-backed
/// embedder can send a single request instead of one request per text. The
/// returned list must contain exactly one vector per input text, in input
/// order, and all vectors must have the same length.
typedef Embedder = Future<List<List<double>>> Function(List<String> texts);
