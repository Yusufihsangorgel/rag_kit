# rag_kit example

`rag_kit_example.dart` runs the whole retrieval pipeline end to end with no
network: index two documents, retrieve against a query, scope the query to one
source with a metadata filter, and build a prompt-ready context string.

The example embeds text with a tiny deterministic hash embedder (words into 64
buckets), so it needs no model and prints the same numbers every run. In a real
application you swap that one function for a call to an embedding model — the
package README shows Ollama and OpenAI bindings; nothing else changes.

```dart
final retriever = Retriever(
  embedder: hashEmbedder,           // swap for a real model in production
  store: InMemoryVectorStore(),
  chunker: Chunker.paragraphs(),
);

await retriever.addText(dartGuide, sourceId: 'dart-guide');
await retriever.addText(cookbook, sourceId: 'cookbook');

// Retrieve across everything...
final hits = await retriever.retrieve('How does Dart pass data between isolates?', topK: 3);

// ...or scope to one source, so other sources never enter scoring:
final scoped = await retriever.retrieve(query, topK: 3,
    where: (doc) => doc.metadata['sourceId'] == 'dart-guide');

// buildContext takes the same filter and returns a string ready to drop into a prompt:
final context = await retriever.buildContext(query, topK: 2,
    where: (doc) => doc.metadata['sourceId'] == 'dart-guide');
```

Run it:

```
dart run example/rag_kit_example.dart
```

Output (the isolates query pulls the isolates paragraph to the top, and the
cookbook source scores far lower, so the metadata filter and the ranking both
show their effect):

```
Query: How does Dart pass data between isolates?

All sources:
  0.357  dart-guide#0
  0.194  dart-guide#1
  0.076  cookbook#0

Scoped to source "dart-guide":
  0.357  dart-guide#0
  0.194  dart-guide#1

Context for the LLM prompt (dart-guide only):

Dart isolates do not share mutable memory. Each isolate has its own heap,
and isolates exchange data by passing messages over ports.

---

Dart records are immutable and let a function return several values without
a class. Pattern matching destructures them by position.
```

For a semantic version backed by real embeddings, see `semantic_demo.dart`,
which indexes a short handbook with a local Ollama `nomic-embed-text` model.
