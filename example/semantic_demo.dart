// Demo for the write-up: index a short handbook with local embeddings, then
// retrieve the passage that answers a question phrased differently from the
// text. Needs a local Ollama with nomic-embed-text.
// Run with: dart run example/semantic_demo.dart
import 'dart:convert';
import 'dart:io';

import 'package:rag_kit/rag_kit.dart';

Future<List<List<double>>> ollamaEmbedder(List<String> texts) async {
  final client = HttpClient();
  final out = <List<double>>[];
  for (final text in texts) {
    final req = await client.postUrl(
      Uri.parse('http://localhost:11434/api/embeddings'),
    );
    req.headers.contentType = ContentType.json;
    req.write(jsonEncode({'model': 'nomic-embed-text', 'prompt': text}));
    final resp = await req.close();
    final body = await resp.transform(utf8.decoder).join();
    final map = jsonDecode(body) as Map<String, dynamic>;
    out.add(
      (map['embedding'] as List).map((n) => (n as num).toDouble()).toList(),
    );
  }
  client.close();
  return out;
}

const handbook = '''
New hires get a company laptop on day one. Pick your OS in the onboarding form and IT ships it before your start date.

Employees accrue paid leave every month. To take a vacation, open a request in the HR portal at least two weeks in advance and your manager approves it.

Reimbursements go through the expenses tool. Upload the receipt, pick a category, and finance pays out on the next cycle.

Report security issues in the security channel, not by email. Anything that looks like a leaked credential is treated as urgent.
''';

Future<void> main() async {
  final retriever = Retriever(
    embedder: ollamaEmbedder,
    store: InMemoryVectorStore(),
    chunker: Chunker.paragraphs(),
  );

  stdout.write('Indexing the handbook with local embeddings...  ');
  await retriever.addText(handbook.trim(), sourceId: 'handbook');
  stdout.writeln('done');
  stdout.writeln('');

  const query = 'how do I take time off?';
  stdout.writeln('Query:  "$query"');
  stdout.writeln('        (note: the text never says "time off")');
  stdout.writeln('');

  final results = await retriever.retrieve(query, topK: 1);
  final top = results.first;
  stdout.writeln('Best match (score ${top.score.toStringAsFixed(2)}):');
  stdout.writeln('  ${top.document.text}');
}
