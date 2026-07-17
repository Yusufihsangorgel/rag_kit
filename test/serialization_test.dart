import 'dart:convert';
import 'dart:typed_data';

import 'package:rag_kit/rag_kit.dart';
import 'package:test/test.dart';

Future<InMemoryVectorStore> sampleStore() async {
  final store = InMemoryVectorStore();
  await store.upsert([
    Document(
      id: 'doc-1',
      text: 'first text',
      embedding: [0.25, -1.5, 3.0],
      metadata: {'lang': 'en', 'page': 1},
    ),
    Document(
      id: 'doc-2',
      text: 'ikinci metin, unicode: ği üş çö',
      embedding: [0.1, 0.2, 0.3],
      metadata: {
        'tags': ['a', 'b'],
        'nested': {'x': 1.5},
        'missing': null,
      },
    ),
  ]);
  return store;
}

/// Builds bytes in the store format by hand, for corruption tests.
Uint8List buildBytes({
  List<int> magic = const [0x52, 0x47, 0x4b, 0x31],
  required int dimension,
  required List<(String, String, String, List<double>)> documents,
  int? claimedCount,
}) {
  final builder = BytesBuilder();
  final header = ByteData(12);
  for (var i = 0; i < 4; i++) {
    header.setUint8(i, magic[i]);
  }
  header.setUint32(4, dimension, Endian.little);
  header.setUint32(8, claimedCount ?? documents.length, Endian.little);
  builder.add(header.buffer.asUint8List());
  void writeString(String value) {
    final bytes = utf8.encode(value);
    final length = ByteData(4)..setUint32(0, bytes.length, Endian.little);
    builder.add(length.buffer.asUint8List());
    builder.add(bytes);
  }

  for (final (id, text, metadataJson, embedding) in documents) {
    writeString(id);
    writeString(text);
    writeString(metadataJson);
    final data = ByteData(embedding.length * 4);
    for (var i = 0; i < embedding.length; i++) {
      data.setFloat32(i * 4, embedding[i], Endian.little);
    }
    builder.add(data.buffer.asUint8List());
  }
  return builder.toBytes();
}

void main() {
  group('InMemoryVectorStore binary serialization', () {
    test('roundtrip preserves documents, metadata, and dimension', () async {
      final store = await sampleStore();
      final loaded = InMemoryVectorStore.fromBytes(store.toBytes());

      expect(await loaded.count(), 2);
      expect(loaded.dimension, 3);

      final original = (await store.search([
        1,
        1,
        1,
      ], topK: 2)).map((r) => r.document);
      final restored = (await loaded.search([
        1,
        1,
        1,
      ], topK: 2)).map((r) => r.document);
      for (final (a, b) in [
        for (var i = 0; i < 2; i++)
          (original.elementAt(i), restored.elementAt(i)),
      ]) {
        expect(b.id, a.id);
        expect(b.text, a.text);
        expect(b.metadata, a.metadata);
        expect(b.embedding, a.embedding);
      }
    });

    test('roundtrip preserves search scores exactly', () async {
      final store = await sampleStore();
      final loaded = InMemoryVectorStore.fromBytes(store.toBytes());
      final query = [0.5, -0.25, 1.0];
      final before = await store.search(query, topK: 2);
      final after = await loaded.search(query, topK: 2);
      for (var i = 0; i < before.length; i++) {
        expect(after[i].score, before[i].score);
        expect(after[i].document.id, before[i].document.id);
      }
    });

    test('empty store roundtrips', () async {
      final loaded = InMemoryVectorStore.fromBytes(
        InMemoryVectorStore().toBytes(),
      );
      expect(await loaded.count(), 0);
      expect(loaded.dimension, isNull);
      expect(await loaded.search([1, 2]), isEmpty);
    });

    test('rejects data that is too short for a header', () {
      expect(
        () => InMemoryVectorStore.fromBytes(Uint8List.fromList([1, 2, 3])),
        throwsFormatException,
      );
    });

    test('rejects bad magic bytes', () {
      final bytes = buildBytes(
        magic: const [0x00, 0x11, 0x22, 0x33],
        dimension: 2,
        documents: [
          ('a', 't', '{}', [1.0, 2.0]),
        ],
      );
      expect(
        () => InMemoryVectorStore.fromBytes(bytes),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('magic'),
          ),
        ),
      );
    });

    test('rejects truncated data', () async {
      final store = await sampleStore();
      final bytes = store.toBytes();
      final truncated = Uint8List.sublistView(bytes, 0, bytes.length - 5);
      expect(
        () => InMemoryVectorStore.fromBytes(truncated),
        throwsFormatException,
      );
    });

    test('rejects trailing bytes', () async {
      final store = await sampleStore();
      final bytes = store.toBytes();
      final padded = Uint8List(bytes.length + 3)
        ..setRange(0, bytes.length, bytes);
      expect(
        () => InMemoryVectorStore.fromBytes(padded),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('trailing'),
          ),
        ),
      );
    });

    test('rejects a zero dimension with a nonzero count', () {
      final bytes = buildBytes(dimension: 0, documents: [('a', 't', '{}', [])]);
      expect(() => InMemoryVectorStore.fromBytes(bytes), throwsFormatException);
    });

    test('rejects a count larger than the actual data', () {
      final bytes = buildBytes(
        dimension: 2,
        documents: [
          ('a', 't', '{}', [1.0, 2.0]),
        ],
        claimedCount: 5,
      );
      expect(() => InMemoryVectorStore.fromBytes(bytes), throwsFormatException);
    });

    test('rejects malformed metadata JSON', () {
      final bytes = buildBytes(
        dimension: 2,
        documents: [
          ('a', 't', '{not json', [1.0, 2.0]),
        ],
      );
      expect(
        () => InMemoryVectorStore.fromBytes(bytes),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('metadata'),
          ),
        ),
      );
    });

    test('rejects metadata that is not a JSON object', () {
      final bytes = buildBytes(
        dimension: 2,
        documents: [
          ('a', 't', '[1, 2]', [1.0, 2.0]),
        ],
      );
      expect(
        () => InMemoryVectorStore.fromBytes(bytes),
        throwsA(
          isA<FormatException>().having(
            (e) => e.message,
            'message',
            contains('not a map'),
          ),
        ),
      );
    });
  });
}
