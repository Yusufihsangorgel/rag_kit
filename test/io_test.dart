@TestOn('vm')
library;

import 'dart:io';

import 'package:rag_kit/io.dart';
import 'package:test/test.dart';

void main() {
  late Directory tempDir;

  setUp(() {
    tempDir = Directory.systemTemp.createTempSync('rag_kit_test');
  });

  tearDown(() {
    tempDir.deleteSync(recursive: true);
  });

  group('InMemoryVectorStore file persistence', () {
    test('save and load roundtrip through a file', () async {
      final store = InMemoryVectorStore();
      await store.upsert([
        Document(
          id: 'a',
          text: 'saved text',
          embedding: [1.0, 2.0, 3.0],
          metadata: {'source': 'file'},
        ),
      ]);
      final path = '${tempDir.path}/store.bin';
      await store.save(path);

      final loaded = await InMemoryVectorStoreFiles.load(path);
      expect(await loaded.count(), 1);
      final results = await loaded.search([1.0, 2.0, 3.0], topK: 1);
      expect(results.single.document.id, 'a');
      expect(results.single.document.text, 'saved text');
      expect(results.single.document.metadata, {'source': 'file'});
      expect(results.single.score, closeTo(1, 1e-6));
    });

    test('load throws FormatException on a corrupt file', () async {
      final path = '${tempDir.path}/corrupt.bin';
      File(path).writeAsBytesSync([9, 9, 9, 9, 9, 9]);
      await expectLater(
        InMemoryVectorStoreFiles.load(path),
        throwsFormatException,
      );
    });

    test('load throws FileSystemException when the file is missing', () async {
      await expectLater(
        InMemoryVectorStoreFiles.load('${tempDir.path}/missing.bin'),
        throwsA(isA<FileSystemException>()),
      );
    });
  });
}
