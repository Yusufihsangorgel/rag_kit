/// File persistence for [InMemoryVectorStore], for the Dart VM and Flutter.
///
/// This library re-exports all of `package:rag_kit/rag_kit.dart` and adds
/// [InMemoryVectorStoreFiles]. It depends on `dart:io` and therefore does
/// not work on the web; web applications should import
/// `package:rag_kit/rag_kit.dart` and persist the result of
/// [InMemoryVectorStore.toBytes] themselves, for example in IndexedDB.
library;

import 'dart:io';

import 'src/in_memory_vector_store.dart';

export 'rag_kit.dart';

/// Saves and loads an [InMemoryVectorStore] as a file in the rag_kit binary
/// format described at [InMemoryVectorStore.toBytes].
extension InMemoryVectorStoreFiles on InMemoryVectorStore {
  /// Writes the store to the file at [path], overwriting it if it exists.
  ///
  /// The write goes to a temporary file that is renamed into place, so a
  /// crash mid-write cannot destroy a previously saved index.
  Future<void> save(String path) async {
    final temp = File('$path.tmp.$pid');
    await temp.writeAsBytes(toBytes());
    await temp.rename(path);
  }

  /// Reads a store previously written with [save].
  ///
  /// ```dart
  /// final store = await InMemoryVectorStoreFiles.load('index.bin');
  /// ```
  ///
  /// Throws a [FormatException] when the file content is not a valid store
  /// and a [FileSystemException] when the file cannot be read.
  static Future<InMemoryVectorStore> load(String path) async {
    final bytes = await File(path).readAsBytes();
    return InMemoryVectorStore.fromBytes(bytes);
  }
}
