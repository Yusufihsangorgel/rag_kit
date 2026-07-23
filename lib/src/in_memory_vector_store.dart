import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'document.dart';
import 'scored_chunk.dart';
import 'vector_store.dart';

/// A [VectorStore] that keeps everything in memory.
///
/// Embeddings are stored as [Float32List], which halves memory compared to
/// doubles at a precision cost that is irrelevant for similarity search.
/// Each document's vector norm is computed once at upsert time, so a search
/// costs one dot product per candidate document.
///
/// Search is exact: every candidate is scored. There is no approximate
/// index, which keeps results deterministic and works well up to roughly
/// 100k chunks. See the package README for the measured trade-off.
///
/// The store can be serialized with [toBytes] and restored with
/// [InMemoryVectorStore.fromBytes]. On the Dart VM and Flutter (not web),
/// `package:rag_kit/io.dart` adds file-based `save` and `load` on top of
/// these.
class InMemoryVectorStore extends VectorStore {
  /// Creates an empty store.
  ///
  /// The embedding dimension is fixed by the first upserted document.
  InMemoryVectorStore();

  /// Restores a store from bytes produced by [toBytes].
  ///
  /// Throws a [FormatException] if [bytes] is not a valid serialized store.
  factory InMemoryVectorStore.fromBytes(Uint8List bytes) {
    final reader = _ByteReader(bytes);
    final magic = reader.readBytes(4);
    if (magic[0] != _magic0 ||
        magic[1] != _magic1 ||
        magic[2] != _magic2 ||
        magic[3] != _magic3) {
      throw const FormatException(
        'Not a rag_kit vector store: bad magic bytes.',
      );
    }
    final dimension = reader.readUint32();
    final count = reader.readUint32();
    if (count > 0 && dimension == 0) {
      throw const FormatException(
        'Invalid vector store: zero dimension with a nonzero document count.',
      );
    }
    final store = InMemoryVectorStore();
    for (var i = 0; i < count; i++) {
      final id = reader.readString();
      final text = reader.readString();
      final metadataJson = reader.readString();
      final Object? decoded;
      try {
        decoded = jsonDecode(metadataJson);
      } on FormatException {
        throw FormatException(
          'Invalid vector store: malformed metadata for document "$id".',
        );
      }
      if (decoded is! Map<String, dynamic>) {
        throw FormatException(
          'Invalid vector store: metadata for document "$id" is not a map.',
        );
      }
      final embedding = reader.readFloat32List(dimension);
      for (var i = 0; i < embedding.length; i++) {
        if (!embedding[i].isFinite) {
          throw FormatException(
            'Invalid vector store: non-finite embedding component in '
            'document "$id".',
          );
        }
      }
      store._insert(
        Document(id: id, text: text, embedding: embedding, metadata: decoded),
      );
    }
    if (!reader.atEnd) {
      throw const FormatException(
        'Invalid vector store: trailing bytes after the last document.',
      );
    }
    if (count > 0) {
      store._dimension = dimension;
    }
    return store;
  }

  /// The largest finite float32 value. Finite doubles beyond this would
  /// silently overflow to infinity when stored as float32, so they are
  /// rejected up front together with NaN and infinity.
  static const double _maxFloat32 = 3.4028234663852886e38;

  // Serialization format "RGK1", see toBytes.
  static const int _magic0 = 0x52; // R
  static const int _magic1 = 0x47; // G
  static const int _magic2 = 0x4b; // K
  static const int _magic3 = 0x31; // 1

  final Map<String, _Entry> _entries = <String, _Entry>{};
  int? _dimension;

  /// The embedding dimension of the stored documents, or null while the
  /// store is empty.
  ///
  /// Set by the first upsert and reset when the store becomes empty again,
  /// after which documents of a different dimension are accepted.
  int? get dimension => _dimension;

  @override
  Future<void> upsert(List<Document> documents) async {
    var dimension = _dimension;
    for (final document in documents) {
      final length = document.embedding.length;
      if (length == 0) {
        throw ArgumentError(
          'Document "${document.id}" has an empty embedding.',
        );
      }
      if (dimension != null && length != dimension) {
        throw ArgumentError(
          'Document "${document.id}" has $length dimensions, but the store '
          'holds $dimension-dimensional embeddings.',
        );
      }
      dimension ??= length;
      for (var i = 0; i < length; i++) {
        final value = document.embedding[i];
        if (!value.isFinite || value.abs() > _maxFloat32) {
          throw ArgumentError(
            'Document "${document.id}" has an embedding component that is '
            'not representable as a finite float32: $value at index $i. '
            'A NaN or infinite score would corrupt every search ranking.',
          );
        }
      }
    }
    for (final document in documents) {
      _insert(document);
    }
    _dimension = dimension;
  }

  @override
  Future<List<ScoredChunk>> search(
    List<double> query, {
    int topK = 5,
    double? minScore,
    bool Function(Document document)? where,
  }) async {
    if (topK < 1) {
      throw ArgumentError.value(topK, 'topK', 'must be at least 1');
    }
    if (_entries.isEmpty) return const [];
    final dimension = _dimension!;
    if (query.length != dimension) {
      throw ArgumentError(
        'Query has ${query.length} dimensions, but the store holds '
        '$dimension-dimensional embeddings.',
      );
    }
    for (var i = 0; i < query.length; i++) {
      final value = query[i];
      if (!value.isFinite || value.abs() > _maxFloat32) {
        throw ArgumentError(
          'Query has a component that is not representable as a finite '
          'float32: $value at index $i.',
        );
      }
    }
    final queryVector = Float64List.fromList(query);
    var querySumSquares = 0.0;
    for (var i = 0; i < queryVector.length; i++) {
      querySumSquares += queryVector[i] * queryVector[i];
    }
    final queryNorm = math.sqrt(querySumSquares);
    final top = _TopK(topK);
    var order = 0;
    for (final entry in _entries.values) {
      if (where != null && !where(entry.document)) continue;
      final score = _score(queryVector, queryNorm, entry);
      if (minScore != null && score < minScore) continue;
      top.add(_Hit(score, order++, entry.document));
    }
    return top.drain();
  }

  @override
  Future<int> removeWhere(bool Function(Document document) test) async {
    final before = _entries.length;
    _entries.removeWhere((_, entry) => test(entry.document));
    if (_entries.isEmpty) {
      _dimension = null;
    }
    return before - _entries.length;
  }

  @override
  Future<int> count() async => _entries.length;

  @override
  Future<void> clear() async {
    _entries.clear();
    _dimension = null;
  }

  /// Serializes the store to the rag_kit binary format.
  ///
  /// The format is compact and self-contained (all integers and floats are
  /// little-endian):
  ///
  /// ```text
  /// bytes 0-3    magic "RGK1"
  /// bytes 4-7    embedding dimension, uint32
  /// bytes 8-11   document count, uint32
  /// per document:
  ///   uint32 byte length + UTF-8 bytes   id
  ///   uint32 byte length + UTF-8 bytes   text
  ///   uint32 byte length + UTF-8 bytes   metadata as JSON
  ///   dimension x 4 bytes                embedding as float32
  /// ```
  ///
  /// Document metadata must be JSON-encodable; otherwise a
  /// [JsonUnsupportedObjectError] is thrown.
  Uint8List toBytes() {
    final builder = BytesBuilder(copy: false);
    final header = ByteData(12)
      ..setUint8(0, _magic0)
      ..setUint8(1, _magic1)
      ..setUint8(2, _magic2)
      ..setUint8(3, _magic3)
      ..setUint32(4, _dimension ?? 0, Endian.little)
      ..setUint32(8, _entries.length, Endian.little);
    builder.add(header.buffer.asUint8List());
    for (final entry in _entries.values) {
      _writeString(builder, entry.document.id);
      _writeString(builder, entry.document.text);
      _writeString(builder, jsonEncode(entry.document.metadata));
      final vector = entry.vector;
      final vectorData = ByteData(vector.length * 4);
      for (var i = 0; i < vector.length; i++) {
        vectorData.setFloat32(i * 4, vector[i], Endian.little);
      }
      builder.add(vectorData.buffer.asUint8List());
    }
    return builder.toBytes();
  }

  static void _writeString(BytesBuilder builder, String value) {
    final bytes = utf8.encode(value);
    final length = ByteData(4)..setUint32(0, bytes.length, Endian.little);
    builder.add(length.buffer.asUint8List());
    builder.add(bytes);
  }

  void _insert(Document document) {
    final vector = Float32List.fromList(document.embedding);
    _entries[document.id] = _Entry(
      document: Document(
        id: document.id,
        text: document.text,
        embedding: vector,
        // Copied for the same reason embedding is: without it, the caller's
        // map would alias the stored one, so mutating it later (or mutating
        // a document handed back by search/retrieve, since that is this
        // same instance) would silently rewrite data already in the store.
        // Unmodifiable turns any such mutation attempt into an immediate
        // error instead of a silent one.
        metadata: Map<String, Object?>.unmodifiable(document.metadata),
      ),
      vector: vector,
      norm: _norm(vector),
    );
  }

  static double _norm(Float32List vector) {
    var sumSquares = 0.0;
    for (var i = 0; i < vector.length; i++) {
      final value = vector[i];
      sumSquares += value * value;
    }
    return math.sqrt(sumSquares);
  }

  static double _score(Float64List query, double queryNorm, _Entry entry) {
    if (queryNorm == 0 || entry.norm == 0) return 0;
    final vector = entry.vector;
    var dot = 0.0;
    for (var i = 0; i < vector.length; i++) {
      dot += query[i] * vector[i];
    }
    return dot / (queryNorm * entry.norm);
  }
}

class _Entry {
  _Entry({required this.document, required this.vector, required this.norm});

  final Document document;
  final Float32List vector;
  final double norm;
}

class _Hit {
  _Hit(this.score, this.order, this.document);

  final double score;
  final int order;
  final Document document;
}

/// A bounded min-heap that keeps the [capacity] best hits seen so far.
///
/// The root is the worst kept hit, so deciding whether a new hit belongs in
/// the result is O(1) and inserting is O(log capacity). This avoids sorting
/// all candidates when only the top few are needed.
class _TopK {
  _TopK(this.capacity);

  final int capacity;
  final List<_Hit> _heap = <_Hit>[];

  /// Whether [a] ranks strictly higher than [b] in the final result order:
  /// higher score first, earlier insertion first on ties.
  static bool _beats(_Hit a, _Hit b) =>
      a.score > b.score || (a.score == b.score && a.order < b.order);

  void add(_Hit hit) {
    if (_heap.length < capacity) {
      _heap.add(hit);
      _siftUp(_heap.length - 1);
    } else if (_beats(hit, _heap[0])) {
      _heap[0] = hit;
      _siftDown();
    }
  }

  void _siftUp(int index) {
    var child = index;
    while (child > 0) {
      final parent = (child - 1) >> 1;
      if (!_beats(_heap[parent], _heap[child])) break;
      _swap(parent, child);
      child = parent;
    }
  }

  void _siftDown() {
    final length = _heap.length;
    var parent = 0;
    while (true) {
      var lowest = parent;
      final left = 2 * parent + 1;
      final right = left + 1;
      if (left < length && _beats(_heap[lowest], _heap[left])) {
        lowest = left;
      }
      if (right < length && _beats(_heap[lowest], _heap[right])) {
        lowest = right;
      }
      if (lowest == parent) return;
      _swap(parent, lowest);
      parent = lowest;
    }
  }

  void _swap(int a, int b) {
    final tmp = _heap[a];
    _heap[a] = _heap[b];
    _heap[b] = tmp;
  }

  List<ScoredChunk> drain() {
    _heap.sort((a, b) {
      final byScore = b.score.compareTo(a.score);
      if (byScore != 0) return byScore;
      return a.order.compareTo(b.order);
    });
    return [
      for (final hit in _heap)
        ScoredChunk(document: hit.document, score: hit.score),
    ];
  }
}

class _ByteReader {
  _ByteReader(this._bytes) : _data = ByteData.sublistView(_bytes);

  final Uint8List _bytes;
  final ByteData _data;
  int _offset = 0;

  bool get atEnd => _offset == _bytes.length;

  void _require(int length) {
    if (_bytes.length - _offset < length) {
      throw const FormatException(
        'Invalid vector store: unexpected end of data.',
      );
    }
  }

  Uint8List readBytes(int length) {
    _require(length);
    final result = Uint8List.sublistView(_bytes, _offset, _offset + length);
    _offset += length;
    return result;
  }

  int readUint32() {
    _require(4);
    final value = _data.getUint32(_offset, Endian.little);
    _offset += 4;
    return value;
  }

  String readString() {
    final length = readUint32();
    final bytes = readBytes(length);
    try {
      return utf8.decode(bytes);
    } on FormatException {
      throw const FormatException(
        'Invalid vector store: malformed UTF-8 string.',
      );
    }
  }

  Float32List readFloat32List(int count) {
    _require(count * 4);
    final result = Float32List(count);
    for (var i = 0; i < count; i++) {
      result[i] = _data.getFloat32(_offset, Endian.little);
      _offset += 4;
    }
    return result;
  }
}
