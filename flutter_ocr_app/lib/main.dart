import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'dart:convert';

const String _defaultBaseUrl = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: 'http://192.168.1.77:8001',
);

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'IA Document OCR & Translation',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const OCRHomePage(),
    );
  }
}

class OCRHomePage extends StatefulWidget {
  const OCRHomePage({super.key});

  @override
  State<OCRHomePage> createState() => _OCRHomePageState();
}

class _OCRHomePageState extends State<OCRHomePage> {
  File? _selectedFile;
  String? _fileType; 
  String _resultText = "";
  String? _visualUrl;
  String? _reconstructedPdfUrl;
  List<dynamic>? _lastFullStructure;
  List<Map<String, dynamic>> _hierarchicalByPage = [];
  List<Map<String, dynamic>> _translatedByPage = [];
  bool _isLoading = false;
  bool _isReconstructing = false;
  bool _isTranslating = false;
  bool _forceAI = false;
  String _selectedLang = 'Aucune';
  String _baseUrl = _defaultBaseUrl;
  late final TextEditingController _baseUrlController;
  
  final List<String> _languages = ['Aucune', 'Français', 'Spanish', 'English', 'German', 'Italian'];
  final Map<String, String?> _langCodes = {
    'Aucune': null,
    'Français': 'French',
    'Spanish': 'Spanish',
    'English': 'English',
    'German': 'German',
    'Italian': 'Italian'
  };

  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _baseUrlController = TextEditingController(text: _baseUrl);
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    super.dispose();
  }

  void _applyBaseUrl() {
    final raw = _baseUrlController.text.trim();
    if (raw.isEmpty) return;
    final normalized = raw.endsWith('/') ? raw.substring(0, raw.length - 1) : raw;
    setState(() {
      _baseUrl = normalized;
    });
  }

  void _setActiveBaseUrl(String value) {
    final normalized = value.endsWith('/') ? value.substring(0, value.length - 1) : value;
    setState(() {
      _baseUrl = normalized;
      _baseUrlController.text = normalized;
    });
  }

  List<String> _candidateBaseUrls() {
    final candidates = <String>[];
    void add(String v) {
      final n = v.endsWith('/') ? v.substring(0, v.length - 1) : v;
      if (!candidates.contains(n)) {
        candidates.add(n);
      }
    }

    add(_baseUrlController.text.trim().isNotEmpty ? _baseUrlController.text.trim() : _baseUrl);
    add(_defaultBaseUrl);

    // Useful fallbacks for USB debug / emulator setups.
    try {
      final parsed = Uri.parse(candidates.first);
      final int port = parsed.hasPort ? parsed.port : 8001;
      final String scheme = parsed.scheme.isEmpty ? "http" : parsed.scheme;
      add("$scheme://127.0.0.1:$port");
      add("$scheme://10.0.2.2:$port");
      add("$scheme://10.0.3.2:$port");
    } catch (_) {}
    return candidates;
  }

  Future<String?> _resolveReachableBaseUrl() async {
    for (final base in _candidateBaseUrls()) {
      try {
        final resp = await http
            .get(Uri.parse("$base/openapi.json"))
            .timeout(const Duration(seconds: 3));
        if (resp.statusCode == 200) {
          _setActiveBaseUrl(base);
          return base;
        }
      } catch (_) {}
    }
    return null;
  }

  Future<void> _testServerConnection() async {
    _applyBaseUrl();
    final resolved = await _resolveReachableBaseUrl();
    if (!mounted) return;
    if (resolved != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Connexion OK: $resolved"), backgroundColor: Colors.green),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Aucun endpoint joignable. Essaye adb reverse + 127.0.0.1:8001"),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _selectedFile = File(pickedFile.path);
        _fileType = 'image';
        _resultText = ""; _visualUrl = null; _reconstructedPdfUrl = null; _lastFullStructure = null; _hierarchicalByPage = []; _translatedByPage = [];
      });
    }
  }

  Future<void> _pickDocument() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'doc', 'docx', 'ppt', 'pptx'],
    );
    if (result != null) {
      setState(() {
        _selectedFile = File(result.files.single.path!);
        _fileType = path.extension(_selectedFile!.path).toLowerCase() == '.pdf' ? 'doc' : 'image';
        _resultText = ""; _visualUrl = null; _reconstructedPdfUrl = null; _lastFullStructure = null; _hierarchicalByPage = []; _translatedByPage = [];
      });
    }
  }

  Future<void> _uploadFile() async {
    if (_selectedFile == null) return;
    _applyBaseUrl();
    setState(() { _isLoading = true; _resultText = ""; _visualUrl = null; _reconstructedPdfUrl = null; _hierarchicalByPage = []; _translatedByPage = []; });

    try {
      final resolvedBase = await _resolveReachableBaseUrl();
      if (resolvedBase == null) {
        throw const SocketException("Aucun endpoint serveur joignable");
      }

      var uri = Uri.parse("$resolvedBase/ocr?force_ai=$_forceAI");
      var request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', _selectedFile!.path));

      var streamedResponse = await request.send().timeout(const Duration(minutes: 15));
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'success') {
          List<dynamic> results = data['results'];
          StringBuffer sb = StringBuffer();
          List<dynamic> structures = [];
          List<Map<String, dynamic>> hierarchicalByPage = [];
          if (results.isNotEmpty) _visualUrl = resolvedBase + results[0]['visual_url'];
          
          for (var page in results) {
            sb.writeln(page['content']);
            if (page['structure'] != null) structures.add(page['structure']);
            final dynamic hier = page['hierarchical_extraction'];
            if (hier is Map) {
              hierarchicalByPage.add({
                'page': page['page'] ?? (hierarchicalByPage.length + 1),
                'data': Map<String, dynamic>.from(hier),
              });
            }
          }
          setState(() { _resultText = sb.toString(); _lastFullStructure = structures; _hierarchicalByPage = hierarchicalByPage; _translatedByPage = []; });
        }
      } else {
        setState(() { _resultText = "Erreur serveur (${response.statusCode})"; });
      }
    } catch (e) {
      setState(() {
        _resultText =
            "Erreur de connexion. URL testées: ${_candidateBaseUrls().join(' | ')}\n"
            "Détail: $e";
      });
    } finally {
      setState(() { _isLoading = false; });
    }
  }

  Future<void> _reconstructDocument() async {
    if (_lastFullStructure == null) return;
    _applyBaseUrl();
    setState(() { _isReconstructing = true; _reconstructedPdfUrl = null; });

    try {
      final resolvedBase = await _resolveReachableBaseUrl();
      if (resolvedBase == null) {
        throw const SocketException("Aucun endpoint serveur joignable");
      }
      String? langCode = _langCodes[_selectedLang];
      var uriStr = "$resolvedBase/reconstruct";
      if (langCode != null) uriStr += "?target_lang=$langCode";
      var url = Uri.parse(uriStr);
      
      var response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"pages": _lastFullStructure}),
      ).timeout(const Duration(minutes: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'success') {
          setState(() { _reconstructedPdfUrl = resolvedBase + data['pdf_url']; });
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text("Document reconstruit avec succès !"), backgroundColor: Colors.green),
          );
        }
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Erreur reconstruction: $e"), backgroundColor: Colors.red),
      );
    } finally {
      setState(() { _isReconstructing = false; });
    }
  }

  Future<void> _translateHierarchicalResults() async {
    if (_hierarchicalByPage.isEmpty) return;
    if (_selectedLang == 'Aucune') {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Choisis une langue avant de traduire"), backgroundColor: Colors.orange),
      );
      return;
    }
    _applyBaseUrl();
    setState(() { _isTranslating = true; _translatedByPage = []; });
    try {
      final String targetLang = _langCodes[_selectedLang] ?? _selectedLang;
      final resolvedBase = await _resolveReachableBaseUrl();
      if (resolvedBase == null) {
        throw const SocketException("Aucun endpoint serveur joignable");
      }
      final pagesPayload = _hierarchicalByPage.map((e) {
        final pageNum = e['page'];
        final data = (e['data'] is Map<String, dynamic>) ? (e['data'] as Map<String, dynamic>) : <String, dynamic>{};
        return {
          "page": pageNum,
          "phrases": data['phrases'] ?? const [],
          "groupes_mots": data['groupes_mots'] ?? const [],
          "mots": data['mots'] ?? const [],
        };
      }).toList();

      final response = await http.post(
        Uri.parse("$resolvedBase/translate-units"),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "target_lang": targetLang,
          "pages": pagesPayload,
        }),
      ).timeout(const Duration(minutes: 10));

      if (response.statusCode != 200) {
        throw Exception("Erreur serveur (${response.statusCode})");
      }
      final data = jsonDecode(response.body);
      if (data is! Map || data['status'] != 'success') {
        throw Exception("Réponse invalide du serveur");
      }

      final List<dynamic> pages = (data['pages'] is List) ? data['pages'] as List<dynamic> : const [];
      final translated = <Map<String, dynamic>>[];
      for (final p in pages) {
        if (p is! Map) continue;
        translated.add({
          'page': p['page'],
          'data': {
            'phrases': (p['phrases'] is List) ? p['phrases'] : const [],
            'groupes_mots': (p['groupes_mots'] is List) ? p['groupes_mots'] : const [],
            'mots': (p['mots'] is List) ? p['mots'] : const [],
          },
        });
      }
      setState(() { _translatedByPage = translated; });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Traduction terminée (${_selectedLang})"), backgroundColor: Colors.green),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Erreur traduction: $e"), backgroundColor: Colors.red),
      );
    } finally {
      if (mounted) {
        setState(() { _isTranslating = false; });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('IA Document OCR & Translation'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Container(
              height: 300,
              decoration: BoxDecoration(
                color: Colors.grey[100],
                border: Border.all(color: Colors.grey.shade300),
                borderRadius: BorderRadius.circular(12),
              ),
              child: _visualUrl != null 
                ? Image.network(_visualUrl!, fit: BoxFit.contain, key: ValueKey(_visualUrl))
                : (_selectedFile != null ? const Icon(Icons.description, size: 50) : const Icon(Icons.cloud_upload, size: 50)),
            ),
            const SizedBox(height: 20),

            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const Text("Serveur API", style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _baseUrlController,
                      decoration: const InputDecoration(
                        border: OutlineInputBorder(),
                        hintText: "http://192.168.1.77:8001",
                      ),
                      keyboardType: TextInputType.url,
                      onSubmitted: (_) => _applyBaseUrl(),
                    ),
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 10,
                      runSpacing: 8,
                      children: [
                        OutlinedButton.icon(
                          onPressed: _testServerConnection,
                          icon: const Icon(Icons.wifi_tethering),
                          label: const Text("Tester connexion"),
                        ),
                        OutlinedButton.icon(
                          onPressed: () {
                            _applyBaseUrl();
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text("URL API appliquée: $_baseUrl")),
                            );
                          },
                          icon: const Icon(Icons.save),
                          label: const Text("Appliquer URL"),
                        ),
                        OutlinedButton.icon(
                          onPressed: () {
                            _setActiveBaseUrl("http://127.0.0.1:8001");
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text("Mode USB activé: http://127.0.0.1:8001")),
                            );
                          },
                          icon: const Icon(Icons.usb),
                          label: const Text("Mode USB localhost"),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text("URL active: $_baseUrl", style: const TextStyle(fontSize: 12)),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(onPressed: () => _pickImage(ImageSource.gallery), icon: const Icon(Icons.image), label: const Text("Galerie")),
                ElevatedButton.icon(onPressed: _pickDocument, icon: const Icon(Icons.folder), label: const Text("Fichiers")),
              ],
            ),
            const SizedBox(height: 20),

            FilledButton.icon(
              onPressed: (_selectedFile != null && !_isLoading) ? _uploadFile : null,
              icon: _isLoading ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)) : const Icon(Icons.search),
              label: Text(_isLoading ? 'Analyse en cours...' : '1. Analyser le document'),
            ),
            
            const Divider(height: 40),

            if (_lastFullStructure != null) ...[
              const Text("Traduction & Reconstruction", style: TextStyle(fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              DropdownButtonFormField<String>(
                value: _selectedLang,
                decoration: const InputDecoration(labelText: 'Langue de destination', border: OutlineInputBorder()),
                items: _languages.map((l) => DropdownMenuItem(value: l, child: Text(l))).toList(),
                onChanged: (val) => setState(() { _selectedLang = val!; }),
              ),
              const SizedBox(height: 10),
              OutlinedButton.icon(
                onPressed: _isReconstructing ? null : _reconstructDocument,
                icon: _isReconstructing ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.layers),
                label: const Text('2. Reconstruire le document'),
              ),
            ],

            if (_reconstructedPdfUrl != null)
              Padding(
                padding: const EdgeInsets.only(top: 15.0),
                child: SelectableText(
                  "Lien du PDF : $_reconstructedPdfUrl",
                  style: const TextStyle(color: Colors.blue, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
              ),

            if (_resultText.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Row(
                          children: [
                            const Expanded(
                              child: Text(
                                "Résultat OCR",
                                style: TextStyle(fontWeight: FontWeight.bold),
                              ),
                            ),
                            IconButton(
                              tooltip: "Copier le résultat",
                              onPressed: () async {
                                await Clipboard.setData(ClipboardData(text: _resultText));
                                if (!mounted) return;
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text("Résultat copié")),
                                );
                              },
                              icon: const Icon(Icons.copy),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        SelectableText(
                          _resultText,
                          style: const TextStyle(fontSize: 11, fontFamily: 'monospace'),
                        ),
                      ],
                    ),
                  ),
                ),
              ),

            if (_hierarchicalByPage.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 20),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final controls = Wrap(
                      spacing: 10,
                      runSpacing: 8,
                      children: [
                        FilledButton.icon(
                          onPressed: _isTranslating ? null : _translateHierarchicalResults,
                          icon: _isTranslating
                              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                              : const Icon(Icons.translate),
                          label: Text(_isTranslating ? "Traduction..." : "Traduire"),
                        ),
                        OutlinedButton.icon(
                          onPressed: () async {
                            await Clipboard.setData(ClipboardData(text: _serializeHierarchical(_hierarchicalByPage, includeResiduals: true)));
                            if (!mounted) return;
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text("Résultats source copiés")),
                            );
                          },
                          icon: const Icon(Icons.copy),
                          label: const Text("Copier source"),
                        ),
                        OutlinedButton.icon(
                          onPressed: _translatedByPage.isEmpty ? null : () async {
                            await Clipboard.setData(ClipboardData(text: _serializeHierarchical(_translatedByPage, includeResiduals: false)));
                            if (!mounted) return;
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text("Résultats traduits copiés")),
                            );
                          },
                          icon: const Icon(Icons.copy_all),
                          label: const Text("Copier traduction"),
                        ),
                      ],
                    );

                    final sourceCard = _buildHierarchicalCard(
                      title: "Extraction hiérarchique (source)",
                      pages: _hierarchicalByPage,
                      includeResiduals: true,
                    );
                    final translatedCard = _buildHierarchicalCard(
                      title: "Traduction (phrases/groupes/mots)",
                      pages: _translatedByPage,
                      includeResiduals: false,
                    );

                    if (constraints.maxWidth >= 980) {
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          controls,
                          const SizedBox(height: 12),
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(child: sourceCard),
                              const SizedBox(width: 12),
                              Expanded(child: translatedCard),
                            ],
                          ),
                        ],
                      );
                    }
                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        controls,
                        const SizedBox(height: 12),
                        sourceCard,
                        const SizedBox(height: 12),
                        translatedCard,
                      ],
                    );
                  },
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildHierarchicalCard({
    required String title,
    required List<Map<String, dynamic>> pages,
    required bool includeResiduals,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(title, style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            if (pages.isEmpty)
              const Text("Aucun résultat", style: TextStyle(color: Colors.black54))
            else
              ..._buildHierarchicalWidgets(pages, includeResiduals: includeResiduals),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildHierarchicalWidgets(List<Map<String, dynamic>> pages, {required bool includeResiduals}) {
    final List<Widget> widgets = [];
    for (final pageEntry in pages) {
      final int pageNum = (pageEntry['page'] is int) ? pageEntry['page'] as int : 0;
      final Map<String, dynamic> data = (pageEntry['data'] is Map<String, dynamic>)
          ? pageEntry['data'] as Map<String, dynamic>
          : <String, dynamic>{};
      final List<dynamic> phrases = (data['phrases'] is List) ? data['phrases'] as List<dynamic> : const [];
      final List<dynamic> groupes = (data['groupes_mots'] is List) ? data['groupes_mots'] as List<dynamic> : const [];
      final List<dynamic> mots = (data['mots'] is List) ? data['mots'] as List<dynamic> : const [];
      final List<dynamic> chiffres = (data['chiffres'] is List) ? data['chiffres'] as List<dynamic> : const [];
      final List<dynamic> nombres = (data['nombres'] is List) ? data['nombres'] as List<dynamic> : const [];
      final List<dynamic> lettres = (data['lettres'] is List) ? data['lettres'] as List<dynamic> : const [];
      final List<dynamic> symboles = (data['symboles'] is List) ? data['symboles'] as List<dynamic> : const [];

      widgets.add(Text("Page $pageNum", style: const TextStyle(fontWeight: FontWeight.w600)));
      widgets.add(const SizedBox(height: 6));
      widgets.addAll(_buildIndexedTextSection("1) Phrases", phrases));
      widgets.addAll(_buildIndexedTextSection("2) Groupes de mots restants (hors phrases)", groupes));
      widgets.addAll(_buildIndexedTextSection("3) Mots restants (hors phrases/groupes)", mots));
      if (includeResiduals) {
        widgets.addAll(_buildIndexedTextSection("4) Chiffres restants", chiffres));
        widgets.addAll(_buildIndexedTextSection("5) Nombres restants", nombres));
        widgets.addAll(_buildIndexedTextSection("6) Lettres restantes", lettres));
        widgets.addAll(_buildIndexedTextSection("7) Symboles restants", symboles));
      }
      widgets.add(const Divider(height: 24));
    }
    if (widgets.isNotEmpty) {
      widgets.removeLast();
    }
    return widgets;
  }

  List<Widget> _buildIndexedTextSection(String title, List<dynamic> items) {
    final List<Widget> widgets = [Text(title, style: const TextStyle(fontWeight: FontWeight.w500))];
    if (items.isEmpty) {
      widgets.add(const Padding(
        padding: EdgeInsets.only(top: 4, bottom: 8),
        child: Text("Aucun", style: TextStyle(color: Colors.black54)),
      ));
      return widgets;
    }
    for (int i = 0; i < items.length; i++) {
      widgets.add(Padding(
        padding: const EdgeInsets.only(top: 2, bottom: 2),
        child: SelectableText("${i + 1}. ${items[i]}"),
      ));
    }
    widgets.add(const SizedBox(height: 8));
    return widgets;
  }

  String _serializeHierarchical(List<Map<String, dynamic>> pages, {required bool includeResiduals}) {
    final StringBuffer sb = StringBuffer();
    for (final pageEntry in pages) {
      final int pageNum = (pageEntry['page'] is int) ? pageEntry['page'] as int : 0;
      final Map<String, dynamic> data = (pageEntry['data'] is Map<String, dynamic>)
          ? pageEntry['data'] as Map<String, dynamic>
          : <String, dynamic>{};
      final List<dynamic> phrases = (data['phrases'] is List) ? data['phrases'] as List<dynamic> : const [];
      final List<dynamic> groupes = (data['groupes_mots'] is List) ? data['groupes_mots'] as List<dynamic> : const [];
      final List<dynamic> mots = (data['mots'] is List) ? data['mots'] as List<dynamic> : const [];
      sb.writeln("Page $pageNum");
      sb.writeln("Phrases:");
      for (int i = 0; i < phrases.length; i++) {
        sb.writeln("${i + 1}. ${phrases[i]}");
      }
      sb.writeln("Groupes de mots:");
      for (int i = 0; i < groupes.length; i++) {
        sb.writeln("${i + 1}. ${groupes[i]}");
      }
      sb.writeln("Mots:");
      for (int i = 0; i < mots.length; i++) {
        sb.writeln("${i + 1}. ${mots[i]}");
      }
      if (includeResiduals) {
        final List<dynamic> chiffres = (data['chiffres'] is List) ? data['chiffres'] as List<dynamic> : const [];
        final List<dynamic> nombres = (data['nombres'] is List) ? data['nombres'] as List<dynamic> : const [];
        final List<dynamic> lettres = (data['lettres'] is List) ? data['lettres'] as List<dynamic> : const [];
        final List<dynamic> symboles = (data['symboles'] is List) ? data['symboles'] as List<dynamic> : const [];
        sb.writeln("Chiffres:");
        for (int i = 0; i < chiffres.length; i++) {
          sb.writeln("${i + 1}. ${chiffres[i]}");
        }
        sb.writeln("Nombres:");
        for (int i = 0; i < nombres.length; i++) {
          sb.writeln("${i + 1}. ${nombres[i]}");
        }
        sb.writeln("Lettres:");
        for (int i = 0; i < lettres.length; i++) {
          sb.writeln("${i + 1}. ${lettres[i]}");
        }
        sb.writeln("Symboles:");
        for (int i = 0; i < symboles.length; i++) {
          sb.writeln("${i + 1}. ${symboles[i]}");
        }
      }
      sb.writeln("");
    }
    return sb.toString().trim();
  }

}
