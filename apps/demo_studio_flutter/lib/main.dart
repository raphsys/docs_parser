import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

const String docsParserRootFromDefine = String.fromEnvironment('DOCS_PARSER_ROOT');
const String docsParserPythonFromDefine = String.fromEnvironment('DOCS_PARSER_PYTHON');
const String vsenseAppNameFromDefine = String.fromEnvironment('VSENSE_APP_NAME', defaultValue: 'vSense Studio');

void main() {
  runApp(const DemoStudioApp());
}

class DemoStudioApp extends StatelessWidget {
  const DemoStudioApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'docs_parser vSense Studio',
      themeMode: ThemeMode.system,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF2F6FED),
        brightness: Brightness.light,
        scaffoldBackgroundColor: const Color(0xFFF4F7FB),
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
        ),
      ),
      darkTheme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF7AA2FF),
        brightness: Brightness.dark,
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
        ),
      ),
      home: const DemoStudioHome(),
    );
  }
}

class DemoStudioHome extends StatefulWidget {
  const DemoStudioHome({super.key});

  @override
  State<DemoStudioHome> createState() => _DemoStudioHomeState();
}

class _DemoStudioHomeState extends State<DemoStudioHome> {
  final String root = docsParserRootFromDefine.isNotEmpty ? docsParserRootFromDefine : Directory.current.path;
  final TextEditingController pdfController = TextEditingController();
  final TextEditingController pagesController = TextEditingController(text: '1');
  final TextEditingController randomController = TextEditingController(text: '0');
  final TextEditingController seedController = TextEditingController(text: '20260616');
  final TextEditingController outNameController = TextEditingController();
  final TextEditingController modelController = TextEditingController(text: 'opus_mt_tc_big_en_fr');
  final TextEditingController sourceLangController = TextEditingController(text: 'en');
  final TextEditingController targetLangController = TextEditingController(text: 'fr');

  final List<String> stages = const [
    'full',
    'pageprint',
    'pagetranslate',
    'pagereconstruct',
    'view_background',
    'audit_translation_selection',
    'audit_text_survival',
  ];
  final List<String> engines = const ['ct2', 'dummy'];
  final List<String> pubreadyModes = const ['debug', 'review', 'publication'];

  String selectedStage = 'full';
  String selectedEngine = 'ct2';
  String pubreadyMode = 'review';
  bool useOcr = false;
  bool running = false;
  int pageCount = 0;
  Set<int> selectedPages = {1};
  String? lastOut;
  String? lastContactSheet;
  String? lastReport;
  final List<String> logs = [];

  String get pythonExecutable {
    if (docsParserPythonFromDefine.isNotEmpty) {
      return docsParserPythonFromDefine;
    }
    return _resolvePythonExecutable(root);
  }

  String _resolvePythonExecutable(String projectRoot) {
    final candidates = <String>[
      '$projectRoot/.docs_parse/bin/python',
      '$projectRoot/.docs-parser/bin/python',
      '$projectRoot/.venv/bin/python',
      '$projectRoot/venv/bin/python',
    ];
    for (final candidate in candidates) {
      if (File(candidate).existsSync()) {
        return candidate;
      }
    }
    return 'python3';
  }


  @override
  void dispose() {
    pdfController.dispose();
    pagesController.dispose();
    randomController.dispose();
    seedController.dispose();
    outNameController.dispose();
    modelController.dispose();
    sourceLangController.dispose();
    targetLangController.dispose();
    super.dispose();
  }

  void log(String line) {
    setState(() => logs.add(line));
  }

  Future<void> pickPdf() async {
    final result = await FilePicker.platform.pickFiles(
      dialogTitle: 'Choisir un PDF à tester',
      type: FileType.custom,
      allowedExtensions: ['pdf'],
    );
    if (result == null || result.files.single.path == null) return;
    pdfController.text = result.files.single.path!;
    await inspectPdf();
  }

  Future<void> inspectPdf() async {
    final pdf = pdfController.text.trim();
    if (pdf.isEmpty) return;
    setState(() {
      logs.clear();
      pageCount = 0;
      selectedPages = {1};
      pagesController.text = '1';
    });
    log('Inspection du PDF…');
    try {
      final res = await Process.run(
        pythonExecutable,
        ['tools/demo_studio_backend.py', '--inspect-pdf', pdf],
        workingDirectory: root,
      );
      if (res.exitCode != 0) {
        log('ERREUR inspection: ${res.stderr}');
        return;
      }
      final data = jsonDecode(res.stdout.toString()) as Map<String, dynamic>;
      final count = (data['page_count'] as num).toInt();
      setState(() {
        pageCount = count;
        selectedPages = {1};
        pagesController.text = '1';
      });
      log('PDF chargé: ${data['name']} · $count pages');
    } catch (e) {
      log('ERREUR inspection: $e');
    }
  }

  void updatePagesFromSelection() {
    final pages = selectedPages.toList()..sort();
    pagesController.text = _compactPages(pages);
  }

  String _compactPages(List<int> pages) {
    if (pages.isEmpty) return '';
    final parts = <String>[];
    var start = pages.first;
    var prev = pages.first;
    for (final p in pages.skip(1)) {
      if (p == prev + 1) {
        prev = p;
        continue;
      }
      parts.add(start == prev ? '$start' : '$start-$prev');
      start = prev = p;
    }
    parts.add(start == prev ? '$start' : '$start-$prev');
    return parts.join(',');
  }

  Future<void> runPipeline() async {
    final pdf = pdfController.text.trim();
    if (pdf.isEmpty) {
      log('Choisis d’abord un PDF.');
      return;
    }
    setState(() {
      running = true;
      logs.clear();
      lastOut = null;
      lastContactSheet = null;
      lastReport = null;
    });

    final args = <String>[
      'tools/demo_studio_backend.py',
      '--pdf', pdf,
      '--stage', selectedStage,
      '--pages', pagesController.text.trim().isEmpty ? '1' : pagesController.text.trim(),
      '--engine', selectedEngine,
      '--model', modelController.text.trim(),
      '--source-lang', sourceLangController.text.trim(),
      '--target-lang', targetLangController.text.trim(),
      '--pubready-mode', pubreadyMode,
      '--seed', seedController.text.trim(),
    ];
    final randomCount = int.tryParse(randomController.text.trim()) ?? 0;
    if (randomCount > 0) {
      args.addAll(['--random-count', '$randomCount']);
    }
    final outName = outNameController.text.trim();
    if (outName.isNotEmpty) {
      args.addAll(['--out', '$root/results/$outName']);
    }
    if (useOcr) args.add('--ocr');

    log('Commande: $pythonExecutable ${args.join(' ')}');
    try {
      final proc = await Process.start(pythonExecutable, args, workingDirectory: root);
      final subOut = proc.stdout.transform(utf8.decoder).transform(const LineSplitter()).listen((line) {
        _handleBackendLine(line);
      });
      final subErr = proc.stderr.transform(utf8.decoder).transform(const LineSplitter()).listen((line) {
        log('ERR: $line');
      });
      final code = await proc.exitCode;
      await subOut.cancel();
      await subErr.cancel();
      log('Process terminé avec code $code');
    } catch (e) {
      log('ERREUR lancement: $e');
    } finally {
      if (mounted) setState(() => running = false);
    }
  }

  void _handleBackendLine(String line) {
    if (line.trim().isEmpty) return;
    try {
      final data = jsonDecode(line) as Map<String, dynamic>;
      final event = data['event'];
      if (event == 'run_start') {
        setState(() => lastOut = data['out'] as String?);
        log('Démarrage ${data['stage']} → ${data['out']}');
      } else if (event == 'selection') {
        log('Pages: ${(data['pages'] as List).join(', ')}');
      } else if (event == 'progress') {
        log('[${data['current']}/${data['total']}] ${data['message']}');
      } else if (event == 'page_done') {
        log('OK ${data['tag']} · traduits=${data['translated_text_count']} · protégés=${data['protected_region_count']} · audit=${data['status']}');
      } else if (event == 'page_error') {
        log('KO ${data['tag']}: ${data['error']}');
      } else if (event == 'warning') {
        log('ATTENTION ${data['tag'] ?? ''}: ${data['message']}');
      } else if (event == 'run_done') {
        setState(() {
          lastOut = data['out'] as String?;
          lastContactSheet = data['contact_sheet'] as String?;
          lastReport = data['report'] as String?;
        });
        log('Résultats: ${data['out']}');
      } else {
        log(line);
      }
    } catch (_) {
      log(line);
    }
  }

  Future<void> openPath(String? path) async {
    if (path == null || path.isEmpty) return;
    try {
      await Process.start('xdg-open', [path]);
    } catch (e) {
      log('Impossible d’ouvrir $path: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      body: SafeArea(
        child: Row(
          children: [
            _SideBar(root: root, running: running),
            Expanded(
              child: CustomScrollView(
                slivers: [
                  SliverToBoxAdapter(child: _Header(onRun: running ? null : runPipeline)),
                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
                    sliver: SliverList(
                      delegate: SliverChildListDelegate([
                        Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(flex: 3, child: _inputColumn(scheme)),
                            const SizedBox(width: 18),
                            Expanded(flex: 2, child: _statusColumn(scheme)),
                          ],
                        ),
                        const SizedBox(height: 18),
                        _LogCard(logs: logs),
                      ]),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _inputColumn(ColorScheme scheme) {
    return Column(
      children: [
        _SectionCard(
          title: 'Document',
          subtitle: 'Choix du PDF et des pages à traiter',
          icon: Icons.picture_as_pdf_rounded,
          child: Column(
            children: [
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: pdfController,
                      decoration: const InputDecoration(
                        labelText: 'PDF source',
                        hintText: '/chemin/vers/document.pdf',
                        border: OutlineInputBorder(),
                      ),
                      onSubmitted: (_) => inspectPdf(),
                    ),
                  ),
                  const SizedBox(width: 10),
                  FilledButton.icon(onPressed: running ? null : pickPdf, icon: const Icon(Icons.folder_open), label: const Text('Choisir')),
                ],
              ),
              const SizedBox(height: 14),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: pagesController,
                      decoration: const InputDecoration(
                        labelText: 'Pages',
                        hintText: '1,3,10-12',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  SizedBox(
                    width: 125,
                    child: TextField(
                      controller: randomController,
                      decoration: const InputDecoration(
                        labelText: 'Random',
                        helperText: '0 = non',
                        border: OutlineInputBorder(),
                      ),
                      keyboardType: TextInputType.number,
                    ),
                  ),
                  const SizedBox(width: 10),
                  SizedBox(
                    width: 150,
                    child: TextField(
                      controller: seedController,
                      decoration: const InputDecoration(labelText: 'Seed', border: OutlineInputBorder()),
                      keyboardType: TextInputType.number,
                    ),
                  ),
                ],
              ),
              if (pageCount > 0) ...[
                const SizedBox(height: 12),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text('Sélection visuelle · $pageCount pages', style: Theme.of(context).textTheme.labelLarge),
                ),
                const SizedBox(height: 8),
                Container(
                  constraints: const BoxConstraints(maxHeight: 190),
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: scheme.surfaceContainerHighest.withOpacity(0.45),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: SingleChildScrollView(
                    child: Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: List.generate(pageCount, (i) {
                        final p = i + 1;
                        return FilterChip(
                          label: Text('$p'),
                          selected: selectedPages.contains(p),
                          onSelected: running
                              ? null
                              : (v) {
                                  setState(() {
                                    if (v) {
                                      selectedPages.add(p);
                                    } else {
                                      selectedPages.remove(p);
                                    }
                                    updatePagesFromSelection();
                                  });
                                },
                        );
                      }),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
        const SizedBox(height: 18),
        _SectionCard(
          title: 'Pipeline',
          subtitle: 'Niveau d’exécution et options techniques',
          icon: Icons.account_tree_rounded,
          child: Column(
            children: [
              Row(
                children: [
                  Expanded(
                    child: DropdownButtonFormField<String>(
                      value: selectedStage,
                      decoration: const InputDecoration(labelText: 'Niveau', border: OutlineInputBorder()),
                      items: stages.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
                      onChanged: running ? null : (v) => setState(() => selectedStage = v ?? selectedStage),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: DropdownButtonFormField<String>(
                      value: selectedEngine,
                      decoration: const InputDecoration(labelText: 'Moteur traduction', border: OutlineInputBorder()),
                      items: engines.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
                      onChanged: running ? null : (v) => setState(() => selectedEngine = v ?? selectedEngine),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              TextField(
                controller: modelController,
                decoration: const InputDecoration(labelText: 'Modèle', border: OutlineInputBorder()),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(child: TextField(controller: sourceLangController, decoration: const InputDecoration(labelText: 'Source', border: OutlineInputBorder()))),
                  const SizedBox(width: 10),
                  Expanded(child: TextField(controller: targetLangController, decoration: const InputDecoration(labelText: 'Cible', border: OutlineInputBorder()))),
                  const SizedBox(width: 10),
                  Expanded(
                    child: DropdownButtonFormField<String>(
                      value: pubreadyMode,
                      decoration: const InputDecoration(labelText: 'PubReady', border: OutlineInputBorder()),
                      items: pubreadyModes.map((s) => DropdownMenuItem(value: s, child: Text(s))).toList(),
                      onChanged: running ? null : (v) => setState(() => pubreadyMode = v ?? pubreadyMode),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: outNameController,
                      decoration: const InputDecoration(
                        labelText: 'Nom du dossier résultat optionnel',
                        hintText: 'ex: essai_page_82',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  FilterChip(
                    label: const Text('OCR'),
                    selected: useOcr,
                    onSelected: running ? null : (v) => setState(() => useOcr = v),
                  ),
                ],
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _statusColumn(ColorScheme scheme) {
    return Column(
      children: [
        _SectionCard(
          title: 'Résultats',
          subtitle: 'Sorties générées dans results/',
          icon: Icons.dashboard_customize_rounded,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _MetricTile(label: 'Dossier', value: lastOut ?? '—', icon: Icons.folder_rounded),
              const SizedBox(height: 10),
              FilledButton.tonalIcon(onPressed: lastOut == null ? null : () => openPath(lastOut), icon: const Icon(Icons.folder_open), label: const Text('Ouvrir le dossier')),
              const SizedBox(height: 8),
              FilledButton.tonalIcon(onPressed: lastContactSheet == null ? null : () => openPath(lastContactSheet), icon: const Icon(Icons.image_rounded), label: const Text('Voir contact_sheet')),
              const SizedBox(height: 8),
              OutlinedButton.icon(onPressed: lastReport == null ? null : () => openPath(lastReport), icon: const Icon(Icons.article_rounded), label: const Text('Rapport Markdown')),
            ],
          ),
        ),
        const SizedBox(height: 18),
        _SectionCard(
          title: 'Unités suivies',
          subtitle: 'Fichiers attendus selon le niveau choisi',
          icon: Icons.fact_check_rounded,
          child: Column(
            children: [
              _StageHint(stage: selectedStage),
              const SizedBox(height: 16),
              if (running) const LinearProgressIndicator(),
              if (!running)
                Text(
                  'Prêt. Lancement non-web, local, sans serveur.',
                  style: TextStyle(color: scheme.onSurfaceVariant),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _SideBar extends StatelessWidget {
  const _SideBar({required this.root, required this.running});

  final String root;
  final bool running;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: 260,
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [scheme.primaryContainer, scheme.surfaceContainerHighest],
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CircleAvatar(
            radius: 28,
            backgroundColor: scheme.primary,
            child: Icon(Icons.auto_awesome_motion, color: scheme.onPrimary),
          ),
          const SizedBox(height: 18),
          Text('vSense Studio', style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800)),
          const SizedBox(height: 6),
          Text('Contrôle local du pipeline docs_parser', style: Theme.of(context).textTheme.bodyMedium),
          const SizedBox(height: 24),
          _NavPill(icon: Icons.schema_rounded, label: 'PAGEPRINT'),
          _NavPill(icon: Icons.translate_rounded, label: 'PAGETRANSLATE'),
          _NavPill(icon: Icons.layers_rounded, label: 'BACKGROUND'),
          _NavPill(icon: Icons.draw_rounded, label: 'PAGERECONSTRUCT'),
          _NavPill(icon: Icons.verified_rounded, label: 'AUDITS'),
          const Spacer(),
          Text('Racine projet', style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: 6),
          SelectableText(root, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 16),
          Chip(
            avatar: Icon(running ? Icons.sync : Icons.check_circle_outline, size: 18),
            label: Text(running ? 'En cours' : 'Disponible'),
          ),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.onRun});

  final VoidCallback? onRun;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(24, 24, 24, 18),
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(28),
          gradient: LinearGradient(
            colors: [Theme.of(context).colorScheme.primary, Theme.of(context).colorScheme.tertiary],
          ),
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('docs_parser vSense Studio', style: Theme.of(context).textTheme.headlineMedium?.copyWith(color: Colors.white, fontWeight: FontWeight.w800)),
                  const SizedBox(height: 8),
                  Text('Sélectionne les pages, exécute une unité du pipeline, puis inspecte les artefacts dans results/.', style: Theme.of(context).textTheme.bodyLarge?.copyWith(color: Colors.white.withOpacity(.92))),
                ],
              ),
            ),
            const SizedBox(width: 18),
            FilledButton.icon(
              onPressed: onRun,
              style: FilledButton.styleFrom(backgroundColor: Colors.white, foregroundColor: Theme.of(context).colorScheme.primary, padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 18)),
              icon: const Icon(Icons.play_arrow_rounded),
              label: const Text('Lancer'),
            ),
          ],
        ),
      ),
    );
  }
}

class _SectionCard extends StatelessWidget {
  const _SectionCard({required this.title, required this.subtitle, required this.icon, required this.child});

  final String title;
  final String subtitle;
  final IconData icon;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      color: scheme.surface,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(backgroundColor: scheme.primaryContainer, child: Icon(icon, color: scheme.onPrimaryContainer)),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(title, style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
                      Text(subtitle, style: Theme.of(context).textTheme.bodySmall?.copyWith(color: scheme.onSurfaceVariant)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 18),
            child,
          ],
        ),
      ),
    );
  }
}

class _LogCard extends StatelessWidget {
  const _LogCard({required this.logs});

  final List<String> logs;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      color: scheme.surface,
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.terminal_rounded),
                const SizedBox(width: 8),
                Text('Journal', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w800)),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              height: 260,
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black,
                borderRadius: BorderRadius.circular(16),
              ),
              child: SingleChildScrollView(
                reverse: true,
                child: SelectableText(
                  logs.isEmpty ? 'Aucun événement.' : logs.join('\n'),
                  style: const TextStyle(color: Color(0xFFE6EDF3), fontFamily: 'monospace', fontSize: 13, height: 1.35),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricTile extends StatelessWidget {
  const _MetricTile({required this.label, required this.value, required this.icon});

  final String label;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest.withOpacity(.55),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Icon(icon, color: scheme.primary),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: Theme.of(context).textTheme.labelMedium),
                const SizedBox(height: 3),
                SelectableText(value, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _NavPill extends StatelessWidget {
  const _NavPill({required this.icon, required this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface.withOpacity(.55),
          borderRadius: BorderRadius.circular(99),
        ),
        child: Row(children: [Icon(icon, size: 18), const SizedBox(width: 8), Text(label)]),
      ),
    );
  }
}

class _StageHint extends StatelessWidget {
  const _StageHint({required this.stage});

  final String stage;

  @override
  Widget build(BuildContext context) {
    final files = switch (stage) {
      'pageprint' => ['source_*.png', 'pageprint_*.json', 'pageprint_bboxes_*.png'],
      'pagetranslate' => ['pageprint_*.json', 'pagetranslate_*.json', 'translated_input_data_*.json'],
      'pagereconstruct' => ['cleanbg_*.png', 'pagereconstruct_plan_*.json', 'pagereconstruct_overlay_*.png', 'reconstructed_*.png'],
      'view_background' => ['source_*.png', 'background_preview_*.png', 'cleanbg_*.png', 'background_compare_*.jpg'],
      'audit_translation_selection' => ['audit_translation_selection_*.json', 'audit_translation_selection_*.md'],
      'audit_text_survival' => ['audit_text_survival_*.json', 'text_survival_*.md', 'text_survival_*.csv'],
      _ => ['source_*.png', 'cleanbg_*.png', 'pageprint_bboxes_*.png', 'pagereconstruct_overlay_*.png', 'reconstructed_*.png', 'audit_*.json', 'pubready_*'],
    };
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(stage, style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w800)),
        const SizedBox(height: 8),
        ...files.map((f) => Padding(
              padding: const EdgeInsets.only(bottom: 5),
              child: Row(children: [const Icon(Icons.insert_drive_file_rounded, size: 16), const SizedBox(width: 7), Expanded(child: Text(f))]),
            )),
      ],
    );
  }
}
