> Copie versionnée du handoff. Le plan approuvé est dans `docs/plan-phase1.md` (copie de `~/.claude/plans/pasted-content-id-28cb-projet-squishy-ember.md`). Dépôt : https://github.com/AMoncade/gesture-remote (privé).

# Handoff → prochaine session ADMIN de gesture-remote (écrit le 2026-09-23)

**ÉTAT : étape 0 partielle, commitée sur `main` au SHA `f3a8b31`. Aucun worktree, aucune session exécutante, console deck arrêtée. Tu reprends le rôle d'admin (skill `claude-deck:deck`, régime `SOIS AUTONOME`) au point « finir l'étape 0 ».**

## 1. À lire d'abord, en entier (pas seulement les titres)
1. Le plan approuvé, **à ne pas modifier** : `C:\Users\adrie\.claude\plans\pasted-content-id-28cb-projet-squishy-ember.md`. Surtout sa dernière section, « Exécution — régie `/deck` », qui fixe la topologie, les 3 lots, les contrats et le déroulé.
2. Les skills `claude-deck:deck`, `session-tasking`, `parallel-sessions` et `session-handoff`. Cette dernière exige de charger les deux autres **avant** d'écrire un brief.
3. Le journal du tour : `C:\Users\adrie\.claude\plugins\cache\claude-deck\claude-deck\2.0.0\deck-journal.jsonl`. Les décisions déjà prises y sont, avec leur motif.

## 2. Ce que l'utilisateur a demandé (ses mots)
- « change nothing, use automode, but use my /deck skill before doing anything to set up how we gonna do it » : le plan reste tel quel, on est en mode auto, la régie deck s'applique. Je l'ai lu comme `SOIS AUTONOME` ; le plan annonçait cette lecture et l'utilisateur l'a approuvé.
- « ouvre d'autres sessions pour t'aider avec /deck » : **c'est l'admin qui ouvre les sessions**, sans attendre que l'utilisateur ouvre des terminaux.
- Sa priorité, en fonctionnalités : ouvrir des apps, ouvrir des pages web, lancer des macros Python. Apple Music et les touches média sont un bonus.

## 3. Ce qui est fait, et ce qui le prouve
Chaque ligne indique comment je l'ai vérifié.
- **Venv** `C:\Users\adrie\gesture-remote\.venv` (Python 3.13.7), installation éditable `-e ".[dev]"`. *Vérifié* : j'ai importé mediapipe 1.0.1, opencv-contrib-python 5.0.0.93, numpy 2.5.3, pyautogui 0.9.54, pydantic 2.13.5, pyyaml 6.0.3, pytest 9.1.1 et ruff 0.16.8, puis cherché des octets nuls dans `site-packages` : **0**.
  - ⚠️ Le premier venv avait été corrompu par l'interruption de la session (508 fichiers `.py` remplis d'octets nuls). Je l'ai supprimé et recréé. **Leçon : ne jamais lancer `pip install` en tâche de fond dans une session qui peut se fermer.**
- **`requirements.lock`** : 39 lignes, écrit via `cmd /c "... pip freeze --exclude-editable > requirements.lock"`, donc en UTF-8 et pas en UTF-16.
- **Modèles et images d'exemple** dans `models/` (gitignorés). Leurs SHA-256 sont **épinglés** dans `tools/download_models.py`. *Vérifié* : relancé, les 6 fichiers affichent « pinned OK ».
- **Porte (a) franchie.** *Vérifié* par un script ad hoc, pas encore par `check_setup.py`, qui n'existe pas. Le GestureRecognizer (mode IMAGE, `model_asset_buffer`) sur les 4 images officielles donne `Thumb_Up` 0.73, `Thumb_Down` 0.77, `Victory` 0.91 et `Pointing_Up` 0.82, chaque fois avec 21 points.
- **Fait établi, qui contredit la doc** : la classe 0 s'appelle **`None`**, pas `Unknown`. Méthode : lecture du `labels.txt` embarqué dans `gesture_recognizer.task`, un zip imbriqué dans `hand_gesture_recognizer.task/canned_gesture_classifier.tflite/labels.txt`. Liste exacte : `None, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou`. La latéralité vaut `Left`/`Right` (`hand_landmarks_detector.tflite/handedness.txt`). La table du plan garde `None` **et** `Unknown` → `none` : c'est sans risque.
- **Deux vérifications que le plan avait laissées ouvertes** : `hasattr(pyautogui, "isValidKey")` vaut **True**, et OpenCV 5 garde `CAP_DSHOW` et `CAP_MSMF` (vérifié avec `hasattr`, **pas encore** avec la caméra ouverte).
- **Contrats du tour, commités dans `f3a8b31`** :
  - `src/gesture_remote/observation.py` : `NONE_LABEL`, `Handedness`, `HandObservation` (avec `handedness_score`) et `FrameObservation.primary()`. `hands` est un **tuple**, pas une liste comme dans l'esquisse du plan ; les dataclasses sont gelées.
  - `src/gesture_remote/config.py` : la **couche modèles seulement**. Réglages, `KeysAction`/`LaunchAction`/`UrlAction`/`ScriptAction`, l'union `ActionSpec` discriminée sur `type`, `Config`, `AppResolutionError`. Les validateurs y sont sans contexte : une seule cible pour `launch`, URL en http(s) avec hôte, script en `.py`, touches non vides. *Vérifié* par un essai rapide : la config d'exemple du plan passe, et les cas `type: cmd`, `launch` vide, `ftp://`, `.txt` et `confirm:` sont refusés. Les messages « arrive en phase 3 » restent à écrire par le lot B, dans le chargeur.
  - `tests/conftest.py` : garde d'import (`pytest_sessionstart` lève `UsageError` si `gesture_remote` ne vient pas du `src/` de l'arbre courant) et `FakeClock`, plus `pythonpath = ["src"]` dans `pyproject.toml`.
  - `tests/test_privacy.py` scanne `src/`, `tools/` et `macros/` ; le détecteur est éprouvé sur un exemple positif. `macros/example_hello.py` affiche une MessageBox en `SETFOREGROUND|TOPMOST`.
- **Qualité au SHA `f3a8b31`** : `pytest` 2 passés, 0 sauté ; `ruff check` et `ruff format --check` propres.
  - Ce que cette suite ne voit pas : elle ne teste que le garde anti-écriture. **Le garde d'import n'a jamais été éprouvé en échec** (voir 4.3).
- Journal deck : 3 décisions inscrites (régime, découpage, topologie). **Aucun** `deck-claims.json` n'existe encore.

## 4. Ce qui reste à faire dans l'étape 0, dans l'ordre
1. **Écrire `tools/check_setup.py`** (spécification dans le plan, section Arborescence). Il est autonome et n'importe pas le paquet. Sections :
   - modèles et images d'exemple : noms bruts des classes, plus la lecture du `labels.txt` décrite en 3 ;
   - caméra : DSHOW puis MSMF, 640×480, 60 frames, fps mesurés, luminosité moyenne, aucune frame écrite ;
   - `--apps TEXTE` : `powershell -NoProfile -NonInteractive -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-StartApps | ConvertTo-Json -Compress"`, sans fenêtre, décodé en `utf-8-sig`, en acceptant une liste **ou** un objet unique ;
   - `--press TOUCHE` : `FAILSAFE=False`, `PAUSE=0`.

   ⚠️ `test_privacy.py` échoue si les mots `imwrite`, `VideoWriter` ou `imencode` apparaissent dans `tools/`, même en commentaire.
2. **Portes (b) et (c), plus le contrôle UTF-8** :
   - `check_setup.py` : la caméra s'ouvre ; noter le backend retenu et les fps ;
   - `--apps "Apple Music"` doit retrouver `AppleInc.AppleMusicWin_nzyj5cx40ttqa!App` (AppID *vérifié* au planning avec `Get-StartApps`) ;
   - `--apps Param` : un nom accentué, pour éprouver l'UTF-8.

   La caméra est une **ressource exclusive** : seul l'admin l'ouvre.
3. **Éprouver le garde d'import** : c'est la seule chose qui rend les worktrees fiables. Une fois un worktree créé :
   - `C:\Users\adrie\gesture-remote\.venv\Scripts\python -m pytest -q` lancé **depuis le worktree** doit passer ;
   - relancé avec `-p no:python_path`, il doit **échouer** en `UsageError`, parce que l'import retombe sur le `src/` de `main` via l'installation éditable.

   Si le premier cas échoue, les worktrees ne sont pas isolés : un venv par worktree devient nécessaire (le repli prévu au plan). *Raisonné, pas observé* : l'installation éditable de setuptools passerait par un `.pth` ajouté en fin de `sys.path`, donc `pythonpath` devrait gagner.
4. **Écrire `CLAUDE.md` du projet** (en anglais ; contenu décrit dans le plan) : règles dures, commandes, architecture, sémantique du moteur, décisions et pièges (dont les faits de la section 3 ci-dessus), avancement, et un tableau de propriété pour le tour 1. C'est la **source des commandes** pour les sessions.
5. Commit par chemins explicites, puis `git worktree add ../gesture-remote-vision -b lot/vision`, et pareil pour `decision` et `actions`, depuis ce commit.

## 5. Ensuite : ouvrir les sessions et déléguer
- **Ouvrir les 3 sessions toi-même.** Vérifié dans `claude --help` le 2026-09-23 :
  - `--permission-mode auto` existe, ainsi que `-n/--name` et un `prompt` positionnel ;
  - `--bg` lance une session en arrière-plan ;
  - `wt.exe` est présent.

  Exemple (non essayé) :
  `wt -w new -d C:\Users\adrie\gesture-remote-vision claude --permission-mode auto -n gr-vision "Lis et exécute le brief C:\Users\adrie\.claude\plans\gesture-remote-brief-vision.md"`.

  Vérifier ensuite qu'elles apparaissent dans `ListAgents` et dans `node C:\Users\adrie\.claude\plugins\cache\claude-deck\claude-deck\2.0.0\scripts\sessions.js`.
- **Briefs** : un fichier par lot à côté du plan. Contenu et règles : section « Chaque brief contient » du plan, avec la revendication en tête, le raisonnement, les fichiers possédés et lus, les dépendances croisées, « ce que je n'ai pas vérifié » et la définition de fini. Y reporter les faits de la section 3, dont le SHA de base.
- **Attributions** dans `deck-claims.json` (même dossier que le journal) ; chaque décision va au journal.
- **Console** : `node C:\Users\adrie\.claude\plugins\cache\claude-deck\claude-deck\2.0.0\scripts\serve.js`, port 7788, verrou de lancement unique. Elle est morte avec l'ancienne session : il faut la relancer.
- Après les lots : compte rendu de chaque session, puis fusion par `claude-admin` (ou par toi), puis étape 5 et balayage des coutures, puis étape 6 avec l'utilisateur.

## 6. Pièges constatés sur ce poste
- Les **agents d'arrière-plan `Plan` ont calé deux fois** (« no progress for 600s ») pendant la planification. Préférer de vraies sessions `claude` ; si un agent cale, reprendre soi-même.
- **CRLF** : git avertit « LF will be replaced by CRLF ». `core.autocrlf` n'est pas réglé au niveau global, ce qui laisse soupçonner un réglage `system` (non vérifié ; `git config --show-origin core.autocrlf` le dira). Les worktrees auront donc des fichiers en CRLF. Python s'en accommode, mais un remplacement de texte doit normaliser les fins de ligne.
- `python` sur le PATH est l'alias du Store : toujours utiliser `.venv\Scripts\python`, par chemin absolu depuis les worktrees.
- Écrire avec l'outil Write ce qui contient des `\`, par exemple `shell:AppsFolder\` : les heredocs Bash les abîment.
- Quota au moment du planning : 20 % de la fenêtre de 5 h. Relancer `quota.js` avant de distribuer du travail.

**État : je n'ai rien en cours, je suis en attente, et plus rien ne dépend de moi.**
