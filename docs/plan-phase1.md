# gesture-remote — Plan de la phase 1

## Contexte

Projet perso pour apprendre la vision et le ML, 100 % local et gratuit. Une app Windows en arrière-plan regarde la webcam et transforme des gestes de la main en commandes.

**Ta priorité : les fonctions**, c'est-à-dire ouvrir des apps, ouvrir des pages web et lancer des macros (des scripts Python). Les touches média (Apple Music) sont un bonus.

La phase 1 livre donc toute la chaîne : webcam → MediaPipe (gestes intégrés) → machine à états → actions `keys`, `launch`, `url` et `script`, pilotée par `config.yaml`. Elle reste simple : pas encore de court/tenu, de combos, de modes, de confirmations ni de barre d'état.

**Critère de réussite** : au bureau (< 1,5 m), chaque geste mappé déclenche sa fonction, et rien ne se déclenche pendant une activité normale. L'architecture accueille les phases 2 à 4 sans réécriture. **Aucun code avant ton approbation.**

Tes réponses :
- distance : au bureau (< 1,5 m) ;
- 🤟 arme et désarme dès la phase 1, armé au démarrage ;
- ✊ = muet ;
- « reste simple » ;
- « le plus important c'est les fonctions ».

État du poste (vérifié aujourd'hui) :
- Windows 11, git 2.45.1 (identité configurée), pas d'uv.
- Python 3.13.7 64 bits est le seul installé (lanceur `py`). Le `python` du PATH est l'alias du Microsoft Store, pas un vrai Python.
- Une seule caméra (ACER HD User Facing Camera), donc index 0 : la question est réglée. L'accès caméra des applications de bureau est autorisé.
- Apple Music est installé (AppID `AppleInc.AppleMusicWin_nzyj5cx40ttqa!App`). Le runtime VC++ est présent.
- `C:\Users\adrie\gesture-remote` n'existe pas encore (tes projets sont dans `C:\Users\adrie`).

## Écarts par rapport à ta spec (à corriger si je t'ai mal compris)

- `launch`, `url` et `script` passent de la phase 3 à la phase 1. `cmd`, `lock`, `macro` (liste d'étapes) et `confirm` restent en phase 3.
- Tes « macros » deviennent le type **`script`** : un fichier Python dans `macros/`. Le type `macro` (liste d'étapes) de ton exemple reste possible en phase 3, s'il sert encore.
- Ton mapping passe sous `bindings:`, à côté de `settings:`, avec la même syntaxe.
- ✌️ passe de « piste suivante » à « page web ». Sur les 7 gestes intégrés, 🤟 sert déjà à l'armement ; il faut donc de la place pour les fonctions. Tout se change dans `config.yaml`.

## Faits vérifiés → décisions

| Fait vérifié le 2026-09-23 (source) | Décision |
|---|---|
| **mediapipe 1.0.1** (publié le 2026-08-14). Depuis la 0.10.30 (déc. 2025), un seul wheel Windows `py3-none-win_amd64`, avec des bindings ctypes vers une API C. La doc officielle dit « Python 3.9 and later » (PyPI, `setup_python`). | **Python 3.13** + `mediapipe==1.0.1`. Si la porte (a) échoue, repli sur 0.10.35, puis sur Python 3.12 + 0.10.21 (dernier wheel `cp312`). |
| `mp.solutions` n'existe plus depuis la 0.10.30 (issues GitHub #6200, #6204, #6261). | Uniquement `mediapipe.tasks.python.vision`. Les 21 points sont dessinés avec notre propre table de connexions. |
| `recognize_for_video(image, timestamp_ms)`, avec des timestamps *monotonically increasing* (source `master`). Le modèle peut être passé en mémoire (`model_asset_buffer`). | Mode **VIDEO**, synchrone. Un helper testé garantit des ms strictement croissantes. |
| Classe 0 = `Unknown` selon la doc (`None` dans les anciens résultats). Les autres : `Closed_Fist`, `Open_Palm`, `Pointing_Up`, `Thumb_Down`, `Thumb_Up`, `Victory`, `ILoveYou`. | Table explicite vers `none`, `closed_fist`, `open_palm`, `pointing_up`, `thumb_down`, `thumb_up`, `victory`, `i_love_you`. `None` **et** `Unknown` donnent `none`. Un nom inconnu déclenche un avertissement unique. |
| Modèles `…/float16/1/gesture_recognizer.task` et `…/float16/1/hand_landmarker.task` : ils existent (8 373 440 et 7 819 105 octets, identiques à `/latest/`, requêtes HEAD), tout comme les 4 images d'exemple officielles. Entrées : 192×192 et 224×224. | Téléchargement épinglé sur `/1/`, SHA-256 affiché puis épinglé. Au bureau, 640×480 suffit. |
| Latéralité « déterminée en supposant une image miroir » (doc Hands). | On retourne l'image **avant** l'inférence. Vérifié au jour 1. |
| pyautogui 0.9.54 : `press()` ignore **en silence** une touche inconnue. `FAILSAFE` lève une exception si la souris est en (0,0). `PAUSE` ajoute 0,1 s. | Touches validées au chargement (`isValidKey`, avec repli sur `KEYBOARD_KEYS` s'il manquait dans la 0.9.54), `FAILSAFE=False`, `PAUSE=0`. |
| mediapipe installe `opencv-contrib-python` **5.0**.0.93 (numpy ≥ 2). | Ne jamais ajouter `opencv-python`. Backend caméra `auto` (voir Risques). |
| GPU « limited to Ubuntu ». `BaseOptions` passe un bundle certifi à la couche C. | CPU. On vérifie qu'il n'y a **aucune** connexion réseau à l'exécution. |
| `launch`, `url` et `script` n'utilisent que la bibliothèque standard (`os.startfile` avec `arguments` depuis Python 3.10, `subprocess`, `webbrowser`). | Aucune dépendance en plus. |

## Architecture

```
Camera ─BGR miroir─▶ IdleThrottle ─▶ Recognizer ─FrameObservation─▶ GestureEngine ─EngineEvent─▶ ActionDispatcher ─▶ keys | launch | url | script
(capture.py)                          (recognition.py)               (segments.py, engine.py)     (actions/)          └─▶ Feedback (bips)
                                          └──────────▶ DebugView (--debug, rien n'est jamais écrit sur disque)
```

Types et interfaces stables :
```python
class HandObservation:      label: str; score: float; handedness: Handedness
                            landmarks: np.ndarray; world_landmarks: np.ndarray   # (21, 3) chacun
class FrameObservation:     hands: list[HandObservation]; image_size: tuple[int, int]
class Recognizer(Protocol): labels: frozenset[str]; def recognize(rgb, timestamp_ms) -> FrameObservation; def close() -> None
class GestureEngine:        def update(hand: HandObservation | None, now: float) -> list[EngineEvent]   # Triggered | Ignored | ArmedChanged
class ActionDispatcher:     def dispatch(action: ActionSpec) -> None   # non bloquant
```
- La boucle principale est `Pipeline.run(stop: threading.Event)`. En phase 3, pystray pourra prendre le thread principal sans réécriture.
- Le temps est injecté (`now`), ce qui rend la machine à états testable avec une horloge scriptée.

| Phase | Ajout | Ce qui ne change pas |
|---|---|---|
| 2 | `recognition/` v1 : HandLandmarker, `features` et un modèle sklearn. `tools/record.py` écrit des données **brutes** en CSV : points image et monde, latéralité et son score, taille d'image, label, session, t. Un fichier par session dans `data/`, jamais écrasé. `tools/train.py` embarque `FEATURES_VERSION` et les labels dans le modèle, et le chargeur refuse un modèle périmé. | `Recognizer`, le moteur, les actions. La config accepte les nouveaux labels via `recognizer.labels`. |
| 3 | Clés `x_hold` et `a>b` ; dans `engine.py`, court au relâchement, tenu à `hold_s`, historique des onsets pour les combos, modes, état de confirmation. Types `cmd`, `lock`, `macro`, `confirm`. Barre d'état, raccourci global (`RegisterHotKey`), bascule caméra, démarrage avec Windows. | `segments.py`, la signature de `dispatch`, les handlers existants. |
| 4 | Détecteurs continus qui consomment les mêmes `landmarks` : pincement vers pycaw, balayages. Le mode diapos est un mode de la phase 3. | Tout le reste. |

## Machine à états (phase 1) : pure, temps injecté

1. **Vote** : chaque frame vote pour son label si une main est détectée, si le label n'est pas `none` et si `score ≥ min_score(label)`. Sinon, elle ne vote pour rien.
2. **Un segment par geste** : après `stable_frames(label)` votes identiques consécutifs, si ce geste n'a pas de segment vivant, son segment commence. Un segment reste vivant tant que son geste a été vu dans les dernières `release_s`. Conséquences :
   - un geste qui revient avant `release_s`, même après un passage par un autre geste, **continue** son segment (pas de nouveau tir) ;
   - un autre geste peut démarrer son propre segment sans attendre.
3. **Tir à l'onset seulement** : il faut que le geste soit mappé, que le moteur soit armé et qu'il ne soit pas en cooldown. Sinon, le segment est **ignoré toute sa vie**, et le moteur émet `Ignored(label, raison)`. Cet événement va dans le journal et l'overlay, sans son : il ne faut pas biper pendant un appel quand tu es désarmé. Ce qui en découle :
   - une tenue de 5 s ne déclenche qu'**un** tir ;
   - un geste stabilisé pendant le cooldown ne fait rien tant qu'on ne l'a pas relâché puis refait. C'est la variante prudente : avec un tir différé à la fin du cooldown, ✊ partirait si tu posais le poing sur le bureau juste après un geste.
4. **`repeat_while_held`** (`keys` seulement) : nouveau tir après `repeat_delay_s`, puis toutes les `repeat_interval_s`, uniquement sur les frames où le geste est visible.
5. **Chaque tir** relance le cooldown global.
6. **Armement** : 🤟 bascule entre armé et désarmé, avec un bip montant ou descendant. Désarmé, seul 🤟 compte.
   - 🤟 **échappe au cooldown** : sinon, il serait ignoré juste après un 👍 répété, justement quand tu veux désarmer.
   - Il reste limité à une bascule par tenue, et une bascule relance le cooldown des autres gestes.
7. **(Re)démarrage** du moteur (lancement, rechargement de la config) : il démarre en cooldown, pour qu'un geste déjà tenu ne se déclenche pas.

`segments.py` expose, pour chaque segment, l'onset, `started_at` et `last_seen`. La phase 3 (court au relâchement, tenu, combos) s'appuie dessus sans le modifier.

| `settings.engine` | Défaut | Rôle |
|---|---|---|
| `min_score` | 0.6 | confiance minimale pour qu'une frame vote |
| `stable_frames` | 10 | votes consécutifs avant l'onset (≈ 0,33 s à 30 fps) |
| `release_s` | 0.8 | absence qui termine un segment. Plus long = moins de doubles tirs ; refaire le même geste demande 0,8 s de pause |
| `cooldown_s` | 1.0 | pause globale après chaque tir |
| `repeat_delay_s` / `repeat_interval_s` | 0.5 / 0.2 | cadence de répétition |
| `arm_gesture` / `start_armed` | `i_love_you` / `true` | armement |
| `per_gesture` | `i_love_you: {stable_frames: 15}` | surcharges (`min_score`, `stable_frames`) |

Inférence réduite : tant qu'une main a été vue dans la dernière seconde, chaque frame passe à l'inférence. Sinon, environ 5 Hz. La caméra est lue à chaque frame, donc son tampon ne vieillit jamais.

## Actions (phase 1)

| Type | Champs | Exécution | Garde-fous |
|---|---|---|---|
| `keys` | `keys: [playpause]` ou une combinaison `[ctrl, shift, esc]` ; `repeat_while_held` | `pyautogui.hotkey(*keys)` | noms validés au chargement |
| `launch` | exactement **un** champ parmi : `app:` (nom du menu Démarrer), `app_id:`, `path:` (.exe, .lnk ou fichier, `%VAR%` développées, `args` optionnels) | app → `explorer.exe shell:AppsFolder\<AppID>` ; path → `os.startfile(path, arguments=…)` | voir ci-dessous |
| `url` | `url:` | `webbrowser.open_new_tab` (navigateur par défaut) | http(s) et hôte validés avec `urlsplit` |
| `script` | `path: macros/x.py`, `args` optionnels | le Python du venv (`sys.executable`) dans un **processus séparé** | voir ci-dessous |

Garde-fous de `launch` :
- `app:` est résolu **au chargement** via `Get-StartApps`, lancé avec PowerShell `-NoProfile -NonInteractive` et sans fenêtre.
- La sortie est forcée en UTF-8 et décodée en `utf-8-sig`, au cas où un BOM précède le JSON.
- Le résultat est mis en cache pour la session, puis relu une fois si le nom est introuvable.
- Nom introuvable → erreur avec suggestions. Nom ambigu → erreur qui liste les AppID.
- Le code de sortie d'explorer n'est pas interprété.

Garde-fous de `script` :
- lancé sans fenêtre (`CREATE_NO_WINDOW`), avec `cwd` = dossier du script, `PYTHONUTF8=1` et `PYTHONUNBUFFERED=1`. Sans ce dernier, la sortie d'un script bloqué n'arriverait jamais dans le log ;
- la sortie va dans `logs/scripts/<nom>.log`, et le code de sortie ainsi que la durée sont journalisés ;
- **une seule instance à la fois par script** : un 2ᵉ tir pendant qu'il tourne est ignoré et journalisé ;
- le fichier doit exister et finir par `.py` ;
- un script en cours continue de tourner si l'app se ferme. C'est voulu : le processus est détaché.

Le dispatcher :
- tient un registre type → handler ;
- exécute sur un seul thread de travail, pour que la vision ne bloque jamais ;
- journalise les exceptions au lieu de les propager ; `--dry-run` journalise sans exécuter ;
- **vit toute la session**, avec son thread, le registre des scripts en cours et le cache du menu Démarrer. Un rechargement de config ne change que le mapping ; sinon, un script en cours pourrait être relancé en double ;
- à la sortie, annule les actions en file (`cancel_futures=True`).

Pour éprouver la chaîne, une macro d'exemple est livrée : `macros/example_hello.py`, qui ouvre une boîte de dialogue Windows via ctypes.

## Config (`config.yaml`, versionnée, rechargée à chaud)

```yaml
settings:
  camera:      { index: 0, width: 640, height: 480, mirror: true, backend: auto }   # auto = DSHOW puis MSMF
  recognition: { model: models/gesture_recognizer.task, num_hands: 1 }
  engine:
    min_score: 0.6
    stable_frames: 10
    release_s: 0.8
    cooldown_s: 1.0
    repeat_delay_s: 0.5
    repeat_interval_s: 0.2
    arm_gesture: i_love_you
    start_armed: true
    per_gesture: { i_love_you: { stable_frames: 15 } }
  idle:     { fps: 5, after_s: 1.0 }
  feedback: { sound: true }
bindings:
  open_palm:   { type: keys,   keys: [playpause] }
  victory:     { type: url,    url: "https://studium.umontreal.ca" }
  pointing_up: { type: launch, app: "Apple Music" }
  thumb_down:  { type: script, path: macros/example_hello.py }
  thumb_up:    { type: keys,   keys: [volumeup], repeat_while_held: true }
  closed_fist: { type: keys,   keys: [volumemute] }
```

**Validation** (pydantic v2, union discriminée sur `type`, `extra="forbid"`). Les chemins relatifs sont résolus depuis le dossier de `config.yaml`. Tout est **refusé avec un message clair**, jamais ignoré :
- champ inconnu ;
- label hors de `recognizer.labels`, que ce soit dans `bindings`, `per_gesture` ou `arm_gesture` ;
- `none` mappé ;
- 🤟 mappé ;
- touche invalide ;
- `launch` avec zéro ou plusieurs champs parmi `app`/`app_id`/`path`, ou chemin absent ;
- URL invalide ;
- script absent ou qui n'est pas un `.py` ;
- `repeat_while_held` ailleurs que sur `keys` ;
- `x_hold`, `a>b`, `cmd`, `lock`, `macro`, `confirm` : message « arrive en phase 3 ».

**Rechargement à chaud** : chaque seconde, on regarde `(mtime_ns, taille)`.
- Fichier valide : on remplace le mapping et on recrée le moteur (en cooldown). Le dispatcher garde son état.
- Fichier invalide : on garde l'ancienne config et on journalise l'erreur une seule fois.
- Changement dans `camera` ou `recognition` : message « redémarrage requis ».

## Arborescence (phase 1)

```
gesture-remote/
├─ .gitignore  CLAUDE.md  README.md  pyproject.toml  requirements.lock  config.yaml
├─ macros/example_hello.py         # tes macros Python vivent ici (versionnées)
├─ models/.gitkeep                 # *.task et samples/ ignorés
├─ tools/
│  ├─ download_models.py           # 2 modèles + 4 images d'exemple, fichier temporaire puis renommage, SHA-256
│  └─ check_setup.py               # mediapipe + images d'exemple (noms bruts des classes), caméra (backend, résolution, fps, image noire ?),
│                                  #   --apps TEXTE (noms et AppID du menu Démarrer), --press TOUCHE
├─ src/gesture_remote/
│  ├─ __init__.py  __main__.py     # python -m gesture_remote [--config] [--debug] [--dry-run] [-v]
│  ├─ app.py                       # câblage + Pipeline.run(stop)
│  ├─ observation.py  capture.py  recognition.py  features.py
│  ├─ segments.py                  # votes → segments par geste
│  ├─ engine.py                    # segments → tirs (onset, répétition, cooldown, armement)
│  ├─ config.py  feedback.py  debug_view.py  logging_setup.py
│  └─ actions/  dispatcher.py  keys.py  launch.py  start_apps.py  url.py  script.py
└─ tests/  conftest.py (FakeClock, scripts de frames, faux presseur/popen/startfile) + un fichier par module
```

`features.py` :
- correction d'aspect : on multiplie x et z par w/h ;
- origine au poignet ;
- division par |p9| en 3D ;
- miroir de x si main gauche ;
- résultat : `float32[63]` ; exception si l'échelle est dégénérée.

Volontairement **pas** invariant en rotation, puisque 👍 et 👎 ne diffèrent que par une rotation. Testé dès maintenant, utilisé en phase 2.

## Dépendances (Python 3.13, dernières versions compatibles au 2026-09-23)

| Paquet | Version | Rôle |
|---|---|---|
| mediapipe | ==1.0.1 | reconnaissance. Tire numpy 2.5.x, opencv-contrib-python 5.0.0.93, matplotlib, sounddevice, absl-py, flatbuffers, certifi |
| pyautogui | ==0.9.54 | touches (source seulement : pip construit un wheel pur Python) |
| pydantic | >=2.13,<3 (2.13.5) | schéma et validation |
| PyYAML | >=6.0.3,<7 | `safe_load` |
| *dev* pytest / ruff | >=9.1,<10 / >=0.16,<0.17 | tests, lint et format |
| *plus tard* | scikit-learn 1.9.1 (ph. 2) · pystray 0.19.5 + Pillow 12.3.0 (ph. 3) · pycaw 20260921 + comtypes 1.4.17 (ph. 4) | disponibles en cp313, épinglés au début de leur phase |

Les versions exactes sont figées dans `requirements.lock` (`pip freeze --exclude-editable`, en UTF-8) et versionnées. Pas de mypy en phase 1 : l'inspection de PyCharm suffit.

## Setup

```powershell
New-Item -ItemType Directory $HOME\gesture-remote | Out-Null; Set-Location $HOME\gesture-remote
git init -b main
py -3.13 -m venv .venv                          # PyCharm détecte .venv tout seul
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
cmd /c ".venv\Scripts\python -m pip freeze --exclude-editable > requirements.lock"   # UTF-8 (le > de PowerShell 5.1 écrit en UTF-16)
.\.venv\Scripts\python tools\download_models.py
.\.venv\Scripts\python tools\check_setup.py                        # modèle, images d'exemple, caméra
.\.venv\Scripts\python tools\check_setup.py --apps "Apple Music"   # résolution menu Démarrer
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m gesture_remote --debug --dry-run
```
On appelle toujours `.venv\Scripts\python`, jamais `python`, qui est l'alias du Store.

`.gitignore` : `.venv/`, `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`, `models/*.task`, `models/samples/`, `data/`, `logs/`, `.idea/`.

## `CLAUDE.md` du projet (en anglais, comme le code)

- **Purpose** et **règles dures** :
  - 100 % local : aucun réseau à l'exécution, sauf `tools/download_models.py` ;
  - les images restent en mémoire : ni `imwrite`, ni `VideoWriter`, ni `imencode` ;
  - on ne journalise jamais d'image ni de points ;
  - code en anglais, type hints, petits modules, `logging` (pas de `print` dans `src/`).
- **Architecture** : chaîne, rôle de chaque module, interfaces stables, coutures par phase.
- **Conventions** :
  - labels en snake_case ;
  - ajouter un type d'action = modèle + handler + test (le test d'exhaustivité l'impose) ;
  - les macros sont des scripts dans `macros/`, leurs `print()` vont dans `logs/scripts/` ;
  - toute affirmation « tests verts » cite le SHA du commit.
- **Commandes**, puis **décisions et pièges** : le tableau « Faits vérifiés », la sémantique du moteur, l'UTF-8 de `Get-StartApps`, l'état du dispatcher qui survit au rechargement.
- **Avancement** : cases par phase, cochées avec le SHA à la fin de chaque phase.

## Tests pytest (sans webcam ; chacun doit pouvoir échouer)

- `test_features` :
  - invariance par translation et par échelle ;
  - main gauche = miroir de la droite → vecteurs identiques ;
  - poignet = 0 et |p9| = 1 ;
  - une main tournée de 90° dans une image 4:3 garde ses distances (échoue sans la correction d'aspect) ;
  - échelle dégénérée → exception ;
  - forme et type.
- `test_segments` / `test_engine` (horloge scriptée à 30 fps) :
  - tir à la 10ᵉ frame, pas à la 9ᵉ ;
  - une frame sous le seuil remet le compte à zéro ;
  - une alternance de labels ne tire jamais ;
  - tenue de 5 s → **exactement 1** tir ;
  - trou de 0,5 s → toujours 1 tir ;
  - ✋ → ☝️ (0,4 s) → ✋ : pas de 2ᵉ tir de ✋ ;
  - relâcher ≥ `release_s` puis refaire → 2ᵉ tir, jamais avant la fin du cooldown ;
  - geste stabilisé pendant le cooldown → `Ignored("cooldown")`, puis rien jusqu'au relâchement ;
  - répétition : nombre exact de tirs sur 2 s, arrêt au relâchement ;
  - `none` et geste non mappé → jamais ;
  - 🤟 : désarmé → `Ignored("disarmed")`, puis réarmement ;
  - 🤟 désarme même juste après une répétition de 👍 (il échappe au cooldown) ;
  - surcharges `per_gesture` ;
  - pas de tir au redémarrage si le geste est déjà tenu.
- `test_config` :
  - le `config.yaml` livré se charge (avec le vrai `isValidKey` et un faux menu Démarrer) ;
  - chaque règle de validation ci-dessus produit son message ;
  - `ConfigStore` : valide → invalide (ancienne gardée, une seule erreur journalisée) → valide.
- `test_actions` :
  - `keys` → combinaison exacte au faux presseur ;
  - `url` → URL exacte ;
  - `launch` `path` → `startfile(path, arguments)` ;
  - `launch` `app` → `["explorer.exe", "shell:AppsFolder\\<AppID>"]` ;
  - `script` → argv, `cwd`, drapeaux et environnement via un faux popen ;
  - un 2ᵉ tir pendant que le script tourne est ignoré, puis accepté après sa fin ;
  - **après un rechargement de config, le script en cours est toujours vu comme en cours** ;
  - **un vrai sous-processus** : un mini-script dans `tmp_path` écrit « é » sur sa sortie, et on vérifie le log, ce qui teste à la fois l'UTF-8 et la redirection ;
  - dry-run → rien d'exécuté ;
  - une exception est journalisée, pas propagée ;
  - l'adaptateur pyautogui met `FAILSAFE=False` et `PAUSE=0` (faux module, aucune touche envoyée) ;
  - **exhaustivité** : chaque type du schéma a un handler.
- `test_start_apps` :
  - JSON en liste **et** en objet unique (piège de `ConvertTo-Json`) ;
  - sortie précédée d'un BOM ;
  - noms accentués ;
  - correspondance exacte ou insensible à la casse ;
  - nom ambigu → AppID listés ;
  - nom absent → suggestions.
- `test_recognition` :
  - conversion d'un faux résultat MediaPipe (`None`/`Unknown` → `none`, `ILoveYou` → `i_love_you`) ;
  - nom inconnu → avertissement unique ;
  - liste de gestes vide ;
  - latéralité ;
  - `StrictTimestamps` (horloge égale ou qui recule).
- `test_capture` : `IdleThrottle`.
- `test_privacy` : aucun `imwrite`, `VideoWriter` ou `imencode` dans `src/` et `tools/`, et le détecteur est éprouvé sur un exemple positif.
- `test_mediapipe_integration` (marqueur `integration`, sauté si les fichiers manquent) : les 4 images officielles donnent `thumb_up`, `thumb_down`, `victory`, `pointing_up`.

## Étapes (un commit local par étape verte, pas de push, pas de remote)

0. **Squelette + portes** : dossier, `git init`, `.gitignore`, `pyproject.toml`, venv, lock, `CLAUDE.md`, les deux outils. Trois portes à franchir :
   - (a) mediapipe s'importe, les 4 images sont reconnues, et on relève les noms bruts des classes ;
   - (b) la caméra s'ouvre en 640×480 ; on relève le backend retenu et les fps réels ;
   - (c) `--apps "Apple Music"` retrouve l'AppID connu, sans accents cassés.

   En bonus : `--press playpause`. Si (a) échoue, on applique le repli prévu.
1. `observation.py` + `features.py` + tests.
2. `config.py` + `config.yaml` + tests.
3. `segments.py` + `engine.py` + tests.
4. `actions/` (dispatcher, keys, launch + start_apps, url, script) + `macros/example_hello.py` + `feedback.py` + tests.
5. `recognition.py`, `capture.py`, `debug_view.py`, `logging_setup.py`, `app.py`, plus le test d'intégration. Premier lancement `--debug --dry-run`.
6. **Mesures et réglage**, voir « Vérification ». Puis ajustement de `config.yaml` et mise à jour de `CLAUDE.md` (phase 1 ✅ + SHA).

Tous les fichiers sont écrits avec l'outil Write, jamais via un heredoc Bash qui abîme les `\` (dont `shell:AppsFolder\`). Ce projet est personnel : il reste hors de `~/work-kit`.

## Risques et parades

| Risque | Parade |
|---|---|
| mediapipe 1.0.x : empaquetage ctypes récent, version majeure d'août 2026. Échec possible du chargement de la DLL. | Runtime VC++ présent (vérifié). Porte (a), sinon replis 0.10.35, puis 3.12 + 0.10.21. |
| OpenCV 5.0 (arrivé avec mediapipe) : je ne sais pas s'il garde le backend DirectShow. MSMF peut être lent à ouvrir la caméra. | `backend: auto` : DSHOW, puis MSMF si l'ouverture échoue. La porte (b) note le backend retenu. |
| Dérive des labels (`None`/`Unknown`, renommages). | Table explicite, avertissement sur un nom inconnu, test d'intégration. |
| Faux positifs : ✋ est un geste naturel, ✊ une pose de repos, ☝️ le doigt qu'on lève pour parler. | Stabilité + seuil par geste + un tir par tenue + segments par geste + cooldown + 🤟 pour désarmer (par exemple avant un appel vidéo). On mesure à sec, puis on durcit via `per_gesture`. |
| Geste « qui ne marche pas » parce qu'il s'est stabilisé pendant le cooldown (variante prudente). | `Ignored` visible dans le journal et l'overlay. On mesure la gêne à l'étape 6 : si elle est réelle, on réduit `cooldown_s`, sans passer au tir différé. |
| Fenêtre qui s'ouvre **derrière** : Windows limite le passage au premier plan pour un processus en arrière-plan (la barre des tâches clignote). | À vérifier au jour 1 (critère « au premier plan »). Si c'est le cas, correctif en phase 3 : ramener la fenêtre lancée au premier plan. |
| Apps du Store : `Get-StartApps` est lent (1 à 2 s) ; noms français accentués ; `ConvertTo-Json` renvoie un objet et non une liste quand il n'y a qu'un résultat ; noms en double ; `explorer.exe` renvoie peut-être 1 même quand tout va bien. | Résolution une seule fois au chargement, puis cache. UTF-8 + `utf-8-sig`. Parseur qui accepte les deux formes (testé). Erreur qui liste les AppID en cas d'ambiguïté, et `app_id:` possible. Code de sortie ignoré. |
| Scripts : ils peuvent bloquer, planter, être relancés en double, ou faire des dégâts s'ils partent sur un faux positif. | Processus séparé, une instance par script (registre qui survit au rechargement), sortie et code de sortie journalisés. Garder les macros inoffensives et idempotentes tant que `confirm` n'existe pas (phase 3). L'armement protège tous les gestes. |
| Touches : une faute de frappe est ignorée en silence ; FAILSAFE ; les raccourcis n'atteignent pas une fenêtre lancée en admin (UIPI) ; les touches média vont à la session média active. Pas de preuve qu'Apple Music réagisse (≈ 70 %, bonus). | Validation au chargement, `FAILSAFE=False`. Si besoin, repli en phase 3 via l'API `GlobalSystemMediaTransportControlsSessionManager` (paquets `winrt-Windows.Media.Control` et `winrt-runtime` 3.2.1, disponibles en cp313). |
| CPU : si tes mains restent visibles quand tu tapes, l'inférence tourne à plein régime en continu. | CPU relevé à l'étape 6. Si besoin, un plafond `active_fps` (par exemple 15), avec `stable_frames` ajusté. |
| Caméra occupée par Teams ou Zoom. | Boucle de reconnexion avec un journal clair. Désarmer ne libère **pas** la caméra : il faut quitter l'app (bascule caméra en phase 3). |
| Mode VIDEO : ms non croissantes → erreur. Inférence plus lente que la caméra → léger retard. | `StrictTimestamps` testé. Temps d'inférence affiché dans l'overlay ; s'il dépasse la période d'une frame, on ajoute un lecteur « dernière frame » dans un thread. |
| Latéralité inversée. | Miroir avant l'inférence, vérifié au jour 1. De toute façon, la cohérence entre enregistrement et inférence suffit pour les features. |
| Vie privée. | Test statique anti-écriture d'images, aucune image ni aucun point dans les logs, vérification réseau, fenêtre debug seulement sur demande. |

## Vérification (critères d'acceptation de la phase 1)

- `pytest` vert (SHA du commit cité), test d'intégration compris. Portes (a), (b) et (c) franchies.
- **Fiabilité** (`--dry-run`, à environ 1 m) : 10 essais par geste mappé → au moins 9 tirs et **0 doublon** par geste, compté sur les lignes `TRIGGER` du log.
- **Faux positifs** : 30 min de `--dry-run` en activité normale (clavier, souris, téléphone, boire, se toucher le visage). Objectif : **0** ligne `TRIGGER`, et aucune ligne `ARMED`/`DISARMED` non voulue. Le bip signale chaque faux positif en direct.
- **Fonctions** (en réel, une fois chacune) :
  - ✌️ ouvre StudiUM au premier plan ;
  - ☝️ ouvre Apple Music ;
  - 👎 affiche la boîte de la macro d'exemple, avec le code de sortie 0 dans le log ;
  - une liaison `path:` temporaire ouvre notepad.exe ;
  - ✊ coupe le son ;
  - 👍 monte le volume par paliers ;
  - 🤟 désarme (plus rien ne marche), puis réarme ;
  - ✋ met Apple Music sur pause (bonus).
- **CPU** relevé (repos et main visible) et noté dans `CLAUDE.md`.
- **Réseau** : pendant l'exécution, `Get-NetTCPConnection` et `Get-NetUDPEndpoint` filtrés sur le PID ne renvoient rien.
- Latéralité : la main droite levée s'affiche « Right » dans l'overlay.

## Questions pour plus tard (sans effet sur la phase 1)

- Tes vraies apps, pages et macros : `config.yaml` se recharge à chaud, tu pourras les ajouter toi-même.
- Phase 3 :
  - quelles actions exigent une confirmation, et avec quel geste ;
  - combinaison du raccourci global pour couper la caméra ;
  - démarrage par le dossier Startup ou par le Planificateur de tâches ;
  - ✊ est mappé seul (muet) **et** sert de préfixe au combo ✊>👍 : il faudra choisir.
- Phase 2 : quels signes perso, et une main ou les deux. Plus de gestes = plus de fonctions, ce qui rend les phases 2 et 3 utiles pour ta priorité.

---

## Exécution — régie `/deck` (ajout : le plan ci-dessus reste inchangé)

**En bref** :
- 1 admin (cette session) + **3 sessions à ouvrir**, une par lot de fichiers disjoints, chacune dans son worktree.
- L'admin fait seul l'étape 0, qui fixe les contrats, puis délègue.
- Les fusions passent par l'agent `claude-admin`.

**Régime** : tu as écrit « use automode », que je lis comme `SOIS AUTONOME`. Après ton approbation, je ne pose plus aucune question, et chaque décision est inscrite avec son motif dans `deck-journal.jsonl`. Tu n'interviens que deux fois :
1. pour ouvrir 3 sessions quand je te le demande (sinon, je les engendre au bout de 5 min) ;
2. à l'étape 6, devant la caméra.

Si « automode » voulait dire autre chose, corrige-moi dans ta réponse.

État mesuré le 2026-09-23 :
- `ListAgents` : aucune autre session sur la machine ;
- `quota.js` : ouvert, 20 % de la fenêtre de 5 h consommés, reset dans 33 min ;
- le dépôt n'existe pas encore.

### Topologie

| Arbre | Branche | Propriétaire |
|---|---|---|
| `C:\Users\adrie\gesture-remote` | `main` | l'admin, plus `claude-admin` pour les fusions. Personne d'autre n'y écrit. |
| `C:\Users\adrie\gesture-remote-vision` | `lot/vision` | session A |
| `C:\Users\adrie\gesture-remote-decision` | `lot/decision` | session B |
| `C:\Users\adrie\gesture-remote-actions` | `lot/actions` | session C |

Ce sont des worktrees frères, comme tes `synchro-calendrier-*`. Chaque session commence par `git worktree list` et `git status -sb`, et rapporte ce qu'elle voit : le brief n'est pas une preuve.

**Un worktree n'est pas un clone** : `.venv`, `models/*.task` et `models/samples/` sont gitignorés, donc absents des worktrees.
- **venv** : celui de `main`, par chemin absolu et en lecture seule. Aucun `pip install` dans les lots ; une dépendance manquante se demande à l'admin.
- **Piège** : l'installation éditable pointe vers le `src/` de `main`. Lancés dans un worktree, les tests testeraient donc le code de `main`, et resteraient verts. Parade posée à l'étape 0 : `pythonpath = ["src"]` dans la config pytest, plus un garde dans `conftest.py` qui échoue si `gesture_remote` n'est pas importé depuis l'arbre courant.
- **Modèles** : A copie `models/` dans son worktree. Un test d'intégration sauté n'est pas un vert : chaque compte rendu donne le nombre de tests sautés.

### Étape 0 (admin, séquentielle) : elle fixe les contrats

Tout le squelette du plan, plus des contrats **gelés pour le tour**. Seul l'admin les modifie, sur demande :
- `pyproject.toml`, `tests/conftest.py` (garde d'import et `FakeClock`), `tests/test_privacy.py`, `CLAUDE.md` (source des commandes), `README.md`, `.gitignore`, `requirements.lock`, les deux outils ;
- `observation.py` (les types) ;
- dans `config.py`, la **couche modèles** : réglages, les 4 modèles d'action et l'union `ActionSpec`, la racine `Config`, `AppResolutionError`. Les noms publics sont gelés. B possède ensuite le fichier et y ajoute le chargement, la validation et `ConfigStore` ;
- `macros/example_hello.py`, avancé depuis l'étape 4 parce que `config.yaml` le référence et qu'il doit donc exister dans tous les arbres.

Aux portes (a), (b) et (c) s'ajoutent deux vérifications que je n'ai pas faites en planifiant :
- `hasattr(pyautogui, "isValidKey")` ;
- `--apps Param`, un nom accentué, pour éprouver l'UTF-8.

Leurs résultats et le SHA de l'étape 0 entrent dans les briefs.

### Lots parallèles (fichiers disjoints)

| Lot | Possède | Lit seulement | Expose |
|---|---|---|---|
| **A · vision** | `features.py`, `recognition.py`, `capture.py` + `test_features`, `test_recognition`, `test_capture`, `test_mediapipe_integration` | `observation.py`, les réglages caméra et reconnaissance, `models/` (copiés) | `CANNED_LABELS`, `MediaPipeGestureRecognizer`, `StrictTimestamps`, `Camera`, `IdleThrottle`, `normalize_landmarks`, `FEATURES_VERSION` |
| **B · décision** | `config.py` (sauf les noms publics des modèles), `config.yaml`, `segments.py`, `engine.py` + `test_config`, `test_segments`, `test_engine` | `observation.py` | `load_config(path, *, labels, is_valid_key, resolve_app)`, `ConfigStore`, `GestureEngine`, `Triggered`/`Ignored`/`ArmedChanged`, `snapshot()` |
| **C · actions** | `actions/*`, `feedback.py` + `test_actions`, `test_start_apps`, `test_feedback` | les modèles d'action et `AppResolutionError` | `ActionDispatcher`, les handlers, `StartAppsIndex`, `ScriptRunner`, `Feedback` |

**Dépendances croisées**, écrites dans les deux briefs concernés :
- **A → B** : les 7 labels. B les reçoit par injection (`labels=`) et ses tests les codent en dur. Le raccord réel est testé à l'étape 5.
- **C → B** : la résolution `app:` → AppID. B la reçoit par injection (`resolve_app=`), et C lève `AppResolutionError`. B ne met **aucune** validation qui dépend du contexte (labels, touches, apps, fichiers) dans les validateurs pydantic : C construit ces modèles directement dans ses tests.
- **B ↔ C** : les noms de champs des modèles d'action ne changent pas pendant le tour. Sinon, on annonce les appels touchés **avant** d'enregistrer le fichier (règle de `parallel-sessions`).

**Ressources exclusives**, avec l'admin comme courtier :
- **la caméra** : seul l'admin l'ouvre (aux portes et à l'étape 5) ;
- **le clavier** : aucun test n'envoie de vraie touche, elle partirait dans le terminal d'une autre session ;
- aucun test ne lance de vraie app ni de vraie URL.

Deux exceptions : le vrai sous-processus du test de script (dans `tmp_path`), et un vrai `Get-StartApps` (lecture seule, marqueur `integration`).

### Chaque brief contient

Selon `session-handoff` :
- la revendication en tête ;
- les sections du plan qui le concernent, **avec leur raisonnement** (par exemple : pourquoi le tir est prudent, pourquoi le registre des scripts survit au rechargement) ;
- les fichiers possédés et les fichiers lus ;
- les dépendances croisées ;
- chaque affirmation marquée vérifiée (avec la commande), raisonnée ou supposée ;
- **« ce que je n'ai pas vérifié »** ;
- la définition de « fini » :
  - tests verts **avec le SHA**, nombre de tests sautés ;
  - un contrôle qui prouve qu'un test pouvait échouer ;
  - `ruff` propre ;
  - `git add` par chemins explicites, commit sur la branche du lot ;
  - état final « commité, en attente d'intégration ».

### Déroulé

1. Après ton approbation : décision inscrite au journal, console `serve.js`, étape 0 et ses portes.
2. Une fois les worktrees créés, je te demande d'ouvrir 3 sessions, par une notification et un `pendingRequest` dans `deck-claims.json` : `cd C:\Users\adrie\gesture-remote-vision; claude`, puis pareil avec `-decision` et `-actions`. Sans session au bout de 5 min, je les engendre en agents.
3. J'envoie les briefs par `SendMessage` et j'inscris les attributions dans `deck-claims.json`.
4. Quand une session se libère : d'abord son compte rendu (fait, pas fait, échoué), inscrit au journal, puis la suite.
5. `claude-admin` intègre A, B et C sur `main` : vérification par ascendance, suite complète avec les modèles présents, SHA.
6. Deux sessions libérées se partagent l'après-fusion :
   - l'étape 5 : `app.py`, `__main__.py`, `debug_view.py`, `logging_setup.py`, sur la branche `lot/app` ;
   - **le balayage des coutures**, en lecture seule : labels de A ↔ `config.yaml`, `AppResolutionError`, union ↔ handlers, dispatcher ↔ rechargement, événements ↔ `app`. Que chaque branche soit verte seule ne prouve rien sur l'ensemble.
7. L'étape 6 avec toi (les gestes). Puis `CLAUDE.md` marque la phase 1 ✅ avec le SHA, et le bilan va au journal.

**Gouverneur de quota** : `quota.js` avant chaque relance.
- À 90 % ou plus : gel. Le travail en cours va à son terme, mais aucune tâche neuve n'est distribuée.
- Reprise 60 s après le reset.

**Ce que je ne sais pas, et qui pourrait changer le découpage** :
- Si la porte (a) échoue, le repli de version se fait à l'étape 0, avant tout brief : le découpage ne bouge pas.
- Si le garde d'import montre que les worktrees ne s'isolent pas malgré `pythonpath` : un venv par worktree, soit environ 200 Mo et quelques minutes chacun.
- Si les agents engendrés calent comme les deux relectures d'aujourd'hui (arrêtées net après 600 s sans progrès), les lots attendent tes sessions, ou je les fais moi-même l'un après l'autre. Même repli si `claude-admin` cale : je fusionne moi-même, puisque `main` est à moi.
