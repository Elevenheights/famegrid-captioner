# Famegrid Captioner

A Python desktop app for batch captioning images for Krea 2 / LoRA training. It writes a UTF-8 `.txt` caption beside each image.

## Windows setup

1. Install Python 3.10 or newer, including Tcl/Tk, and enable **Add Python to PATH**.
2. Download and extract this repository, then double-click `Setup.bat`.
3. Edit the newly created `.env` file. Set `OPENAI_API_KEY` for OpenAI or `XAI_API_KEY` (alternatively `GROK_API_KEY`) for xAI/Grok. One provider is enough. Keep this file private.
4. Double-click `Start Famegrid Captioner.bat`.

## macOS / Linux setup

Install Python 3.10 or newer with Tkinter (some Linux distributions provide it in a separate `python3-tk` package). Then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env to add your provider key.
python caption_photos.py
```

The app has been smoke-tested on Windows. macOS and Linux have not been visually tested.

## Captioning images

- Add one or more image folders; optionally include their subfolders.
- Add caption hints directly to folder names using underscore-prefixed phrases, such as `_red jacket _standing` (see **Folder-name hints** below).
- Select a provider and an image-capable model available to your account. The model name is editable; **Refresh models** lists candidates, but does not verify image support. Defaults are inherited from the original script.
- Set the trigger word and subject class, or enable **Class varies** to choose a class per image.
- Choose caption categories, detail level, target length, and parallel image count.
- Optionally edit the prompt override or enable automatic Grok fallback when both providers are configured.
- Start captioning. Existing sidecar captions are skipped unless **Overwrite existing .txt captions** is enabled.

Supported filename extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, and `.bmp`. Providers may reject formats or images they do not support. Images are sent to the selected API provider; successful captions can incur API charges, including retries and fallback calls.

Highlight/shadow caption analysis has been removed. There is no toggle or API parameter for it. All prompt paths, including custom prompts, retries, and fallback requests, instruct the model to omit this analysis. Model output remains probabilistic; existing captions are not rewritten automatically.

**Delete orphaned captions** is off by default. Enabling it deletes `.txt` files without matching images in scanned folders. Use it only in dedicated image/caption folders with backups.

## Optional local image tagger

With the virtual environment active, install:

```sh
python -m pip install -r requirements-tagger.txt
```

On Windows, use `.venv\Scripts\python.exe -m pip install -r requirements-tagger.txt`.

Enable the local WD EVA02-Large v3 pre-scan in the app. The first run downloads model files from `SmilingWolf/wd-eva02-large-tagger-v3` on Hugging Face; allow disk space and download time. Review the model's own license and usage terms separately. The tagger is optional; ordinary captioning does not require these dependencies.

## Folder-name hints

You can tell the captioner what to include by putting an underscore before a phrase in an image folder's name. No CSV or JSON file is required.

For example, name a folder `_red jacket`, put the relevant images inside, and add that folder in the app. The captioner asks the model to include the exact phrase `red jacket` naturally in each image's caption.

Use a space followed by another underscore to separate multiple hints:

```text
_red jacket                    -> red jacket
_red jacket _standing          -> red jacket; standing
Session 01 _blue dress _garden -> blue dress; garden
_long-haired                   -> long haired
my_photos                      -> no hints
```

The underscore must begin a phrase: put it at the start of the folder name or after a space. Use spaces within a phrase; use ` _` to start the next hint. Hyphens within a hint become spaces. All words after a hint's underscore belong to that hint until the next ` _` or the end of the name, so put ordinary folder labels before the first hint.

When scanning subfolders, hints are collected from the image's own folder up through the folder you added to the queue. For example:

```text
Dataset _red jacket/           <- add this folder with Include subfolders enabled
    _standing/
        image001.jpg           <- gets standing and red jacket
    _seated/
        image002.jpg           <- gets seated and red jacket
```

Folders above the queued folder do not contribute hints. If you add only `_standing` to the queue, the parent folder's `red jacket` hint will not be included.

Folder hints take priority over conflicting visual interpretations, per-image metadata, and machine-generated tags. The app checks for each hint's phrase in the generated caption and retries once if a phrase is missing; it reports a failure if the retry still omits it. Use accurate, non-conflicting hints that apply to the images in that folder. Highlight/shadow analysis remains excluded.

Existing captions are skipped by default. To apply newly added folder hints to images that already have `.txt` captions, enable **Overwrite existing .txt captions** and run captioning again.

## Optional metadata files

In an image folder, add `subject_metadata.csv` with these columns:

```csv
filename,hint,identity
image001.jpg,red jacket,example_subject
```

Folder hints can also be supplied in `caption_folder_hints.json`:

```json
{
  ".": ["red jacket"],
  "outdoors": ["garden"]
}
```

JSON hints are combined with folder-name hints. Hints and tags influence the description; only supply accurate labels. The highlight/shadow exclusion applies to these inputs too.

## Development

```sh
python -m unittest discover -s tests -v
```

Tests use mocked API responses and make no paid requests. No credentials, photos, captions, logs, historical backups, or personal configuration files are included in this distribution. Supply your own `.env`; explicitly set environment variables take priority.

No license has been selected for this release. Choose and add a license before inviting reuse under specific terms.
