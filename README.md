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

## Optional metadata

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

Underscore-prefixed folder phrases also become hints (for example `_red jacket`). Hints and tags influence the description; only supply accurate labels. The highlight/shadow exclusion applies to these inputs too.

## Development

```sh
python -m unittest discover -s tests -v
```

Tests use mocked API responses and make no paid requests. No credentials, photos, captions, logs, historical backups, or personal configuration files are included in this distribution. Supply your own `.env`; explicitly set environment variables take priority.

No license has been selected for this release. Choose and add a license before inviting reuse under specific terms.
