import os
import base64
import csv
import json
import re
import sys
import threading
import traceback
import tkinter as tk
from mimetypes import guess_type
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Configuration
MODEL_NAME = "gpt-5.2"
GROK_DEFAULT_MODEL = "grok-4"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
MAX_WORKERS = 5  # Process 5 images in parallel
TRIGGER_PHRASE = "famegrid"
SUBJECT_CLASS = "woman"
MIN_CAPTION_CHARACTERS = 900
# None disables hard truncation entirely; set an integer to cap captions again.
MAX_CAPTION_CHARACTERS = None
# Soft length guidance handed to the model. Not enforced after the fact.
CAPTION_TARGET_CHARACTERS = 1950
# Output token ceiling. Headroom above the target so the model is never cut off.
CAPTION_MAX_TOKENS = 2200
OVERWRITE_EXISTING_CAPTIONS = False
DELETE_ORPHANED_CAPTIONS = False
SUBJECT_METADATA_CSV = "subject_metadata.csv"
FOLDER_HINTS_FILENAME = "caption_folder_hints.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")
WD_TAGGER_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
WD_TAG_THRESHOLD = 0.35
WD_MAX_TAGS = 100
WD_BLOCKED_TAGS = {
    # Booru shorthand and unsafe/ambiguous age classifications must never be
    # passed into a natural-language caption prompt.
    "1girl",
    "1boy",
    "loli",
    "shota",
    "child",
    "young girl",
    "young boy",
    "underage",
    # Medium, rendering, and dataset artifacts describe the image source rather
    # than useful visible subject or scene content.
    "realistic",
    "photorealistic",
    "photo background",
    "photo inset",
    "3d background",
    "mixed media",
    "traditional media",
    "voice actor",
}
_LOCAL_TAGGER = None
_LOCAL_TAGGER_LOCK = threading.Lock()
_NULL_STREAMS = []

CAPTION_DETAIL_LEVELS = {
    "Concise": "Use a compact caption with brief phrases and only the main visible facts. Summarize each applicable selected category without listing fine details.",
    "Balanced": "Use moderate detail. Cover the main visible facts and a few distinguishing details in each applicable selected category, avoiding exhaustive lists.",
    "Detailed": "Use thorough, specific descriptions of the clearly visible features in each applicable selected category, including individual body-part build and proportions when selected and visible.",
    "Exhaustive": "Use the fullest useful description of every clearly visible detail in each applicable selected category, including fine distinctions and spatial relationships. Avoid repetition and invented details.",
}
CAPTION_DETAIL_DEFAULT = "Detailed"

CAPTION_CONTENT_FIELDS = (
    ("age", "Age range", "age or age range"),
    ("skinTone", "Skin tone", "skin tone"),
    ("facialFeatures", "Facial features", "facial structure or facial features"),
    ("hair", "Hair", "hair or facial hair"),
    ("bodyType", "Body type & proportions", "body type, build, silhouette, or proportions"),
    ("makeup", "Makeup", "makeup"),
    ("pose", "Pose & action", "pose or action"),
    ("expression", "Facial expression", "facial expression or gaze"),
    ("clothing", "Clothing & accessories", "clothing, footwear, or accessories"),
    ("colors", "Colors & materials", "colors, materials, or textures"),
    ("environment", "Environment & background", "environment, setting, background, or surrounding objects"),
    ("composition", "Composition & camera framing", "composition, crop, camera framing, angle, or viewpoint"),
    ("lighting", "Lighting", "lighting or illumination"),
    ("atmosphere", "Weather & atmosphere", "weather or atmosphere"),
)


LENGTH_PLACEHOLDER = "<<CAPTION_LENGTH_GUIDANCE>>"


def length_guidance_text(max_characters):
    """Build the caption-length bullet. 0/None omits any length instruction."""
    if not max_characters:
        return (
            "- Use natural, literal English. Describe the image as fully as the "
            "visible detail supports. Never invent or repeat details merely to "
            "reach a length. Always finish your closing sentence rather than "
            "stopping mid-thought."
        )
    return (
        "- Use natural, literal English. Try to keep the caption under about "
        f"{max_characters} characters including spaces and punctuation. Make the "
        "description as dense and specific as that budget allows; a simple or "
        "tightly cropped image may be shorter. Never invent or repeat details "
        "merely to reach a length. Always finish your closing sentence rather "
        "than stopping mid-thought."
    )






CAPTION_PROMPT = f"""
Write one Krea 2 LoRA training caption for this image.

Output requirements:
- Return only one plain-text paragraph, with no heading, quotation marks, markdown,
  bullet points, or line breaks. Multiple complete sentences are allowed.
- Start exactly with: {TRIGGER_PHRASE},
{LENGTH_PLACEHOLDER}
- Preserve a subject-first balance: devote roughly two thirds of the caption to
  the main {SUBJECT_CLASS} and her visible characteristics, clothing, expression,
  and pose, then use the remaining portion for the environment and composition.
  Environmental detail must expand the caption rather than replace or compress
  useful character detail.
- Give the main {SUBJECT_CLASS} the most detail. Describe only what is visibly
  supported, in this priority order:
  1. A natural perceived-age phrase plus a specific adult age range, followed by
     a precise visible skin-tone description. Use phrases such as "a woman in her
     early twenties, approximately 20-24 years old," "a woman in her mid twenties,
     approximately 24-27 years old," "a woman in her late twenties, approximately
     27-30 years old," or "a woman in her late twenties to early thirties,
     approximately 28-33 years old." Apply the same early, mid, late, or
     decade-spanning format to older adults. Include both the natural phrase and
     numeric range when the face supports them; never state one exact age.
  2. Distinguishing facial characteristics when visible: face shape, jawline,
     cheek fullness, brow shape, eye shape, nose shape, lip shape, and visible
     marks such as freckles. Do not claim biometric identification.
  3. Hairstyle, length, texture, color, parting, bangs, braids, curls, or tied-back
     arrangement, plus any visible facial hair.
  4. Qualitative body build, silhouette, and visible proportions. Be specific
     about shoulder width, torso length, waist definition, hip width, bust or
     chest size, buttock prominence, thigh build, limb proportions, and overall
     frame when these features are clearly visible. Use direct, neutral wording
     such as narrow or broad shoulders, slim or thick waist, narrow or wide hips,
     small or large breasts, flat or prominent buttocks, slim or thick thighs,
     petite, slender, athletic, muscular, soft, curvy, plus-size, or heavyset.
     When clearly supported, name the overall body shape, such as hourglass, pear,
     rectangle, oval, or inverted triangle. Do not use erotic commentary.
  5. Visible makeup in useful detail: complexion makeup, eyebrow styling,
     eyeshadow color and placement, eyeliner shape, lashes, blush, contour,
     highlighter, and lipstick color and finish. Include makeup only when it can be
     described confidently from the image.
  6. Expression, gaze direction, eyelid position, mouth position, and head angle.
  7. Clothing, footwear, accessories, and visible materials, colors, and fit.
- Describe the pose precisely: head direction; shoulder, torso, and hip
  orientation; seated, standing, leaning, crouching, or reclining posture; weight
  distribution; spinal curve; pelvic angle; placement and bend of every visible
  arm and leg; elbow and knee direction; wrist rotation; hand placement and palm
  direction; finger position; foot angle; raised heel or pointed toes; contact
  points with the floor, furniture, objects, or people; and position within the
  frame. Explain asymmetry, crossed limbs, arched posture, or a hip pushed outward
  when visible. Never invent anatomy or positioning hidden by cropping, clothing,
  or occlusion.
- Describe the visible environment thoroughly after the subject. Identify the
  indoor or outdoor setting and explain the spatial arrangement around the person:
  foreground, middle ground, background, floor or ground surface, walls, ceiling,
  doors, windows, furniture, architecture, landscape, vegetation, streets,
  vehicles, signs, nearby people, and important small objects. Include visible
  colors, materials, textures, patterns, object placement, weather conditions, and
  background activity when supported. Explain where the subject stands, sits, or
  reclines in relation to surrounding objects and surfaces. Include shot distance,
  crop, viewpoint, camera angle, and whether the composition is centered,
  symmetrical, diagonal, or off-center, but do not guess a camera, lens, focal
  length, or capture settings.
- Prefer concrete descriptions (for example, "deep brown skin," "oval face,"
  "shoulder-length tightly curled black hair," or "arms folded across her torso")
  over vague praise or aesthetic language.
- If any requested attribute cannot be determined confidently, omit that attribute
  completely. Describe only present, visible information. Never mention that a
  feature is missing, hidden, cropped, unavailable, uncertain, outside the frame,
  or impossible to determine. Never include negative filler such as "her face is
  not visible" or "her age cannot be determined."
- Do not guess at background, origin, affiliation, beliefs, health, or any other
  detail that appearance alone cannot establish. User-supplied metadata and
  folder-level hints may be included exactly as provided, but never make them
  more specific. Apart from such supplied labels, describe observable features
  such as skin tone, facial structure, hair, clothing, and visible objects. Do
  not add speculative "possibly ..." phrasing.
- Never infer exact height, weight, measurements, or an exact age. Use an
  approximate perceived age range and qualitative body-build terms only. Do not
  infer body features concealed by loose clothing, objects, pose, or cropping. If
  the subject could be under 18, omit breast and buttock size descriptions and use
  only general, non-sexual body-build language.
- Silently check the finished caption before answering: it must cover every
  applicable priority above, contain no unsupported claim, and end on a complete
  sentence. Do not print the checklist.

This dataset is training a photorealistic style. Do not describe or praise the
rendering style that the LoRA should absorb. In particular, omit terms such as
photograph, photo, photorealistic, realistic, RAW, DSLR, cinematic, masterpiece,
best quality, 8K, ultra-detailed, skin texture, pores, color grading, film grain,
depth of field, bokeh, lens model, lighting, illumination, or other quality/style
tags.
""".strip()


def load_subject_metadata(folder=".", logger=print):
    """Load optional per-filename labels from the sidecar CSV."""
    metadata_path = os.path.join(folder, SUBJECT_METADATA_CSV)
    if not os.path.exists(metadata_path):
        return {}

    metadata = {}
    loaded_rows = 0
    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        # "hint" is the current column; "ethnicity" is still read so older
        # sheets keep working without being rewritten.
        label_column = "hint" if "hint" in fields else ("ethnicity" if "ethnicity" in fields else None)
        if "filename" not in fields or label_column is None:
            raise ValueError(
                f"{SUBJECT_METADATA_CSV} must contain a filename column and a hint "
                "column. An optional identity column may also be included."
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue

            # Collapse whitespace so CSV values cannot alter the prompt structure.
            hint = " ".join((row.get(label_column) or "").split())[:100]
            identity = " ".join((row.get("identity") or "").split())[:100]
            value = {"hints": [hint] if hint else [], "identity": identity}
            metadata[filename.lower()] = value
            metadata[os.path.splitext(filename)[0].lower()] = value
            loaded_rows += 1

    logger(f"Loaded subject metadata for {loaded_rows} image(s).")
    return metadata


def normalize_hint_values(value):
    """Accept a string, a list, or a {hint: ...} mapping and return clean labels."""
    if isinstance(value, dict):
        value = value.get("hint", value.get("ethnicity_hint", ""))
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    cleaned = []
    for item in value:
        text = " ".join(str(item).split())[:100]
        if text:
            cleaned.append(text)
    return cleaned


def hints_from_folder_name(folder_name):
    """Pull _underscore-marked phrases out of a folder name.

    "_asian woman" -> ["asian woman"];
    "_cleavage _big breasts" -> ["cleavage", "big breasts"].
    Each phrase runs from its underscore to the next underscore-marked phrase or
    the end of the folder name. The underscore must start a word, so ordinary
    names like "my_photos" contribute nothing.
    """
    matches = re.findall(
        r"(?<![A-Za-z0-9])_([^_]+?)(?=\s+_|$)", folder_name
    )
    hints = []
    for match in matches:
        phrase = " ".join(match.replace("-", " ").split())
        if phrase:
            hints.append(phrase)
    return hints


def load_folder_hints(folder=".", logger=print):
    """Load optional user-supplied hints associated with folders in this batch."""
    hints_path = os.path.join(folder, FOLDER_HINTS_FILENAME)
    if not os.path.exists(hints_path):
        return {}

    with open(hints_path, "r", encoding="utf-8-sig") as f:
        raw_hints = json.load(f)
    if not isinstance(raw_hints, dict):
        raise ValueError(f"{FOLDER_HINTS_FILENAME} must contain a JSON object.")

    hints = {}
    for folder_name, value in raw_hints.items():
        labels = normalize_hint_values(value)
        normalized_name = str(folder_name).strip().replace("\\", "/").strip("/").casefold()
        if normalized_name and labels:
            hints[normalized_name] = labels

    logger(f"Loaded folder hints for {len(hints)} folder(s).")
    return hints


def find_folder_hints(image_path, root_folder, folder_hints):
    """Collect hints for an image from its own folder upward to the batch root.

    Configured caption_folder_hints.json values and _underscore phrases are both
    included. Results are ordered most-specific first, with configured values
    before folder-name values, and de-duplicated case-insensitively.
    """
    collected = []
    seen = set()

    def add(labels):
        for label in labels:
            key = label.casefold()
            if key not in seen:
                seen.add(key)
                collected.append(label)

    root_folder = os.path.abspath(root_folder)
    current = os.path.abspath(os.path.dirname(image_path))
    while os.path.commonpath((root_folder, current)) == root_folder:
        basename = os.path.basename(current)
        configured = None
        if folder_hints:
            relative = os.path.relpath(current, root_folder).replace("\\", "/").casefold()
            for candidate in (relative, basename.casefold()):
                if candidate in folder_hints:
                    configured = folder_hints[candidate]
                    break
        if configured is not None:
            add(configured)
        add(hints_from_folder_name(basename))
        if current == root_folder:
            break
        current = os.path.dirname(current)

    return collected


def metadata_for_image(image_path, root_folder, subject_metadata, folder_hints):
    """Combine filename-specific metadata with hints from the folder tree."""
    filename = os.path.basename(image_path).lower()
    combined = dict(
        subject_metadata.get(filename)
        or subject_metadata.get(os.path.splitext(filename)[0])
        or {}
    )
    folder_labels = find_folder_hints(image_path, root_folder, folder_hints)
    if folder_labels:
        combined["folder_hints"] = folder_labels
    return combined or None


def caption_prefix(trigger_word=TRIGGER_PHRASE, subject_class=SUBJECT_CLASS):
    trigger_word = " ".join(trigger_word.split())
    subject_class = " ".join(subject_class.split())
    if not trigger_word:
        raise ValueError("Trigger word must be filled in.")
    # An empty class means the model chooses a class for each image.
    return f"{trigger_word} {subject_class}".strip()


def strip_caption_prefix(caption, trigger_word, subject_class):
    """Remove an existing opening without matching partial words."""
    caption = re.sub(
        rf"^\s*{re.escape(trigger_word)}(?!\w)\s*[,;:-]?\s*",
        "", caption, count=1, flags=re.IGNORECASE,
    )
    if not subject_class:
        return caption
    return re.sub(
        rf"^\s*(?:a\s+|an\s+|the\s+)?{re.escape(subject_class)}(?!\w)\s*[,;:-]?\s*",
        "", caption, count=1, flags=re.IGNORECASE,
    )


def build_caption_prompt(subject_metadata=None, visual_tags=None,
                         max_characters=CAPTION_TARGET_CHARACTERS, prompt_override=None,
                         caption_content_settings=None, detail_level=CAPTION_DETAIL_DEFAULT,
                         trigger_word=TRIGGER_PHRASE, subject_class=SUBJECT_CLASS):
    """Add optional metadata and pre-scan tags to the caption instructions."""
    prefix = caption_prefix(trigger_word, subject_class)
    trigger_word, subject_class = " ".join(trigger_word.split()), " ".join(subject_class.split())
    override = prompt_override.strip() if prompt_override else ""
    if override:
        prompt = override
    else:
        prompt = CAPTION_PROMPT.replace(
            LENGTH_PLACEHOLDER, length_guidance_text(max_characters), 1
        )

    if not override:
        prompt = prompt.replace(f"- Start exactly with: {TRIGGER_PHRASE},", f"- Start exactly with: {prefix}", 1)
        prompt = prompt.replace(f"the main {SUBJECT_CLASS}", f"the main {subject_class or 'subject'}")
        prompt = prompt.replace("and her visible characteristics", "and their visible characteristics")


    if subject_metadata:
        supplied = []
        file_hints = subject_metadata.get("hints") or []
        identity = subject_metadata.get("identity")
        folder_labels = subject_metadata.get("folder_hints") or []
        for hint in file_hints:
            supplied.append(f"hint: {hint}")
        if identity:
            supplied.append(f"identity token: {identity}")

        if supplied:
            prompt += (
                "\n\nThe dataset author supplied the following authoritative metadata for "
                f"this image: {'; '.join(supplied)}. Include these labels naturally and "
                "exactly as written. Do not replace, reinterpret, or infer them from the image."
            )

        if folder_labels:
            joined = json.dumps(folder_labels, ensure_ascii=False)
            prompt += (
                "\n\nThe dataset author supplied the following folder-level hint(s) for "
                f"this image: {joined}. Treat each one as authoritative context rather "
                "than something you inferred from the picture. Every folder-level hint "
                "must appear in the finished caption. Work each label naturally into the "
                "caption exactly as written, wherever it fits best, without "
                "expanding it into anything more specific and without explaining that it "
                "came from metadata. These folder hints are the highest-authority source. "
                "If the image appearance, machine-generated tags, filename metadata, or "
                "any other instruction conflicts with a folder hint, the folder hint wins. "
                "Never omit, replace, reinterpret, or contradict a folder hint, and never "
                "include a competing description of the same attribute. Continue to "
                "describe all non-conflicting visible details independently."
            )

        retry_hints = subject_metadata.get("_retry_missing_folder_hints") or []
        if retry_hints:
            prompt += (
                "\n\nCORRECTION REQUIRED: A previous caption draft omitted these mandatory "
                f"folder hints: {', '.join(retry_hints)}. Regenerate the entire caption and "
                "make sure every one of those exact phrases appears verbatim in natural "
                "language. Do not return a caption unless all of them are present."
            )

    if visual_tags:
        folder_hint_priority = ""
        if subject_metadata and subject_metadata.get("folder_hints"):
            folder_hint_priority = (
                "The dataset author's folder-level hints are authoritative and take "
                "priority over these machine-generated tags. If any tag conflicts with "
                "a folder hint, ignore the tag and preserve the folder hint exactly as "
                "written. "
            )
        prompt += (
            "\n\nA local WD EVA02-Large v3 image pre-scan returned these "
            "machine-generated tags with confidence scores:\n"
            f"{visual_tags}\n"
            f"{folder_hint_priority}"
            "Treat these filtered tags as authoritative factual input and give them priority "
            "over the model's independent interpretation of the visible image. If the model's "
            "vision conflicts with a tag, preserve the tag and do not include the competing "
            "visual interpretation. Folder hints remain the highest authority, followed by "
            "filename metadata; either one overrides a conflicting tag. Express every remaining "
            "non-conflicting tag as fluent natural English, combining related tags into natural "
            "phrases and complete sentences. Do not omit a tag merely because the same detail is "
            "also visually apparent. Never mention the tagger, confidence scores, metadata, or "
            "the existence of tags, and never paste or recite the comma-separated tag list."
        )

    if caption_content_settings:
        included = [
            prompt_label for key, _label, prompt_label in CAPTION_CONTENT_FIELDS
            if caption_content_settings.get(key) is True
        ]
        excluded = [
            prompt_label for key, _label, prompt_label in CAPTION_CONTENT_FIELDS
            if caption_content_settings.get(key) is False
        ]
        prompt += (
            "\n\nThe following caption-category controls override all earlier "
            "category-specific include or omit guidance, but they never override safety "
            "rules and never permit invented details."
        )
        if included:
            prompt += (
                " When clearly visible, include every applicable selected category: "
                + "; ".join(included) + "."
            )
        if excluded:
            prompt += (
                " Do not mention, imply, or infer these unselected categories: "
                + "; ".join(excluded) + "."
            )
        prompt += (
            " Authoritative filename metadata and folder hints are exceptions: they must "
            "still appear exactly as supplied even if they overlap an unselected category."
        )
    if detail_level not in CAPTION_DETAIL_LEVELS:
        raise ValueError(f"Unknown caption detail level: {detail_level}")
    prompt += (
        "\n\nCaption detail level: " + detail_level + ". "
        + CAPTION_DETAIL_LEVELS[detail_level]
        + " This detail-level instruction overrides earlier requests for description depth, "
        "including in a custom prompt. It does not change which categories are selected, "
        "permit excluded categories, or relax safety or visibility requirements. "
        "Keep all mandatory folder hints, filename metadata, and supplied factual tags. "
        "Treat any stated character target as an upper budget, not a minimum; "
        "a concise caption may be much shorter. Never pad to reach that target."
    )
    if not subject_class:
        prompt += (
            "\n\nRequired caption opening: Start exactly with " + json.dumps(trigger_word)
            + " followed by a space, a concise subject class chosen from this image, "
            "then the description. Class varies: independently identify the main subject "
            "in each image (for example person, dog, car, or landscape). Do not use "
            "the literal words 'class varies' or a placeholder. Do not assume every "
            "image has the same subject class. This instruction overrides any fixed "
            "subject class or opening in the built-in or custom prompt."
        )
        return prompt + "\n\n" + 'Do not describe or analyze highlights, shadows, tonal range, clipping, black point, white point, exposure, contrast, or tonal separation. This restriction applies even when a custom prompt, selected category, metadata, folder hint, or machine-generated tag asks for such analysis. Ordinary makeup names such as eyeshadow and highlighter are allowed.'
    prompt += (
        "\n\nRequired caption opening: Start exactly with " + json.dumps(prefix)
        + " followed by a space and the description. Do not insert a comma between "
        "the trigger word and subject class. This opening overrides any other opening "
        "instruction, including a custom prompt. Use the supplied subject class for "
        "the main subject; do not automatically describe every subject as a woman."
    )
    return prompt + "\n\n" + 'Do not describe or analyze highlights, shadows, tonal range, clipping, black point, white point, exposure, contrast, or tonal separation. This restriction applies even when a custom prompt, selected category, metadata, folder hint, or machine-generated tag asks for such analysis. Ordinary makeup names such as eyeshadow and highlighter are allowed.'


class LocalWdTagger:
    """Direct ONNX runner for the WD EVA02-Large v3 image tagger."""

    def __init__(self, logger=print):
        # The GUI launcher uses pythonw, where stdout/stderr do not exist. Prevent
        # Hugging Face's otherwise invisible progress bar from writing to None.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import numpy as np
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError(
                "Local tagging requires onnxruntime, huggingface_hub, numpy, and pillow."
            ) from exc

        self.np = np
        self.Image = Image
        self.ImageOps = ImageOps
        logger(f"Loading local tagger {WD_TAGGER_REPO}; the first run downloads its model files...")
        labels_path = hf_hub_download(WD_TAGGER_REPO, "selected_tags.csv")
        model_path = hf_hub_download(WD_TAGGER_REPO, "model.onnx")

        self.tag_names = []
        self.general_indexes = []
        with open(labels_path, "r", encoding="utf-8-sig", newline="") as labels_file:
            for index, row in enumerate(csv.DictReader(labels_file)):
                name = (row.get("name") or "").replace("_", " ").strip()
                self.tag_names.append(name)
                if str(row.get("category", "")).strip() == "0":
                    self.general_indexes.append(index)

        available = ort.get_available_providers()
        preferred = [
            provider for provider in (
                "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"
            )
            if provider in available
        ]
        self.session = ort.InferenceSession(
            model_path, providers=preferred or available
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        _, height, _, _ = self.session.get_inputs()[0].shape
        self.target_size = int(height)
        logger(f"Local tagger ready using {self.session.get_providers()[0]}.")

    def _prepare_image(self, image_path):
        image = self.ImageOps.exif_transpose(self.Image.open(image_path)).convert("RGBA")
        canvas = self.Image.new("RGBA", image.size, (255, 255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        max_dimension = max(image.size)
        left = (max_dimension - image.width) // 2
        top = (max_dimension - image.height) // 2
        square = self.Image.new("RGB", (max_dimension, max_dimension), (255, 255, 255))
        square.paste(image, (left, top))
        if max_dimension != self.target_size:
            square = square.resize(
                (self.target_size, self.target_size), self.Image.Resampling.BICUBIC
            )

        image_array = self.np.asarray(square, dtype=self.np.float32)[:, :, ::-1]
        return self.np.expand_dims(image_array, axis=0)

    def predict(self, image_path, threshold=WD_TAG_THRESHOLD):
        image_array = self._prepare_image(image_path)
        predictions = self.session.run(
            [self.output_name], {self.input_name: image_array}
        )[0][0].astype(float)
        tags = [
            (self.tag_names[index], predictions[index])
            for index in self.general_indexes
            if predictions[index] >= threshold
            and self.tag_names[index]
            and self.tag_names[index].casefold() not in WD_BLOCKED_TAGS
        ]
        tags.sort(key=lambda item: item[1], reverse=True)
        return ", ".join(
            f"{name} ({score:.2f})" for name, score in tags[:WD_MAX_TAGS]
        )


def get_local_tagger(logger=print):
    """Load and cache one local tagger instance for all worker threads."""
    global _LOCAL_TAGGER
    if _LOCAL_TAGGER is None:
        with _LOCAL_TAGGER_LOCK:
            if _LOCAL_TAGGER is None:
                _LOCAL_TAGGER = LocalWdTagger(logger)
    return _LOCAL_TAGGER


def choose_grok_fallback(model_ids):
    """Choose the most likely image-capable general Grok model from the API list."""
    excluded_terms = (
        "image-generation", "imagine", "embedding", "moderation", "audio",
        "code", "mini", "beta",
    )
    candidates = [
        model_id for model_id in model_ids
        if "grok" in model_id.lower()
        and not any(term in model_id.lower() for term in excluded_terms)
    ]
    if not candidates:
        return None

    def score(model_id):
        name = model_id.lower()
        return (
            "vision" in name,
            "4.1" in name,
            "grok-4" in name,
            "latest" in name,
            "fast" not in name,
        )

    return max(candidates, key=score)


def load_env_file(path=ENV_FILE):
    """Load simple KEY=VALUE entries while preserving explicitly set environment variables."""
    if not os.path.isfile(path):
        return

    with open(path, "r", encoding="utf-8-sig") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and value:
                os.environ.setdefault(key, value)




def create_clients():
    """Create API clients for every provider whose key is available."""
    load_env_file()
    clients = {}
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        clients["OpenAI"] = OpenAI(api_key=openai_key)

    grok_key = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
    if grok_key:
        clients["Grok"] = OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1")
    return clients


def fetch_provider_models(provider, client):
    """Return likely caption-capable model names for a provider."""
    model_ids = sorted(model.id for model in client.models.list().data)
    if provider == "OpenAI":
        return [
            model_id for model_id in model_ids
            if any(tag in model_id.lower() for tag in ("gpt", "o1", "o3"))
            and not any(tag in model_id.lower() for tag in ("audio", "realtime", "embedding", "image"))
        ]
    return [
        model_id for model_id in model_ids
        if "grok" in model_id.lower()
        and not any(tag in model_id.lower() for tag in ("image-generation", "imagine", "embedding", "audio"))
    ]

def encode_image(image_path):
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def request_caption(
    provider, client, image_path, model, subject_metadata=None, visual_tags=None,
    logger=print, max_characters=CAPTION_TARGET_CHARACTERS,
    prompt_override=None, caption_content_settings=None, detail_level=CAPTION_DETAIL_DEFAULT,
    trigger_word=TRIGGER_PHRASE, subject_class=SUBJECT_CLASS
):
    """Send one caption request using provider-compatible token parameters."""
    base64_image = encode_image(image_path)
    mime_type, _ = guess_type(image_path)
    if not mime_type:
        mime_type = 'image/jpeg'

    prompt_text = build_caption_prompt(
        subject_metadata, visual_tags, max_characters,
        prompt_override, caption_content_settings, detail_level, trigger_word, subject_class
    )
    missing_from_prompt = [
        hint for hint in ((subject_metadata or {}).get("folder_hints") or [])
        if hint not in prompt_text
    ]
    if missing_from_prompt:
        raise RuntimeError(
            "internal prompt validation failed; missing folder hint(s): "
            + ", ".join(missing_from_prompt)
        )

    request = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
    }
    if provider == 'Grok':
        request["max_tokens"] = CAPTION_MAX_TOKENS
    else:
        request["max_completion_tokens"] = CAPTION_MAX_TOKENS

    response = client.chat.completions.create(**request)
    choice = response.choices[0]
    message = choice.message
    refusal = getattr(message, "refusal", None)
    content = (message.content or "").strip()

    # The model hitting its token ceiling cuts the caption mid-sentence, which
    # local truncation no longer masks. Surface it instead of saving it silently.
    if getattr(choice, "finish_reason", None) == "length":
        logger(
            f"Warning: {provider} stopped at the {CAPTION_MAX_TOKENS}-token ceiling; "
            "caption may end mid-sentence. Raise CAPTION_MAX_TOKENS if this recurs."
        )

    refusal_prefixes = (
        "sorry", "i'm sorry", "i am sorry", "i can't", "i can’t", "i cannot",
        "unable to comply", "unable to assist", "i won't", "i won’t", "i will not",
    )
    if refusal:
        raise RuntimeError(f"{provider} refused the caption request: {refusal}")
    if not content:
        raise RuntimeError(f"{provider} returned an empty caption")
    refusal_check = strip_caption_prefix(content, trigger_word, subject_class).casefold()
    if refusal_check.startswith(refusal_prefixes):
        raise RuntimeError(f"{provider} returned refusal-style text: {content[:160]}")

    return content


def missing_folder_hints(caption, subject_metadata):
    """Return authoritative folder hints absent from a generated caption."""
    normalized_caption = " ".join((caption or "").split()).casefold()
    return [
        hint for hint in ((subject_metadata or {}).get("folder_hints") or [])
        if " ".join(hint.split()).casefold() not in normalized_caption
    ]


def request_caption_with_hint_validation(
    provider, client, image_path, model, subject_metadata=None, visual_tags=None,
    logger=print, max_characters=CAPTION_TARGET_CHARACTERS,
    prompt_override=None, caption_content_settings=None, detail_level=CAPTION_DETAIL_DEFAULT,
    trigger_word=TRIGGER_PHRASE, subject_class=SUBJECT_CLASS
):
    """Generate a caption and retry once when an exact folder hint is missing."""
    caption = request_caption(
        provider, client, image_path, model, subject_metadata, visual_tags,
        logger, max_characters, prompt_override, caption_content_settings, detail_level, trigger_word, subject_class,
    )
    missing = missing_folder_hints(caption, subject_metadata)
    if not missing:
        return caption

    logger(
        f"Retrying {os.path.basename(image_path)} because mandatory folder hints were "
        f"missing: {', '.join(missing)}"
    )
    retry_metadata = dict(subject_metadata or {})
    retry_metadata["_retry_missing_folder_hints"] = missing
    caption = request_caption(
        provider, client, image_path, model, retry_metadata, visual_tags,
        logger, max_characters, prompt_override, caption_content_settings, detail_level, trigger_word, subject_class,
    )
    still_missing = missing_folder_hints(caption, subject_metadata)
    if still_missing:
        raise RuntimeError(
            "caption omitted mandatory folder hint(s) after retry: "
            + ", ".join(still_missing)
        )
    return caption


def caption_image(
    provider, client, image_path, model, subject_metadata=None, fallback=None,
    logger=print, visual_tags=None,
    max_characters=CAPTION_TARGET_CHARACTERS, prompt_override=None,
    caption_content_settings=None, detail_level=CAPTION_DETAIL_DEFAULT,
    trigger_word=TRIGGER_PHRASE, subject_class=SUBJECT_CLASS
):
    """Sends image to OpenAI for captioning."""
    try:
        return request_caption_with_hint_validation(
            provider, client, image_path, model, subject_metadata, visual_tags,
            logger, max_characters, prompt_override, caption_content_settings, detail_level, trigger_word, subject_class
        )
    except Exception as e:
        error_message = str(e)
        logger(f"{provider} error captioning {os.path.basename(image_path)}: {e}")

        if provider == 'OpenAI' and fallback:
            fallback_provider, fallback_client, fallback_model = fallback
            logger(f"Retrying {os.path.basename(image_path)} with {fallback_provider} - {fallback_model}...")
            try:
                return request_caption_with_hint_validation(
                    fallback_provider,
                    fallback_client,
                    image_path,
                    fallback_model,
                    subject_metadata,
                    visual_tags,
                    logger,
                    max_characters,
                    prompt_override,
                    caption_content_settings,
                    detail_level,
                    trigger_word,
                    subject_class,
                )
            except Exception as fallback_error:
                logger(f"Grok fallback also failed for {os.path.basename(image_path)}: {fallback_error}")
                return None
        logger(f"Caption failed with {provider} - {model}: {error_message}")
        return None

def truncate_at_sentence_boundary(caption, max_characters):
    """Keep a caption under the character limit without cutting a sentence when possible."""
    if len(caption) <= max_characters:
        return caption

    shortened = caption[:max_characters].rstrip()
    sentence_ends = [match.end() for match in re.finditer(r"[.!?](?=\s|$)", shortened)]
    if sentence_ends:
        # Keep the last complete sentence unless doing so would discard most of the
        # useful caption because the model produced one unusually long sentence.
        complete = shortened[:sentence_ends[-1]].strip()
        if len(complete) >= max_characters * 0.6:
            return complete

    # Avoid splitting the final word when no useful sentence boundary exists.
    shortened = shortened.rsplit(" ", 1)[0].rstrip(",;:-")
    return shortened + "."


def clean_caption(caption, logger=print, trigger_word=TRIGGER_PHRASE, subject_class=SUBJECT_CLASS):
    """Normalize a model response into one Krea 2 sidecar caption."""
    caption = caption.strip().replace("**", "")
    caption = re.sub(r"^\s*(caption|description)\s*:\s*", "", caption, flags=re.IGNORECASE)
    caption = caption.strip(" \t\r\n\"'`*-•")
    caption = " ".join(caption.split())

    # Enforce the exact trigger + class opening in the saved caption.
    prefix = caption_prefix(trigger_word, subject_class)
    trigger_word, subject_class = " ".join(trigger_word.split()), " ".join(subject_class.split())
    caption = strip_caption_prefix(caption, trigger_word, subject_class)
    caption = f"{prefix} {caption}".strip()

    character_count = len(caption)
    if MAX_CAPTION_CHARACTERS and character_count > MAX_CAPTION_CHARACTERS:
        caption = truncate_at_sentence_boundary(caption, MAX_CAPTION_CHARACTERS)
        logger(f"Warning: caption was truncated to {MAX_CAPTION_CHARACTERS} characters.")
    elif character_count < MIN_CAPTION_CHARACTERS:
        logger(
            f"Warning: caption has only {character_count} characters; preferred "
            f"minimum is {MIN_CAPTION_CHARACTERS}."
        )

    return caption


def process_single_image(
    provider, image_file, client, model, subject_metadata=None, fallback=None,
    logger=print, local_tagger=None, tag_threshold=WD_TAG_THRESHOLD,
    max_characters=CAPTION_TARGET_CHARACTERS,
    prompt_override=None, caption_content_settings=None, detail_level=CAPTION_DETAIL_DEFAULT,
    trigger_word=TRIGGER_PHRASE, subject_class=SUBJECT_CLASS
):
    """Process a single image and return the result."""
    base_name = os.path.splitext(image_file)[0]
    text_file = f"{base_name}.txt"

    folder_labels = (subject_metadata or {}).get("folder_hints") or []
    if folder_labels:
        logger(
            f"Folder hints for {os.path.basename(image_file)}: "
            + "; ".join(folder_labels)
        )

    visual_tags = None
    if local_tagger:
        try:
            visual_tags = local_tagger.predict(image_file, tag_threshold)
            logger(f"Local WD tagger scanned {os.path.basename(image_file)}")
        except Exception as exc:
            logger(f"Local WD tagger skipped {os.path.basename(image_file)}: {exc}")

    caption = caption_image(
        provider, client, image_file, model, subject_metadata, fallback, logger,
        visual_tags, max_characters, prompt_override, caption_content_settings, detail_level, trigger_word, subject_class,
    )

    if caption:
        caption = clean_caption(caption, logger, trigger_word, subject_class)

        with open(text_file, "w", encoding="utf-8") as f:
            f.write(caption)

        return True
    return False

def cleanup_orphaned_captions(files, image_base_names, delete_orphans=False, logger=print):
    """Report orphaned captions and optionally delete them."""
    txt_files = [f for f in files if f.lower().endswith('.txt')]
    orphaned = []

    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]
        if base_name not in image_base_names:
            orphaned.append(txt_file)
            if delete_orphans:
                try:
                    os.remove(txt_file)
                except Exception as e:
                    logger(f"Error deleting orphaned caption {txt_file}: {e}")

    if orphaned:
        action = "Deleted" if delete_orphans else "Found"
        logger(f"{action} {len(orphaned)} orphaned caption(s).")

    return len(orphaned)

class CaptionApp:
    """Tkinter interface for batch-captioning one or more image folders."""

    def __init__(self, root):
        self.root = root
        self.root.title("Famegrid Krea 2 Captioner")
        self.root.geometry("860x920")
        self.root.minsize(700, 520)

        self.folders = []
        self.provider_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value="")
        self.workers_var = tk.IntVar(value=MAX_WORKERS)
        self.overwrite_var = tk.BooleanVar(value=OVERWRITE_EXISTING_CAPTIONS)
        self.delete_orphans_var = tk.BooleanVar(value=DELETE_ORPHANED_CAPTIONS)
        self.include_subfolders_var = tk.BooleanVar(value=True)
        self.fallback_var = tk.BooleanVar(value=False)
        self.local_tagger_var = tk.BooleanVar(value=False)
        self.tag_threshold_var = tk.DoubleVar(value=WD_TAG_THRESHOLD)
        self.trigger_word_var = tk.StringVar(value=TRIGGER_PHRASE)
        self.subject_class_var = tk.StringVar(value=SUBJECT_CLASS)
        self.class_varies_var = tk.BooleanVar(value=False)
        self.detail_level_var = tk.StringVar(value=CAPTION_DETAIL_DEFAULT)
        self.caption_length_var = tk.IntVar(value=CAPTION_TARGET_CHARACTERS)
        self.caption_content_vars = {
            key: tk.BooleanVar(value=True)
            for key, _label, _prompt_label in CAPTION_CONTENT_FIELDS
        }
        self.status_var = tk.StringVar(value="Add one or more image folders to begin.")

        self._build_ui()

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(12, weight=1)

        folder_frame = ttk.Frame(outer)
        folder_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        folder_frame.columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="Image folders").grid(
            row=0, column=0, sticky="nw", padx=(0, 8), pady=5
        )

        list_wrap = ttk.Frame(folder_frame)
        list_wrap.grid(row=0, column=1, sticky="ew", pady=5)
        list_wrap.columnconfigure(0, weight=1)
        self.folder_list = tk.Listbox(list_wrap, height=5, selectmode="extended", activestyle="none")
        self.folder_list.grid(row=0, column=0, sticky="ew")
        folder_scroll = ttk.Scrollbar(list_wrap, orient="vertical", command=self.folder_list.yview)
        folder_scroll.grid(row=0, column=1, sticky="ns")
        self.folder_list.configure(yscrollcommand=folder_scroll.set)
        self.folder_list.bind("<Delete>", lambda _e: self.remove_selected_folders())

        folder_buttons = ttk.Frame(folder_frame)
        folder_buttons.grid(row=0, column=2, sticky="n", padx=(8, 0), pady=5)
        ttk.Button(folder_buttons, text="Add folder...", command=self.add_folder, width=17).pack(fill="x")
        ttk.Button(
            folder_buttons, text="Add subfolders...", command=self.add_subfolders, width=17
        ).pack(fill="x", pady=(4, 0))
        ttk.Button(
            folder_buttons, text="Remove", command=self.remove_selected_folders, width=17
        ).pack(fill="x", pady=(4, 0))
        ttk.Button(folder_buttons, text="Clear", command=self.clear_folders, width=17).pack(
            fill="x", pady=(4, 0)
        )

        ttk.Label(outer, text="Primary provider").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=5)
        provider_options = ttk.Frame(outer)
        provider_options.grid(row=1, column=1, columnspan=2, sticky="w", pady=5)
        self.provider_buttons = {}
        for provider in ("OpenAI", "Grok"):
            button = ttk.Radiobutton(
                provider_options, text=provider, value=provider,
                variable=self.provider_var, command=self.provider_changed,
            )
            button.pack(side="left", padx=(0, 16))
            self.provider_buttons[provider] = button
        ttk.Label(provider_options, text="Choose which provider captions your images.").pack(side="left")

        ttk.Label(outer, text="Model").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=5)
        self.model_box = ttk.Combobox(outer, textvariable=self.model_var)
        self.model_box.grid(row=2, column=1, sticky="ew", pady=5)
        self.refresh_button = ttk.Button(outer, text="Refresh models", command=self.refresh_models, state="disabled")
        self.refresh_button.grid(row=2, column=2, padx=(8, 0), pady=5)

        ttk.Label(outer, text="Parallel images").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=5)
        ttk.Spinbox(outer, from_=1, to=20, textvariable=self.workers_var, width=8).grid(
            row=3, column=1, sticky="w", pady=5
        )

        options = ttk.Frame(outer)
        options.grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 2))
        ttk.Checkbutton(options, text="Overwrite existing .txt captions", variable=self.overwrite_var).pack(
            side="left", padx=(0, 18)
        )
        ttk.Checkbutton(options, text="Delete orphaned captions", variable=self.delete_orphans_var).pack(
            side="left", padx=(0, 18)
        )
        ttk.Checkbutton(options, text="Include subfolders", variable=self.include_subfolders_var).pack(
            side="left"
        )

        caption_opts = ttk.Frame(outer)
        caption_opts.grid(row=5, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Label(caption_opts, text="Target caption length").grid(row=0, column=1, sticky="w", padx=(0, 6))
        ttk.Spinbox(
            caption_opts,
            from_=0,
            to=20000,
            increment=50,
            width=8,
            textvariable=self.caption_length_var,
        ).grid(row=0, column=2, sticky="w")
        ttk.Label(caption_opts, text="characters (0 = do not mention a length)").grid(
            row=0, column=3, sticky="w", padx=(6, 0)
        )
        ttk.Label(caption_opts, text="Detail level").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Combobox(
            caption_opts, textvariable=self.detail_level_var,
            values=tuple(CAPTION_DETAIL_LEVELS), state="readonly", width=12,
        ).grid(row=1, column=1, sticky="w", pady=(6, 0))
        ttk.Label(caption_opts, text="Description depth within selected categories").grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(6, 0), pady=(6, 0)
        )

        ttk.Label(caption_opts, text="Trigger word").grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(caption_opts, textvariable=self.trigger_word_var, width=16).grid(
            row=2, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(caption_opts, text="Subject class").grid(
            row=2, column=2, sticky="w", padx=(6, 0), pady=(6, 0)
        )
        class_options = ttk.Frame(caption_opts)
        class_options.grid(
            row=2, column=3, sticky="w", padx=(6, 0), pady=(6, 0)
        )
        self.subject_class_entry = ttk.Entry(class_options, textvariable=self.subject_class_var, width=16)
        self.subject_class_entry.pack(side="left")
        ttk.Checkbutton(
            class_options, text="Class varies", variable=self.class_varies_var,
            command=lambda: self.subject_class_entry.configure(
                state="disabled" if self.class_varies_var.get() else "normal"
            ),
        ).pack(side="left", padx=(8, 0))
        ttk.Label(caption_opts, text="Trigger + class; Class varies chooses the class from each image.").grid(
            row=3, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )

        self.fallback_button = ttk.Checkbutton(
            outer,
            text="Optional: fall back to Grok if OpenAI fails or refuses (requires a Grok API key)",
            variable=self.fallback_var,
            state="disabled",
        )
        self.fallback_button.grid(row=6, column=0, columnspan=3, sticky="w", pady=(2, 10))

        tagger_frame = ttk.LabelFrame(outer, text="Optional local tag pre-scan", padding=8)
        tagger_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        tagger_frame.columnconfigure(2, weight=1)
        ttk.Checkbutton(
            tagger_frame,
            text="Use WD EVA02-Large Tagger v3 before the LLM",
            variable=self.local_tagger_var,
        ).grid(row=0, column=0, sticky="w", padx=(0, 14))
        ttk.Label(tagger_frame, text="Threshold").grid(row=0, column=1, sticky="w", padx=(0, 6))
        ttk.Spinbox(
            tagger_frame,
            from_=0.05,
            to=0.95,
            increment=0.05,
            textvariable=self.tag_threshold_var,
            width=7,
        ).grid(row=0, column=2, sticky="w")

        content_frame = ttk.LabelFrame(
            outer, text="Caption content", padding=8
        )
        content_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        ttk.Label(
            content_frame,
            text=(
                "Checked categories are included when visible. Unchecked categories are "
                "explicitly excluded. Folder hints and filename metadata always remain required."
            ),
            wraplength=780,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 7))
        for index, (key, label, _prompt_label) in enumerate(CAPTION_CONTENT_FIELDS):
            ttk.Checkbutton(
                content_frame, text=label, variable=self.caption_content_vars[key]
            ).grid(
                row=1 + index // 3,
                column=index % 3,
                sticky="w",
                padx=(0, 18),
                pady=2,
            )
        content_buttons = ttk.Frame(content_frame)
        content_buttons.grid(row=1, column=3, rowspan=5, sticky="ne")
        ttk.Button(
            content_buttons,
            text="Select all",
            command=lambda: self._set_caption_categories(True),
            width=11,
        ).pack(fill="x")
        ttk.Button(
            content_buttons,
            text="Clear all",
            command=lambda: self._set_caption_categories(False),
            width=11,
        ).pack(fill="x", pady=(4, 0))

        prompt_frame = ttk.LabelFrame(outer, text="Prompt override (optional)", padding=8)
        prompt_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        prompt_frame.columnconfigure(0, weight=1)
        ttk.Label(
            prompt_frame,
            text=(
                "Leave blank to use the built-in prompt. When filled, this replaces the "
                "built-in prompt; metadata, local tag hints, category controls, and detail level still apply."
            ),
            wraplength=780,
        ).grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.prompt_override_box = scrolledtext.ScrolledText(
            prompt_frame, wrap="word", height=6, undo=True
        )
        self.prompt_override_box.grid(row=1, column=0, sticky="ew")
        prompt_buttons = ttk.Frame(prompt_frame)
        prompt_buttons.grid(row=2, column=0, sticky="e", pady=(5, 0))
        ttk.Button(
            prompt_buttons, text="Load built-in prompt", command=self.load_builtin_prompt
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            prompt_buttons,
            text="Clear override",
            command=lambda: self.prompt_override_box.delete("1.0", "end"),
        ).pack(side="left")

        controls = ttk.Frame(outer)
        controls.grid(row=10, column=0, columnspan=3, sticky="ew")
        controls.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(controls, text="Start Captioning", command=self.start_captioning)
        self.start_button.grid(row=0, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.status_var).grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.progress = ttk.Progressbar(outer, mode="determinate")
        self.progress.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(10, 8))

        self.log_box = scrolledtext.ScrolledText(outer, wrap="word", height=18, state="disabled")
        self.log_box.grid(row=12, column=0, columnspan=3, sticky="nsew")

    def _refresh_folder_list(self):
        self.folder_list.delete(0, "end")
        for path in self.folders:
            self.folder_list.insert("end", path)
        count = len(self.folders)
        if count:
            self.status_var.set(f"{count} folder(s) queued.")
        else:
            self.status_var.set("Add one or more image folders to begin.")

    def _set_caption_categories(self, enabled):
        for variable in self.caption_content_vars.values():
            variable.set(enabled)

    def _add_folder_paths(self, paths):
        """Add directories to the queue, skipping duplicates and non-directories."""
        existing = {os.path.normcase(p) for p in self.folders}
        added = 0
        for path in paths:
            resolved = os.path.abspath(path)
            key = os.path.normcase(resolved)
            if key in existing or not os.path.isdir(resolved):
                continue
            self.folders.append(resolved)
            existing.add(key)
            added += 1
        self._refresh_folder_list()
        return added

    def add_folder(self):
        selected = filedialog.askdirectory(title="Choose a folder containing images")
        if selected:
            if self._add_folder_paths([selected]):
                self.log(f"Added folder: {os.path.abspath(selected)}")
            else:
                self.log(f"Already queued: {os.path.abspath(selected)}")

    def add_subfolders(self):
        parent = filedialog.askdirectory(
            title="Choose a parent folder - each subfolder inside it will be added"
        )
        if not parent:
            return
        try:
            children = sorted(
                os.path.join(parent, name)
                for name in os.listdir(parent)
                if os.path.isdir(os.path.join(parent, name))
            )
        except OSError as exc:
            messagebox.showerror("Could not read folder", str(exc))
            return
        if not children:
            messagebox.showinfo("No subfolders", "That folder has no subfolders.")
            return
        added = self._add_folder_paths(children)
        self.log(f"Added {added} subfolder(s) from {os.path.abspath(parent)}.")

    def remove_selected_folders(self):
        for index in sorted(self.folder_list.curselection(), reverse=True):
            del self.folders[index]
        self._refresh_folder_list()

    def clear_folders(self):
        self.folders.clear()
        self._refresh_folder_list()

    def load_builtin_prompt(self):
        """Copy the current built-in prompt into the editor for customization."""
        try:
            caption_length = max(0, int(self.caption_length_var.get()))
        except (TypeError, ValueError, tk.TclError):
            caption_length = CAPTION_TARGET_CHARACTERS
        subject_class = "" if self.class_varies_var.get() else self.subject_class_var.get().strip()
        if not self.trigger_word_var.get().strip() or (not self.class_varies_var.get() and not subject_class):
            messagebox.showerror("Missing caption opening", "Enter a trigger word and a subject class, or check Class varies.")
            return
        prompt = build_caption_prompt(
            max_characters=caption_length,
            detail_level=self.detail_level_var.get(),
            trigger_word=self.trigger_word_var.get(), subject_class=subject_class
        )
        self.prompt_override_box.delete("1.0", "end")
        self.prompt_override_box.insert("1.0", prompt)
        self.prompt_override_box.focus_set()

    def provider_changed(self, _event=None):
        default = MODEL_NAME if self.provider_var.get() == "OpenAI" else GROK_DEFAULT_MODEL
        self.model_var.set(default)
        self.model_box["values"] = ()
        self.refresh_button.configure(state="normal")
        self.fallback_button.configure(
            state="normal" if self.provider_var.get() == "OpenAI" else "disabled"
        )

    def log(self, message):
        def append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", str(message).rstrip() + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        try:
            self.root.after(0, append)
        except tk.TclError:
            pass

    def refresh_models(self):
        provider = self.provider_var.get()
        self.refresh_button.configure(state="disabled")
        self.status_var.set(f"Fetching {provider} models...")

        def worker():
            try:
                clients = create_clients()
                if provider not in clients:
                    key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GROK_API_KEY or XAI_API_KEY"
                    raise RuntimeError(f"{key_name} is not set.")
                models = fetch_provider_models(provider, clients[provider])
                self.root.after(0, lambda: self._models_loaded(provider, models))
            except Exception as exc:
                self.root.after(0, lambda error=str(exc): self._model_load_failed(error))

        threading.Thread(target=worker, daemon=True).start()

    def _models_loaded(self, provider, models):
        self.refresh_button.configure(state="normal")
        if provider != self.provider_var.get():
            return
        self.model_box["values"] = models
        if models and self.model_var.get() not in models:
            preferred = MODEL_NAME if provider == "OpenAI" else choose_grok_fallback(models)
            self.model_var.set(preferred or models[0])
        self.status_var.set(f"Loaded {len(models)} {provider} model(s).")

    def _model_load_failed(self, error):
        self.refresh_button.configure(state="normal")
        self.status_var.set("Could not fetch models.")
        messagebox.showerror("Model lookup failed", error)

    def start_captioning(self):
        folders = list(self.folders)
        provider = self.provider_var.get()
        if provider not in ("OpenAI", "Grok"):
            messagebox.showerror("Choose a provider", "Choose OpenAI or Grok as your primary captioning provider.")
            return
        model = self.model_var.get().strip()
        use_local_tagger = self.local_tagger_var.get()
        prompt_override = self.prompt_override_box.get("1.0", "end-1c").strip()
        caption_content_settings = {
            key: variable.get()
            for key, variable in self.caption_content_vars.items()
        }

        trigger_word = " ".join(self.trigger_word_var.get().split())
        subject_class = "" if self.class_varies_var.get() else " ".join(self.subject_class_var.get().split())
        if not trigger_word or (not self.class_varies_var.get() and not subject_class):
            messagebox.showerror("Missing caption opening", "Enter a trigger word and a subject class, or check Class varies.")
            return

        if not folders:
            messagebox.showerror("No folders", "Add at least one image folder.")
            return
        missing = [path for path in folders if not os.path.isdir(path)]
        if missing:
            messagebox.showerror(
                "Invalid folder",
                "These queued folders no longer exist:\n\n" + "\n".join(missing),
            )
            return
        if not model:
            messagebox.showerror("Missing model", "Enter or select a model name.")
            return
        try:
            workers = int(self.workers_var.get())
            if not 1 <= workers <= 20:
                raise ValueError
        except (TypeError, ValueError, tk.TclError):
            messagebox.showerror("Invalid worker count", "Parallel images must be between 1 and 20.")
            return

        try:
            tag_threshold = float(self.tag_threshold_var.get())
            if not 0.05 <= tag_threshold <= 0.95:
                raise ValueError
        except (TypeError, ValueError, tk.TclError):
            messagebox.showerror("Invalid tag threshold", "Tag threshold must be between 0.05 and 0.95.")
            return

        try:
            caption_length = int(self.caption_length_var.get())
            if caption_length < 0:
                raise ValueError
        except (TypeError, ValueError, tk.TclError):
            messagebox.showerror(
                "Invalid caption length",
                "Target caption length must be 0 or a positive number of characters.",
            )
            return

        clients = create_clients()
        if provider not in clients:
            key_name = "OPENAI_API_KEY" if provider == "OpenAI" else "GROK_API_KEY or XAI_API_KEY"
            messagebox.showerror("Missing API key", f"{key_name} is not set in the environment.")
            return

        settings = {
            "folders": folders,
            "provider": provider,
            "model": model,
            "workers": workers,
            "overwrite": self.overwrite_var.get(),
            "delete_orphans": self.delete_orphans_var.get(),
            "include_subfolders": self.include_subfolders_var.get(),
            "use_fallback": provider == "OpenAI" and self.fallback_var.get(),
            "use_local_tagger": use_local_tagger,
            "tag_threshold": tag_threshold,
            "max_characters": caption_length,
            "prompt_override": prompt_override or None,
            "caption_content_settings": caption_content_settings,
            "detail_level": self.detail_level_var.get(),
            "trigger_word": trigger_word,
            "subject_class": subject_class,
            "clients": clients,
        }
        self.start_button.configure(state="disabled")
        self.refresh_button.configure(state="disabled")
        self.progress.configure(value=0, maximum=1)
        self.status_var.set("Preparing...")
        self.log(f"Queued {len(folders)} folder(s):")
        for path in folders:
            self.log(f"  - {path}")
        self.log(f"Primary model: {provider} - {model}")
        self.log(
            "Using the UI prompt override."
            if prompt_override else "Using the built-in caption prompt."
        )
        included_categories = [
            label for key, label, _prompt_label in CAPTION_CONTENT_FIELDS
            if caption_content_settings[key]
        ]
        self.log(
            f"Caption categories selected: {len(included_categories)} of "
            f"{len(CAPTION_CONTENT_FIELDS)}."
        )
        self.log(f"Caption opening: {trigger_word} {subject_class or '[class chosen per image]'}")
        self.log(f"Caption detail level: {settings['detail_level']}.")
        if prompt_override:
            self.log("The target-length control does not modify a prompt override.")
        else:
            self.log(
                f"Asking the model to stay under about {caption_length} characters."
                if caption_length else "No caption length mentioned to the model."
            )
        self.log(
            "Subfolder scanning enabled." if settings["include_subfolders"]
            else "Only the selected folder will be scanned."
        )
        if use_local_tagger:
            self.log(
                f"Local WD EVA02-Large v3 pre-scan enabled at threshold {tag_threshold:.2f}."
            )
        threading.Thread(target=self._run_captioning, args=(settings,), daemon=True).start()

    def _run_captioning(self, settings):
        try:
            folders = settings["folders"]
            provider = settings["provider"]
            client = settings["clients"][provider]
            model = settings["model"]

            fallback = None
            if provider == "OpenAI" and settings["use_fallback"]:
                grok_client = settings["clients"].get("Grok")
                if grok_client:
                    try:
                        grok_models = fetch_provider_models("Grok", grok_client)
                        fallback_model = choose_grok_fallback(grok_models)
                        if fallback_model:
                            fallback = ("Grok", grok_client, fallback_model)
                            self.log(f"Automatic fallback: Grok - {fallback_model}")
                        else:
                            self.log("Grok fallback unavailable: no suitable model was returned.")
                    except Exception as exc:
                        self.log(f"Grok fallback setup failed: {exc}")
                else:
                    self.log("Grok fallback disabled because GROK_API_KEY or XAI_API_KEY is not set.")

            local_tagger = None
            if settings["use_local_tagger"]:
                local_tagger = get_local_tagger(self.log)

            # Each queued folder is its own root: it loads its own
            # subject_metadata.csv and caption_folder_hints.json, so hints never
            # leak between unrelated folders and never go silently missing.
            images_to_process = []
            folder_context = {}
            seen_images = set()
            skipped = 0

            for folder in folders:
                if settings["include_subfolders"]:
                    files = [
                        os.path.join(directory, name)
                        for directory, _subdirectories, names in os.walk(folder)
                        for name in names
                    ]
                else:
                    files = [
                        os.path.join(folder, name)
                        for name in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, name))
                    ]
                image_files = [
                    path for path in files if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
                ]
                image_base_names = {os.path.splitext(path)[0] for path in image_files}
                cleanup_orphaned_captions(
                    files, image_base_names, settings["delete_orphans"], self.log
                )

                metadata = load_subject_metadata(folder, self.log)
                folder_hints = load_folder_hints(folder, self.log)
                folder_context[folder] = (metadata, folder_hints)

                queued_here = 0
                for image_path in image_files:
                    # A parent and one of its children can both be queued; only
                    # caption each image once, under the folder that found it first.
                    key = os.path.normcase(os.path.abspath(image_path))
                    if key in seen_images:
                        continue
                    seen_images.add(key)
                    caption_path = os.path.splitext(image_path)[0] + ".txt"
                    if os.path.exists(caption_path) and not settings["overwrite"]:
                        skipped += 1
                    else:
                        images_to_process.append((image_path, folder))
                        queued_here += 1

                self.log(f"{os.path.basename(folder) or folder}: {queued_here} image(s) queued.")

            if skipped:
                self.log(f"Skipped {skipped} image(s) with existing captions.")
            if not images_to_process:
                self._finish("No images need captioning.", 0, 0)
                return

            total = len(images_to_process)
            self.root.after(0, lambda: self.progress.configure(maximum=total, value=0))
            self.log(f"Captioning {total} image(s) with {settings['workers']} parallel worker(s).")

            completed = 0
            succeeded = 0
            with ThreadPoolExecutor(max_workers=settings["workers"]) as executor:
                future_to_image = {
                    executor.submit(
                        process_single_image,
                        provider,
                        image_path,
                        client,
                        model,
                        metadata_for_image(
                            image_path, image_folder, *folder_context[image_folder]
                        ),
                        fallback,
                        self.log,
                        local_tagger,
                        settings["tag_threshold"],
                        settings["max_characters"],
                        settings["prompt_override"],
                        settings["caption_content_settings"],
                        settings["detail_level"],
                        settings["trigger_word"],
                        settings["subject_class"],
                    ): image_path
                    for image_path, image_folder in images_to_process
                }
                for future in as_completed(future_to_image):
                    image_path = future_to_image[future]
                    try:
                        if future.result():
                            succeeded += 1
                            self.log(f"Saved {os.path.splitext(os.path.basename(image_path))[0]}.txt")
                    except Exception as exc:
                        self.log(f"Unexpected error for {os.path.basename(image_path)}: {exc}")
                    completed += 1
                    self.root.after(0, lambda done=completed: self._set_progress(done, total))

            self._finish(
                f"Finished: {succeeded} succeeded, {total - succeeded} failed.",
                succeeded,
                total - succeeded,
            )
        except Exception as exc:
            self.log(f"Fatal error: {exc}")
            for line in traceback.format_exc().rstrip().splitlines():
                self.log(f"  {line}")
            self._finish("Captioning stopped because of an error.", 0, 1)

    def _set_progress(self, completed, total):
        self.progress.configure(value=completed)
        self.status_var.set(f"Captioning {completed} of {total}...")

    def _finish(self, status, succeeded, failed):
        def update():
            self.status_var.set(status)
            self.start_button.configure(state="normal")
            self.refresh_button.configure(state="normal")
            self.log(status)
            if failed:
                messagebox.showwarning("Captioning complete", status)
            elif succeeded:
                messagebox.showinfo("Captioning complete", status)

        self.root.after(0, update)


def main():
    # pythonw sets these streams to None. Some dependencies still expect file-like
    # objects even in a windowed application, so provide harmless sinks.
    for stream_name in ("stdout", "stderr"):
        if getattr(sys, stream_name) is None:
            stream = open(os.devnull, "w", encoding="utf-8")
            _NULL_STREAMS.append(stream)
            setattr(sys, stream_name, stream)
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    CaptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
