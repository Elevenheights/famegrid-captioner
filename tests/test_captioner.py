import inspect
import os
from pathlib import Path
import tempfile
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import caption_photos as app


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=text, refusal=None), finish_reason='stop'
    )])


class CaptionerTests(unittest.TestCase):
    def assert_exclusion(self, prompt):
        self.assertIn('Do not describe or analyze highlights, shadows', prompt)
        self.assertNotIn('Technical descriptions of highlight and shadow reproduction are required', prompt)
        self.assertNotIn('Describe highlights and shadows using', prompt)

    def test_default_and_custom_prompt_paths(self):
        for subject_class in ('woman', ''):
            for override in (None, 'Describe highlights and shadows.'):
                with self.subTest(subject_class=subject_class, override=override):
                    prompt = app.build_caption_prompt(
                        prompt_override=override, subject_class=subject_class,
                        caption_content_settings={'lighting': True},
                        visual_tags='cast shadow: 0.9',
                        subject_metadata={'folder_hints': ['red jacket']},
                    )
                    self.assert_exclusion(prompt)
                    self.assertIn('red jacket', prompt)

    def test_removed_parameter(self):
        for function in (app.build_caption_prompt, app.request_caption,
                         app.request_caption_with_hint_validation, app.caption_image,
                         app.process_single_image):
            self.assertNotIn('include_tonal', inspect.signature(function).parameters)

    def test_primary_and_fallback_requests(self):
        primary, fallback = Mock(), Mock()
        primary.chat.completions.create.side_effect = RuntimeError('mock primary outage')
        fallback.chat.completions.create.return_value = response('famegrid woman in a red jacket.')
        with patch.object(app, 'encode_image', return_value='fake-image'):
            caption = app.caption_image(
                'OpenAI', primary, 'sample.png', 'primary-model',
                fallback=('Grok', fallback, 'fallback-model'), logger=lambda _: None,
                prompt_override='Caption the person.',
            )
        self.assertIn('red jacket', caption)
        for client in (primary, fallback):
            request = client.chat.completions.create.call_args.kwargs
            self.assert_exclusion(request['messages'][0]['content'][0]['text'])
        self.assertIn('max_tokens', fallback.chat.completions.create.call_args.kwargs)
        self.assertIn('max_completion_tokens', primary.chat.completions.create.call_args.kwargs)

    def test_hint_retry_preserves_exclusion(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            response('famegrid woman standing.'),
            response('famegrid woman in a red jacket.'),
        ]
        with patch.object(app, 'encode_image', return_value='fake-image'):
            caption = app.request_caption_with_hint_validation(
                'OpenAI', client, 'sample.png', 'model',
                subject_metadata={'folder_hints': ['red jacket']}, logger=lambda _: None,
            )
        self.assertIn('red jacket', caption)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        for call in client.chat.completions.create.call_args_list:
            self.assert_exclusion(call.kwargs['messages'][0]['content'][0]['text'])

    def test_sidecar_output(self):
        client = Mock()
        client.chat.completions.create.return_value = response('famegrid woman in a red jacket.')
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'sample.png'
            image.write_bytes(b'mock-image')
            self.assertTrue(app.process_single_image(
                'OpenAI', str(image), client, 'model', logger=lambda _: None,
            ))
            caption = image.with_suffix('.txt').read_text(encoding='utf-8')
            self.assertTrue(caption.startswith('famegrid woman'))
            self.assertIn('red jacket', caption)

    def test_environment_overrides_env_file(self):
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / '.env'
            env.write_text('OPENAI_API_KEY=example-file-key\nXAI_API_KEY=example-xai-key\n')
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'example-environment-key'}, clear=True):
                app.load_env_file(str(env))
                self.assertEqual(os.environ['OPENAI_API_KEY'], 'example-environment-key')
                self.assertEqual(os.environ['XAI_API_KEY'], 'example-xai-key')


class ProviderSelectionTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.ui = app.CaptionApp(self.root)
        self.root.update_idletasks()

    def tearDown(self):
        self.root.destroy()

    def test_requires_explicit_provider(self):
        self.assertEqual(self.ui.provider_var.get(), '')
        self.assertFalse(self.ui.fallback_var.get())
        self.assertTrue(self.ui.fallback_button.instate(['disabled']))
        with patch.object(app.messagebox, 'showerror') as error, patch.object(app, 'create_clients') as clients:
            self.ui.start_captioning()
            error.assert_called_once()
            clients.assert_not_called()

    def test_primary_selection_and_optional_fallback(self):
        for provider, fallback_enabled, expected_fallback in (
            ('OpenAI', False, False), ('OpenAI', True, True), ('Grok', True, False),
        ):
            with self.subTest(provider=provider, fallback=fallback_enabled), tempfile.TemporaryDirectory() as folder:
                self.ui.provider_buttons[provider].invoke()
                self.ui.fallback_var.set(fallback_enabled)
                self.assertEqual(self.ui.provider_var.get(), provider)
                self.assertEqual(self.ui.model_var.get(), app.MODEL_NAME if provider == 'OpenAI' else app.GROK_DEFAULT_MODEL)
                self.assertEqual(self.ui.fallback_button.instate(['disabled']), provider == 'Grok')
                self.ui.folders = [folder]
                with patch.object(app, 'create_clients', return_value={provider: Mock()}), patch.object(app.threading, 'Thread') as thread:
                    self.ui.start_captioning()
                    settings = thread.call_args.kwargs['args'][0]
                    self.assertEqual(settings['provider'], provider)
                    self.assertEqual(settings['use_fallback'], expected_fallback)
                    thread.return_value.start.assert_called_once()

    def test_grok_requires_only_grok_key(self):
        with patch.dict(os.environ, {'XAI_API_KEY': 'example-grok-key'}, clear=True), patch.object(app, 'load_env_file'), patch.object(app, 'OpenAI') as constructor:
            clients = app.create_clients()
            self.assertEqual(set(clients), {'Grok'})
            constructor.assert_called_once_with(api_key='example-grok-key', base_url='https://api.x.ai/v1')

    def test_stale_model_results_do_not_replace_selected_provider(self):
        self.ui.provider_buttons['Grok'].invoke()
        self.ui._models_loaded('OpenAI', ['openai-only-model'])
        self.assertEqual(self.ui.model_var.get(), app.GROK_DEFAULT_MODEL)
        self.assertEqual(tuple(self.ui.model_box['values']), ())


if __name__ == '__main__':
    unittest.main()
