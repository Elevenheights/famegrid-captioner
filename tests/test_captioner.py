import inspect
import os
from pathlib import Path
import tempfile
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


if __name__ == '__main__':
    unittest.main()
