import importlib
import unittest

import torch


nodes = importlib.import_module("custom_nodes.comfyui-SelfLift.nodes")
h3_tiling = importlib.import_module("custom_nodes.comfyui-SelfLift.h3_tiling")


class SelfLiftH3ContinuationTests(unittest.TestCase):
    def test_low_resolution_preserves_audio_and_resizes_video_tail(self):
        video = torch.zeros(1, 24, 7, 6, 8)
        audio = torch.zeros(1, 32, 2, 37)
        cond = [[torch.zeros(1, 2, 3), {"minimax_keyframes": [{
            "resolved_frame_index": 0,
            "latent": video,
            "audio_latent": audio,
        }]}]]
        resized = nodes._resize_keyframes(cond, 4, 6)
        keyframe = resized[0][1]["minimax_keyframes"][0]
        self.assertEqual(tuple(keyframe["latent"].shape), (1, 24, 7, 4, 6))
        self.assertIs(keyframe["audio_latent"], audio)
        self.assertEqual(keyframe["resolved_frame_index"], 0)

    def test_high_resolution_tile_keeps_tail_and_reference_rows_aligned(self):
        video = torch.zeros(1, 24, 7, 20, 30)
        audio = torch.zeros(1, 32, 2, 40)
        context = torch.zeros(1, 12, 96)
        tail = torch.zeros_like(video)
        reference = torch.zeros(1, 24, 1, 16, 18)
        payload = {
            "keyframes": [{"latent": tail, "resolved_frame_index": 0}],
            "refs": [{"kind": "image", "latent": reference, "latent_h": 16, "latent_w": 18}],
            "cond_video_latents": [tail, reference],
        }
        tiled = h3_tiling._tile_payload(payload, context, video, audio, 4, 0, 20)
        rows = sum(
            latent.shape[2] * ((latent.shape[3] + 1) // 2) * ((latent.shape[4] + 1) // 2)
            for latent in tiled["cond_video_latents"]
        )
        self.assertEqual(rows, int((~tiled["layout"].img_update).sum()))


if __name__ == "__main__":
    unittest.main()
