import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hud_step import StepHud, _status_style


class GoalHudTests(unittest.TestCase):
    def test_only_final_goal_is_shown_as_completed(self):
        for status in ('waiting', 'in_progress', 'completed', 'confirming_completed',
                       'confirming_goal_completed', 'failed', 'confirming_failed'):
            label, background, _, _ = _status_style(status)
            self.assertNotIn('CONCLUÍDO', label)
            self.assertLessEqual(background[1], background[2])  # not a green completion panel
        self.assertEqual(_status_style('goal_completed')[0], 'OBJETIVO CONCLUÍDO')
        for status in ('in_progress', 'completed', 'confirming_completed', None):
            self.assertEqual(_status_style(status)[0], '')

    def test_final_goal_renders_green_and_preserves_camera_below_banner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'step.json'
            payload = {'step_index': 2, 'description': 'Colocar xícara',
                       'agent': 'manipulation', 'status': 'goal_completed'}
            path.write_text(json.dumps(payload))
            hud = StepHud(path)
            frame = np.full((480, 640, 3), 100, dtype=np.uint8)
            result = hud.draw(frame.copy())
            self.assertGreater(result[10, 1, 1], result[10, 1, 2])
            np.testing.assert_array_equal(result[150:], frame[150:])
            payload['status'] = 'completed'
            replacement = path.with_suffix('.tmp')
            replacement.write_text(json.dumps(payload))
            replacement.replace(path)
            active = hud.draw(frame.copy())
            self.assertLess(active[10, 1, 1], active[10, 1, 2])


if __name__ == '__main__':
    unittest.main()
