"""Generate clearly synthetic schema examples for the three VLA domains."""
import argparse
import json
from pathlib import Path
from PIL import Image


def create(root):
    for domain in ('robotics', 'simulation', 'desktop'):
        directory = root / domain
        (directory / 'frames').mkdir(parents=True, exist_ok=True)
        train, validation = [], []
        for index in range(32):
            red = index % 2 == 0
            value = .2 if red else -.2
            Image.new('RGB', (64, 64), 'red' if red else 'blue').save(directory / 'frames' / f'{index}.png')
            if domain == 'robotics':
                action = [value, -value, 0., 0., 0., 0., float(red)]
                state = [0.] * 7
            elif domain == 'simulation':
                action = {'continuous': [value, .5], 'discrete': [1 if red else 3]}
                state = []
            else:
                action = {'continuous': [20. if red else -20., 0., 0.], 'discrete': [1 if red else 2, 5 if red else 7]}
                state = []
            row = {'image': f'frames/{index}.png', 'instruction': 'move toward red' if red else 'move toward blue',
                   'state': state, 'actions': [action] * (1 if index % 8 == 7 else 8 if domain == 'robotics' else 4),
                   'episode_id': f'synthetic_{domain}_{index // 8}'}
            (train if index < 24 else validation).append(row)
        for name, rows in [('train', train), ('validation', validation)]:
            (directory / f'{name}.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    (root / 'README.txt').write_text('Synthetic colored-image fixtures for checking VLA data schemas and training mechanics. Not recorded robot/game/desktop demonstrations and not a real-world policy quality benchmark.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('examples/vla'))
    create(parser.parse_args().output)
