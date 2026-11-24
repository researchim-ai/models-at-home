"""Generate independent toy episodes with action-controlled square motion."""
import json
from pathlib import Path
import random
from PIL import Image, ImageDraw
root = Path(__file__).resolve().parents[1] / 'examples' / 'world-models'
root.mkdir(parents=True, exist_ok=True)
(root / 'frames').mkdir(exist_ok=True)
rng = random.Random(42)
for split, count in [('train', 16), ('validation', 4)]:
    rows = []
    for ep in range(count):
        x, y = rng.randrange(3, 22), rng.randrange(3, 22)
        frames, actions, states, rewards, dones = [], [], [[x, y]], [], []
        for t in range(9):
            name = f'frames/{split}-{ep}-{t}.png'
            im = Image.new('RGB', (32, 32), '#182338')
            ImageDraw.Draw(im).rectangle((x, y, x+6, y+6), fill='#51e3b4')
            im.save(root / name); frames.append(name)
            if t < 8:
                dx, dy = rng.choice([-2, 0, 2]), rng.choice([-2, 0, 2])
                actions.append([dx, dy])
                x, y = max(0, min(25, x+dx)), max(0, min(25, y+dy))
                states.append([x, y]); rewards.append(float(x >= 20)); dones.append(t == 7)
        rows.append(dict(episode_id=f'{split}-{ep}', frames=frames, actions=actions, states=states, rewards=rewards, dones=dones))
    (root / f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
