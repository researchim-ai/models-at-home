import base64
from io import BytesIO
import pytest
pytest.importorskip('PIL')
from PIL import Image
from backend.services.world_foundation_runtime import seed_image


def test_seed_decode():
    b=BytesIO();Image.new('RGB',(32,32),'green').save(b,format='PNG')
    assert seed_image({'image_base64':base64.b64encode(b.getvalue()).decode()}).size==(32,32)
    with pytest.raises(ValueError):seed_image({'image_base64':'invalid'})
