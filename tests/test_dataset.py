import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_dataset import parking_boxes


def test_el_contorno_de_pklot_pasa_a_una_caja_yolo(tmp_path):
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 50), "white").save(image_path)
    xml_path = tmp_path / "frame.xml"
    xml_path.write_text(
        """<?xml version="1.0"?>
        <parking>
          <space id="1" occupied="0">
            <contour>
              <point x="10" y="10"/>
              <point x="30" y="10"/>
              <point x="30" y="20"/>
              <point x="10" y="20"/>
            </contour>
          </space>
          <space id="2" occupied="1">
            <contour>
              <point x="40" y="10"/>
              <point x="80" y="10"/>
              <point x="80" y="30"/>
              <point x="40" y="30"/>
            </contour>
          </space>
        </parking>
        """,
        encoding="utf-8",
    )
    boxes = parking_boxes(xml_path, image_path)
    assert boxes[0][0] == 0
    assert boxes[1][0] == 1
    assert abs(boxes[0][1] - 0.20) < 1e-3
    assert abs(boxes[0][2] - 0.30) < 1e-3
