import base64

from astra_direct.policy import observation_content


def test_observation_content_excludes_paths_and_embeds_image(tmp_path):
    image = tmp_path / "head.png"
    image.write_bytes(b"png")
    content = observation_content({
        "task": "grasp",
        "next_call": "secret command",
        "images": [{"camera": "head", "path": str(image)}],
    })
    assert "secret command" not in content[0]["text"]
    assert str(image) not in content[0]["text"]
    assert content[-1]["image_url"] == "data:image/png;base64," + base64.b64encode(b"png").decode()

