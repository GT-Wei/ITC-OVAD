"""
ITC-OVAD — open-vocabulary aerial object detection demo (inference only).

Examples
--------
# detect a custom open vocabulary
python demo.py -r weights/itc_ovad_dior_dota.pth -i path/to/image.jpg \
    --class-names "airplane,ship,storage tank,harbor,bridge"

# or use a ready-made class list (JSON: [["cls1"], ["cls2"], ...])
python demo.py -r weights/itc_ovad_dior_dota.pth -i path/to/image.jpg \
    --vocab class_text_dict/eval_detail_v1.5/DIOR_GZSD.json

If the machine has no internet access, the CLIP text/image weights are loaded
from the local Hugging Face cache; export HF_HUB_OFFLINE=1 to skip update checks.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

import src.misc.dist  # noqa: F401  (registers nothing; kept for parity)
from src.core import YAMLConfig


def load_vocabulary(args):
    """Return (names, class_text) where class_text is the [[name], ...] form the model expects."""
    if args.vocab:
        with open(args.vocab, "r") as f:
            items = json.load(f)
        names = [x[0] if isinstance(x, (list, tuple)) else x for x in items]
    elif args.class_names:
        names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    else:
        raise SystemExit("Provide either --class-names 'a,b,c' or --vocab path/to/list.json")
    # the text encoder expects one sub-list holding the whole vocabulary: [[c1, c2, ..., cN]]
    class_text = [names]
    return names, class_text


def load_weights(path):
    ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        return ckpt["model"]
    if "ema" in ckpt:
        return ckpt["ema"]["module"]
    return ckpt


_PALETTE = [
    (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0), (175, 82, 222),
    (255, 45, 85), (90, 200, 250), (255, 204, 0), (48, 176, 199), (162, 132, 94),
]


def _load_font(size):
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(here, "assets", "DejaVuSans-Bold.ttf"),
                 "DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw(image, names, labels, boxes, scores, thrh, out_path):
    W, H = image.size
    drawer = ImageDraw.Draw(image)
    line_w = max(2, round(min(W, H) / 400))
    font = _load_font(max(14, round(min(W, H) / 45)))
    keep = scores > thrh
    n = 0
    for box, lab, scr in zip(boxes[keep], labels[keep], scores[keep]):
        n += 1
        x1, y1, x2, y2 = (float(v) for v in box)
        color = _PALETTE[int(lab) % len(_PALETTE)]
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        tag = f"{names[int(lab)]} {scr:.2f}"
        l, t, r, b = drawer.textbbox((0, 0), tag, font=font)
        tw, th = r - l, b - t
        ty = y1 - th - 5 if y1 - th - 5 >= 0 else y1 + 1
        drawer.rectangle([x1, ty, x1 + tw + 6, ty + th + 5], fill=color)
        drawer.text((x1 + 3, ty + 1), tag, fill="white", font=font)
    image.save(out_path)
    print(f"[ITC-OVAD] {n} detections (thr={thrh}) -> {out_path}")


def main(args):
    names, class_text = load_vocabulary(args)

    cfg = YAMLConfig(args.config)
    cfg.model.load_state_dict(load_weights(args.resume))

    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            outputs = self.model(images, class_text=class_text)
            return self.postprocessor(outputs, orig_sizes)

    detector = Detector().to(args.device).eval()

    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    orig_sizes = torch.tensor([[w, h]], dtype=torch.float32).to(args.device)

    transform = T.Compose([T.Resize((800, 800)), T.ToTensor()])
    data = transform(image)[None].to(args.device)

    with torch.no_grad():
        labels, boxes, scores = detector(data, orig_sizes)

    draw(
        image, names,
        labels[0].cpu().numpy(), boxes[0].cpu().numpy(), scores[0].cpu().numpy(),
        args.thresh, args.output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITC-OVAD open-vocabulary detection demo")
    parser.add_argument("-c", "--config", default="configs/itc-ovad/DOTAv1.5/itc_ovad_r50.yml")
    parser.add_argument("-r", "--resume", required=True, help="path to an ITC-OVAD checkpoint")
    parser.add_argument("-i", "--image", required=True, help="input image path")
    parser.add_argument("--class-names", default=None, help="comma-separated open vocabulary")
    parser.add_argument("--vocab", default=None, help="JSON file with [[name], ...] class list")
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-t", "--thresh", type=float, default=0.4)
    parser.add_argument("-o", "--output", default="result.jpg")
    main(parser.parse_args())
