import argparse
import html
import io
import os
import shutil
import uuid
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import quote

import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template_string, request, send_file, url_for
from PIL import Image
from werkzeug.utils import secure_filename

from dataset.patternnet_dataset import PatternNetDataset, build_train_query_db_splits
from models.hash_model import HashModel
from utils.retrieval import hamming_distance, sign_hash


DEFAULT_ROOT = "data/CLRS"
DEFAULT_WEIGHTS = "model_cb1_CLRS_if0.05_bits32.pth"
DEFAULT_SPLIT = "split_CLRS.json"
DEFAULT_IMB_FACTOR = 0.05
DEFAULT_QUERY_RATIO = 0.2
DEFAULT_HASH_BITS = 32


def parse_args():
    parser = argparse.ArgumentParser(description="CLRS remote sensing image retrieval app")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--split_path", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--imb_factor", type=float, default=DEFAULT_IMB_FACTOR)
    parser.add_argument("--query_ratio", type=float, default=DEFAULT_QUERY_RATIO)
    parser.add_argument("--hash_bits", type=int, default=DEFAULT_HASH_BITS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--upload_dir", type=str, default=".tmp_uploads")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_checkpoint(weights_path: str):
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"], checkpoint.get("num_classes"), checkpoint.get("hash_bits", DEFAULT_HASH_BITS)
    return checkpoint, None, DEFAULT_HASH_BITS


@dataclass
class RetrievalIndex:
    root: str
    model: torch.nn.Module
    device: torch.device
    transform: object
    db_codes: np.ndarray
    db_labels: np.ndarray
    db_paths: List[str]
    idx_to_class: dict

    @classmethod
    def build(
        cls,
        root: str,
        weights_path: str,
        split_path: str,
        imb_factor: float,
        query_ratio: float,
        device_name: str,
        hash_bits: int,
    ):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        device = choose_device(device_name)
        transform = build_transform()

        train_items, _, db_items, class_to_idx = build_train_query_db_splits(
            root=root,
            long_tail=imb_factor < 1.0,
            imb_factor=imb_factor,
            query_ratio=query_ratio,
            split_path=split_path,
            seed=42,
        )
        del train_items

        state_dict, num_classes_from_ckpt, hash_bits_from_ckpt = load_checkpoint(weights_path)
        num_classes = num_classes_from_ckpt if num_classes_from_ckpt is not None else len(class_to_idx)

        model = HashModel(hash_bits=hash_bits_from_ckpt, num_classes=num_classes, pretrained=False).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        db_dataset = PatternNetDataset(root=root, transform=transform, items=db_items)
        db_loader = torch.utils.data.DataLoader(
            db_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        db_codes, db_labels, db_paths = [], [], []
        with torch.no_grad():
            for batch_imgs, batch_labels in db_loader:
                batch_imgs = batch_imgs.to(device, non_blocking=True)
                codes, _ = model(batch_imgs)
                db_codes.append(codes.cpu().numpy())
                db_labels.append(batch_labels.numpy())

        db_codes_arr = np.concatenate(db_codes, axis=0)
        db_labels_arr = np.concatenate(db_labels, axis=0)
        db_paths = [path for path, _ in db_items]

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        return cls(
            root=os.path.abspath(root),
            model=model,
            device=device,
            transform=transform,
            db_codes=db_codes_arr,
            db_labels=db_labels_arr,
            db_paths=db_paths,
            idx_to_class=idx_to_class,
        )

    def encode_query(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            codes, _ = self.model(tensor)
        return codes.cpu().numpy()[0]

    def search(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        query_code = sign_hash(self.encode_query(image_path))
        db_codes = sign_hash(self.db_codes)
        distances = hamming_distance(query_code, db_codes)
        order = np.argsort(distances, kind="stable")[:top_k]

        results = []
        for idx in order:
            label = int(self.db_labels[idx])
            class_name = self.idx_to_class.get(label, str(label))
            results.append((self.db_paths[idx], float(distances[idx]), class_name))
        return results


def make_gallery_items(results: List[Tuple[str, float, str]]):
    cards = []
    for rank, (path, distance, class_name) in enumerate(results, start=1):
        cards.append(
            {
                "rank": rank,
                "distance": f"{distance:.1f}",
                "class_name": class_name,
                "path": path,
            }
        )
    return cards


def create_flask_app(index: RetrievalIndex, upload_dir: str, top_k: int):
    app = Flask(__name__)
    app.config["UPLOAD_DIR"] = upload_dir
    app.config["TOP_K"] = max(1, int(top_k))
    os.makedirs(upload_dir, exist_ok=True)

    template = """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>CLRS 遥感图像检索</title>
            <style>
                :root { --bg:#0f1115; --panel:#151a22; --line:rgba(255,255,255,.08); --text:#eef2f7; --muted:#a7b0bf; }
                body { margin:0; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; color:var(--text);
                    background: radial-gradient(circle at top left, rgba(125,211,252,.14), transparent 28%), linear-gradient(135deg, #0b0d11 0%, #121621 56%, #0e1017 100%); }
                .wrap { max-width: 1200px; margin: 24px auto; padding: 0 16px; }
                .hero, .panel { border: 1px solid var(--line); border-radius: 18px; background: var(--panel); box-shadow: 0 14px 40px rgba(0,0,0,.25); }
                .hero { padding: 20px 22px; }
                .hero h1 { margin: 0 0 10px; font-size: 30px; }
                .hero p { margin: 6px 0; color: var(--muted); }
                .panel { margin-top: 16px; padding: 16px; }
                .row { display:flex; gap:14px; flex-wrap:wrap; align-items:center; }
                input[type=file] { color: var(--text); }
                button { border:0; border-radius: 10px; padding: 10px 16px; font-weight:700; cursor:pointer;
                    background: linear-gradient(135deg, #7dd3fc, #f7c873); color:#0b0d11; }
                .msg { color:#ffd7a1; margin-top:10px; }
                .query img { max-width: 280px; border-radius: 12px; border:1px solid var(--line); }
                .grid { display:grid; grid-template-columns: repeat(auto-fill,minmax(210px,1fr)); gap:12px; margin-top:14px; }
                .card { border:1px solid var(--line); border-radius: 12px; padding: 10px; background: rgba(255,255,255,.02); }
                .card img { width:100%; height:145px; object-fit:cover; border-radius:10px; }
                .meta { margin-top:8px; font-size:13px; color: var(--muted); line-height:1.4; }
            </style>
        </head>
        <body>
            <div class="wrap">
                <section class="hero">
                    <h1>CLRS 遥感图像检索</h1>
                    <p>模型：model_cb1_CLRS_if0.05_bits32.pth · if=0.05 · bits=32 · Top-{{ top_k }}</p>
                    <p>上传一张图像，自动返回 CLRS 数据集中最相似的 {{ top_k }} 张图片。</p>
                </section>

                <section class="panel">
                    <form method="post" enctype="multipart/form-data" class="row">
                        <input type="file" name="query_image" accept="image/*" required />
                        <button type="submit">开始检索</button>
                    </form>
                    {% if message %}<div class="msg">{{ message }}</div>{% endif %}

                    {% if query_name %}
                    <div class="query" style="margin-top:14px;">
                        <div style="margin-bottom:8px; color:var(--muted);">查询图像：{{ query_name }}</div>
                        <img src="{{ url_for('uploaded_image', filename=query_name) }}" alt="query" />
                    </div>
                    {% endif %}

                    {% if results %}
                    <div class="grid">
                        {% for item in results %}
                        <div class="card">
                            <img src="{{ url_for('dataset_image') }}?path={{ item.path_q }}" alt="result" />
                            <div class="meta">
                                #{{ item.rank }} · {{ item.class_name }}<br/>
                                Hamming={{ item.distance }}<br/>
                                {{ item.path_show }}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                </section>
            </div>
        </body>
        </html>
        """

    def _is_under_root(target_path: str, root_path: str) -> bool:
        target_abs = os.path.normpath(os.path.abspath(target_path))
        root_abs = os.path.normpath(os.path.abspath(root_path))
        try:
            return os.path.commonpath([target_abs, root_abs]) == root_abs
        except ValueError:
            return False

    def _send_image_browser_friendly(file_path: str):
        # Browsers often fail to render TIFF directly; convert non-web-native formats to PNG.
        ext = os.path.splitext(file_path)[1].lower()
        web_native_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        if ext in web_native_exts:
            return send_file(file_path)

        img = Image.open(file_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    @app.route("/", methods=["GET", "POST"])
    def home():
        message = ""
        query_name = ""
        view_results = []

        if request.method == "POST":
            f = request.files.get("query_image")
            if f is None or f.filename.strip() == "":
                message = "请先选择一张图片。"
            else:
                safe_name = secure_filename(f.filename)
                if not safe_name:
                    safe_name = "query.jpg"
                ext = os.path.splitext(safe_name)[1]
                query_name = f"{uuid.uuid4().hex}{ext}"
                query_path = os.path.join(app.config["UPLOAD_DIR"], query_name)
                f.save(query_path)

                raw_results = index.search(query_path, top_k=app.config["TOP_K"])
                cards = make_gallery_items(raw_results)
                for item in cards:
                    path_norm = os.path.normpath(item["path"])
                    view_results.append(
                        {
                            "rank": item["rank"],
                            "distance": item["distance"],
                            "class_name": item["class_name"],
                            "path_show": html.escape(path_norm),
                            "path_q": quote(path_norm),
                        }
                    )
                message = f"已检索出 {len(view_results)} 张最相似图片。"

        return render_template_string(
            template,
            top_k=app.config["TOP_K"],
            message=message,
            query_name=query_name,
            results=view_results,
        )

    @app.route("/uploaded/<path:filename>")
    def uploaded_image(filename: str):
        file_path = os.path.normpath(os.path.join(app.config["UPLOAD_DIR"], filename))
        if not _is_under_root(file_path, app.config["UPLOAD_DIR"]):
            return "invalid path", 400
        if not os.path.isfile(file_path):
            return "file not found", 404
        return _send_image_browser_friendly(file_path)

    @app.route("/dataset-image")
    def dataset_image():
        path = request.args.get("path", "")
        path = os.path.normpath(path)
        root_norm = os.path.normpath(index.root)
        abs_path = os.path.normpath(os.path.abspath(path))
        if not _is_under_root(abs_path, root_norm):
            return "invalid path", 400
        if not os.path.isfile(abs_path):
            return "file not found", 404
        return _send_image_browser_friendly(abs_path)

    return app


def main():
    args = parse_args()
    index = RetrievalIndex.build(
        root=args.root,
        weights_path=args.weights,
        split_path=args.split_path,
        imb_factor=args.imb_factor,
        query_ratio=args.query_ratio,
        device_name=args.device,
        hash_bits=args.hash_bits,
    )

    upload_dir = os.path.abspath(args.upload_dir)
    if os.path.isdir(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)
    os.makedirs(upload_dir, exist_ok=True)

    app = create_flask_app(index=index, upload_dir=upload_dir, top_k=args.top_k)
    app.run(host=args.server_name, port=args.server_port, debug=False)


if __name__ == "__main__":
    main()