import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict


class ImageOverlapMerger:
    """画像の重複領域を検出して自動結合クラス（高速KMP + Auto + 低信頼/曖昧警告）"""

    def __init__(self, seed: int = 12345):
        self.seed = seed
        self._weight_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def pil_to_numpy(self, pil_image: Image.Image) -> np.ndarray:
        return np.array(pil_image)

    def numpy_to_pil(self, numpy_array: np.ndarray) -> Image.Image:
        return Image.fromarray(numpy_array.astype(np.uint8))

    # -------------------------
    # Fast overlap detection
    # -------------------------
    def _get_weights(self, ncols: int) -> np.ndarray:
        """row fingerprint用の重み（uint64）をキャッシュ"""
        key = (ncols, self.seed)
        w = self._weight_cache.get(key)
        if w is None:
            rng = np.random.default_rng(self.seed)
            w = rng.integers(1, np.iinfo(np.uint64).max, size=ncols, dtype=np.uint64)
            self._weight_cache[key] = w
        return w

    def _row_fingerprint_u64(self, img: np.ndarray) -> np.ndarray:
        """
        画像の各「行」を uint64 で指紋化
        - 衝突ゼロ保証はないので、最後に候補kをピクセル完全一致で検証する
        """
        img = np.ascontiguousarray(img)
        h = img.shape[0]
        flat = img.reshape(h, -1).astype(np.uint64)  # (h, w*ch)

        w = self._get_weights(flat.shape[1])
        return (flat * w).sum(axis=1, dtype=np.uint64)

    def _prefix_function(self, arr: np.ndarray) -> np.ndarray:
        """KMPのprefix function"""
        pi = np.zeros(len(arr), dtype=np.int32)
        j = 0
        for i in range(1, len(arr)):
            while j > 0 and arr[i] != arr[j]:
                j = pi[j - 1]
            if arr[i] == arr[j]:
                j += 1
            pi[i] = j
        return pi

    def _longest_prefix_of_pattern_that_is_suffix_of_text(
        self, pattern: np.ndarray, text: np.ndarray
    ) -> int:
        """patternのprefixがtextのsuffixに一致する最大長"""
        sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
        if np.any(pattern == sentinel) or np.any(text == sentinel):
            sentinel = np.uint64(0xFFFFFFFFFFFFFFFE)

        concat = np.concatenate([pattern, np.array([sentinel], dtype=np.uint64), text])
        pi = self._prefix_function(concat)
        return int(pi[-1])

    def find_best_overlap_vertical_fast(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        *,
        min_overlap_px: int = 1,
    ) -> Tuple[Optional[int], int, str, str]:
        """
        縦方向の最適重複（両方向）を高速に検出（制限なし）
        - normal: A bottom + B top
        - reverse: A top + B bottom
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]

        if w_a != w_b:
            return None, 0, f"Width mismatch (A: {w_a}px, B: {w_b}px)", "none"

        a_rows = self._row_fingerprint_u64(img_a)
        b_rows = self._row_fingerprint_u64(img_b)

        # normal: B prefix == A suffix の最大k
        k1 = self._longest_prefix_of_pattern_that_is_suffix_of_text(b_rows, a_rows)
        # reverse: A prefix == B suffix の最大k
        k2 = self._longest_prefix_of_pattern_that_is_suffix_of_text(a_rows, b_rows)

        best_k = 0
        best_dir = "none"

        # 候補は“最大k”のみ検証（PNG完全一致前提）
        if k1 >= min_overlap_px and np.array_equal(img_a[-k1:, :, :], img_b[:k1, :, :]):
            best_k = k1
            best_dir = "normal"

        if k2 >= min_overlap_px and np.array_equal(img_a[:k2, :, :], img_b[-k2:, :, :]):
            if k2 > best_k:
                best_k = k2
                best_dir = "reverse"

        if best_k > 0:
            direction_desc = (
                "A bottom + B top" if best_dir == "normal" else "A top + B bottom"
            )
            msg = f"Vertical merge ({direction_desc}): {best_k}px overlap (match rate: 100.0%)"
            return best_k, best_k, msg, best_dir

        return None, 0, "No matching region found for vertical merge", "none"

    def find_best_overlap_horizontal_fast(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        *,
        min_overlap_px: int = 1,
    ) -> Tuple[Optional[int], int, str, str]:
        """
        横方向の最適重複（両方向）を高速に検出（転置して縦検出を流用）
        - normal: A right + B left
        - reverse: A left + B right
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]

        if h_a != h_b:
            return None, 0, f"Height mismatch (A: {h_a}px, B: {h_b}px)", "none"

        a_t = np.transpose(img_a, (1, 0, 2))  # (w, h, ch)
        b_t = np.transpose(img_b, (1, 0, 2))

        overlap, _, _, direction = self.find_best_overlap_vertical_fast(
            a_t, b_t, min_overlap_px=min_overlap_px
        )
        if overlap is None:
            return None, 0, "No matching region found for horizontal merge", "none"

        direction_desc = (
            "A right + B left" if direction == "normal" else "A left + B right"
        )
        msg = f"Horizontal merge ({direction_desc}): {overlap}px overlap (match rate: 100.0%)"
        return overlap, overlap, msg, direction

    # -------------------------
    # Merge / Mask
    # -------------------------
    def merge_vertical(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        overlap: int,
        priority: str = "B",
        direction: str = "normal",
    ) -> np.ndarray:
        if direction == "normal":
            if priority == "A":
                return np.vstack([img_a, img_b[overlap:]])
            else:
                return np.vstack([img_a[:-overlap], img_b])
        else:
            if priority == "A":
                return np.vstack([img_b[:-overlap], img_a])
            else:
                return np.vstack([img_b, img_a[overlap:]])

    def merge_horizontal(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        overlap: int,
        priority: str = "B",
        direction: str = "normal",
    ) -> np.ndarray:
        if direction == "normal":
            if priority == "A":
                return np.hstack([img_a, img_b[:, overlap:]])
            else:
                return np.hstack([img_a[:, :-overlap], img_b])
        else:
            if priority == "A":
                return np.hstack([img_b[:, :-overlap], img_a])
            else:
                return np.hstack([img_b, img_a[:, overlap:]])

    def create_overlap_mask(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        merge_type: str,
        overlap: int,
        direction: str,
    ) -> Tuple[Image.Image, Image.Image]:
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]

        if merge_type == "vertical":
            if direction == "normal":
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[h_a - overlap : h_a, :] = 255
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[:overlap, :] = 255
            else:
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[:overlap, :] = 255
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[h_b - overlap : h_b, :] = 255

        elif merge_type == "horizontal":
            if direction == "normal":
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[:, w_a - overlap : w_a] = 255
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[:, :overlap] = 255
            else:
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[:, :overlap] = 255
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[:, w_b - overlap : w_b] = 255

        return Image.fromarray(mask_a), Image.fromarray(mask_b)

    # -------------------------
    # Public API
    # -------------------------
    def merge_images(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        merge_type: str,
        priority: str = "B",
        *,
        min_overlap_px: int = 1,  # 鎖状マージ想定なら 1〜8 でもOK
        warn_overlap_ratio: float = 0.10,  # 10%未満なら「低信頼」警告
        ambiguous_margin_px: int = 16,  # 縦横差が近いと「曖昧」警告
    ) -> Tuple[
        Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]
    ]:
        np_a = self.pil_to_numpy(img_a)
        np_b = self.pil_to_numpy(img_b)

        h_a, w_a = np_a.shape[:2]
        h_b, w_b = np_b.shape[:2]

        size_info = f"Image A: {h_a}×{w_a}px, Image B: {h_b}×{w_b}px (Priority: Image {priority})\n"

        # Auto: 縦横を両方試す
        if merge_type == "auto":
            v_overlap, _, v_msg, v_dir = self.find_best_overlap_vertical_fast(
                np_a, np_b, min_overlap_px=min_overlap_px
            )
            h_overlap, _, h_msg, h_dir = self.find_best_overlap_horizontal_fast(
                np_a, np_b, min_overlap_px=min_overlap_px
            )

            v = int(v_overlap or 0)
            h = int(h_overlap or 0)

            if v == 0 and h == 0:
                return (
                    None,
                    f"{size_info}❌ No matching overlap found (min_overlap_px={min_overlap_px})",
                    None,
                    None,
                )

            chosen = "vertical" if v >= h else "horizontal"

            # 警告メッセージだけ出す（プレビューはしない）
            warns = []
            if v > 0 and h > 0 and abs(v - h) <= ambiguous_margin_px:
                warns.append(
                    f"⚠️ Ambiguous: vertical={v}px, horizontal={h}px (±{ambiguous_margin_px}px)."
                )

            if chosen == "vertical":
                ratio = v / max(h_a, 1)
                if ratio < warn_overlap_ratio:
                    warns.append(
                        f"⚠️ Low confidence: vertical overlap is small ({v}px, {ratio * 100:.1f}%)."
                    )
            else:
                ratio = h / max(w_a, 1)
                if ratio < warn_overlap_ratio:
                    warns.append(
                        f"⚠️ Low confidence: horizontal overlap is small ({h}px, {ratio * 100:.1f}%)."
                    )

            size_info += (
                f"Auto decision: vertical={v}px / horizontal={h}px → {chosen}\n"
            )
            if warns:
                size_info += "\n".join(warns) + "\n"

            merge_type = chosen

        # Vertical
        if merge_type == "vertical":
            overlap, _, message, direction = self.find_best_overlap_vertical_fast(
                np_a, np_b, min_overlap_px=min_overlap_px
            )
            if overlap is None:
                return None, f"{size_info}❌ {message}", None, None

            mask_a, mask_b = self.create_overlap_mask(
                np_a, np_b, "vertical", overlap, direction
            )
            merged = self.merge_vertical(np_a, np_b, overlap, priority, direction)
            final_image = self.numpy_to_pil(merged)
            final_h, final_w = merged.shape[:2]
            result_message = (
                f"{size_info}✅ {message}\nMerged size: {final_h}×{final_w}px"
            )
            return final_image, result_message, mask_a, mask_b

        # Horizontal
        if merge_type == "horizontal":
            overlap, _, message, direction = self.find_best_overlap_horizontal_fast(
                np_a, np_b, min_overlap_px=min_overlap_px
            )
            if overlap is None:
                return None, f"{size_info}❌ {message}", None, None

            mask_a, mask_b = self.create_overlap_mask(
                np_a, np_b, "horizontal", overlap, direction
            )
            merged = self.merge_horizontal(np_a, np_b, overlap, priority, direction)
            final_image = self.numpy_to_pil(merged)
            final_h, final_w = merged.shape[:2]
            result_message = (
                f"{size_info}✅ {message}\nMerged size: {final_h}×{final_w}px"
            )
            return final_image, result_message, mask_a, mask_b

        return (
            None,
            f"❌ Invalid merge type: {merge_type} (Use 'auto'/'vertical'/'horizontal')",
            None,
            None,
        )


def process_two_images(
    img_a, img_b, merge_type, priority
) -> Tuple[Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]]:
    if not img_a or not img_b:
        return (
            None,
            "❌ Please upload both Image A (base) and Image B (overlay)",
            None,
            None,
        )

    try:
        if img_a.mode != "RGB":
            img_a = img_a.convert("RGB")
        if img_b.mode != "RGB":
            img_b = img_b.convert("RGB")

        merger = ImageOverlapMerger(seed=12345)

        # 鎖状マージも想定 → min_overlap_pxは小さめでOK（誤判定が怖ければ 8〜32）
        result_image, status_message, mask_a, mask_b = merger.merge_images(
            img_a,
            img_b,
            merge_type,
            priority,
            min_overlap_px=1,  # 1〜8 推奨。安全寄りなら 8/16 に上げる
            warn_overlap_ratio=0.10,  # 10%未満で「低信頼」表示
            ambiguous_margin_px=16,  # 縦横が近いと「曖昧」表示
        )

        return result_image, status_message, mask_a, mask_b

    except Exception as e:
        return None, f"❌ Processing error: {str(e)}", None, None


def create_gradio_interface():
    with gr.Blocks(title="Image Overlap Merger") as demo:
        gr.Markdown("<h2 style='text-align:center;'>Image Overlap Merger</h2>")
        gr.Markdown(
            "Auto-detect and merge overlapping regions of two images. "
            "Masks show the detected overlap regions. "
            "If overlap is small or ambiguous, a warning is shown."
        )

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    img_a = gr.Image(type="pil", label="Image A")
                    img_b = gr.Image(type="pil", label="Image B")

                merge_type = gr.Radio(
                    choices=[
                        ("Auto", "auto"),
                        ("Vertical", "vertical"),
                        ("Horizontal", "horizontal"),
                    ],
                    value="auto",
                    label="Merge Direction",
                )

                overlap_priority = gr.Radio(
                    choices=[("Image A", "A"), ("Image B", "B")],
                    value="B",
                    label="Overlap Priority",
                )

                with gr.Row():
                    cancel_btn = gr.Button("Cancel", variant="secondary")
                    run_btn = gr.Button("Submit", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image", format="png")
                with gr.Row():
                    mask_a = gr.Image(
                        type="pil", label="Image A Overlap Mask", format="png"
                    )
                    mask_b = gr.Image(
                        type="pil", label="Image B Overlap Mask", format="png"
                    )
                message = gr.Textbox(label="Message", lines=8)

        run_btn.click(
            fn=process_two_images,
            inputs=[img_a, img_b, merge_type, overlap_priority],
            outputs=[output_image, message, mask_a, mask_b],
        )

        cancel_btn.click(
            fn=lambda: (None, None, "auto", "B"),
            inputs=[],
            outputs=[img_a, img_b, merge_type, overlap_priority],
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
