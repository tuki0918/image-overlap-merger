import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict


class ImageOverlapMerger:
    """
    4つの検出モードを提供します。

    [Edge系] 端と端（suffix/prefix）だけを重ねる
      - edge_auto      : 基本FAST、重複が小さい時だけSLOW_EXACTにフォールバック
      - edge_fast      : 行/列を指紋化 + KMPで最長suffix/prefix一致をO(N)で取得 → 最後はnp.array_equalで確定
      - edge_slow_exact: 最大kから降順でnp.array_equalを総当たり（遅いが確実）

    [ShiftMatch] Aの途中にBを「ずらして重ねる」(ABCD + BCCEF -> ABCCEF のような挙動)
      - shift_match    : 1Dシフトを総当たりし、重なり領域で一致ピクセルが最大のシフトを採用（重なりはpriorityで上書き）
    """

    def __init__(self, seed: int = 12345):
        self.seed = seed
        self._weight_cache: Dict[Tuple[int, int], np.ndarray] = {}

    # -------------------------
    # Utils
    # -------------------------
    def pil_to_numpy(self, pil_image: Image.Image) -> np.ndarray:
        return np.array(pil_image)

    def numpy_to_pil(self, numpy_array: np.ndarray) -> Image.Image:
        return Image.fromarray(numpy_array.astype(np.uint8))

    # -------------------------
    # FAST (Edge): fingerprints + KMP
    # -------------------------
    def _get_weights(self, ncols: int) -> np.ndarray:
        key = (ncols, self.seed)
        w = self._weight_cache.get(key)
        if w is None:
            rng = np.random.default_rng(self.seed)
            w = rng.integers(1, np.iinfo(np.uint64).max, size=ncols, dtype=np.uint64)
            self._weight_cache[key] = w
        return w

    def _row_fingerprint_u64(self, img: np.ndarray) -> np.ndarray:
        img = np.ascontiguousarray(img)
        h = img.shape[0]
        flat = img.reshape(h, -1).astype(np.uint64)
        w = self._get_weights(flat.shape[1])
        return (flat * w).sum(axis=1, dtype=np.uint64)

    def _prefix_function(self, arr: np.ndarray) -> np.ndarray:
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
        sentinel = np.uint64(0xFFFFFFFFFFFFFFFF)
        if np.any(pattern == sentinel) or np.any(text == sentinel):
            sentinel = np.uint64(0xFFFFFFFFFFFFFFFE)

        concat = np.concatenate([pattern, np.array([sentinel], dtype=np.uint64), text])
        pi = self._prefix_function(concat)
        return int(pi[-1])

    def find_best_overlap_vertical_edge_fast(
        self, img_a: np.ndarray, img_b: np.ndarray, *, min_overlap_px: int = 1
    ) -> Tuple[Optional[int], str, str]:
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        if w_a != w_b:
            return None, "none", f"Width mismatch (A:{w_a}px, B:{w_b}px)"

        a_rows = self._row_fingerprint_u64(img_a)
        b_rows = self._row_fingerprint_u64(img_b)

        k1 = self._longest_prefix_of_pattern_that_is_suffix_of_text(
            b_rows, a_rows
        )  # normal
        k2 = self._longest_prefix_of_pattern_that_is_suffix_of_text(
            a_rows, b_rows
        )  # reverse

        best_k = 0
        best_dir = "none"

        if k1 >= min_overlap_px and np.array_equal(img_a[-k1:, :, :], img_b[:k1, :, :]):
            best_k = k1
            best_dir = "normal"

        if k2 >= min_overlap_px and np.array_equal(img_a[:k2, :, :], img_b[-k2:, :, :]):
            if k2 > best_k:
                best_k = k2
                best_dir = "reverse"

        if best_k > 0:
            desc = "A bottom + B top" if best_dir == "normal" else "A top + B bottom"
            msg = f"Vertical edge ({desc}): {best_k}px overlap (match rate: 100.0%) [EDGE_FAST]"
            return best_k, best_dir, msg

        return None, "none", "No matching region found [EDGE_FAST]"

    def find_best_overlap_horizontal_edge_fast(
        self, img_a: np.ndarray, img_b: np.ndarray, *, min_overlap_px: int = 1
    ) -> Tuple[Optional[int], str, str]:
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        if h_a != h_b:
            return None, "none", f"Height mismatch (A:{h_a}px, B:{h_b}px)"

        a_t = np.transpose(img_a, (1, 0, 2))
        b_t = np.transpose(img_b, (1, 0, 2))

        overlap, direction, _ = self.find_best_overlap_vertical_edge_fast(
            a_t, b_t, min_overlap_px=min_overlap_px
        )
        if overlap is None:
            return None, "none", "No matching region found [EDGE_FAST]"

        desc = "A right + B left" if direction == "normal" else "A left + B right"
        msg = f"Horizontal edge ({desc}): {overlap}px overlap (match rate: 100.0%) [EDGE_FAST]"
        return overlap, direction, msg

    # -------------------------
    # SLOW_EXACT (Edge): brute force descending
    # -------------------------
    def find_best_overlap_vertical_edge_slow_exact(
        self, img_a: np.ndarray, img_b: np.ndarray, *, min_overlap_px: int = 1
    ) -> Tuple[Optional[int], str, str]:
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        if w_a != w_b:
            return None, "none", f"Width mismatch (A:{w_a}px, B:{w_b}px)"

        max_k = min(h_a, h_b)

        for k in range(max_k, min_overlap_px - 1, -1):
            if np.array_equal(img_a[-k:, :, :], img_b[:k, :, :]):
                msg = f"Vertical edge (A bottom + B top): {k}px overlap (match rate: 100.0%) [EDGE_SLOW_EXACT]"
                return k, "normal", msg

        for k in range(max_k, min_overlap_px - 1, -1):
            if np.array_equal(img_a[:k, :, :], img_b[-k:, :, :]):
                msg = f"Vertical edge (A top + B bottom): {k}px overlap (match rate: 100.0%) [EDGE_SLOW_EXACT]"
                return k, "reverse", msg

        return None, "none", "No matching region found [EDGE_SLOW_EXACT]"

    def find_best_overlap_horizontal_edge_slow_exact(
        self, img_a: np.ndarray, img_b: np.ndarray, *, min_overlap_px: int = 1
    ) -> Tuple[Optional[int], str, str]:
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        if h_a != h_b:
            return None, "none", f"Height mismatch (A:{h_a}px, B:{h_b}px)"

        max_k = min(w_a, w_b)

        for k in range(max_k, min_overlap_px - 1, -1):
            if np.array_equal(img_a[:, -k:, :], img_b[:, :k, :]):
                msg = f"Horizontal edge (A right + B left): {k}px overlap (match rate: 100.0%) [EDGE_SLOW_EXACT]"
                return k, "normal", msg

        for k in range(max_k, min_overlap_px - 1, -1):
            if np.array_equal(img_a[:, :k, :], img_b[:, -k:, :]):
                msg = f"Horizontal edge (A left + B right): {k}px overlap (match rate: 100.0%) [EDGE_SLOW_EXACT]"
                return k, "reverse", msg

        return None, "none", "No matching region found [EDGE_SLOW_EXACT]"

    # -------------------------
    # ShiftMatch: best shift by match pixels (supports internal overlap like ABCD + BCCEF)
    # -------------------------
    def best_shift_horizontal_by_matches(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        *,
        min_overlap_px: int = 1,
        search_ratio: float = 1.0,
    ) -> Tuple[Optional[int], int, float, int]:
        """
        s: BをA基準で右(+)/左(-)にどれだけずらすか
        戻り: (best_s, overlap_w, match_ratio, match_pixels)
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        if h_a != h_b:
            return None, 0, 0.0, 0

        # 探索範囲（フル総当たりが嫌なら ratio で絞れる）
        s_min_full = -(w_b - min_overlap_px)
        s_max_full = w_a - min_overlap_px

        span = s_max_full - s_min_full
        if span <= 0:
            return None, 0, 0.0, 0

        if search_ratio < 1.0:
            half = int((span * search_ratio) / 2)
            center = (s_min_full + s_max_full) // 2
            s_min = max(s_min_full, center - half)
            s_max = min(s_max_full, center + half)
        else:
            s_min, s_max = s_min_full, s_max_full

        best_key = None
        best = (None, 0, 0.0, 0)  # s, overlap_w, ratio, match_pixels

        for s in range(s_min, s_max + 1):
            x0 = max(0, s)
            x1 = min(w_a, s + w_b)
            overlap_w = x1 - x0
            if overlap_w < min_overlap_px:
                continue

            a_slice = img_a[:, x0:x1, :]
            b_slice = img_b[:, (x0 - s) : (x1 - s), :]

            match_pixels = int(np.sum(np.all(a_slice == b_slice, axis=2)))
            ratio = match_pixels / (overlap_w * h_a)

            # 優先順位：一致ピクセル最大 → 重なり幅大 → 一致率高 → |シフト|小（安定化）
            key = (match_pixels, overlap_w, ratio, -abs(s))
            if best_key is None or key > best_key:
                best_key = key
                best = (s, overlap_w, ratio, match_pixels)

        return best

    def best_shift_vertical_by_matches(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        *,
        min_overlap_px: int = 1,
        search_ratio: float = 1.0,
    ) -> Tuple[Optional[int], int, float, int]:
        # transposeして横問題にする
        a_t = np.transpose(img_a, (1, 0, 2))
        b_t = np.transpose(img_b, (1, 0, 2))
        return self.best_shift_horizontal_by_matches(
            a_t, b_t, min_overlap_px=min_overlap_px, search_ratio=search_ratio
        )

    def merge_with_shift_horizontal(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        s: int,
        *,
        priority: str = "B",
    ) -> np.ndarray:
        h, w_a = img_a.shape[:2]
        w_b = img_b.shape[1]

        left = min(0, s)
        right = max(w_a, s + w_b)
        out_w = right - left

        out = np.zeros((h, out_w, 3), dtype=np.uint8)

        a_x = -left
        b_x = s - left

        out[:, a_x : a_x + w_a, :] = img_a

        if priority == "B":
            out[:, b_x : b_x + w_b, :] = img_b
        else:
            # A優先：A領域外だけBを貼る
            if b_x < a_x:
                out[:, b_x:a_x, :] = img_b[:, : a_x - b_x, :]
            a_right = a_x + w_a
            b_right = b_x + w_b
            if b_right > a_right:
                out[:, a_right:b_right, :] = img_b[:, (a_right - b_x) :, :]

        return out

    def merge_with_shift_vertical(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        s: int,
        *,
        priority: str = "B",
    ) -> np.ndarray:
        a_t = np.transpose(img_a, (1, 0, 2))
        b_t = np.transpose(img_b, (1, 0, 2))
        out_t = self.merge_with_shift_horizontal(a_t, b_t, s, priority=priority)
        return np.transpose(out_t, (1, 0, 2))

    def create_shift_overlap_mask_horizontal(
        self, img_a: np.ndarray, img_b: np.ndarray, s: int
    ) -> Tuple[Image.Image, Image.Image, int]:
        """
        ShiftMatch用：重なり領域（矩形）をA/B各画像サイズ内で白にしたマスクを返す
        戻り: (mask_a, mask_b, overlap_w)
        """
        h, w_a = img_a.shape[:2]
        w_b = img_b.shape[1]

        x0 = max(0, s)
        x1 = min(w_a, s + w_b)
        overlap_w = x1 - x0
        if overlap_w <= 0:
            return (
                Image.fromarray(np.zeros((h, w_a), np.uint8)),
                Image.fromarray(np.zeros((h, w_b), np.uint8)),
                0,
            )

        # A上の重なりは [x0, x1)
        mask_a = np.zeros((h, w_a), dtype=np.uint8)
        mask_a[:, x0:x1] = 255

        # B上の重なりは [x0 - s, x1 - s)
        bx0 = x0 - s
        bx1 = x1 - s
        mask_b = np.zeros((h, w_b), dtype=np.uint8)
        mask_b[:, bx0:bx1] = 255

        return Image.fromarray(mask_a), Image.fromarray(mask_b), overlap_w

    def create_shift_overlap_mask_vertical(
        self, img_a: np.ndarray, img_b: np.ndarray, s: int
    ) -> Tuple[Image.Image, Image.Image, int]:
        # transposeして横マスクを流用
        a_t = np.transpose(img_a, (1, 0, 2))
        b_t = np.transpose(img_b, (1, 0, 2))
        mask_a_t, mask_b_t, overlap_h = self.create_shift_overlap_mask_horizontal(
            a_t, b_t, s
        )
        # maskもtransposeして戻す
        mask_a = np.transpose(np.array(mask_a_t), (1, 0))
        mask_b = np.transpose(np.array(mask_b_t), (1, 0))
        return Image.fromarray(mask_a), Image.fromarray(mask_b), overlap_h

    # -------------------------
    # Edge Mask (existing style)
    # -------------------------
    def create_edge_overlap_mask(
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

        else:  # horizontal
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
    # Edge Merge (existing)
    # -------------------------
    def merge_vertical_edge(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        overlap: int,
        priority: str,
        direction: str,
    ) -> np.ndarray:
        if direction == "normal":
            return (
                np.vstack([img_a, img_b[overlap:]])
                if priority == "A"
                else np.vstack([img_a[:-overlap], img_b])
            )
        else:
            return (
                np.vstack([img_b[:-overlap], img_a])
                if priority == "A"
                else np.vstack([img_b, img_a[overlap:]])
            )

    def merge_horizontal_edge(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        overlap: int,
        priority: str,
        direction: str,
    ) -> np.ndarray:
        if direction == "normal":
            return (
                np.hstack([img_a, img_b[:, overlap:]])
                if priority == "A"
                else np.hstack([img_a[:, :-overlap], img_b])
            )
        else:
            return (
                np.hstack([img_b[:, :-overlap], img_a])
                if priority == "A"
                else np.hstack([img_b, img_a[:, overlap:]])
            )

    # -------------------------
    # Public: merge_images
    # -------------------------
    def merge_images(
        self,
        img_a: Image.Image,
        img_b: Image.Image,
        merge_type: str,
        priority: str,
        *,
        detection_mode: str = "edge_auto",  # edge_auto / edge_fast / edge_slow_exact / shift_match
        min_overlap_px: int = 1,
        warn_overlap_ratio: float = 0.10,
        ambiguous_margin_px: int = 16,
        slow_fallback_ratio: float = 0.10,  # edge_auto時：重複が小さいとSLOW_EXACTで再探索
        shift_search_ratio: float = 1.0,  # shift_match時：探索範囲(1.0でフル)
    ) -> Tuple[
        Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]
    ]:
        np_a = self.pil_to_numpy(img_a)
        np_b = self.pil_to_numpy(img_b)

        h_a, w_a = np_a.shape[:2]
        h_b, w_b = np_b.shape[:2]

        header = (
            f"Image A: {h_a}×{w_a}px, Image B: {h_b}×{w_b}px (Priority: Image {priority})\n"
            f"Merge type: {merge_type}, Detection mode: {detection_mode}\n"
        )

        # -------------------------
        # ShiftMatch branch
        # -------------------------
        if detection_mode == "shift_match":

            def decide_axis_by_shift() -> Tuple[str, dict]:
                # vertical
                sv, ov, rv, mv = self.best_shift_vertical_by_matches(
                    np_a,
                    np_b,
                    min_overlap_px=min_overlap_px,
                    search_ratio=shift_search_ratio,
                )
                # horizontal
                sh, oh, rh, mh = self.best_shift_horizontal_by_matches(
                    np_a,
                    np_b,
                    min_overlap_px=min_overlap_px,
                    search_ratio=shift_search_ratio,
                )

                # スコア比較：一致ピクセル最大 → 重なり幅大 → 一致率高 → |シフト|小
                key_v = (mv, ov, rv, -abs(sv) if sv is not None else -(10**9))
                key_h = (mh, oh, rh, -abs(sh) if sh is not None else -(10**9))

                if sv is None and sh is None:
                    return "none", {
                        "msg": "No shift found (check min_overlap_px / sizes)."
                    }

                chosen = "vertical" if key_v >= key_h else "horizontal"

                # 曖昧警告（match_pixelsが近い）
                warns = []
                if (
                    sv is not None
                    and sh is not None
                    and abs(mv - mh) <= (ambiguous_margin_px * max(h_a, w_a))
                ):
                    warns.append(
                        f"⚠️ Ambiguous (ShiftMatch): match_pixels vertical={mv}, horizontal={mh} (picked {chosen})."
                    )

                # 低信頼警告（重なり比率が小さい）
                if chosen == "vertical" and sv is not None:
                    overlap_ratio = ov / max(h_a, 1)  # 縦は「重なり高さ」/H
                    if overlap_ratio < warn_overlap_ratio:
                        warns.append(
                            f"⚠️ Low overlap (ShiftMatch vertical): overlap={ov}px ({overlap_ratio * 100:.1f}%), match_ratio={rv * 100:.1f}%."
                        )
                if chosen == "horizontal" and sh is not None:
                    overlap_ratio = oh / max(w_a, 1)
                    if overlap_ratio < warn_overlap_ratio:
                        warns.append(
                            f"⚠️ Low overlap (ShiftMatch horizontal): overlap={oh}px ({overlap_ratio * 100:.1f}%), match_ratio={rh * 100:.1f}%."
                        )

                info = {
                    "sv": sv,
                    "ov": ov,
                    "rv": rv,
                    "mv": mv,
                    "sh": sh,
                    "oh": oh,
                    "rh": rh,
                    "mh": mh,
                    "warns": warns,
                    "chosen": chosen,
                }
                return chosen, info

            # merge_type が auto のときは縦横を決める
            if merge_type == "auto":
                chosen, info = decide_axis_by_shift()
                if chosen == "none":
                    return None, header + "❌ " + info["msg"], None, None

                msg = header
                msg += (
                    f"ShiftMatch candidates:\n"
                    f"  vertical: shift={info['sv']}, overlap={info['ov']}px, match_ratio={info['rv'] * 100:.1f}%, match_pixels={info['mv']}\n"
                    f"  horizontal: shift={info['sh']}, overlap={info['oh']}px, match_ratio={info['rh'] * 100:.1f}%, match_pixels={info['mh']}\n"
                    f"Auto decision (ShiftMatch): → {chosen}\n"
                )
                if info["warns"]:
                    msg += "\n".join(info["warns"]) + "\n"

                if chosen == "vertical":
                    s = info["sv"]
                    if s is None:
                        return None, msg + "❌ Vertical ShiftMatch failed.", None, None
                    mask_a, mask_b, ov = self.create_shift_overlap_mask_vertical(
                        np_a, np_b, s
                    )
                    merged = self.merge_with_shift_vertical(
                        np_a, np_b, s, priority=priority
                    )
                    final = self.numpy_to_pil(merged)
                    fh, fw = merged.shape[:2]
                    msg += f"✅ ShiftMatch vertical: shift={s}, overlap={ov}px\nMerged size: {fh}×{fw}px"
                    return final, msg, mask_a, mask_b

                else:
                    s = info["sh"]
                    if s is None:
                        return (
                            None,
                            msg + "❌ Horizontal ShiftMatch failed.",
                            None,
                            None,
                        )
                    mask_a, mask_b, ow = self.create_shift_overlap_mask_horizontal(
                        np_a, np_b, s
                    )
                    merged = self.merge_with_shift_horizontal(
                        np_a, np_b, s, priority=priority
                    )
                    final = self.numpy_to_pil(merged)
                    fh, fw = merged.shape[:2]
                    msg += f"✅ ShiftMatch horizontal: shift={s}, overlap={ow}px\nMerged size: {fh}×{fw}px"
                    return final, msg, mask_a, mask_b

            # merge_type が vertical/horizontal 指定のとき
            if merge_type == "vertical":
                s, overlap_h, ratio, match_pixels = self.best_shift_vertical_by_matches(
                    np_a,
                    np_b,
                    min_overlap_px=min_overlap_px,
                    search_ratio=shift_search_ratio,
                )
                if s is None:
                    return (
                        None,
                        header + "❌ No vertical shift found (ShiftMatch).",
                        None,
                        None,
                    )

                warn = ""
                overlap_ratio = overlap_h / max(h_a, 1)
                if overlap_ratio < warn_overlap_ratio:
                    warn = f"⚠️ Low overlap (ShiftMatch vertical): overlap={overlap_h}px ({overlap_ratio * 100:.1f}%), match_ratio={ratio * 100:.1f}%.\n"

                mask_a, mask_b, _ = self.create_shift_overlap_mask_vertical(
                    np_a, np_b, s
                )
                merged = self.merge_with_shift_vertical(
                    np_a, np_b, s, priority=priority
                )
                final = self.numpy_to_pil(merged)
                fh, fw = merged.shape[:2]
                msg = (
                    header
                    + warn
                    + f"✅ ShiftMatch vertical: shift={s}, overlap={overlap_h}px, match_ratio={ratio * 100:.1f}%, match_pixels={match_pixels}\n"
                    f"Merged size: {fh}×{fw}px"
                )
                return final, msg, mask_a, mask_b

            if merge_type == "horizontal":
                s, overlap_w, ratio, match_pixels = (
                    self.best_shift_horizontal_by_matches(
                        np_a,
                        np_b,
                        min_overlap_px=min_overlap_px,
                        search_ratio=shift_search_ratio,
                    )
                )
                if s is None:
                    return (
                        None,
                        header + "❌ No horizontal shift found (ShiftMatch).",
                        None,
                        None,
                    )

                warn = ""
                overlap_ratio = overlap_w / max(w_a, 1)
                if overlap_ratio < warn_overlap_ratio:
                    warn = f"⚠️ Low overlap (ShiftMatch horizontal): overlap={overlap_w}px ({overlap_ratio * 100:.1f}%), match_ratio={ratio * 100:.1f}%.\n"

                mask_a, mask_b, _ = self.create_shift_overlap_mask_horizontal(
                    np_a, np_b, s
                )
                merged = self.merge_with_shift_horizontal(
                    np_a, np_b, s, priority=priority
                )
                final = self.numpy_to_pil(merged)
                fh, fw = merged.shape[:2]
                msg = (
                    header
                    + warn
                    + f"✅ ShiftMatch horizontal: shift={s}, overlap={overlap_w}px, match_ratio={ratio * 100:.1f}%, match_pixels={match_pixels}\n"
                    f"Merged size: {fh}×{fw}px"
                )
                return final, msg, mask_a, mask_b

            return (
                None,
                header + "❌ Invalid merge type. Use auto/vertical/horizontal.",
                None,
                None,
            )

        # -------------------------
        # Edge branch (edge_auto / edge_fast / edge_slow_exact)
        # -------------------------
        def edge_detect_vertical(mode: str):
            if mode == "edge_slow_exact":
                return self.find_best_overlap_vertical_edge_slow_exact(
                    np_a, np_b, min_overlap_px=min_overlap_px
                )
            else:
                return self.find_best_overlap_vertical_edge_fast(
                    np_a, np_b, min_overlap_px=min_overlap_px
                )

        def edge_detect_horizontal(mode: str):
            if mode == "edge_slow_exact":
                return self.find_best_overlap_horizontal_edge_slow_exact(
                    np_a, np_b, min_overlap_px=min_overlap_px
                )
            else:
                return self.find_best_overlap_horizontal_edge_fast(
                    np_a, np_b, min_overlap_px=min_overlap_px
                )

        # merge_type=auto の場合：縦横を両方試して大きいoverlapを採用
        if merge_type == "auto":
            base_mode = (
                "edge_fast"
                if detection_mode in ("edge_auto", "edge_fast")
                else "edge_slow_exact"
            )

            v_overlap, v_dir, v_msg = edge_detect_vertical(base_mode)
            h_overlap, h_dir, h_msg = edge_detect_horizontal(base_mode)

            v = int(v_overlap or 0)
            h = int(h_overlap or 0)

            if v == 0 and h == 0:
                return (
                    None,
                    header
                    + f"❌ No edge overlap found (min_overlap_px={min_overlap_px}).",
                    None,
                    None,
                )

            chosen = "vertical" if v >= h else "horizontal"
            warns = []

            if v > 0 and h > 0 and abs(v - h) <= ambiguous_margin_px:
                warns.append(
                    f"⚠️ Ambiguous (Edge): vertical={v}px, horizontal={h}px (±{ambiguous_margin_px}px)."
                )

            if chosen == "vertical":
                ratio = v / max(h_a, 1)
                if ratio < warn_overlap_ratio:
                    warns.append(
                        f"⚠️ Low overlap (Edge vertical): {v}px ({ratio * 100:.1f}%)."
                    )
            else:
                ratio = h / max(w_a, 1)
                if ratio < warn_overlap_ratio:
                    warns.append(
                        f"⚠️ Low overlap (Edge horizontal): {h}px ({ratio * 100:.1f}%)."
                    )

            # edge_auto なら小さい重複で SLOW_EXACT 再探索
            if detection_mode == "edge_auto":
                need_fallback = False
                if (
                    chosen == "vertical"
                    and v > 0
                    and (v / max(h_a, 1)) < slow_fallback_ratio
                ):
                    need_fallback = True
                if (
                    chosen == "horizontal"
                    and h > 0
                    and (h / max(w_a, 1)) < slow_fallback_ratio
                ):
                    need_fallback = True

                if need_fallback:
                    if chosen == "vertical":
                        v2, v_dir2, v_msg2 = (
                            self.find_best_overlap_vertical_edge_slow_exact(
                                np_a, np_b, min_overlap_px=min_overlap_px
                            )
                        )
                        if v2 is not None:
                            v, v_dir, v_msg = int(v2), v_dir2, v_msg2
                            warns.append(
                                "ℹ️ Fallback: re-detected vertical using EDGE_SLOW_EXACT."
                            )
                    else:
                        h2, h_dir2, h_msg2 = (
                            self.find_best_overlap_horizontal_edge_slow_exact(
                                np_a, np_b, min_overlap_px=min_overlap_px
                            )
                        )
                        if h2 is not None:
                            h, h_dir, h_msg = int(h2), h_dir2, h_msg2
                            warns.append(
                                "ℹ️ Fallback: re-detected horizontal using EDGE_SLOW_EXACT."
                            )

            msg = header
            msg += (
                f"Auto decision (Edge): vertical={v}px / horizontal={h}px → {chosen}\n"
            )
            if warns:
                msg += "\n".join(warns) + "\n"

            if chosen == "vertical":
                overlap = v
                direction = v_dir
                if overlap <= 0 or direction == "none":
                    return None, msg + "❌ Vertical edge detection failed.", None, None
                mask_a, mask_b = self.create_edge_overlap_mask(
                    np_a, np_b, "vertical", overlap, direction
                )
                merged = self.merge_vertical_edge(
                    np_a, np_b, overlap, priority, direction
                )
                final = self.numpy_to_pil(merged)
                fh, fw = merged.shape[:2]
                msg += f"✅ {v_msg}\nMerged size: {fh}×{fw}px"
                return final, msg, mask_a, mask_b

            else:
                overlap = h
                direction = h_dir
                if overlap <= 0 or direction == "none":
                    return (
                        None,
                        msg + "❌ Horizontal edge detection failed.",
                        None,
                        None,
                    )
                mask_a, mask_b = self.create_edge_overlap_mask(
                    np_a, np_b, "horizontal", overlap, direction
                )
                merged = self.merge_horizontal_edge(
                    np_a, np_b, overlap, priority, direction
                )
                final = self.numpy_to_pil(merged)
                fh, fw = merged.shape[:2]
                msg += f"✅ {h_msg}\nMerged size: {fh}×{fw}px"
                return final, msg, mask_a, mask_b

        # merge_typeがvertical/horizontal指定
        if merge_type == "vertical":
            mode = (
                "edge_slow_exact"
                if detection_mode == "edge_slow_exact"
                else "edge_fast"
            )
            overlap, direction, det_msg = edge_detect_vertical(mode)
            if overlap is None:
                return None, header + f"❌ {det_msg}", None, None

            ratio = overlap / max(h_a, 1)
            warn = ""
            if ratio < warn_overlap_ratio:
                warn = f"⚠️ Low overlap (Edge vertical): {overlap}px ({ratio * 100:.1f}%).\n"

            mask_a, mask_b = self.create_edge_overlap_mask(
                np_a, np_b, "vertical", overlap, direction
            )
            merged = self.merge_vertical_edge(np_a, np_b, overlap, priority, direction)
            final = self.numpy_to_pil(merged)
            fh, fw = merged.shape[:2]
            msg = header + warn + f"✅ {det_msg}\nMerged size: {fh}×{fw}px"
            return final, msg, mask_a, mask_b

        if merge_type == "horizontal":
            mode = (
                "edge_slow_exact"
                if detection_mode == "edge_slow_exact"
                else "edge_fast"
            )
            overlap, direction, det_msg = edge_detect_horizontal(mode)
            if overlap is None:
                return None, header + f"❌ {det_msg}", None, None

            ratio = overlap / max(w_a, 1)
            warn = ""
            if ratio < warn_overlap_ratio:
                warn = f"⚠️ Low overlap (Edge horizontal): {overlap}px ({ratio * 100:.1f}%).\n"

            mask_a, mask_b = self.create_edge_overlap_mask(
                np_a, np_b, "horizontal", overlap, direction
            )
            merged = self.merge_horizontal_edge(
                np_a, np_b, overlap, priority, direction
            )
            final = self.numpy_to_pil(merged)
            fh, fw = merged.shape[:2]
            msg = header + warn + f"✅ {det_msg}\nMerged size: {fh}×{fw}px"
            return final, msg, mask_a, mask_b

        return (
            None,
            header + "❌ Invalid merge type. Use auto/vertical/horizontal.",
            None,
            None,
        )


def process_two_images(
    img_a,
    img_b,
    merge_type,
    detection_mode,
    priority,
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

        # Advanced settings are intentionally hardcoded (no user controls).
        min_overlap_px = 1
        warn_overlap_ratio = 0.10
        ambiguous_margin_px = 16
        slow_fallback_ratio = 0.10
        shift_search_ratio = 1.0

        # Overlap priority is only meaningful for ShiftMatch.
        if detection_mode != "shift_match":
            priority = "B"

        merger = ImageOverlapMerger(seed=12345)
        return merger.merge_images(
            img_a,
            img_b,
            merge_type,
            priority,
            detection_mode=detection_mode,
            min_overlap_px=int(min_overlap_px),
            warn_overlap_ratio=float(warn_overlap_ratio),
            ambiguous_margin_px=int(ambiguous_margin_px),
            slow_fallback_ratio=float(slow_fallback_ratio),
            shift_search_ratio=float(shift_search_ratio),
        )

    except Exception as e:
        return None, f"❌ Processing error: {str(e)}", None, None


def create_gradio_interface():
    with gr.Blocks(title="Image Overlap Merger") as demo:
        gr.Markdown("<h2 style='text-align:center;'>Image Overlap Merger</h2>")
        gr.Markdown(
            "- **Edge**: 端と端だけを重ねる（ABCD + CDEF → ABCDEF）\n"
            "- **ShiftMatch**: 途中をずらして重ねる（ABCD + BCCEF → ABCCEF）\n"
            "- 曖昧/低信頼のときはメッセージで警告します"
        )

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    img_a = gr.Image(type="pil", label="画像A（ベース）")
                    img_b = gr.Image(type="pil", label="画像B（重ねる）")

                merge_type = gr.Radio(
                    choices=[
                        ("自動（推奨）", "auto"),
                        ("縦方向", "vertical"),
                        ("横方向", "horizontal"),
                    ],
                    value="auto",
                    label="結合方向",
                )

                detection_mode = gr.Radio(
                    choices=[
                        (
                            "Edge 自動（推奨）",
                            "edge_auto",
                        ),
                        ("ShiftMatch（ずらし一致）", "shift_match"),
                    ],
                    value="edge_auto",
                    label="検出モード",
                )

                with gr.Row():
                    cancel_btn = gr.Button("クリア", variant="secondary")
                    run_btn = gr.Button("実行", variant="primary")

                priority = gr.Radio(
                    choices=[("画像Aを優先", "A"), ("画像Bを優先", "B")],
                    value="B",
                    label="重なり優先（ShiftMatchのみ有効）",
                    interactive=False,
                )
                gr.Markdown("- Edge は常に **画像B優先**（ShiftMatchのみ切替可）")

            with gr.Column():
                output_image = gr.Image(type="pil", label="結果画像", format="png")
                with gr.Row():
                    mask_a = gr.Image(
                        type="pil", label="重なりマスク（画像A）", format="png"
                    )
                    mask_b = gr.Image(
                        type="pil", label="重なりマスク（画像B）", format="png"
                    )
                message = gr.Textbox(label="メッセージ", lines=12)

        run_btn.click(
            fn=process_two_images,
            inputs=[
                img_a,
                img_b,
                merge_type,
                detection_mode,
                priority,
            ],
            outputs=[output_image, message, mask_a, mask_b],
        )

        def _toggle_priority(mode: str):
            if mode == "shift_match":
                return gr.update(interactive=True)
            return gr.update(value="B", interactive=False)

        detection_mode.change(
            fn=_toggle_priority,
            inputs=[detection_mode],
            outputs=[priority],
        )

        cancel_btn.click(
            fn=lambda: (None, None, "auto", "edge_auto", "B"),
            inputs=[],
            outputs=[
                img_a,
                img_b,
                merge_type,
                detection_mode,
                priority,
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",  # 重要: 外部(同一LAN)から到達できるようにする
        server_port=7860,
        share=False,
    )
