import gradio as gr
import numpy as np
from PIL import Image
from typing import Tuple, Optional

class ImageOverlapMerger:
    """画像の重複領域を検出して自動結合クラス"""
    
    def __init__(self):
        pass
    
    def pil_to_numpy(self, pil_image: Image.Image) -> np.ndarray:
        """PIL画像をNumPy配列に変換"""
        return np.array(pil_image)
    
    def numpy_to_pil(self, numpy_array: np.ndarray) -> Image.Image:
        """NumPy配列をPIL画像に変換"""
        return Image.fromarray(numpy_array.astype(np.uint8))
    
    def find_best_overlap_vertical(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[Optional[int], int, str, str]:
        """
        縦方向結合での最適な重複位置を検出（両方向を自動検出）
        A下部+B上部 と A上部+B下部 の両方を試して最適な方を選択
        
        Returns: (重複サイズ, オフセット位置, 詳細メッセージ, 方向)
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        
        # 幅チェック
        if w_a != w_b:
            return None, 0, f"Width mismatch (A: {w_a}px, B: {w_b}px)", "none"
        
        best_overlap = 0
        best_offset = 0
        best_match_pixels = 0
        best_direction = "normal"
        
        # パターン1: img_aの下部 + img_bの上部
        max_search_range = min(h_a, h_b)
        
        for offset in range(1, max_search_range + 1):
            # img_aの下部offset行
            region_a = img_a[h_a - offset:h_a, :]
            # img_bの上部offset行
            region_b = img_b[:offset, :]
            
            # 一致ピクセル数をカウント
            match_pixels = np.sum(np.all(region_a == region_b, axis=2))
            
            # より良い一致が見つかった場合
            if match_pixels > best_match_pixels:
                best_match_pixels = match_pixels
                best_overlap = offset
                best_offset = offset
                best_direction = "normal"
        
        # パターン2: img_aの上部 + img_bの下部（逆方向）
        for offset in range(1, max_search_range + 1):
            # img_aの上部offset行
            region_a = img_a[:offset, :]
            # img_bの下部offset行
            region_b = img_b[h_b - offset:h_b, :]
            
            # 一致ピクセル数をカウント
            match_pixels = np.sum(np.all(region_a == region_b, axis=2))
            
            # より良い一致が見つかった場合
            if match_pixels > best_match_pixels:
                best_match_pixels = match_pixels
                best_overlap = offset
                best_offset = offset
                best_direction = "reverse"
        
        if best_overlap > 0:
            match_ratio = (best_match_pixels / (best_overlap * w_a)) * 100
            direction_desc = "A bottom + B top" if best_direction == "normal" else "A top + B bottom"
            return best_overlap, best_offset, f"Vertical merge ({direction_desc}): {best_overlap}px overlap (match rate: {match_ratio:.1f}%)", best_direction
        else:
            return None, 0, "No matching region found for vertical merge", "none"
    
    def find_best_overlap_horizontal(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[Optional[int], int, str, str]:
        """
        横方向結合での最適な重複位置を検出（両方向を自動検出）
        A右部+B左部 と A左部+B右部 の両方を試して最適な方を選択
        
        Returns: (重複サイズ, オフセット位置, 詳細メッセージ, 方向)
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        
        # 高さチェック
        if h_a != h_b:
            return None, 0, f"Height mismatch (A: {h_a}px, B: {h_b}px)", "none"
        
        best_overlap = 0
        best_offset = 0
        best_match_pixels = 0
        best_direction = "normal"
        
        # パターン1: img_aの右部 + img_bの左部
        max_search_range = min(w_a, w_b)
        
        for offset in range(1, max_search_range + 1):
            # img_aの右部offset列
            region_a = img_a[:, w_a - offset:w_a]
            # img_bの左部offset列
            region_b = img_b[:, :offset]
            
            # 一致ピクセル数をカウント
            match_pixels = np.sum(np.all(region_a == region_b, axis=2))
            
            # より良い一致が見つかった場合
            if match_pixels > best_match_pixels:
                best_match_pixels = match_pixels
                best_overlap = offset
                best_offset = offset
                best_direction = "normal"
        
        # パターン2: img_aの左部 + img_bの右部（逆方向）
        for offset in range(1, max_search_range + 1):
            # img_aの左部offset列
            region_a = img_a[:, :offset]
            # img_bの右部offset列
            region_b = img_b[:, w_b - offset:w_b]
            
            # 一致ピクセル数をカウント
            match_pixels = np.sum(np.all(region_a == region_b, axis=2))
            
            # より良い一致が見つかった場合
            if match_pixels > best_match_pixels:
                best_match_pixels = match_pixels
                best_overlap = offset
                best_offset = offset
                best_direction = "reverse"
        
        if best_overlap > 0:
            match_ratio = (best_match_pixels / (best_overlap * h_a)) * 100
            direction_desc = "A right + B left" if best_direction == "normal" else "A left + B right"
            return best_overlap, best_offset, f"Horizontal merge ({direction_desc}): {best_overlap}px overlap (match rate: {match_ratio:.1f}%)", best_direction
        else:
            return None, 0, "No matching region found for horizontal merge", "none"
    
    def merge_vertical(self, img_a: np.ndarray, img_b: np.ndarray, overlap: int, priority: str = "B", direction: str = "normal") -> np.ndarray:
        """縦方向に画像を結合（両方向対応）"""
        if direction == "normal":
            # A下部 + B上部（従来）
            if priority == "A":
                # img_aの全体 + img_bの重複部分を除いた部分
                result = np.vstack([img_a, img_b[overlap:]])
            else:
                # img_aの重複部分を除いた部分 + img_bの全体
                result = np.vstack([img_a[:-overlap], img_b])
        else:  # direction == "reverse"
            # A上部 + B下部（逆方向）
            if priority == "A":
                # Bの重複部分を除いた部分 + img_aの全体
                result = np.vstack([img_b[:-overlap], img_a])
            else:
                # img_bの全体 + img_aの重複部分を除いた部分
                result = np.vstack([img_b, img_a[overlap:]])
        return result
    
    def merge_horizontal(self, img_a: np.ndarray, img_b: np.ndarray, overlap: int, priority: str = "B", direction: str = "normal") -> np.ndarray:
        """横方向に画像を結合（両方向対応）"""
        if direction == "normal":
            # A右部 + B左部（従来）
            if priority == "A":
                # img_aの全体 + img_bの重複部分を除いた部分
                result = np.hstack([img_a, img_b[:, overlap:]])
            else:
                # img_aの重複部分を除いた部分 + img_bの全体
                result = np.hstack([img_a[:, :-overlap], img_b])
        else: 
            # A左部 + B右部（逆方向）
            if priority == "A":
                # Bの重複部分を除いた部分 + img_aの全体
                result = np.hstack([img_b[:, :-overlap], img_a])
            else:  # priority == "B" (デフォルト)
                # img_bの全体 + img_aの重複部分を除いた部分
                result = np.hstack([img_b, img_a[:, overlap:]])
        return result
    
    def create_overlap_mask(self, img_a: np.ndarray, img_b: np.ndarray, merge_type: str, overlap: int, direction: str) -> Tuple[Image.Image, Image.Image]:
        """
        重複領域のマスクを作成
        
        Args:
            img_a: ベース画像
            img_b: オーバーレイ画像
            merge_type: "vertical" (縦) / "horizontal" (横)
            overlap: 重複ピクセル数
            direction: "normal" / "reverse"
            
        Returns: (Image Aのマスク, Image Bのマスク)
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        
        if merge_type == "vertical":
            if direction == "normal":
                # 縦方向結合の場合 - Aの下部 + Bの上部
                # Image Aのマスク：下部のoverlap行が白（重複領域）
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[h_a-overlap:h_a, :] = 255
                
                # Image Bのマスク：上部のoverlap行が白（重複領域）
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[:overlap, :] = 255
            else:  # direction == "reverse"
                # 縦方向結合（逆）の場合 - Aの上部 + Bの下部
                # Image Aのマスク：上部のoverlap行が白（重複領域）
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[:overlap, :] = 255
                
                # Image Bのマスク：下部のoverlap行が白（重複領域）
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[h_b-overlap:h_b, :] = 255
                
        elif merge_type == "horizontal":
            if direction == "normal":
                # 横方向結合の場合 - Aの右部 + Bの左部
                # Image Aのマスク：右部のoverlap列が白（重複領域）
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[:, w_a-overlap:w_a] = 255
                
                # Image Bのマスク：左部のoverlap列が白（重複領域）
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[:, :overlap] = 255
            else:  # direction == "reverse"
                # 横方向結合（逆）の場合 - Aの左部 + Bの右部
                # Image Aのマスク：左部のoverlap列が白（重複領域）
                mask_a = np.zeros((h_a, w_a), dtype=np.uint8)
                mask_a[:, :overlap] = 255
                
                # Image Bのマスク：右部のoverlap列が白（重複領域）
                mask_b = np.zeros((h_b, w_b), dtype=np.uint8)
                mask_b[:, w_b-overlap:w_b] = 255
        
        return Image.fromarray(mask_a), Image.fromarray(mask_b)
    
    def merge_images(self, img_a: Image.Image, img_b: Image.Image, merge_type: str, priority: str = "B") -> Tuple[Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]]:
        """
        2枚の画像を指定方向で結合
        
        Args:
            img_a: ベース画像
            img_b: オーバーレイ画像
            merge_type: "vertical" (縦) or "horizontal" (横)
            priority: "A" (Image A優先) or "B" (Image B優先, デフォルト)
            
        Returns: (結合画像, ステータスメッセージ, Image Aマスク, Image Bマスク)
        """
        # NumPy配列に変換
        np_a = self.pil_to_numpy(img_a)
        np_b = self.pil_to_numpy(img_b)
        
        # サイズ情報
        h_a, w_a = np_a.shape[:2]
        h_b, w_b = np_b.shape[:2]
        
        size_info = f"Image A: {h_a}×{w_a}px, Image B: {h_b}×{w_b}px (Priority: Image {priority})\n"
        
        if merge_type == "vertical":
            # 縦方向結合 - 両方向を自動検出
            overlap, offset, message, direction = self.find_best_overlap_vertical(np_a, np_b)
            
            if overlap is not None:
                # マスクを作成
                mask_a, mask_b = self.create_overlap_mask(np_a, np_b, merge_type, overlap, direction)
                
                merged = self.merge_vertical(np_a, np_b, overlap, priority, direction)
                final_image = self.numpy_to_pil(merged)
                
                final_h, final_w = merged.shape[:2]
                result_message = f"{size_info}✅ {message}\nMerged size: {final_h}×{final_w}px"
                
                return final_image, result_message, mask_a, mask_b
            else:
                return None, f"{size_info}❌ {message}", None, None
        
        elif merge_type == "horizontal":
            # 横方向結合 - 両方向を自動検出
            overlap, offset, message, direction = self.find_best_overlap_horizontal(np_a, np_b)
            
            if overlap is not None:
                # マスクを作成
                mask_a, mask_b = self.create_overlap_mask(np_a, np_b, merge_type, overlap, direction)
                
                merged = self.merge_horizontal(np_a, np_b, overlap, priority, direction)
                final_image = self.numpy_to_pil(merged)
                
                final_h, final_w = merged.shape[:2]
                result_message = f"{size_info}✅ {message}\nMerged size: {final_h}×{final_w}px"
                
                return final_image, result_message, mask_a, mask_b
            else:
                return None, f"{size_info}❌ {message}", None, None
        
        else:
            return None, f"❌ Invalid merge type: {merge_type} (Please specify 'vertical' or 'horizontal')", None, None


def process_two_images(img_a, img_b, merge_type, priority) -> Tuple[Optional[Image.Image], str, Optional[Image.Image], Optional[Image.Image]]:
    """画像処理関数"""
    
    if not img_a or not img_b:
        return None, "❌ Please upload both Image A (base) and Image B (overlay)", None, None
    
    if not merge_type:
        return None, "❌ Please select merge type (vertical or horizontal)", None, None
    
    try:
        # RGBモードに変換
        if img_a.mode != 'RGB':
            img_a = img_a.convert('RGB')
        if img_b.mode != 'RGB':
            img_b = img_b.convert('RGB')
        
        # 結合処理
        merger = ImageOverlapMerger()
        result_image, status_message, mask_a, mask_b = merger.merge_images(img_a, img_b, merge_type, priority)
        
        return result_image, status_message, mask_a, mask_b
        
    except Exception as e:
        return None, f"❌ Processing error: {str(e)}", None, None


def create_gradio_interface():
    with gr.Blocks(title="Image Overlap Merger") as demo:
        gr.Markdown("<h2 style='text-align:center;'>Image Overlap Merger</h2>")
        gr.Markdown(
            "Automatically detect and merge overlapping regions of two images. "
            "Masks show the detected overlap regions."
        )

        with gr.Row():
            # 左カラム（入力）
            with gr.Column():
                with gr.Row():
                    img_a = gr.Image(type="pil", label="Image A (Left/Top position)")
                    img_b = gr.Image(type="pil", label="Image B (Right/Bottom position)")

                merge_type = gr.Radio(
                    choices=[("Vertical", "vertical"), ("Horizontal", "horizontal")],
                    value="vertical",
                    label="Merge Direction"
                )

                overlap_priority = gr.Radio(
                    choices=[("Image A", "A"), ("Image B", "B")],
                    value="B",
                    label="Overlap Priority"
                )

                with gr.Row():
                    cancel_btn = gr.Button("Cancel", variant="secondary")
                    run_btn = gr.Button("Submit", variant="primary")

            # 右カラム（出力）
            with gr.Column():
                output_image = gr.Image(type="pil", label="Output Image", format="png")
                with gr.Row():
                    mask_a = gr.Image(type="pil", label="Image A Overlap Mask", format="png")
                    mask_b = gr.Image(type="pil", label="Image B Overlap Mask", format="png")
                message = gr.Textbox(label="Message", lines=5)

        # Run
        run_btn.click(
            fn=process_two_images,
            inputs=[img_a, img_b, merge_type, overlap_priority],
            outputs=[output_image, message, mask_a, mask_b]
        )

        # Cancel
        cancel_btn.click(
            fn=lambda: (None, None, "vertical", "B"),
            inputs=[],
            outputs=[img_a, img_b, merge_type, overlap_priority]
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
