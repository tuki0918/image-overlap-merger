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
    
    def find_best_overlap_vertical(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[Optional[int], int, str]:
        """
        縦方向結合での最適な重複位置を検出
        img_aの下部にimg_bを重ねる場合を想定
        
        Returns: (重複サイズ, オフセット位置, 詳細メッセージ)
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        
        # 幅チェック
        if w_a != w_b:
            return None, 0, f"Width mismatch (A: {w_a}px, B: {w_b}px)"
        
        best_overlap = 0
        best_offset = 0
        best_match_pixels = 0
        
        # img_aの下部からimg_bの上部まで1pxずつスライド
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
        
        if best_overlap > 0:
            match_ratio = (best_match_pixels / (best_overlap * w_a)) * 100
            return best_overlap, best_offset, f"Vertical merge: {best_overlap}px overlap (match rate: {match_ratio:.1f}%)"
        else:
            return None, 0, "No matching region found for vertical merge"
    
    def find_best_overlap_horizontal(self, img_a: np.ndarray, img_b: np.ndarray) -> Tuple[Optional[int], int, str]:
        """
        横方向結合での最適な重複位置を検出
        img_aの右部にimg_bを重ねる場合を想定
        
        Returns: (重複サイズ, オフセット位置, 詳細メッセージ)
        """
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        
        # 高さチェック
        if h_a != h_b:
            return None, 0, f"Height mismatch (A: {h_a}px, B: {h_b}px)"
        
        best_overlap = 0
        best_offset = 0
        best_match_pixels = 0
        
        # img_aの右部からimg_bの左部まで1pxずつスライド
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
        
        if best_overlap > 0:
            match_ratio = (best_match_pixels / (best_overlap * h_a)) * 100
            return best_overlap, best_offset, f"Horizontal merge: {best_overlap}px overlap (match rate: {match_ratio:.1f}%)"
        else:
            return None, 0, "No matching region found for horizontal merge"
    
    def merge_vertical(self, img_a: np.ndarray, img_b: np.ndarray, overlap: int, priority: str = "B") -> np.ndarray:
        """縦方向に画像を結合"""
        if priority == "A":
            # img_aの全体 + img_bの重複部分を除いた部分
            result = np.vstack([img_a, img_b[overlap:]])
        else:  # priority == "B" (デフォルト)
            # img_aの重複部分を除いた部分 + img_bの全体
            result = np.vstack([img_a[:-overlap], img_b])
        return result
    
    def merge_horizontal(self, img_a: np.ndarray, img_b: np.ndarray, overlap: int, priority: str = "B") -> np.ndarray:
        """横方向に画像を結合"""
        if priority == "A":
            # img_aの全体 + img_bの重複部分を除いた部分
            result = np.hstack([img_a, img_b[:, overlap:]])
        else:  # priority == "B" (デフォルト)
            # img_aの重複部分を除いた部分 + img_bの全体
            result = np.hstack([img_a[:, :-overlap], img_b])
        return result
    
    def merge_images(self, img_a: Image.Image, img_b: Image.Image, direction: str, priority: str = "B") -> Tuple[Optional[Image.Image], str]:
        """
        2枚の画像を指定方向で結合
        
        Args:
            img_a: ベース画像（下レイヤー）
            img_b: オーバーレイ画像（上レイヤー - 重複部分で優先される）
            direction: "portrait" (縦) or "horizon" (横)
            priority: "A" (Image A優先) or "B" (Image B優先, デフォルト)
            
        Returns: (結合画像, ステータスメッセージ)
        """
        # NumPy配列に変換
        np_a = self.pil_to_numpy(img_a)
        np_b = self.pil_to_numpy(img_b)
        
        # サイズ情報
        h_a, w_a = np_a.shape[:2]
        h_b, w_b = np_b.shape[:2]
        
        size_info = f"Image A: {h_a}×{w_a}px, Image B: {h_b}×{w_b}px (Priority: Image {priority})\n"
        
        if direction == "portrait":
            # 縦方向結合
            overlap, offset, message = self.find_best_overlap_vertical(np_a, np_b)
            
            if overlap is not None:
                merged = self.merge_vertical(np_a, np_b, overlap, priority)
                final_image = self.numpy_to_pil(merged)
                
                final_h, final_w = merged.shape[:2]
                result_message = f"{size_info}✅ {message}\nMerged size: {final_h}×{final_w}px"
                
                return final_image, result_message
            else:
                return None, f"{size_info}❌ {message}"
        
        elif direction == "horizon":
            # 横方向結合
            overlap, offset, message = self.find_best_overlap_horizontal(np_a, np_b)
            
            if overlap is not None:
                merged = self.merge_horizontal(np_a, np_b, overlap, priority)
                final_image = self.numpy_to_pil(merged)
                
                final_h, final_w = merged.shape[:2]
                result_message = f"{size_info}✅ {message}\nMerged size: {final_h}×{final_w}px"
                
                return final_image, result_message
            else:
                return None, f"{size_info}❌ {message}"
        
        else:
            return None, f"❌ Invalid direction: {direction} (Please specify 'portrait' or 'horizon')"


def process_two_images(img_a, img_b, direction, priority) -> Tuple[Optional[Image.Image], str]:
    """画像処理関数"""
    
    if not img_a or not img_b:
        return None, "❌ Please upload both Image A (base) and Image B (overlay)"
    
    if not direction:
        return None, "❌ Please select merge direction (vertical or horizontal)"
    
    try:
        # RGBモードに変換
        if img_a.mode != 'RGB':
            img_a = img_a.convert('RGB')
        if img_b.mode != 'RGB':
            img_b = img_b.convert('RGB')
        
        # 結合処理
        merger = ImageOverlapMerger()
        result_image, status_message = merger.merge_images(img_a, img_b, direction, priority)
        
        return result_image, status_message
        
    except Exception as e:
        return None, f"❌ Processing error: {str(e)}"


def create_gradio_interface():
    demo = gr.Interface(
        fn=process_two_images,
        inputs=[
            gr.Image(type="pil", label="Image A (Left/Top position)"),
            gr.Image(type="pil", label="Image B (Right/Bottom position)"),
            gr.Radio(
                choices=[
                    ("Vertical", "portrait"),
                    ("Horizontal", "horizon")
                ],
                value="portrait",
                label="Merge Direction"
            ),
            gr.Radio(
                choices=[
                    ("Image A", "A"),
                    ("Image B", "B"),
                ],
                value="B",
                label="Overlap Priority"
            )
        ],
        outputs=[
            gr.Image(type="pil", label="Output Image", format="png"),
            gr.Textbox(label="Message", lines=5)
        ],
        title="Image Overlap Merger",
        description="Automatically detect and merge overlapping regions of two images",
        flagging_mode="never"
    )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
