# 通用模型 Embedding 函數設計
# 下面提供一個通用的函數設計，可以方便地切換不同的語言模型並獲取 embeddings：

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import pandas as pd
import numpy as np
import os
import time
from typing import List

# ----------------------------------------------------------------------------------- #
#   🛠️ 一键修复：升级你的 clean_sentences
#   结合前文清洗 + 类型转换：
#
def ensure_list_of_str(sentences):
    """
    Ensure input is List[str], safe for tokenizer.
    Handles: np.ndarray, pd.Series, list, str, with NaN/None cleaning.
    """
    import numpy as np
    import pandas as pd

    # Convert to list, handling numpy/pandas quirks
    if isinstance(sentences, (pd.Series, pd.DataFrame)):
        sentences = sentences.values
    if isinstance(sentences, np.ndarray):
        sentences = sentences.tolist()  # ← critical for dtype=object!
    
    # Now it's List or scalar
    if isinstance(sentences, str):
        sentences = [sentences]
    elif not isinstance(sentences, list):
        sentences = list(sentences)
    
    # Clean None / NaN / empty
    cleaned = []
    for i, s in enumerate(sentences):
        if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
            print(f"⚠️ Skipping invalid sentence at index {i}: {repr(s)}")
            continue
        cleaned.append(str(s).strip())
    
    if not cleaned:
        raise ValueError("No valid sentences after cleaning.")
    return cleaned

#
#   
#   Q1: 好！整合支持 content-only masking, 並且看 single_embedding 
#       跟 get_sentences_embeddings 是否也可整合
#
#   Qwen3-Max: 模型差异与隐藏状态提取
#   answer to Q1:
#   非常好！你已有一个结构清晰的 ModelEmbedding 类，我们来系统性升级它，使其：
#
#   ✅ 支持 content-only pooling（排除 BOS/EOS/PAD）
#   ✅ 统一 single_embedding() 与 get_sentences_embeddings() 的 pooling 逻辑
#   ✅ 向后兼容（不破坏现有调用）
#   ✅ 显式支持 "content_mean" 作为新 pooling 类型（推荐用于神经科学）
#
#   ✅ 升级方案总览
#
#   1. 修改点: 重构 mean_pooling → 支持 input_ids & tokenizer
#       说明: 新增 exclude_special_tokens=True
#   2. 修改点: 新增 pooling 类型："content_mean"
#       说明: 语义最干净，推荐默认
#   3. 修改点: 统一所有 mean-based pooling 调用新 mean_pooling
#       说明: 避免重复代码（如 last_avg）
#   4. 修改点: 改造 aggregate_hidden_states() 调用新 pooling
#       说明: 兼容 single_embedding
#   5. 修改点: （可选）将 "mean" 默认行为改为 "content_mean"
#       说明: 更科学，可通过 flag 控制
#
# --------------------------------------------------------------- #

from typing import List, Optional, Union
import torch
from transformers import PreTrainedTokenizerBase

def mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    exclude_special_tokens: bool = False,
) -> torch.Tensor:
    """
    支持 content-only mean pooling 的通用函数
    
    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len), 1 for real tokens
        input_ids: (batch, seq_len), required if exclude_special_tokens=True
        tokenizer: required if exclude_special_tokens=True
        exclude_special_tokens: 若 True，排除 BOS/EOS/PAD；否则仅排除 PAD（via attention_mask）
    
    Returns:
        (batch, hidden_dim)
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    if exclude_special_tokens and input_ids is not None and tokenizer is not None:
        # Build content mask: non-pad, non-BOS, non-EOS
        mask = torch.ones_like(input_ids, dtype=torch.float32, device=hidden_states.device)
        
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            mask = mask * (input_ids != pad_id).float()
        
        bos_id = tokenizer.bos_token_id
        if bos_id is not None:
            mask = mask * (input_ids != bos_id).float()
        
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            mask = mask * (input_ids != eos_id).float()
    else:
        # fallback: only mask out padding via attention_mask
        if attention_mask is None:
            mask = torch.ones(batch_size, seq_len, device=hidden_states.device)
        else:
            mask = attention_mask.float()

    mask_expanded = mask.unsqueeze(-1).expand_as(hidden_states)
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# ------------------------------------------------------------------- #
#
#   Q2: 如何可以整合get_sentences_embeddings 跟 single_embedding 像是善用batch
#       想使用single_embedding 可能是因為記憶體或cpu 不足以一次處理所有的句子, 但
#       get_sentences_embeddings 可能會比 single_embedding 速度快很多, 
#       希望可以整合, 盡可能利用記憶體及CPU 可以跑得快, 有效運用記憶體
#
#   Qwen3-Max answer to Q2:
#   你提出了一个非常实际且关键的问题：
#   如何在内存/显存受限时，尽可能高效地利用 batch 处理能力，
#   统一 get_sentences_embeddings（快）与 single_embedding（省内存）？
#

# ------------------------------------------------------------------- #
#
#   Q3: 會跑很多layer, 理論上每層layer 可用的batch_size 都一樣, 
#       不需要每層layer 都重新估計batch_size
#
#       需要
#       加入 tqdm 进度条 + logging
#       实现 tokenization caching（对固定 stimset 极速重复运行）
#
#       可能在不同的GPU跑, 目前主要在 NVIDIA H100 80GB HBM3 跑, 可以有5 個GPU, 視情況
#       跑的模型從 8B 到70B 都有, 典型句子數128句
#
#   Qwen3-Max answer to Q3:
#   太好了！你已经有一个优秀的自适应批处理框架，现在我们要做 三重升级：
#
#   ✅ 1. 全局 batch_size 缓存（避免每层重复估算）
#   ✅ 2. 添加 tqdm + logging（清晰进度 & 诊断）
#   ✅ 3. Tokenization caching（对固定 stimset 极速重复运行）
#   ✅ 4. 多 GPU 感知（H100 80GB ×5 → 智能分配）
#

import os
import time
import logging
from typing import Optional, Union, List, Dict, Any, Tuple
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import hashlib

# ===== 配置 logging =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ModelEmbedding: 
    def __init__(
        self,
        # model_name: str,
        # model_name_or_path: str,
        # max_length=300,
        # cache_dir="../models",
        device: str = "Auto",
        # Load_LLM_method: int = 1,
        **kwargs,
    ):
        """
        初始化模型和tokenizer

        Args:
            device (str): "cpu" 或 "cuda"，若為 None 自動判斷

            kwargs = model_config containing
                model_name: 模型名稱 (例如 'gpt2-xl', 'bert-base-chinese')
                #max_length: 最大序列長度
        """
        # print(f'device = {device}')
        # print(f'kwargs = {kwargs}')

        # ------- expect model_config = kwargs  --------------- #
        gpu_ids = kwargs.get("gpu_ids", [0])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        print(f" --------------  loading ModelEmbedding  ------------")
        print(f"gpu_ids = {gpu_ids}")
        print(
            f'os.environ["CUDA_VISIBLE_DEVICES"] = {os.environ["CUDA_VISIBLE_DEVICES"]}'
        )
        print(f" ----------------------------------------------------")

        model_name = kwargs.get("model_name")
        Load_LLM_method = kwargs.get("Load_LLM_method")

        AutoModel_config = kwargs.get("AutoModel_config")
        model_name_or_path = AutoModel_config["model"]

        # --- 1. 設定設備 ---

        if device == "Auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- 2. 載入模型與 Tokenizer ---
        # 關鍵：output_hidden_states=True 才能獲取所有中間層

        print(f"model_name_or_path = {model_name_or_path} -----------------------")

        if Load_LLM_method == 0:  # use model_directory
            #  --- method 0 ---
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

            # self.model = AutoModel.from_pretrained(
            #     model_name_or_path, output_hidden_states=True
            # ).to(device)
            self.model = AutoModel.from_pretrained(
                model_name_or_path, device_map="auto", output_hidden_states=True
            )

        elif Load_LLM_method == 1:  # use model_name

            #  --- method 1 ---
            # 如果需要，設置模型緩存目錄
            os.makedirs(model_name_or_path, exist_ok=True)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=model_name_or_path
            )

            # self.model = AutoModel.from_pretrained(
            #     model_name, output_hidden_states=True, cache_dir=model_name_or_path
            # ).to(device)
            self.model = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,
                device_map="auto",
                cache_dir=model_name_or_path,
            )

        # ✅ 正确获取设备（此时可能是多设备）
        self.device = self.model.device
        # self.device = device

        # # Update self.device correctly:
        # self.device = next(self.model.parameters()).device

        # 創建一個集合來儲存所有獨特的設備名稱
        used_devices = set()

        # 迭代模型的命名參數
        for name, param in self.model.named_parameters():
            # 檢查參數所在的設備
            device_name = str(param.device)
            used_devices.add(device_name)

        # 打印結果
        print("模型權重被分配到的設備清單:")
        print(used_devices)
        self.used_devices = used_devices

        #  --- method 2 ---
        # self.config = AutoConfig.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name, config=self.config,
        #                                     cache_dir=model_dir)

        self.model_dir = model_name_or_path
        self.model_name = model_name
        # self.max_length = max_length

        self.model.eval()

        # -----------     answer to Q6      --------------------
        # ... [前面所有初始化代码：GPU设置、加载tokenizer、加载model、self.model.eval()] ...

        # ===== 关键：在 model 完全加载后调用 =====
        # ===== ✅ 设备一致性验证（关键诊断点）=====
        try:
            self._validate_device_consistency()  # ← 插在这里！
        except Exception as e:
            logger.warning(f"Device consistency check failed (non-fatal): {e}")
            # 不中断初始化，但记录警告供调试
            self.last_layer_device = getattr(self.model, "device", torch.device("cpu"))
        

        # 設置padding token (如果需要)
        # Setup tokenizer padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("✅ ModelEmbedding initialized successfully")

        # # ===== 新增：类级缓存 =====
        # self._global_batch_size_cache: Dict[str, int] = {}  # {(model_name, pooling): batch_size}
        # self._tokenization_cache: Dict[str, Any] = {}       # {"hash": (input_ids, attention_mask, sentences)}

        # ✅ Replace global caches with safer per-batch cache
        self._batch_token_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._batch_cache_lru: Dict[str, float] = {}  # LRU timestamp
        self._max_cache_size = 100  # 最多缓存 100 个 batch

    def _get_batch_cache_key(self, sentences: List[str], max_length: int = 512) -> str:
        """生成 batch 级缓存 key（按内容+长度）"""
        # 取前 50 字符做摘要（防长句 key 过大）
        sample = "|".join(s[:50] for s in sentences[:10])
        key = f"{hashlib.md5(sample.encode()).hexdigest()[:8]}_len{max_length}_cnt{len(sentences)}"
        return key

    def _tokenize_batch_safe(
        self, 
        sentences: List[str], 
        max_length: int = 512,
        cache_key: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        安全 tokenize batch：无 view 泄漏 + 缓存控制
        """
        if cache_key is None:
            cache_key = self._get_batch_cache_key(sentences, max_length)
        
        # 尝试从缓存读取
        if cache_key in self._batch_token_cache:
            cached = self._batch_token_cache[cache_key]
            # 深拷贝避免 view 泄漏（关键！）
            return {
                "input_ids": cached["input_ids"].clone(),
                "attention_mask": cached["attention_mask"].clone()
            }
        
        # 实际 tokenize
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # ✅ 关键：转为连续内存 + CPU 存储（避免 GPU 泄漏）
        cache_data = {
            "input_ids": encoded["input_ids"].cpu().contiguous(),
            "attention_mask": encoded["attention_mask"].cpu().contiguous()
        }
        
        # LRU cache eviction
        if len(self._batch_token_cache) >= self._max_cache_size:
            oldest_key = min(self._batch_cache_lru, key=self._batch_cache_lru.get)
            self._batch_token_cache.pop(oldest_key, None)
            self._batch_cache_lru.pop(oldest_key, None)
        
        self._batch_token_cache[cache_key] = cache_data
        self._batch_cache_lru[cache_key] = time.time()
        
        # 返回深拷贝（调用方自由移动到 GPU）
        return {
            "input_ids": cache_data["input_ids"].clone(),
            "attention_mask": cache_data["attention_mask"].clone()
        }

    def _clear_token_cache(self, max_age_seconds: float = 300):
        """清理过期缓存（可选）"""
        now = time.time()
        to_remove = [k for k, t in self._batch_cache_lru.items() if now - t > max_age_seconds]
        for k in to_remove:
            self._batch_token_cache.pop(k, None)
            self._batch_cache_lru.pop(k, None)

    # --------------------------------------------------------------- #
    #   Q_Gem4: 那不同size 的模型（1B 到 70B) 要在哪裡開始調整適合的batch_size? 
    #           是在呼叫get_embeddings 的時候？
    #           小size (例如70B) 有可能可以batch_size = 128 or 32
    #   Gemini answer to Q_Gem4:
    #   要讓程式自動適應 1B (大 batch) 到 70B (小 batch)，我們不應該在呼叫時手動寫死一個數字，
    #   而是應該修改 _probe_safe_batch_size 這個「大腦」，讓它根據模型的大小來決定「起手式」。
    #
    #   修改地點：_probe_safe_batch_size
    #   請將 ModelEmbedding 類別中的 _probe_safe_batch_size 替換為以下版本。
    #   這個版本的核心邏輯是利用 hidden_size (隱藏層維度) 來判斷模型大小，進而設定不同的測試候選列表。
    #
    def _probe_safe_batch_size(
        self,
        sentences: List[str],
        layer: int = -1,
        pooling: str = "content_mean",
        max_trials: int = 3
        ) -> int:
        """
        智能 Batch Size 探測器 (兼容 1B ~ 70B)
        根據模型隱藏層大小，自動決定測試範圍。
        """
        # --- 1. 根據 hidden_size 判斷模型量級 ---
        # 獲取 hidden_size (如果獲取失敗，默認當作 7B 處理)
        hidden_size = getattr(self.model.config, "hidden_size", 4096)
        
        # 定義候選列表 (從大到小嘗試)
        if hidden_size >= 8192: 
            # === 70B+ 等級 (hidden_size 通常 >= 8192) ===
            # 70B 模型非常佔顯存，保守起見從 4 開始試
            candidates = [4, 2, 1]
            logger.info(f"📏 Detected Large Model (Hidden Size: {hidden_size}), probing range: {candidates}")
            
        elif hidden_size >= 4096:
            # === 7B ~ 14B 等級 (hidden_size 通常 4096 ~ 5120) ===
            # 這類模型通常可以跑 32 ~ 64
            candidates = [64, 32, 16, 8, 4]
            logger.info(f"📏 Detected Medium Model (Hidden Size: {hidden_size}), probing range: {candidates}")
            
        else:
            # === 1B ~ 3B 小模型 (hidden_size < 4096) ===
            # 小模型可以跑很大，從 128 開始試
            candidates = [128, 64, 32, 16]
            logger.info(f"📏 Detected Small Model (Hidden Size: {hidden_size}), probing range: {candidates}")

        # 如果句子總數很少，不需要測試那麼大的 batch
        candidates = [c for c in candidates if c <= len(sentences)]
        if not candidates:
            return len(sentences)

        # --- 2. 開始探測 (邏輯保持不變) ---
        for bs in candidates:
            success_count = 0
            for trial in range(max_trials):
                try:
                    # 取樣測試
                    test_sents = sentences[:bs]
                    
                    # 這裡調用我們剛剛升級過的 _run_batch
                    # 因為它有 OOM 保護，所以測試過程也是安全的
                    with torch.no_grad():
                         _ = self._run_batch(test_sents, layer=layer, pooling=pooling)
                    
                    success_count += 1
                    # 連續成功 2 次 (或只需要 1 次如果很有把握) 就認為安全
                    if success_count >= 1: 
                        logger.info(f"✅ Auto-detected safe batch_size = {bs}")
                        return bs
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.debug(f"⚠️ Probe batch_size={bs} OOM. Trying smaller...")
                        break  # 當前 batch_size 太大，跳出並嘗試下一個更小的
                    else:
                        logger.warning(f"⚠️ Probe error: {e}")
                        break # 其他錯誤也跳過
                except Exception as e:
                    logger.warning(f"⚠️ Probe error: {e}")
                    break
        
        # 如果全部失敗，回傳 1 (最保守)
        logger.warning("⚠️ All probe attempts failed. Fallback to batch_size=1")
        return 1

    def _validate_device_consistency(self):
        """验证设备映射一致性"""
        if not hasattr(self.model, "device_map") or not self.model.device_map:
            return
        
        # 检查hidden_states设备是否一致
        test_input = self.tokenizer("test", return_tensors="pt")
        test_input = {k: v.to("cuda:0") for k, v in test_input.items()}
        
        with torch.no_grad():
            output = self.model(**test_input, output_hidden_states=True)
        
        devices = []
        for i, hs in enumerate(output.hidden_states):
            devices.append(str(hs.device))
        
        unique_devices = set(devices)
        logger.info(f"Hidden states devices: {unique_devices}")
        
        if len(unique_devices) > 1:
            logger.warning("⚠️ Hidden states span multiple devices - pooling may fail silently")
            # 记录最后一层设备
            self.last_layer_device = devices[-1]
        else:
            self.last_layer_device = devices[0]

    # --------------------------------------------------------------- #
    #   Q_Gem3: 除了像上面更改_run_batch, get_embedding_sentence 跟 _smart_pooling
    #           是否有需要更改 get_embeddings ? 也去除冗餘
    #   Gemini answer to Q_Gem3:
    #   修改思路
    #   1. 移除 use_direct_tokenize 判斷：不再區分是否為 70B 模型，統一走一條路。
    #   2. 移除 _run_batch_from_input_safe 調用：
    #           Tokenize 的動作已經封裝在 _run_batch 裡，
    #           外部不需要再手動 Tokenize。
    #   3. 保留 Batch Size 探測與遞迴重試：這是處理 OOM 的第一道防線（動態縮小 Batch），
    #           非常有保留價值。
    #
    def get_embeddings(
        self,
        sentences: List[str],
        layer: int = -1,
        pooling: str = "content_mean",
        batch_size: Optional[int] = None,
        max_memory_ratio: float = 0.7,
        return_numpy: bool = True,
        use_cache: bool = True, # 雖然保留參數兼容，但內部邏輯已簡化
        **kwargs
        ) -> Union[torch.Tensor, np.ndarray]:

        # --- 1. 基礎檢查 ---
        # (這裡假設你有 _ensure_list_of_str 函數，若無可用簡單列表推導式)
        if hasattr(self, '_ensure_list_of_str'):
            sentences = self._ensure_list_of_str(sentences)
        
        if not sentences:
            raise ValueError("No valid sentences")
        
        # --- 2. 獲取 Batch Size (智能探測) ---
        if batch_size is None:
            # 簡單生成 key 用於緩存 batch_size 結果
            sample_key = f"len{len(sentences)}_L{layer}_P{pooling}"
            
            if hasattr(self, '_bs_cache') and sample_key in self._bs_cache:
                batch_size = self._bs_cache[sample_key]
                logger.info(f"🔁 Using cached batch_size={batch_size}")
            else:
                # 調用你現有的探測函數
                batch_size = self._probe_safe_batch_size(
                    sentences, layer=layer, pooling=pooling
                )
                if not hasattr(self, '_bs_cache'):
                    self._bs_cache = {}
                self._bs_cache[sample_key] = batch_size
        
        logger.info(f"🚀 Processing {len(sentences)} sentences, batch_size={batch_size}")
        
        # --- 3. 分塊處理 (Unified Loop) ---
        all_embeddings = []
        n_batches = (len(sentences) + batch_size - 1) // batch_size
        
        # 這裡不需要再判斷 use_direct_tokenize 了
        # 因為新的 _run_batch 已經能夠處理所有情況
        
        for batch_idx in tqdm(range(n_batches), desc=f"Layer {layer}"):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(sentences))
            batch_sents = sentences[start:end]

            try:
                # ✅ 統一調用升級版的 _run_batch
                # 它內部已經包含了: Tokenize -> GPU Forward -> Smart Pooling -> CPU Fallback
                emb_batch = self._run_batch(batch_sents, layer=layer, pooling=pooling)
                
                # 結果統一轉到 CPU 保存，節省顯存
                all_embeddings.append(emb_batch.cpu())
                
                # 顯式刪除臨時變量，協助垃圾回收
                del emb_batch
                
            except RuntimeError as e:
                # 只有當 _run_batch 內部的 CPU Fallback 也失敗，或者外部顯存真的不夠時
                # 才會走到這裡
                if "out of memory" in str(e).lower():
                    logger.warning(f"⚠️ OOM detected at batch_size={batch_size}")

                    if batch_size > 1:
                        # 策略 A: 遞迴降級 (例如 4 -> 2 -> 1)
                        logger.info("📉 Halving batch size and retrying...")
                        torch.cuda.empty_cache() # 關鍵：清理顯存
                        
                        return self.get_embeddings(
                            sentences, 
                            layer=layer, 
                            pooling=pooling,
                            batch_size=max(1, batch_size // 2), # 減半
                            max_memory_ratio=max_memory_ratio,
                            return_numpy=return_numpy,
                            **kwargs
                        )
                    else:
                        # 策略 B: batch_size=1 也爆了 (極端情況)
                        # 因為 _run_batch 已經有 CPU Fallback，走到這裡代表連 CPU 都跑不動
                        # 或者有其他嚴重問題，必須拋出異常
                        logger.critical("❌ Batch size 1 failed even with CPU fallback.")
                        raise e
                else:
                    # 其他錯誤 (如維度不匹配等) 直接拋出
                    raise e

        # 保存本次成功的 batch_size 供下次參考
        self.run_batch_size = batch_size

        # --- 4. 合併結果 ---
        if not all_embeddings:
            return None # 或拋出異常
            
        final_emb = torch.cat(all_embeddings, dim=0)
        return final_emb.numpy() if return_numpy else final_emb

    # --------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    #   Qwen3-Max answer to Q10:
    #
    #   ✅ 修复方案：三步走
    #   ✅ 步骤 1：修复 _run_batch_from_input（推荐长期方案）
    #
    def _run_batch_from_input_safe(
        self,
        encoded_input: Dict[str, torch.Tensor],
        layer: int,
        pooling: str,
        **kwargs
    ) -> torch.Tensor:
        try:
            with torch.no_grad():
                # ✅ 关键 1：禁用 autocast（70B 多 GPU 下不稳定）
                # with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = self.model(**encoded_input, output_hidden_states=True)
            
            hidden_states_ALL = output.hidden_states
            target_layer = len(hidden_states_ALL) + layer if layer < 0 else layer
            hidden_states = hidden_states_ALL[target_layer]
            
            # ✅ 关键 2：强制统一设备（避免跨 GPU silent zero）
            if hidden_states.device != encoded_input["input_ids"].device:
                hidden_states = hidden_states.to(encoded_input["input_ids"].device)
            
            current_device = hidden_states.device
            
            # --- Pooling ---
            if pooling in ("mean", "content_mean"):
                # ✅ 关键 3：在 GPU 上直接 pooling（70B batch_size=1 显存足够）
                result = mean_pooling(
                    hidden_states,  # 不移 CPU！
                    attention_mask=encoded_input["attention_mask"],
                    input_ids=encoded_input["input_ids"],
                    tokenizer=self.tokenizer,
                    exclude_special_tokens=(pooling == "content_mean")
                )
                return result  # 同设备，无需 .to()
            
            elif pooling == "cls":
                return hidden_states[:, 0]
            
            elif pooling == "last":
                seq_lengths = encoded_input["attention_mask"].sum(dim=1) - 1
                return hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]
            
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # ✅ 关键 4：提供可 action 的错误信息
                raise RuntimeError(
                    f"CUDA OOM in _run_batch_from_input. "
                    f"Suggested fix: use _run_batch for 70B models, or reduce batch_size to 1."
                ) from e
            raise

    def _run_batch_from_input(
            self, 
            encoded_input: Dict[str, torch.Tensor], 
            layer: int, 
            pooling: str, 
            **kwargs):
        try:
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = self.model(**encoded_input, output_hidden_states=True)
            
            hidden_states_ALL = output.hidden_states
            target_layer = len(hidden_states_ALL) + layer if layer < 0 else layer
            hidden_states = hidden_states_ALL[target_layer]
            
            # ✅ 关键修复：获取当前hidden_states的实际设备
            current_device = hidden_states.device
            
            # --- Pooling with correct device handling ---
            if pooling in ("mean", "content_mean"):
                # 移到CPU处理节省显存，但返回时用original device
                result = mean_pooling(
                    hidden_states.cpu(),  # 先移到CPU
                    attention_mask=encoded_input["attention_mask"].cpu(),
                    input_ids=encoded_input["input_ids"].cpu(),
                    tokenizer=self.tokenizer,
                    exclude_special_tokens=(pooling == "content_mean")
                )
                return result.to(current_device)  # ← 关键：用计算设备，不是self.device
            
            elif pooling == "cls":
                return hidden_states[:, 0]  # ← 保持原设备
            
            elif pooling == "last":
                seq_lengths = encoded_input["attention_mask"].sum(dim=1) - 1
                return hidden_states[torch.arange(hidden_states.size(0)), seq_lengths]  # ← 保持原设备
                
            elif pooling == "max":
                # 重型操作移到 CPU 节省 GPU 显存
                hs_cpu = hidden_states.cpu()
                am_cpu = encoded_input["attention_mask"].cpu()
                mask_expanded = am_cpu.unsqueeze(-1).expand(hs_cpu.size())
                hs_masked = hs_cpu.masked_fill(mask_expanded == 0, -1e9)
                result, _ = torch.max(hs_masked, dim=1)
                return result.to(current_device)  # ← 关键：迁移回 hidden_states 原设备


            else:
                raise ValueError(f"Unsupported pooling: {pooling}")
                
        except Exception as e:
            logger.error(f"Batch processing failed on device {hidden_states.device if 'hidden_states' in locals() else 'unknown'}: {e}")
            raise

    # --------------------------------------------------------------- #

    # ===== 辅助方法 =====
    def _ensure_list_of_str(self, sentences):
        """清洗 + 转 list[str]"""
        if isinstance(sentences, (pd.Series, pd.DataFrame)):
            sentences = sentences.values
        if isinstance(sentences, np.ndarray):
            sentences = sentences.tolist()
        if isinstance(sentences, str):
            sentences = [sentences]
        elif not isinstance(sentences, list):
            sentences = list(sentences)
        
        cleaned = []
        for i, s in enumerate(sentences):
            if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
                logger.warning(f"Skipping invalid sentence at index {i}: {repr(s)}")
                continue
            cleaned.append(str(s).strip())
        return cleaned

    def _get_token_cache_key(self, sentences: List[str], pooling: str) -> str:
        """生成 tokenization 缓存 key（基于内容哈希）"""
        import hashlib
        content = "|".join(sentences[:100]) + f"|{len(sentences)}|{pooling}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _estimate_max_batch_size(
        self,
        sentences: Optional[List[str]] = None,
        encoded_input: Optional[Dict] = None,
        layer: int = -1,
        pooling: str = "content_mean",
        max_memory_ratio: float = 0.75
    ) -> int:
        """估算最大 batch_size（支持多 GPU）"""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return min(32, len(sentences) if sentences else 32)  # CPU 安全值

        # ===== 多 GPU 感知 =====
        if hasattr(self.model, "device_map") and self.model.device_map:
            # device_map 情况：估算最紧张的 GPU
            device_mem_used = {}
            for name, device in self.model.device_map.items():
                if str(device) not in device_mem_used:
                    device_mem_used[str(device)] = 0
            # Simplified: assume uniform load
            total_gpus = len(set(str(d) for d in self.model.device_map.values() if "cuda" in str(d)))
            max_mem_per_gpu = torch.cuda.get_device_properties(0).total_memory // total_gpus
        else:
            # Single GPU or DDP
            max_mem_per_gpu = torch.cuda.get_device_properties(0).total_memory

        available_mem = int(max_mem_per_gpu * max_memory_ratio * 0.8)  # extra 20% safety

        # Try batch_size=2
        test_batch = sentences[:2] if sentences else None
        if test_batch is None and encoded_input is not None:
            test_input = {
                k: v[:2].to(self.device) for k, v in encoded_input.items()
            }
        else:
            test_input = self.tokenizer(
                text=test_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

        torch.cuda.reset_peak_memory_stats()
        init_mem = torch.cuda.memory_allocated()
        
        try:
            with torch.no_grad():
                _ = self._run_batch_from_input(test_input, layer=layer, pooling=pooling)
            peak_mem = torch.cuda.max_memory_allocated()
            mem_per_batch = peak_mem - init_mem
            del _
        except Exception as e:
            logger.error(f"Test batch failed: {e}")
            return 1

        if mem_per_batch <= 1e6:  # <1MB? unlikely
            return min(256, len(sentences) if sentences else 256)
        
        # # Q: 這樣 estimated_max_batch 不是最大是1? 這樣 會用到的batch_size 不就最多是1? 這樣不是很沒效率？
        # # A: 您的问题非常关键——这确实是一个严重逻辑错误！让我们仔细分析问题并提供修复方案。
        # estimated_max_batch = max(1, int(available_mem // mem_per_batch))
        # return min(estimated_max_batch, len(sentences) if sentences else 256, 256)
        #   问题出在：当 available_mem // mem_per_batch < 1 时，int() 截断为0，然后 max(1, 0) = 1。
        #   根本问题不是 max(1, ...)，而是内存估算过于保守。

        # # ✅ 正确且简单的修复
        # estimated_max_batch = int(available_mem // mem_per_batch)
        # # 先限制上限，再确保至少为1
        # batch_size = min(estimated_max_batch, len(sentences) if sentences else 256, 256)
        # batch_size = max(1, batch_size)  # 最后确保≥1
        # return batch_size

        # 🐞 问题：mem_per_batch 可能被高估（batch_size=2包含额外开销）
        # ✅ 修复：用更准确的 per-sample 计算
        mem_per_sample = mem_per_batch / 2.0  # 因为测试用的是batch_size=2
        estimated_max_batch = int(available_mem // mem_per_sample)
        
        # 先限制上限，再确保下限
        batch_size = min(estimated_max_batch, len(sentences) if sentences else 256, 256)
        return max(1, batch_size)  # 安全兜底

    # --------------------------------------------------------------- #
    #   Q_Gem2: 可是強制把 hidden_states 和 attention_mask 都搬到 CPU 會不會效率很慢？
    #           這個code 要兼顧各種可能的大語言模型, 像是1B ~ 70B
    #   Gemini answer to Q_Gem2:
    #   最佳解決方案：智能對齊 (Smart Alignment)
    #   我們不預設使用 CPU，而是採取以下邏輯：
    #   1. 檢測輸出在哪：看 hidden_states 在哪個設備 (例如 cuda:1)。
    #   2. 搬運小變數：把 attention_mask (很小) 搬去跟 hidden_states (很大) 同一個 GPU。
    #       這樣就不會報錯，且不用搬運大矩陣，速度最快。
    #   3. OOM 兜底：如果上述操作導致顯存不足 (OOM)，再 自動降級到 CPU 處理。
    #
    #   修改後的 _run_batch (兼容 1B ~ 70B)
    #   請將此函數替換入你的 ModelEmbedding 類別中：

    def _run_batch(
        self,
        sentences: List[str],
        layer: int = -1,
        pooling: str = "content_mean",
        **kwargs
        ) -> torch.Tensor:
        """
        自適應 Batch 處理：
        1. 1B-7B 模型 -> 全程 GPU (速度最快)
        2. 70B 模型 -> 自動對齊設備 (避免 Device Mismatch)
        3. 顯存不足 -> 自動降級到 CPU (保證不崩潰)
        """
        # --- 1. 清洗數據 ---
        if isinstance(sentences, str): sentences = [sentences]
        cleaned = [str(s).strip() for s in sentences if str(s).strip()]
        if not cleaned: raise ValueError("No valid sentences.")

        # ===== Tokenize ===== #
        # 注意：對於 device_map="auto"，input 放在第一張卡通常是安全的，
        # 模型會自動將 Tensor 傳遞到後續設備。
        
        # --- 2. Tokenize (Input 通常放在 cuda:0) ---
        # 注意：對於多卡模型，PyTorch 會自動處理 Input 傳遞，這裡放在 self.device 即可
        encoded_input = self.tokenizer(
            text=cleaned,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,  # 防止极端长句
        ).to(self.device)

        # --- 3. Forward Pass ---
        try:
            with torch.no_grad():
                # 70B 建議關閉 autocast 或保持默認，視具體情況而定
                output = self.model(**encoded_input, output_hidden_states=True)
            
            hidden_states_ALL = output.hidden_states
            target_layer = len(hidden_states_ALL) + layer if layer < 0 else layer
            
            # 取得原始輸出 (這個 Tensor 可能在 cuda:0, 也可能在 cuda:1...)
            # 這是 70B 最大的變數，不要急著 .to(cpu)
            hidden_states = hidden_states_ALL[target_layer]

            # 釋放模型輸出的其他部分，只留需要的層
            del output
            # torch.cuda.empty_cache() # 頻繁調用會變慢，只在 OOM 處理時調用

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(f"OOM during Forward Pass: {e}")
            raise e

        # --- 4. 智能 Pooling (嘗試 GPU -> 失敗轉 CPU) ---
        try:
            return self._smart_pooling(hidden_states, encoded_input, pooling)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("⚠️ GPU Pooling OOM. Falling back to CPU pooling...")
                torch.cuda.empty_cache()
                
                # --- Fallback: CPU Mode ---
                # 這是最後手段，雖然慢但保證能跑
                hs_cpu = hidden_states.cpu().float()
                input_cpu = {k: v.cpu() for k, v in encoded_input.items()}
                return self._smart_pooling(hs_cpu, input_cpu, pooling)
            else:
                raise e

    def _smart_pooling(self, hidden_states, encoded_input, pooling):
        """
        輔助函數：執行 Pooling，自動處理設備對齊
        """
        # 關鍵：以 hidden_states 的設備為準！
        # 因為 hidden_states 體積最大，搬運它成本最高。
        # 我們把 mask (小) 搬過去配合它。
        target_device = hidden_states.device
        
        mask = encoded_input["attention_mask"]
        if mask.device != target_device:
            mask = mask.to(target_device)
            
        # 如果需要 input_ids (例如 content_mean 需要過濾 token)，也搬過去
        input_ids = encoded_input.get("input_ids")
        if input_ids is not None and input_ids.device != target_device:
            input_ids = input_ids.to(target_device)

        if pooling in ("mean", "content_mean"):
            return mean_pooling(
                hidden_states,
                attention_mask=mask,
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                exclude_special_tokens=(pooling == "content_mean")
            )
        
        elif pooling == "cls":
            return hidden_states[:, 0]
            
        elif pooling == "last":
            seq_lengths = mask.sum(dim=1) - 1
            # 確保索引也在同一設備
            indices = torch.arange(hidden_states.size(0), device=target_device)
            return hidden_states[indices, seq_lengths]
            
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

    # --------------------------------------------------------------- #
    #
    #   Qwen3-Max answer to Q2
    #
    #   🔁 替换旧接口（兼容性处理）
    #   ✅ 保留旧方法（但标记 deprecated）
    #
    def get_sentences_embeddings(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "`get_sentences_embeddings` is deprecated. Use `get_embeddings` instead.",
            FutureWarning
        )
        return self.get_embeddings(*args, **kwargs, return_numpy=False)

    def single_embedding(self, sentence: str, **kwargs):
        import warnings
        warnings.warn(
            "`single_embedding` is deprecated. Use `get_embeddings([sentence])` instead.",
            FutureWarning
        )
        emb = self.get_embeddings([sentence], **kwargs, return_numpy=True)
        return emb[0]  # squeeze first dim


    # --------------------------------------------------------------- #
    def aggregate_hidden_states(self, hidden_states: torch.Tensor, pooling: str):
        """
        统一使用 mean_pooling 替代旧逻辑（兼容旧 pooling 名）

        Aggregate hidden states based on the specified pooling method.

        根據指定的池化方法聚合隱藏狀態

        Args:
            hidden_states: 模型輸出的隱藏狀態
            pooling: 池化方法 ('mean', 'extract_mean', 'cls', 'last')

        Returns:
            numpy array: 聚合後的embedding向量
        """
        # tokens_list = self.tokenizer.convert_ids_to_tokens(self.tokens['input_ids'][0])

        # 兼容旧参数名：'mean' → 'content_mean'（更合理），'mean_w_mask' → 'mean'
        if pooling == "mean":
            # 旧 "mean" → 现在等价于 "content_mean"
            pooling = "content_mean"
        elif pooling == "mean_w_mask":
            pooling = "mean"  # pure attention_mask-based

        if pooling in ("mean", "content_mean"):
            emb = mean_pooling(
                hidden_states,
                attention_mask=self.tokens["attention_mask"],
                input_ids=self.tokens["input_ids"],
                tokenizer=self.tokenizer,
                exclude_special_tokens=(pooling == "content_mean")
            )
            return emb.squeeze().cpu().numpy()

        elif pooling == "extract_mean":
            # 保留你原有的 special_token_span 逻辑（未提供，暂略）
            # sentence = self.tokenizer.decode(
            #     self.tokens["input_ids"][0], skip_special_tokens=True
            # )
            # This_layer = special_token_span(
            #     self.tokenizer, self.tokens, hidden_states, sentence
            # )
            # This_layer = special_token_span(...)
            # return This_layer.mean(dim=0).cpu().numpy()
            raise NotImplementedError("extract_mean not implemented here")

        elif pooling == "cls":
            # 使用[CLS] token (適用於BERT類模型)
            return hidden_states[:, 0, :].squeeze().cpu().numpy()
        elif pooling == "last":
            return hidden_states[:, -1, :].squeeze().cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    # --------------------------------------------------------------- #
    def show_config_info(self):
        config = self.model.config
        num_hidden_layers = config.num_hidden_layers
        hidden_size = config.hidden_size

        print("\n ----------- hidden layers info ------------------- ")
        print(f" number of hidden layer = {self.model.config.num_hidden_layers}")
        print(f" hidden_size            = {hidden_size}")
        # print(f' n_layer = {self.model.config.n_layer}')    # 'BertConfig' object has no attribute 'n_layer'

        print("")
        return num_hidden_layers, hidden_size

    def Print_decode_token(self):
        tokens_list = self.tokenizer.convert_ids_to_tokens(self.tokens["input_ids"][0])
        sentence = self.tokenizer.decode(
            self.tokens["input_ids"][0], skip_special_tokens=True
        )

        print(f"tokens_list  = {tokens_list}")
        print(f"sentence     = {sentence}")

    def show_token_info(self):

        if self.process == "single":
            print(f". sentence = {self.sentence}")
            print(f"  length of sentence = {len(self.sentence)}")

        print(" -----------      tokenizer     --------------")
        print(f"tokens['input_ids']             = {self.tokens['input_ids']}")
        print(f"tokens['attention_mask']        = {self.tokens['attention_mask']}")
        print(f"tokens['input_ids'].shape       = {self.tokens['input_ids'].shape}")
        print(
            f"tokens['attention_mask'].shape  = {self.tokens['attention_mask'].shape}"
        )

    def Print_hidden(self):

        for idx, layer in enumerate(self.hs_ALL):
            print(f" --------    idx {idx}   ------------- ")
            print(layer)

# ----------------------------------------------------------------------------- #
#     Main function to run the script and get embeddings from the brain data    #
# ----------------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
#   Q_Gem1: 為何大size 70B 的大語言模型, 一直將batch_size 調到1 了，
#           還是一直得到None 的df_emb
#   Gemini answer to Q_Gem1:
#   這是一個非常典型的 70B 大模型 + 多 GPU (device_map="auto") 遇到的問題。
#   2. 修改 get_embedding_sentence (顯示錯誤原因)
#       修改這部分代碼，確保如果所有嘗試都失敗，你會知道原因，而不是只得到 None。
#
def get_embedding_sentence(
    stimset,
    Mod_emb: ModelEmbedding,
    pooling, 
    layer,
    FileSave,
    batch_size=None, # <--- 3. 接收參數
    **kwargs
):
    sentences = stimset["sentence"].values.tolist()
    
    # ===== 关键升级：自动批处理 ===== #
    t1 = time.time()

    print(f"\n🔍 Processing layer {layer}")

    # 初始化
    embeddings = None
    df_emb = None
    df_embeddings = None
    
    # 第一次嘗試
    try:
        embeddings = Mod_emb.get_embeddings(
            sentences,
            layer=layer,
            pooling=pooling,
            batch_size=batch_size, # <--- 4. 傳入 ModelEmbedding
            max_memory_ratio=0.6,  # ⚠️ 更保守！70B用0.6
            return_numpy=True
        )
    except Exception as e:
        logger.error(f"❌ Initial attempt failed at layer {layer}: {e}")
        logger.warning("⚠️ Switching to RESCUE MODE: batch_size=1")
        
        # 第二次嘗試 (Rescue Mode)
        try:
            embeddings = Mod_emb.get_embeddings(
                sentences,
                layer=layer,
                pooling=pooling,
                batch_size=1, # 強制為 1
                max_memory_ratio=0.5,
                return_numpy=True
            )
            logger.info(f"✅ Rescue successful for layer {layer}")
            
        except Exception as e2:
            # 這是重點：如果這裡也失敗了，我們要印出致命錯誤
            logger.critical(f"❌❌ FATAL ERROR at layer {layer}: {e2}")
            # 強烈建議：這裡可以考慮 raise e2，讓程式停下來，而不是默默回傳 None
            # raise e2 
            return None, None # 如果你不想中斷，就回傳 None

    t2 = time.time()
    print(f"\n⏱️ Total embedding time: {t2 - t1:.2f}s for {len(sentences)} sentences")

    # 如果成功獲取 embeddings
    if embeddings is not None:
        try:
            # 檢查是否全為 0 (這是另一個常見 bug)
            if np.all(embeddings == 0):
                logger.warning("⚠️ WARNING: Returned embeddings are all ZEROS!")

            df_emb = pd.DataFrame(embeddings, index=stimset.index)
            df_embeddings = pd.concat([stimset, df_emb], axis=1)
            
            # 保存
            os.makedirs(os.path.dirname(FileSave), exist_ok=True)
            df_embeddings.to_csv(FileSave, index=False, encoding="utf-8-sig")
            logger.info(f"✅ Saved {FileSave}")
            
            return df_emb, df_embeddings

        except Exception as e:
            logger.error(f"❌ Error during DataFrame creation/saving: {e}")
            return None, None
    else:
        logger.error("❌ Embeddings is None after all attempts.")
        return None, None


#
#          data to embedding
#

def run_llm_and_save(
    model_config,
    DirSave,
    stimset,
    layer_list=None,
    device="Auto",
    manual_batch_size=None,
):
    """
    Load LLM, run inference for specified layers, save each layer's embeddings to CSV.

    Args:
        model_config:       dict loaded from model config JSON
        DirSave:            directory to save CSV files
        stimset:            DataFrame with 'sentence' column and meaningful index
        layer_list:         list of layer indices to extract;
                            None = all layers (0 .. num_hidden_layers)
        device:             'Auto', 'cpu', or 'cuda:N'
        manual_batch_size:  override auto batch-size detection (useful for 70B models)

    Returns:
        Mod_emb, num_hidden_layers, hidden_size, device, used_devices, gpu_usage
    """
    source_model = model_config["model_name"]
    pooling      = model_config["AutoModel_config"]["pooling"]

    Mod_emb = ModelEmbedding(device=device, **model_config)
    num_hidden_layers, hidden_size = Mod_emb.show_config_info()

    if layer_list is None:
        layer_list = model_config["AutoModel_config"].get(
            "layer_list", range(0, num_hidden_layers + 1)
        )

    for layer in layer_list:
        FileSave = os.path.join(DirSave, f"{source_model}_emb_L{layer}_{pooling}_utf8.csv")
        os.makedirs(os.path.dirname(FileSave), exist_ok=True)
        logger.info(f"LLM FileSave = {FileSave}")

        get_embedding_sentence(
            stimset, Mod_emb, pooling, layer, FileSave,
            batch_size=manual_batch_size,
        )

    device      = Mod_emb.device
    used_devices = Mod_emb.used_devices
    gpu_usage   = check_gpu_usage()

    return Mod_emb, num_hidden_layers, hidden_size, device, used_devices, gpu_usage


def load_embeddings_csv(
    DirSave,
    model_config,
    layer,
    stimset_index,
):
    """
    Load pre-computed embeddings from CSV. Does NOT load the LLM.

    Args:
        DirSave:        directory containing CSV files (same path used in run_llm_and_save)
        model_config:   dict loaded from model config JSON (for filename & metadata)
        layer:          layer index to load
        stimset_index:  pandas Index to assign to df_emb rows (e.g. stimset.index)

    Returns:
        df_emb: DataFrame of shape (n_sentences, hidden_size)
        meta:   dict with keys: num_hidden_layers, hidden_size, LLM_info, FileSave
    """
    source_model      = model_config["model_name"]
    pooling           = model_config["AutoModel_config"]["pooling"]
    num_hidden_layers = model_config["num_hidden_layers"]
    hidden_size       = model_config["hidden_size"]

    FileSave = os.path.join(DirSave, f"{source_model}_emb_L{layer}_{pooling}_utf8.csv")
    logger.info(f"Loading embeddings from: {FileSave}")

    df_sent_emb = pd.read_csv(FileSave, encoding="utf-8-sig")
    df_emb = df_sent_emb.iloc[:, 2:]
    df_emb.index = stimset_index
    df_emb.columns = df_emb.columns.astype(int)

    logger.info(f"Loaded embeddings: shape={df_emb.shape}")

    LLM_info = f"{source_model}, L={layer}/{num_hidden_layers}, hs={hidden_size}"
    meta = {
        "num_hidden_layers": num_hidden_layers,
        "hidden_size":       hidden_size,
        "LLM_info":          LLM_info,
        "FileSave":          FileSave,
    }
    return df_emb, meta


def check_gpu_usage():

    gpu_usage = {}

    # 1. 取得可見的 GPU 數量 (即邏輯數量)
    num_gpus = torch.cuda.device_count()
    print(f"程式可見 {num_gpus} 張 GPU (邏輯 ID 從 0 到 {num_gpus - 1})")

    print("\n--- 記憶體使用報告 ---")
    for i in range(num_gpus):
        # 設置當前要查詢的設備為邏輯 ID (cuda:0, cuda:1, ...)
        torch.cuda.set_device(i)

        # 取得當前程序已分配的總記憶體 (單位：位元組 Byte)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)

        # 取得當前程序已快取的總記憶體 (單位：位元組 Byte)
        # PyTorch 會保留一些已釋放的記憶體供未來使用 (Cached)
        cached = torch.cuda.memory_reserved(i) / (1024**3)

        # 取得 GPU 總容量 (單位：位元組 Byte)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)

        print(f"cuda:{i} (邏輯 ID):")
        print(f"  總容量: {total_mem:.2f} GiB")
        print(f"  已分配 (實際使用): {allocated:.2f} GiB")
        print(f"  已預留/快取: {cached:.2f} GiB")

        gpu_usage[i] = {
            "total_mem": total_mem,
            "allocated": allocated,
            "cached": cached,
        }
    return gpu_usage

    # 請注意：這些數字只反映了您當前 PyTorch 程序的使用情況，不包含其他程序佔用的記憶體。
