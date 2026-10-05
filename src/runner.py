"""MMLU Benchmark Pipeline 循序推論執行模組（Step 5）。

串接 DataLoader -> Model -> Evaluator 三段，並嚴格遵循單一方向
管線生命週期：

1. 一次性載入：樣本由呼叫端以 ``MMLUDatasetLoader.get_samples()``
   預先一次性載入；本模組不重覆讀取資料。
2. 循序模型迴圈：依設定逐模型執行（``for model in models``），
   嚴禁非同步／多執行緒／多處理，防止 VRAM OOM 與雲端限流放大。
3. 逐題循序推論：以 ``model.predict_batch([sample])``（batch=1）
   取得三鍵預測紀錄；latency 量測與 ``ERROR:`` 哨兵防禦為模型層
   唯一職責，本模組不重複量測。
4. 速率控制：兩筆連續請求之間休眠 ``request_delay`` 秒
   （最後一題之後不休眠）。
5. 資源釋放：每模型迴圈結束後（含異常中断路徑）以
   :func:`release_vram` 釋放 CUDA 快取。
6. 評估對接：將預測交予 ``Evaluator.evaluate()`` 取得
   ``EvaluationResult``。

本模組不執行答案解析、正則過濾或指標計算（職責屬
``src/evaluator.py``），亦不讀取 ``target_letter``（防答案鍵滲漏）。
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

from tqdm import tqdm  # type: ignore[import-untyped]

from src.evaluator import Evaluator, EvaluationResult
from src.models import BaseModelInterface, build_model_interface

LOGGER = logging.getLogger(__name__)

#: 模型層預測紀錄三鍵契約（對齊 ``BaseModelInterface.predict_batch``）。
PREDICTION_RECORD_KEYS: frozenset[str] = frozenset(
    {"question_id", "raw_output", "latency"}
)


def release_vram() -> None:
    """釋放 CUDA 快取，為載入下一個模型準備。

    僅在 CUDA 可用時呼叫 ``torch.cuda.empty_cache()``；torch 未安裝、
    無 GPU 環境或釋放失敗時，最多記錄警告，絕不中斷管線。

    Note:
        本函式為防禦性資源管理，不拋出任何例外。
    """
    try:
        import torch
    except ImportError:
        LOGGER.debug("torch is not installed; skipping VRAM release.")
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.warning(
            "Failed to release the CUDA cache; continuing the pipeline.",
            exc_info=True,
        )


@dataclass(frozen=True)
class ModelRunOutput:
    """單一模型之推論與評估產出。

    Attributes:
        model_name: 模型名稱（``BaseModelInterface.model_name``）。
        predictions: 三鍵預測紀錄清單（與輸入樣本等長同序），每筆含
            ``question_id`` / ``raw_output`` / ``latency``。
        evaluation: ``Evaluator.evaluate()`` 結果（完整指標）。
        total_seconds: 推論迴圈牆鐘總耗時（秒；含 ``request_delay``
            休眠；不含模型建構、資源釋放與評估時間）。
    """

    model_name: str
    predictions: List[Dict[str, object]]
    evaluation: EvaluationResult
    total_seconds: float


class BenchmarkRunner:
    """循序推論編排器：串接 DataLoader -> Model -> Evaluator。

    Attributes:
        evaluator: 評估器實體（純計算、無副作用）。
        request_delay: 兩筆單題推論請求之間的休眠秒數（雲端限流控制）；
            ``0.0`` 為不休眠。

    Example:
        >>> runner = BenchmarkRunner(Evaluator(), request_delay=0.5)
        >>> outputs = runner.run_benchmark(samples, models_cfg, eval_cfg)
    """

    def __init__(self, evaluator: Evaluator, *, request_delay: float = 0.0) -> None:
        """初始化並執行建構期 fail-fast 驗證。

        Args:
            evaluator: 評估器實體（必須為 ``Evaluator`` 實例）。
            request_delay: 單題推論請求間之休眠秒數；必須為有限非負
                數值（拒絕 bool、非數值、NaN/Inf 與負值）；``0.0`` 表示
                不休眠。

        Raises:
            TypeError: 當 ``evaluator`` 非 ``Evaluator`` 實例時。
            ValueError: 當 ``request_delay`` 為 bool、非數值、NaN/Inf
                或負值時。
        """
        if not isinstance(evaluator, Evaluator):
            raise TypeError(
                "evaluator must be an Evaluator instance, "
                f"got {type(evaluator).__name__}."
            )
        if isinstance(request_delay, bool) or not isinstance(
            request_delay, (int, float)
        ):
            raise ValueError(
                "request_delay must be a finite non-negative number, "
                f"got {request_delay!r}."
            )
        if math.isnan(request_delay) or math.isinf(request_delay):
            raise ValueError(f"request_delay must be finite, got {request_delay!r}.")
        if request_delay < 0:
            raise ValueError(f"request_delay must be >= 0, got {request_delay!r}.")
        self.evaluator: Evaluator = evaluator
        self.request_delay: float = float(request_delay)

    def run_model(
        self,
        model: BaseModelInterface,
        samples: Sequence[Mapping[str, object]],
    ) -> ModelRunOutput:
        """執行單模型之逐題循序推論迴圈並交予評估器。

        執行流程（單執行緒、嚴格循序）：
            1. 逐樣本呼叫 ``model.predict_batch([sample])``（batch=1）
               取得三鍵紀錄，並驗證模型層契約（fail-fast）。
            2. 非最後一題之後休眠 ``request_delay`` 秒
               （``0.0`` 時完全不呼叫休眠）。
            3. 以 tqdm 輸出進度（``unit="q"``）；非 TTY 環境下 tqdm
               自動降級為純文字行，不影響管線。
            4. 迴圈結束後（含異常中断路徑，以 try/finally 保證）呼叫
               :func:`release_vram`。
            5. 呼叫 ``evaluator.evaluate(samples, predictions,
               model_name=model.model_name)``。

        Args:
            model: 模型推論介面（必須暴露 ``model_name`` 與
                ``predict_batch``）。
            samples: 預載樣本清單（``MMLUDatasetLoader.get_samples()``
                輸出）；本方法不修改樣本內容，且不讀取
                ``target_letter``。

        Returns:
            ``ModelRunOutput``：預測紀錄＋評估結果＋總耗時。

        Raises:
            RuntimeError: 當模型層違反契約（``predict_batch`` 未恰好
                回傳 1 筆紀錄、紀錄缺必要鍵、或 ``latency`` 非有限數值）
                時。
        """
        total: int = len(samples)
        predictions: List[Dict[str, object]] = []
        start_wall: float = time.perf_counter()
        progress = tqdm(
            samples,
            total=total,
            desc=f"model:{model.model_name}",
            unit="q",
            mininterval=0.5,
        )
        try:
            for index, sample in enumerate(progress):
                predictions.append(self._predict_single(model, sample))
                if index < total - 1 and self.request_delay > 0.0:
                    time.sleep(self.request_delay)
        finally:
            release_vram()
        total_seconds: float = time.perf_counter() - start_wall
        evaluation: EvaluationResult = self.evaluator.evaluate(
            samples, predictions, model_name=model.model_name
        )
        LOGGER.info(
            "Model %s inference complete: samples=%d accuracy=%.4f "
            "avg_latency=%.4fs total_wall=%.2fs",
            model.model_name,
            total,
            evaluation.overall_accuracy,
            evaluation.average_latency,
            total_seconds,
        )
        return ModelRunOutput(
            model_name=model.model_name,
            predictions=predictions,
            evaluation=evaluation,
            total_seconds=total_seconds,
        )

    def run_benchmark(
        self,
        samples: Sequence[Mapping[str, object]],
        model_configs: Sequence[Mapping[str, object]],
        evaluation_config: Mapping[str, object],
    ) -> List[ModelRunOutput]:
        """依模型清單循序執行完整 Benchmark（對齊管線生命週期）。

        依序對每個 ``models[]`` 設定區塊：以 ``build_model_interface``
        建構模型介面 -> 執行 :meth:`run_model`（逐題循序推論 -> 釋放
        VRAM -> 評估對接）。模型建構與推論皆為串行；同一份樣本
        全程重用，不重覆載入。

        Args:
            samples: 預載樣本清單（呼叫端已以
                ``MMLUDatasetLoader.get_samples()`` 完成一次性載入）。
            model_configs: ``configs/eval_config.yaml`` 之 ``models[]``
                設定區塊清單。
            evaluation_config: ``evaluation:`` 全域設定區塊（透傳給工廠
                以讀取 retry 等執行參數）。

        Returns:
            每模型之結果清單（與 ``model_configs`` 同序）。

        Raises:
            ValueError: 當模型設定區塊缺失、類型不支援或雲端 API key
                環境變數缺失時（例外自工廠原樣上拋；fail-fast、不跳過，
                避免評測報告靜默缺漏模型）。
        """
        results: List[ModelRunOutput] = []
        total_models: int = len(model_configs)
        for index, model_config in enumerate(model_configs):
            LOGGER.info(
                "Benchmark: building model %d/%d (%s).",
                index + 1,
                total_models,
                model_config.get("name", "<unnamed>"),
            )
            model: BaseModelInterface = build_model_interface(
                model_config, evaluation_config
            )
            results.append(self.run_model(model, samples))
        return results

    def _predict_single(
        self,
        model: BaseModelInterface,
        sample: Mapping[str, object],
    ) -> Dict[str, object]:
        """以模型層執行單題推論並驗證回傳紀錄契約。

        Args:
            model: 模型推論介面。
            sample: 單一樣本紀錄（原樣透傳，不修改任何欄位）。

        Returns:
            三鍵預測紀錄（``question_id`` / ``raw_output`` /
            ``latency``）。

        Raises:
            RuntimeError: 當 ``predict_batch`` 未恰好回傳 1 筆紀錄、
                紀錄缺必要鍵、或 ``latency`` 非有限數值時。
        """
        records: List[Dict[str, object]] = model.predict_batch([sample])
        if len(records) != 1:
            raise RuntimeError(
                f"Model {model.model_name!r} predict_batch returned "
                f"{len(records)} record(s) for a single sample; expected "
                "exactly 1."
            )
        record: Mapping[str, object] = records[0]
        missing: frozenset[str] = PREDICTION_RECORD_KEYS - record.keys()
        if missing:
            raise RuntimeError(
                f"Model {model.model_name!r} prediction record is missing "
                f"key(s) {sorted(missing)}; model-layer contract violated."
            )
        latency: object = record["latency"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
        ):
            raise RuntimeError(
                f"Model {model.model_name!r} prediction record latency "
                f"{latency!r} is not a finite number; model-layer contract "
                "violated."
            )
        return dict(record)
