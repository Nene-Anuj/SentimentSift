#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Café sentiment pipeline (run‑and‑go version)
-------------------------------------------------
Run simply with:
    python cafe_sentiment_pipeline.py
or override defaults:
    python cafe_sentiment_pipeline.py --reviews my_reviews.json --output_dir out
"""

import os
import math
import json
import sys
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# ------------------------ DEFAULT PATHS ------------------------ #
DEFAULT_REVIEWS = os.path.join("data", "processed", "reviews.json")
DEFAULT_OUTDIR  = os.path.join("data", "sentiment")

# ------------------------ Like weight ------------------------ #
def like_weight(like_cnt: int) -> float:
    x = max(like_cnt, 0)
    return 1.0 + 0.6 * (x > 0) + math.sqrt(x) * 0.3

# ------------------------ Logistic z scale ------------------------ #
def logistic_z_scale(values: List[float], k: float = 1.702,
                     eps: float = 1e-8) -> List[float]:
    arr = np.asarray(values, float)
    z = (arr - arr.mean()) / (arr.std(ddof=0) + eps)
    s = 1.0 / (1.0 + np.exp(-k * z))
    return (10.0 * s).tolist()

# ------------------------ Tier helpers ------------------------ #
def percentile_tier3(arr: np.ndarray) -> List[str]:
    p33, p67 = np.percentile(arr, [33.33, 66.67])
    return np.where(arr >= p67, "Tier‑1",
           np.where(arr >= p33, "Tier‑2", "Tier‑3")).tolist()

def z_tier3(scores: List[float], thr: float = 0.7, eps: float = 1e-8) -> List[str]:
    arr = np.asarray(scores, float)
    sd = arr.std(ddof=0)
    if sd < eps:
        return percentile_tier3(arr)
    z = (arr - arr.mean()) / sd
    hi = np.mean(z >= thr); lo = np.mean(z <= -thr)
    if hi == 0 or lo == 0:
        return percentile_tier3(arr)
    return np.where(z >=  thr, "Tier‑1",
           np.where(z <= -thr, "Tier‑3", "Tier‑2")).tolist()

# ------------------------ Sentiment classifier ------------------------ #
def classify_sentiment(score: float) -> str:
    """将0-1之间的情感分数分类为good、neutral或bad"""
    if score >= 0.67:
        return "good"
    elif score <= 0.33:
        return "bad"
    else:
        return "neutral"

# ------------------------ Sentiment analyzer ------------------------ #
class TextOnlySentimentAnalyzer:
    def __init__(self,
                 model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        print(f"Loading {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        self.aspect_keywords: Dict[str, List[str]] = {
            "service": ["service", "staff", "waiter", "friendly", "rude",
                        "attentive", "helpful", "slow", "barista", "manager"],
            "food": ["coffee", "food", "pastry", "taste", "delicious", "menu",
                     "portion", "price", "expensive", "cheap", "fresh"],
            "ambiance": ["ambiance", "atmosphere", "decor", "environment",
                         "cozy", "loud", "clean", "dirty", "music", "lighting"]
        }

    def extract_aspect_sentences(self, text: str, aspect: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        kws = self.aspect_keywords[aspect]
        return [s for s in sentences if any(kw in s.lower() for kw in kws)]

    def analyze_sentiment(self, text: str) -> float:
        if not text:
            return 0.3
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = softmax(logits.cpu().numpy(), axis=1)[0]
        return float(probs[1])  # positive prob 0‑1

    def aspect_score(self, review_text: str, aspect: str,
                     like_cnt: int = 0) -> Tuple[Optional[float], Optional[str]]:
        aspect_sents = self.extract_aspect_sentences(review_text, aspect)
        if not aspect_sents:
            return None, None
        scores, weights = [], []
        sent_classes = []
        for s in aspect_sents:
            s_score = self.analyze_sentiment(s)
            # 分类每个句子
            sent_class = classify_sentiment(s_score)
            sent_classes.append(sent_class)
            
            conf = abs(s_score - 0.5) * 2.0
            w = like_weight(like_cnt) * (0.5 + conf)
            scores.append(s_score); weights.append(w)
        
        if not weights or sum(weights) == 0:
            return None, None
            
        weighted = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        # 计算每种分类的比例
        sentiment_counts = {
            "good": sent_classes.count("good"),
            "neutral": sent_classes.count("neutral"),
            "bad": sent_classes.count("bad")
        }
        
        # 返回加权分数和情感分类比例
        return weighted * 10.0, sentiment_counts

# ------------------------ Pipeline ------------------------ #
def run_pipeline(reviews_path: str, output_dir: str, thr: float = 0.7) -> None:
    if not os.path.exists(reviews_path):
        print(f"Error: reviews file not found → {reviews_path}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    analyzer = TextOnlySentimentAnalyzer()
    with open(reviews_path, "r", encoding="utf-8") as f:
        all_reviews = json.load(f)

    biz_reviews: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_reviews:
        bid = r.get("business_id")
        if bid:
            biz_reviews.setdefault(bid, []).append(r)

    dims = ["service", "food", "ambiance"]
    biz_raw = {bid: {d: [] for d in dims} for bid in biz_reviews}
    # 为每个维度存储情感分类计数
    biz_sentiment_counts = {bid: {
        d: {"good": 0, "neutral": 0, "bad": 0} for d in dims
    } for bid in biz_reviews}
    # 为每个维度记录存在评论的总数
    biz_aspect_counts = {bid: {d: 0 for d in dims} for bid in biz_reviews}

    print(f"Analyzing {len(biz_reviews)} businesses …")
    for bid, revs in tqdm(biz_reviews.items()):
        for rev in revs:
            text = rev.get("review_text", "")
            likes = int(rev.get("likes", 0) or 0)
            for d in dims:
                score_and_counts = analyzer.aspect_score(text, d, likes)
                if score_and_counts[0] is not None:
                    biz_raw[bid][d].append(score_and_counts[0])
                    
                    # 更新情感分类计数
                    if score_and_counts[1] is not None:
                        # 我们不增加biz_aspect_counts[bid][d]，因为我们现在计算的是句子级别的情感
                        # 而不是评论级别的情感
                        for cls in ["good", "neutral", "bad"]:
                            biz_sentiment_counts[bid][d][cls] += score_and_counts[1][cls]

    # 计算每个维度的平均分数
    biz_avg = {bid: {d: (np.mean(v) if v else 5.0)
                     for d, v in dim_dict.items()}
               for bid, dim_dict in biz_raw.items()}

    # 计算每个维度的情感比例
    biz_sentiment_percentages = {}
    for bid in biz_reviews:
        biz_sentiment_percentages[bid] = {}
        for d in dims:
            # 计算该维度的总句子数
            total_sents = sum(biz_sentiment_counts[bid][d].values())
            if total_sents > 0:
                for cls in ["good", "neutral", "bad"]:
                    key = f"{d}_{cls}_pct"
                    count = biz_sentiment_counts[bid][d][cls]
                    # 确保百分比相加为1.0
                    biz_sentiment_percentages[bid][key] = round(count / total_sents, 4)
            else:
                # 如果没有该维度的评论，则设置为默认值
                for cls in ["good", "neutral", "bad"]:
                    key = f"{d}_{cls}_pct"
                    biz_sentiment_percentages[bid][key] = 0.0
                # 如果没有评论，将neutral设为1.0（默认值）
                biz_sentiment_percentages[bid][f"{d}_neutral_pct"] = 1.0

    values = {d: [biz_avg[bid][d] for bid in biz_reviews] for d in dims}
    scaled = {d: logistic_z_scale(vals) for d, vals in values.items()}
    tiers = {d: z_tier3(scaled[d], thr) for d in dims}
    ids = list(biz_reviews)
    idx_map = {bid: i for i, bid in enumerate(ids)}

    tier_stats = {d: {"Tier‑1": 0, "Tier‑2": 0, "Tier‑3": 0} for d in dims}
    results = []
    for bid in ids:
        i = idx_map[bid]
        item = {"business_id": bid, "scores": {}, "sentiment_percentages": {}}
        
        # 添加分数和层级
        for d in dims:
            sc = round(float(scaled[d][i]), 2)
            tr = tiers[d][i]
            item["scores"][d] = sc
            item["scores"][f"{d}_tier"] = tr
            tier_stats[d][tr] += 1
        
        # 添加情感百分比
        for d in dims:
            for cls in ["good", "neutral", "bad"]:
                key = f"{d}_{cls}_pct"
                item["sentiment_percentages"][key] = biz_sentiment_percentages[bid][key]
        
        results.append(item)

    out_file = os.path.join(output_dir, "cafe_sentiment_with_tiers.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nTier distribution per dimension:")
    for d in dims:
        total = sum(tier_stats[d].values())
        print(f"\n{d.capitalize()}:")
        for t in ("Tier‑1", "Tier‑2", "Tier‑3"):
            cnt = tier_stats[d][t]
            print(f"  {t}: {cnt} cafés ({cnt / total * 100:.1f}%)")

    print(f"\nResult saved to: {out_file}")

# ------------------------ CLI ------------------------ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cafe sentiment pipeline")
    parser.add_argument("--reviews", type=str, default=DEFAULT_REVIEWS,
                        help=f"Path to reviews.json (default {DEFAULT_REVIEWS})")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTDIR,
                        help=f"Directory to save JSON (default {DEFAULT_OUTDIR})")
    parser.add_argument("--thr", type=float, default=0.7,
                        help="z‑score threshold for Tier‑1/3")
    args = parser.parse_args()
    run_pipeline(reviews_path=args.reviews,
                 output_dir=args.output_dir,
                 thr=args.thr)