"""
Minimal inference pipeline for radiology report generation.
Supports Stage A (image only) and Stage B (image + EHR) via multi-pass prompting
with structured JSON outputs for Impression, CheXpert, and ICD predictions.
"""

import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token

# Import our model loader
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from load_finetuned_model import load_finetuned_llava  # noqa: E402

# Label definitions matching the dataset (12 labels)
CHEXPERT = [
    "No Finding", "Enlarged Cardiomediastinum", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

# ICD-10 indicator list
ICD = [
    "Pneumonia", "Pleural_Effusion", "Pneumothorax", "Pulmonary_Edema",
    "Cardiomegaly", "Atelectasis", "Pulmonary_Embolism", "Rib_Fracture"
]


class TokenBiasProcessor(LogitsProcessor):
    """Applies additive bias to specified token ids during generation."""

    def __init__(self, token_bias: Dict[int, float]):
        self.token_bias = token_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.token_bias:
            return scores
        for token_id, bias in self.token_bias.items():
            if 0 <= token_id < scores.shape[-1]:
                scores[:, token_id] += bias
        return scores


class RadiologyInferencePipeline:
    """Minimal I/O pipeline for radiology report generation."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.conv_mode = None
        self._positive_token_ids: List[int] = []
        self._negative_token_ids: List[int] = []
        self._load_model()

    # ------------------------------------------------------------------ #
    # Helper setup

    def _load_model(self) -> None:
        print("🤖 Loading fine-tuned model...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_finetuned_llava(device=self.device)
        name_hint = str(getattr(self.model.config, "_name_or_path", "")).lower()
        model_type = str(getattr(self.model.config, "model_type", "")).lower()
        combo_name = f"{name_hint} {model_type}"
        if "llava" in combo_name or "mistral" in combo_name:
            self.conv_mode = "llava_v1"
        elif "llama" in combo_name:
            self.conv_mode = "llava_llama_2"
        else:
            self.conv_mode = "llava_v1"
        self._setup_token_bias_ids()
        print(f"💬 Using conversation template: {self.conv_mode}")
        print("✅ Model loaded successfully")

    def _collect_token_ids(self, variants: List[str]) -> List[int]:
        ids: List[int] = []
        for text in variants:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if encoded:
                ids.append(encoded[-1])
        return ids

    def _setup_token_bias_ids(self) -> None:
        positive_variants = ["1", " 1", "\"1\"", ":1", ": 1", "1}"]
        negative_variants = ["0", " 0", "\"0\"", ":0", ": 0", "0}"]
        self._positive_token_ids = list({tid for tid in self._collect_token_ids(positive_variants)})
        self._negative_token_ids = list({tid for tid in self._collect_token_ids(negative_variants)})

    # ------------------------------------------------------------------ #
    # Utility helpers

    @staticmethod
    def _str_to_bool(value: Optional[str], default: bool) -> bool:
        if value is None:
            return default
        return value.lower() in ("1", "true", "yes", "y")

    def _normalize_label_value(self, value: Any, allow_negative: bool = False) -> int:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            if allow_negative and value < 0:
                return -1
            if value > 0:
                return 1
        return 0

    def _aggregate_chexpert_votes(
        self,
        votes: List[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        result = {label: 0 for label in CHEXPERT}
        if not votes:
            if fallback:
                for label in CHEXPERT:
                    result[label] = self._normalize_label_value(fallback.get(label, 0), allow_negative=True)
            return result

        total = len(votes)
        pos_threshold = float(os.getenv("CHEXPERT_POS_VOTE_THRESHOLD", "0.34"))
        neg_threshold = float(os.getenv("CHEXPERT_NEG_VOTE_THRESHOLD", "0.34"))
        pos_required = max(1, math.ceil(total * pos_threshold))
        neg_required = max(1, math.ceil(total * neg_threshold))

        for label in CHEXPERT:
            pos_votes = sum(
                1 for v in votes if self._normalize_label_value(v.get(label, 0), allow_negative=True) > 0
            )
            neg_votes = sum(
                1 for v in votes if self._normalize_label_value(v.get(label, 0), allow_negative=True) < 0
            )
            if pos_votes >= pos_required:
                result[label] = 1
            elif neg_votes >= neg_required:
                result[label] = -1
            else:
                result[label] = 0
        return result

    def _aggregate_icd_votes(
        self,
        votes: List[Dict[str, Any]],
        fallback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        result = {label: 0 for label in ICD}
        if not votes:
            if fallback:
                for label in ICD:
                    result[label] = 1 if self._normalize_label_value(fallback.get(label, 0)) > 0 else 0
            return result

        total = len(votes)
        threshold = float(os.getenv("ICD_VOTE_THRESHOLD", "0.34"))
        required = max(1, math.ceil(total * threshold))

        for label in ICD:
            pos_votes = sum(1 for v in votes if self._normalize_label_value(v.get(label, 0)) > 0)
            result[label] = 1 if pos_votes >= required else 0
        return result

    def _make_token_bias(self, positive_bias: float, negative_bias: float) -> Dict[int, float]:
        token_bias: Dict[int, float] = {}
        for tid in self._positive_token_ids:
            token_bias[tid] = token_bias.get(tid, 0.0) + positive_bias
        for tid in self._negative_token_ids:
            token_bias[tid] = token_bias.get(tid, 0.0) + negative_bias
        return token_bias

    # ------------------------------------------------------------------ #
    # Public helpers

    def format_ehr(self, ehr_dict: Dict[str, Any]) -> str:
        keep = ["Age", "Sex", "Vitals", "Labs", "O2_device", "Chronic_conditions"]
        filtered_ehr = {k: ehr_dict.get(k, {}) for k in keep if k in ehr_dict}
        return json.dumps(filtered_ehr, indent=2)

    def _gen_params(self):
        temperature = float(os.getenv("GEN_TEMPERATURE", "0.1"))
        max_new_tokens = int(os.getenv("GEN_MAX_NEW_TOKENS", "512"))
        top_p = float(os.getenv("GEN_TOP_P", "0.9"))
        do_sample = self._str_to_bool(os.getenv("GEN_DO_SAMPLE", "true"), True)
        return temperature, max_new_tokens, top_p, do_sample

    # ------------------------------------------------------------------ #
    # Core generation logic

    def generate(self, image_path: str, ehr_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        stage = "B" if ehr_json else "A"
        img = Image.open(image_path).convert("RGB")
        print(f"✅ Loaded image: {img.size}")

        if self.image_processor is None:
            raise RuntimeError("Image processor not available")

        dtype = torch.float32 if self.model.device.type == "cpu" else torch.float16
        img_tensor = process_images([img], self.image_processor, self.model.config)
        if isinstance(img_tensor, list):
            img_tensor = [tensor.to(self.model.device, dtype=dtype) for tensor in img_tensor]
        else:
            img_tensor = img_tensor.to(self.model.device, dtype=dtype)
        print(
            "✅ Processed image tensor shape:",
            img_tensor.shape if isinstance(img_tensor, torch.Tensor) else "list",
        )

        strict_json = self._str_to_bool(os.getenv("GEN_STRICT_JSON", "true"), True)
        debug_mode = self._str_to_bool(os.getenv("DEBUG_PIPELINE", "false"), False)

        def _run_generation(
            system_prompt: Optional[str],
            user_prompt: str,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            max_tokens: Optional[int] = None,
            do_sample: Optional[bool] = None,
            token_bias: Optional[Dict[int, float]] = None,
        ) -> str:
            conv_key = self.conv_mode if self.conv_mode in conv_templates else "llava_v1"
            conv = conv_templates[conv_key].copy()
            if system_prompt:
                conv.system = system_prompt
            user_role, assistant_role = conv.roles
            conv.append_message(user_role, f"{user_prompt}\n\n{DEFAULT_IMAGE_TOKEN}")
            conv.append_message(assistant_role, None)
            prompt_text = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            if input_ids is None:
                return ""
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self.model.device).contiguous()

            base_temp, base_max_tokens, base_top_p, base_do_sample = self._gen_params()
            t = temperature if temperature is not None else base_temp
            tp = top_p if top_p is not None else base_top_p
            mx = max_tokens if max_tokens is not None else base_max_tokens
            ds = base_do_sample if do_sample is None else do_sample

            logits_processors = None
            if token_bias:
                logits_processors = LogitsProcessorList([TokenBiasProcessor(token_bias)])

            total_max_length = input_ids.shape[1] + mx
            stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(total_max_length)])

            with torch.inference_mode():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=img_tensor,
                    max_new_tokens=mx,
                    temperature=t,
                    top_p=tp,
                    do_sample=ds,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    logits_processor=logits_processors,
                    stopping_criteria=stopping_criteria,
                )
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if strict_json:
            # Pass 1: Impression (text only)
            imp_temp = float(os.getenv("IMP_TEMPERATURE", "0.15"))
            imp_top_p = float(os.getenv("IMP_TOP_P", "0.9"))
            imp_do_sample = self._str_to_bool(os.getenv("IMP_DO_SAMPLE", "false"), False)
            imp_max_tokens = int(os.getenv("IMP_MAX_NEW_TOKENS", "160"))

            if stage == "A":
                system_imp = "You are a board-certified radiologist. Provide concise, clinically precise impressions."
                prompt_imp = "Write a concise radiology IMPRESSION (1-3 sentences) for this chest X-ray. Return text only."
            else:
                ehr_text = self.format_ehr(ehr_json)
                system_imp = "You are a board-certified radiologist who combines imaging and EHR context."
                prompt_imp = (
                    f"Patient EHR summary:\n{ehr_text}\n\n"
                    "Write a concise radiology IMPRESSION (1-3 sentences) that integrates both the image and the EHR. Return text only."
                )

            impression_text = _run_generation(
                system_imp,
                prompt_imp,
                temperature=imp_temp,
                top_p=imp_top_p,
                max_tokens=imp_max_tokens,
                do_sample=imp_do_sample,
            ).strip()
            if debug_mode:
                print("[DEBUG] Impression draft:", impression_text)

            # Pass 2: CheXpert JSON with voting
            chexpert_positive_bias = float(os.getenv("CHEXPERT_POSITIVE_BIAS", "2.0"))
            chexpert_negative_bias = float(os.getenv("CHEXPERT_NEGATIVE_BIAS", "-0.25"))
            chexpert_token_bias = self._make_token_bias(chexpert_positive_bias, chexpert_negative_bias)
            cx_vote = max(1, int(os.getenv("CHEXPERT_VOTE", "3")))
            cx_temp = float(os.getenv("CHEXPERT_TEMPERATURE", "0.7"))
            cx_top_p = float(os.getenv("CHEXPERT_TOP_P", "0.9"))
            cx_do_sample = self._str_to_bool(os.getenv("CHEXPERT_DO_SAMPLE", "true"), True)
            cx_max_tokens = int(os.getenv("CHEXPERT_MAX_NEW_TOKENS", "200"))
            retries = max(0, int(os.getenv("GEN_RETRIES", "2")))

            chexpert_example_pos = (
                "{\n"
                '  "No Finding": 0,\n'
                '  "Enlarged Cardiomediastinum": 1,\n'
                '  "Lung Opacity": 1,\n'
                '  "Lung Lesion": 0,\n'
                '  "Edema": 1,\n'
                '  "Consolidation": 1,\n'
                '  "Pneumonia": 1,\n'
                '  "Pneumothorax": 0,\n'
                '  "Pleural Effusion": 1,\n'
                '  "Pleural Other": 0,\n'
                '  "Fracture": 0,\n'
                '  "Support Devices": 1\n'
                "}"
            )
            chexpert_example_neg = (
                "{\n"
                '  "No Finding": 1,\n'
                '  "Enlarged Cardiomediastinum": 0,\n'
                '  "Lung Opacity": 0,\n'
                '  "Lung Lesion": 0,\n'
                '  "Edema": 0,\n'
                '  "Consolidation": 0,\n'
                '  "Pneumonia": 0,\n'
                '  "Pneumothorax": 0,\n'
                '  "Pleural Effusion": 0,\n'
                '  "Pleural Other": 0,\n'
                '  "Fracture": 0,\n'
                '  "Support Devices": 0\n'
                "}"
            )
            cx_system = "You are assisting with CheXpert labelling. Always respond with JSON only."
            cx_prompt = (
                "Using the chest X-ray and the impression below, fill the CheXpert JSON with integers in {-1,0,1}.\n"
                "Positive example:\n"
                f"{chexpert_example_pos}\n\n"
                "Negative example:\n"
                f"{chexpert_example_neg}\n\n"
                f"Draft impression: {impression_text}\n\n"
                "Now respond with ONLY the JSON object (no commentary)."
            )

            chexpert_votes: List[Dict[str, Any]] = []
            last_chexpert_dict: Optional[Dict[str, Any]] = None
            for _ in range(cx_vote):
                cx_text = _run_generation(
                    cx_system,
                    cx_prompt,
                    temperature=cx_temp,
                    top_p=cx_top_p,
                    max_tokens=cx_max_tokens,
                    do_sample=cx_do_sample,
                    token_bias=chexpert_token_bias,
                )
                if debug_mode:
                    print("[DEBUG] CheXpert response:", cx_text)
                blob = self._extract_json(cx_text)
                attempts = 0
                while blob is None and attempts < retries:
                    attempts += 1
                    cx_text = _run_generation(
                        cx_system,
                        "Your previous answer was not valid JSON. Return ONLY the CheXpert JSON object now.",
                        temperature=cx_temp,
                        top_p=cx_top_p,
                        max_tokens=cx_max_tokens,
                        do_sample=cx_do_sample,
                        token_bias=chexpert_token_bias,
                    )
                    blob = self._extract_json(cx_text)
                if blob:
                    try:
                        cx_dict = json.loads(blob)
                        chexpert_votes.append(cx_dict)
                        last_chexpert_dict = cx_dict
                    except Exception as exc:
                        if debug_mode:
                            print(f"[DEBUG] Failed to parse CheXpert JSON: {exc}")

            # Pass 3: ICD JSON with voting (Stage B only)
            icd_votes: List[Dict[str, Any]] = []
            last_icd_dict: Optional[Dict[str, Any]] = None
            if stage == "B":
                icd_positive_bias = float(os.getenv("ICD_POSITIVE_BIAS", "2.0"))
                icd_negative_bias = float(os.getenv("ICD_NEGATIVE_BIAS", "-0.25"))
                icd_token_bias = self._make_token_bias(icd_positive_bias, icd_negative_bias)
                icd_vote = max(1, int(os.getenv("ICD_VOTE", "3")))
                icd_temp = float(os.getenv("ICD_TEMPERATURE", "0.7"))
                icd_top_p = float(os.getenv("ICD_TOP_P", "0.9"))
                icd_do_sample = self._str_to_bool(os.getenv("ICD_DO_SAMPLE", "true"), True)
                icd_max_tokens = int(os.getenv("ICD_MAX_NEW_TOKENS", "160"))

                icd_example = (
                    "{\n"
                    '  "Pneumonia": 1,\n'
                    '  "Pleural_Effusion": 1,\n'
                    '  "Pneumothorax": 0,\n'
                    '  "Pulmonary_Edema": 1,\n'
                    '  "Cardiomegaly": 1,\n'
                    '  "Atelectasis": 1,\n'
                    '  "Pulmonary_Embolism": 0,\n'
                    '  "Rib_Fracture": 0\n'
                    "}"
                )
                icd_system = "You are assisting with ICD indicator labelling. Always respond with JSON only."
                icd_prompt = (
                    f"Draft impression: {impression_text}\n\n"
                    "Patient EHR indicators:\n"
                    f"{self.format_ehr(ehr_json)}\n\n"
                    "Example ICD JSON with positive findings:\n"
                    f"{icd_example}\n\n"
                    "Return ONLY the ICD JSON for the current case."
                )

                for _ in range(icd_vote):
                    icd_text = _run_generation(
                        icd_system,
                        icd_prompt,
                        temperature=icd_temp,
                        top_p=icd_top_p,
                        max_tokens=icd_max_tokens,
                        do_sample=icd_do_sample,
                        token_bias=icd_token_bias,
                    )
                    if debug_mode:
                        print("[DEBUG] ICD response:", icd_text)
                    blob = self._extract_json(icd_text)
                    attempts = 0
                    while blob is None and attempts < retries:
                        attempts += 1
                        icd_text = _run_generation(
                            icd_system,
                            "Your previous answer was not valid JSON. Return ONLY the ICD JSON object now.",
                            temperature=icd_temp,
                            top_p=icd_top_p,
                            max_tokens=icd_max_tokens,
                            do_sample=icd_do_sample,
                            token_bias=icd_token_bias,
                        )
                        blob = self._extract_json(icd_text)
                    if blob:
                        try:
                            icd_dict = json.loads(blob)
                            icd_votes.append(icd_dict)
                            last_icd_dict = icd_dict
                        except Exception as exc:
                            if debug_mode:
                                print(f"[DEBUG] Failed to parse ICD JSON: {exc}")

            parsed = {
                "impression": impression_text.strip(),
                "chexpert": self._aggregate_chexpert_votes(chexpert_votes, fallback=last_chexpert_dict),
                "icd": self._aggregate_icd_votes(icd_votes, fallback=last_icd_dict) if stage == "B" else {label: 0 for label in ICD},
                "raw_output": impression_text,
            }
            if self._str_to_bool(os.getenv("ENABLE_LABEL_KEYWORDS", "true"), True):
                self._apply_keyword_rules(parsed, stage, ehr_json)
            if debug_mode:
                print("[DEBUG] Final multi-pass result:", parsed)
            return parsed

        # Fallback: single shot JSON (rarely used)
        system_prompt = (
            "You are a radiology assistant. Respond with a strict JSON object containing the requested keys."
        )
        if stage == "A":
            example = (
                "{\n"
                '  "impression": "The lungs are clear. No effusion or pneumothorax.",\n'
                '  "chexpert": {"No Finding": 1, "Enlarged Cardiomediastinum": 0, "Lung Opacity": 0, "Lung Lesion": 0, "Edema": 0, "Consolidation": 0, "Pneumonia": 0, "Pneumothorax": 0, "Pleural Effusion": 0, "Pleural Other": 0, "Fracture": 0, "Support Devices": 0}\n'
                "}"
            )
            user_prompt = (
                "Analyze the chest X-ray image and return ONLY the JSON object shown in the example.\n"
                f"Example:\n{example}\n"
                "Now respond for the current case."
            )
        else:
            ehr_text = self.format_ehr(ehr_json)
            example = (
                "{\n"
                '  "impression": "Moderate pulmonary edema with bilateral pleural effusions.",\n'
                '  "chexpert": {"No Finding": 0, "Enlarged Cardiomediastinum": 1, "Lung Opacity": 1, "Lung Lesion": 0, "Edema": 1, "Consolidation": 0, "Pneumonia": 0, "Pneumothorax": 0, "Pleural Effusion": 1, "Pleural Other": 0, "Fracture": 0, "Support Devices": 1},\n'
                '  "icd": {"Pneumonia": 0, "Pleural_Effusion": 1, "Pneumothorax": 0, "Pulmonary_Edema": 1, "Cardiomegaly": 1, "Atelectasis": 0, "Pulmonary_Embolism": 0, "Rib_Fracture": 0}\n'
                "}"
            )
            user_prompt = (
                f"Patient EHR summary:\n{ehr_text}\n\n"
                "Analyze the chest X-ray image and return ONLY the JSON object shown in the example.\n"
                f"Example:\n{example}\n"
                "Now respond for the current case."
            )

        raw_response = _run_generation(system_prompt, user_prompt)
        parsed = self.parse_output(raw_response, stage, ehr_json)
        if self._str_to_bool(os.getenv("ENABLE_LABEL_KEYWORDS", "true"), True):
            self._apply_keyword_rules(parsed, stage, ehr_json)
        return parsed

    # ------------------------------------------------------------------ #
    # Parsing helpers

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    def parse_output(self, text: str, stage: str, ehr_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = {
            "impression": "",
            "chexpert": {label: 0 for label in CHEXPERT},
            "icd": {label: 0 for label in ICD}
        }

        json_blob = self._extract_json(text)
        if json_blob is None:
            result["impression"] = text.strip()
            return result

        try:
            payload = json.loads(json_blob)
        except json.JSONDecodeError:
            result["impression"] = text.strip()
            return result

        impression = payload.get("impression") or payload.get("Impression") or ""
        result["impression"] = impression.strip()
        if not result["impression"] and text:
            result["impression"] = text.strip()

        chexpert_payload = payload.get("chexpert") or {}
        for label in CHEXPERT:
            result["chexpert"][label] = self._normalize_label_value(chexpert_payload.get(label, 0), allow_negative=True)

        icd_payload = payload.get("icd") or {}
        if stage == "B":
            for label in ICD:
                result["icd"][label] = 1 if self._normalize_label_value(icd_payload.get(label, 0)) > 0 else 0

        result["raw_output"] = text.strip()
        if self._str_to_bool(os.getenv("ENABLE_LABEL_KEYWORDS", "true"), True):
            self._apply_keyword_rules(result, stage, ehr_json)
        return result

    def _apply_keyword_rules(self, parsed: Dict[str, Any], stage: str, ehr_json: Optional[Dict[str, Any]]) -> None:
        impression = (parsed.get("impression") or "").lower()
        if not impression:
            return

        chexpert = parsed.get("chexpert", {})
        icd = parsed.get("icd", {})

        chexpert_keywords = [
            ("pneumothorax", "Pneumothorax"),
            ("tension pneumothorax", "Pneumothorax"),
            ("effusion", "Pleural Effusion"),
            ("pleural effusions", "Pleural Effusion"),
            ("opacity", "Lung Opacity"),
            ("opacities", "Lung Opacity"),
            ("edema", "Edema"),
            ("pulmonary edema", "Edema"),
            ("consolidation", "Consolidation"),
            ("cardiomegaly", "Enlarged Cardiomediastinum"),
            ("enlarged cardiac silhouette", "Enlarged Cardiomediastinum"),
            ("fracture", "Fracture"),
            ("rib fracture", "Fracture"),
            ("pacemaker", "Support Devices"),
            ("line", "Support Devices"),
            ("catheter", "Support Devices"),
            ("tube", "Support Devices"),
        ]
        for keyword, label in chexpert_keywords:
            if keyword in impression:
                chexpert[label] = 1

        if stage == "B":
            icd_keywords = [
                ("pneumonia", "Pneumonia"),
                ("effusion", "Pleural_Effusion"),
                ("pulmonary edema", "Pulmonary_Edema"),
                ("cardiomegaly", "Cardiomegaly"),
                ("atelectasis", "Atelectasis"),
                ("embolism", "Pulmonary_Embolism"),
                ("pneumothorax", "Pneumothorax"),
                ("fracture", "Rib_Fracture"),
            ]
            for keyword, label in icd_keywords:
                if keyword in impression:
                    icd[label] = 1

        if stage == "B" and ehr_json:
            labs = ehr_json.get("Labs", {})
            bnp_info = labs.get("BNP") or labs.get("bnp")
            try:
                bnp_value = float(bnp_info.get("value")) if bnp_info and "value" in bnp_info else None
            except (TypeError, ValueError):
                bnp_value = None
            bnp_threshold = float(os.getenv("BNP_EDEMA_THRESHOLD", "1200"))
            admission_type = (ehr_json.get("admission_type") or "").lower()
            if bnp_value and bnp_value >= bnp_threshold and ("elective" in admission_type):
                chexpert["Edema"] = 1
                chexpert["Pleural Effusion"] = 1
                icd["Pulmonary_Edema"] = 1
                icd["Pleural_Effusion"] = 1
            crp_info = labs.get("CRP") or labs.get("crp")
            try:
                crp_value = float(crp_info.get("value")) if crp_info and "value" in crp_info else None
            except (TypeError, ValueError):
                crp_value = None
            crp_threshold = float(os.getenv("CRP_PNEUMONIA_THRESHOLD", "150"))
            if crp_value and crp_value >= crp_threshold and "pneumonia" in impression:
                icd["Pneumonia"] = 1
            o2_info = ehr_json.get("O2_device") or {}
            device_name = (o2_info.get("device") or "").lower()
            if device_name and device_name not in ("unknown", "room air", "none"):
                chexpert["Support Devices"] = 1

        positive_labels = [label for label, value in chexpert.items() if label != "No Finding" and value != 0]
        chexpert["No Finding"] = 0 if positive_labels else 1

    # ------------------------------------------------------------------ #
    # Convenience wrappers


_pipeline: Optional[RadiologyInferencePipeline] = None


def get_pipeline(device: str = "cpu") -> RadiologyInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RadiologyInferencePipeline(device=device)
    return _pipeline


def generate(image_path: str, ehr_json: Optional[Dict[str, Any]] = None, device: str = "cpu") -> Dict[str, Any]:
    pipeline = get_pipeline(device=device)
    return pipeline.generate(image_path, ehr_json)


if __name__ == "__main__":
    print("🔬 Testing Stage A inference…")
    result_a = generate("src/data/sample_images/sample_xray_1.jpg")
    print("Stage A Result:", json.dumps(result_a, indent=2))

    print("\n🔬 Testing Stage B inference…")
    sample_ehr = {
        "Age": 65,
        "Sex": "M",
        "Vitals": {"heart_rate": 92, "o2_saturation": 90},
        "Labs": {"BNP": 310000, "CRP": 162},
        "Chronic_conditions": ["hypertension", "heart_failure"],
    }
    result_b = generate("src/data/sample_images/sample_xray_1.jpg", sample_ehr)
    print("Stage B Result:", json.dumps(result_b, indent=2))
