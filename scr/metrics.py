"""
metrics.py - Comprehensive metrics for OCR quality evaluation
Includes CER, WER, field-level metrics, and aggregation functions
"""

import re
import json
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import Levenshtein
from jiwer import wer as jiwer_wer
import pandas as pd
from pathlib import Path


# ============= Text Normalization Functions =============

def normalize_text(text: str, level: str = 'basic') -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        level: 'basic' for simple normalization, 'aggressive' for full normalization
    
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    text = str(text).strip()
    
    # Basic normalization
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = text.lower()
    text = text.replace('ё', 'е')
    
    if level == 'aggressive':
        # Remove punctuation
        text = re.sub(r'[^\w\s\d]', '', text)
        # Remove extra spaces again
        text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize date to standard format YYYY-MM-DD.
    
    Args:
        date_str: Date string in various formats
    
    Returns:
        Normalized date or None if parsing fails
    """
    if not date_str:
        return None
    
    date_str = str(date_str).strip()
    
    # Common date formats to try
    formats = [
        '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%d',
        '%d-%m-%Y', '%Y.%m.%d', '%d %m %Y',
        '%d.%m.%y', '%d/%m/%y', '%y-%m-%d'
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    return date_str  # Return as-is if can't parse


def normalize_number(num_str: str) -> Optional[float]:
    """
    Normalize number string to float.
    
    Args:
        num_str: Number string with possible formatting
    
    Returns:
        Float value or None if parsing fails
    """
    if not num_str:
        return None
    
    num_str = str(num_str).strip()
    
    # Remove spaces, replace comma with dot
    num_str = num_str.replace(' ', '').replace(',', '.')
    
    # Remove currency symbols and other non-numeric chars except dot and minus
    num_str = re.sub(r'[^\d\.\-]', '', num_str)
    
    try:
        return float(num_str)
    except ValueError:
        return None


# ============= Core Metric Functions =============

def cer(ref: str, hyp: str, normalize: bool = False) -> float:
    """
    Calculate Character Error Rate (CER).
    
    Args:
        ref: Reference text
        hyp: Hypothesis text
        normalize: Whether to normalize texts before comparison
    
    Returns:
        CER score (0.0 = perfect match, 1.0+ = many errors)
    """
    if not ref:
        return 0.0 if not hyp else 1.0
    
    if normalize:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    
    distance = Levenshtein.distance(ref, hyp)
    return distance / max(1, len(ref))


def wer(ref: str, hyp: str, normalize: bool = False) -> float:
    """
    Calculate Word Error Rate (WER).
    
    Args:
        ref: Reference text
        hyp: Hypothesis text
        normalize: Whether to normalize texts before comparison
    
    Returns:
        WER score (0.0 = perfect match, 1.0+ = many errors)
    """
    if not ref:
        return 0.0 if not hyp else 1.0
    
    if normalize:
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)
    
    try:
        return jiwer_wer(ref, hyp)
    except:
        # Fallback to word-level Levenshtein if jiwer fails
        ref_words = ref.split()
        hyp_words = hyp.split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        distance = Levenshtein.distance(' '.join(ref_words), ' '.join(hyp_words))
        return distance / max(1, len(' '.join(ref_words)))


def normalized_levenshtein(ref: str, hyp: str) -> float:
    """
    Calculate normalized Levenshtein similarity (0 to 1, where 1 is perfect match).
    
    Args:
        ref: Reference text
        hyp: Hypothesis text
    
    Returns:
        Similarity score (1.0 = perfect match, 0.0 = completely different)
    """
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    
    distance = Levenshtein.distance(ref, hyp)
    max_len = max(len(ref), len(hyp))
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (distance / max_len)


# ============= Field-Level Metrics =============

def compare_field_values(ref_value: Any, hyp_value: Any, field_type: str = 'text') -> Dict[str, float]:
    """
    Compare two field values based on field type.
    
    Args:
        ref_value: Reference value
        hyp_value: Hypothesis value
        field_type: Type of field ('text', 'number', 'date', 'list')
    
    Returns:
        Dictionary with comparison metrics
    """
    metrics = {}
    
    if ref_value is None and hyp_value is None:
        return {'exact_match': 1.0, 'normalized_match': 1.0}
    
    if ref_value is None or hyp_value is None:
        return {'exact_match': 0.0, 'normalized_match': 0.0}
    
    # Exact match
    metrics['exact_match'] = 1.0 if str(ref_value) == str(hyp_value) else 0.0
    
    # Type-specific comparison
    if field_type == 'number':
        ref_num = normalize_number(str(ref_value))
        hyp_num = normalize_number(str(hyp_value))
        
        if ref_num is not None and hyp_num is not None:
            # Allow small tolerance for floating point comparison
            metrics['normalized_match'] = 1.0 if abs(ref_num - hyp_num) < 0.01 else 0.0
            if ref_num != 0:
                metrics['relative_error'] = abs(ref_num - hyp_num) / abs(ref_num)
        else:
            metrics['normalized_match'] = 0.0
            
    elif field_type == 'date':
        ref_date = normalize_date(str(ref_value))
        hyp_date = normalize_date(str(hyp_value))
        metrics['normalized_match'] = 1.0 if ref_date == hyp_date else 0.0
        
    elif field_type == 'list':
        # For lists, calculate Jaccard similarity
        if isinstance(ref_value, str):
            ref_items = set(ref_value.split(','))
        else:
            ref_items = set(ref_value) if isinstance(ref_value, list) else {ref_value}
            
        if isinstance(hyp_value, str):
            hyp_items = set(hyp_value.split(','))
        else:
            hyp_items = set(hyp_value) if isinstance(hyp_value, list) else {hyp_value}
        
        intersection = len(ref_items & hyp_items)
        union = len(ref_items | hyp_items)
        metrics['normalized_match'] = intersection / union if union > 0 else 0.0
        
    else:  # text
        ref_norm = normalize_text(str(ref_value), level='aggressive')
        hyp_norm = normalize_text(str(hyp_value), level='aggressive')
        metrics['normalized_match'] = 1.0 if ref_norm == hyp_norm else 0.0
        metrics['similarity'] = normalized_levenshtein(str(ref_value), str(hyp_value))
        metrics['cer'] = cer(str(ref_value), str(hyp_value), normalize=True)
    
    return metrics


def field_metrics(gt: Dict, pred: Dict, field_types: Optional[Dict[str, str]] = None) -> Dict:
    """
    Calculate metrics for each field in the documents.
    
    Args:
        gt: Ground truth dictionary
        pred: Predicted dictionary
        field_types: Optional dict mapping field names to types
    
    Returns:
        Dictionary with metrics for each field
    """
    if field_types is None:
        field_types = {}
    
    results = {}
    all_fields = set(gt.keys()) | set(pred.keys())
    
    for field in all_fields:
        gt_value = gt.get(field)
        pred_value = pred.get(field)
        field_type = field_types.get(field, 'text')
        
        # Basic presence metrics
        field_metrics = {
            'in_gt': field in gt,
            'in_pred': field in pred,
            'both_present': field in gt and field in pred
        }
        
        # Value comparison metrics
        if field in gt and field in pred:
            comparison = compare_field_values(gt_value, pred_value, field_type)
            field_metrics.update(comparison)
        else:
            field_metrics['exact_match'] = 0.0
            field_metrics['normalized_match'] = 0.0
        
        results[field] = field_metrics
    
    # Calculate precision, recall, F1 for field detection
    detected_fields = set(pred.keys())
    expected_fields = set(gt.keys())
    
    if expected_fields:
        tp = len(detected_fields & expected_fields)
        fp = len(detected_fields - expected_fields)
        fn = len(expected_fields - detected_fields)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results['_field_detection'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    return results


def document_exact_match(gt: Dict, pred: Dict, required_fields: Optional[List[str]] = None) -> bool:
    """
    Check if all required fields match exactly between GT and prediction.
    
    Args:
        gt: Ground truth dictionary
        pred: Predicted dictionary
        required_fields: List of required field names (if None, use all GT fields)
    
    Returns:
        True if all required fields match exactly
    """
    if required_fields is None:
        required_fields = list(gt.keys())
    
    for field in required_fields:
        if field not in pred:
            return False
        if str(gt.get(field, '')).strip() != str(pred.get(field, '')).strip():
            return False
    
    return True


# ============= Aggregation Functions =============

def aggregate_metrics(results: List[Dict], weights: Optional[Dict[str, float]] = None) -> Dict:
    """
    Aggregate metrics across multiple documents.
    
    Args:
        results: List of metric dictionaries for each document
        weights: Optional weights for different metrics
    
    Returns:
        Aggregated metrics dictionary
    """
    if not results:
        return {}
    
    aggregated = {}
    
    # Collect all metric keys
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    
    # Calculate means for numeric metrics
    for key in all_keys:
        values = []
        for result in results:
            if key in result:
                value = result[key]
                if isinstance(value, (int, float)):
                    values.append(value)
                elif isinstance(value, dict):
                    # Recursively aggregate nested dicts
                    nested_results = [result[key] for result in results if key in result]
                    aggregated[key] = aggregate_metrics(nested_results, weights)
        
        if values:
            aggregated[key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': pd.Series(values).std() if len(values) > 1 else 0.0,
                'count': len(values)
            }
    
    # Apply weights if provided
    if weights:
        weighted_score = 0.0
        weight_sum = 0.0
        
        for metric, weight in weights.items():
            if metric in aggregated and 'mean' in aggregated[metric]:
                weighted_score += aggregated[metric]['mean'] * weight
                weight_sum += weight
        
        if weight_sum > 0:
            aggregated['weighted_score'] = weighted_score / weight_sum
    
    return aggregated


# ============= Evaluation Pipeline =============

def evaluate_document(gt_path: Path, pred_path: Path, 
                      field_types: Optional[Dict[str, str]] = None) -> Dict:
    """
    Evaluate a single document's OCR and extraction quality.
    
    Args:
        gt_path: Path to ground truth JSON
        pred_path: Path to predicted JSON
        field_types: Optional field type specifications
    
    Returns:
        Dictionary with all metrics for the document
    """
    # Load files
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    
    # Calculate metrics
    metrics = {
        'document': gt_path.stem,
        'field_metrics': field_metrics(gt_data, pred_data, field_types),
        'exact_match': document_exact_match(gt_data, pred_data)
    }
    
    # If there's a 'text' field, calculate text-level metrics
    if 'text' in gt_data and 'text' in pred_data:
        metrics['text_cer'] = cer(gt_data['text'], pred_data['text'])
        metrics['text_wer'] = wer(gt_data['text'], pred_data['text'])
        metrics['text_similarity'] = normalized_levenshtein(gt_data['text'], pred_data['text'])
    
    return metrics


def create_comparison_table(baseline_results: List[Dict], 
                           our_results: List[Dict],
                           output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a comparison table between baseline and our pipeline.
    
    Args:
        baseline_results: List of baseline metric dictionaries
        our_results: List of our pipeline metric dictionaries
        output_path: Optional path to save CSV
    
    Returns:
        DataFrame with comparison metrics
    """
    rows = []
    
    for base, ours in zip(baseline_results, our_results):
        row = {
            'document': base.get('document', 'unknown'),
            'baseline_exact_match': base.get('exact_match', 0),
            'our_exact_match': ours.get('exact_match', 0),
        }
        
        # Add text metrics if available
        for metric in ['text_cer', 'text_wer', 'text_similarity']:
            if metric in base:
                row[f'baseline_{metric}'] = base[metric]
            if metric in ours:
                row[f'our_{metric}'] = ours[metric]
        
        # Add improvement metrics
        if 'text_cer' in base and 'text_cer' in ours:
            row['cer_improvement'] = base['text_cer'] - ours['text_cer']
            row['cer_improvement_pct'] = (row['cer_improvement'] / base['text_cer'] * 100 
                                         if base['text_cer'] > 0 else 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add summary row
    summary = pd.DataFrame([{
        'document': 'AVERAGE',
        **{col: df[col].mean() for col in df.columns if col != 'document'}
    }])
    
    df = pd.concat([df, summary], ignore_index=True)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Comparison table saved to {output_path}")
    
    return df


# ============= Testing Utilities =============

def run_evaluation_pipeline(gt_dir: Path, baseline_dir: Path, our_dir: Path,
                           field_types: Optional[Dict[str, str]] = None,
                           output_dir: Optional[Path] = None) -> Dict:
    """
    Run complete evaluation pipeline on all documents.
    
    Args:
        gt_dir: Directory with ground truth JSONs
        baseline_dir: Directory with baseline predictions
        our_dir: Directory with our pipeline predictions
        field_types: Optional field type specifications
        output_dir: Optional directory to save results
    
    Returns:
        Dictionary with all evaluation results
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_results = []
    our_results = []
    
    # Process each GT file
    for gt_file in sorted(gt_dir.glob('*.json')):
        doc_name = gt_file.stem
        baseline_file = baseline_dir / f'{doc_name}.json'
        our_file = our_dir / f'{doc_name}.json'
        
        if baseline_file.exists() and our_file.exists():
            # Evaluate baseline
            baseline_metrics = evaluate_document(gt_file, baseline_file, field_types)
            baseline_results.append(baseline_metrics)
            
            # Evaluate our pipeline
            our_metrics = evaluate_document(gt_file, our_file, field_types)
            our_results.append(our_metrics)
            
            print(f"Evaluated {doc_name}")
    
    # Aggregate results
    results = {
        'baseline': {
            'individual': baseline_results,
            'aggregated': aggregate_metrics(baseline_results)
        },
        'our_pipeline': {
            'individual': our_results,
            'aggregated': aggregate_metrics(our_results)
        }
    }
    
    # Create comparison table
    comparison_df = create_comparison_table(
        baseline_results, 
        our_results,
        output_dir / 'comparison.csv' if output_dir else None
    )
    results['comparison'] = comparison_df
    
    # Save detailed results
    if output_dir:
        with open(output_dir / 'detailed_results.json', 'w', encoding='utf-8') as f:
            # Convert DataFrame to dict for JSON serialization
            results_copy = results.copy()
            results_copy['comparison'] = comparison_df.to_dict('records')
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("OCR Metrics Module loaded successfully")
    
    # Example field types for Uzbek documents
    UZBEK_DOC_FIELD_TYPES = {
        'passport_number': 'text',
        'full_name': 'text',
        'birth_date': 'date',
        'issue_date': 'date',
        'expiry_date': 'date',
        'gender': 'text',
        'nationality': 'text',
        'document_number': 'text',
        'inn': 'number',
        'amount': 'number',
        'address': 'text'
    }
    
    # Test basic metrics
    ref_text = "SHAXSIY PASPORT"
    hyp_text = "SHAXSY PASSPORT"
    
    print(f"\nExample metrics:")
    print(f"CER: {cer(ref_text, hyp_text):.4f}")
    print(f"WER: {wer(ref_text, hyp_text):.4f}")
    print(f"Normalized Levenshtein: {normalized_levenshtein(ref_text, hyp_text):.4f}")
    
    # Example field comparison
    gt_doc = {
        'passport_number': 'AB1234567',
        'full_name': 'ИВАНОВ ИВАН ИВАНОВИЧ',
        'birth_date': '01.01.1990',
        'amount': '1,500.50'
    }
    
    pred_doc = {
        'passport_number': 'AB1234567',
        'full_name': 'Иванов Иван Иванович',
        'birth_date': '1990-01-01',
        'amount': '1500.5'
    }
    
    print(f"\nField metrics example:")
    metrics = field_metrics(gt_doc, pred_doc, UZBEK_DOC_FIELD_TYPES)
    for field, field_metrics_dict in metrics.items():
        if not field.startswith('_'):
            print(f"  {field}: {field_metrics_dict}")