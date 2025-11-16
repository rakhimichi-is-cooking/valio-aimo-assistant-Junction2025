from typing import List, Dict, Optional
from difflib import SequenceMatcher
import re
import numpy as np

# Neural embedding imports
try:
    # Force PyTorch backend for sentence-transformers (avoids Keras 3 issues)
    import torch
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    EMBEDDINGS_AVAILABLE = False
    _EMBEDDINGS_IMPORT_ERROR = str(e)

# Global embedding model (lazy loaded)
_embedding_model = None


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using sequence matching.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score [0-1]
    """
    if not text1 or not text2:
        return 0.0

    # Normalize text
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()

    # Calculate similarity
    return SequenceMatcher(None, t1, t2).ratio()


def _extract_numeric_attributes(product: Dict) -> Dict[str, float]:
    """
    Extract numeric attributes from product for comparison.

    Args:
        product: Product dictionary

    Returns:
        Dictionary of numeric attributes
    """
    attrs = {}

    # Extract fat percentage (common in dairy products)
    name = product.get("name", "")
    fat_match = re.search(r'(\d+(?:\.\d+)?)\s*%', name)
    if fat_match:
        attrs['fat_pct'] = float(fat_match.group(1))

    # Extract size/volume
    size_str = str(product.get("unit_size") or product.get("pack_size") or "")
    size_match = re.search(r'(\d+(?:\.\d+)?)', size_str)
    if size_match:
        attrs['size'] = float(size_match.group(1))

    return attrs


def _calculate_attribute_similarity(ref_attrs: Dict, cand_attrs: Dict) -> float:
    """
    Calculate similarity based on numeric attributes.

    Args:
        ref_attrs: Reference product attributes
        cand_attrs: Candidate product attributes

    Returns:
        Similarity score [0-1]
    """
    if not ref_attrs or not cand_attrs:
        return 0.5  # Neutral if no attributes

    similarities = []

    # Compare each attribute
    for key in ref_attrs:
        if key in cand_attrs:
            ref_val = ref_attrs[key]
            cand_val = cand_attrs[key]

            # Calculate relative difference
            if ref_val > 0:
                diff = abs(ref_val - cand_val) / ref_val
                similarity = max(0, 1 - diff)
                similarities.append(similarity)

    if similarities:
        return sum(similarities) / len(similarities)

    return 0.5


def _get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the global embedding model (lazy loading).

    Returns:
        SentenceTransformer model for semantic similarity
    """
    global _embedding_model

    if _embedding_model is None:
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Run: pip install sentence-transformers")

        # Use all-MiniLM-L6-v2: Fast, lightweight, good for product matching
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    return _embedding_model


def _calculate_neural_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity using neural embeddings (BERT-based).

    Uses sentence-transformers to compute semantic similarity between product
    descriptions. This captures meaning beyond simple text matching.

    Args:
        text1: First text (e.g., product name)
        text2: Second text (e.g., product name)

    Returns:
        Similarity score [0-1] based on cosine similarity of embeddings
    """
    if not text1 or not text2:
        return 0.0

    try:
        model = _get_embedding_model()

        # Generate embeddings
        embeddings = model.encode([text1, text2])

        # Calculate cosine similarity
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]

        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)

        # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
        normalized = (similarity + 1) / 2

        return float(normalized)

    except Exception as e:
        print(f"Neural similarity calculation failed: {e}")
        # Fallback to text similarity
        return _calculate_text_similarity(text1, text2)


def _batch_calculate_neural_similarity(
    reference_text: str,
    candidate_texts: List[str]
) -> List[float]:
    """
    Batch calculate neural similarities for efficiency.

    Args:
        reference_text: Reference product description
        candidate_texts: List of candidate product descriptions

    Returns:
        List of similarity scores [0-1]
    """
    if not reference_text or not candidate_texts:
        return [0.0] * len(candidate_texts)

    try:
        model = _get_embedding_model()

        # Encode all texts at once (much faster than one-by-one)
        all_texts = [reference_text] + candidate_texts
        embeddings = model.encode(all_texts)

        ref_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]

        similarities = []
        for cand_embedding in candidate_embeddings:
            # Cosine similarity
            dot_product = np.dot(ref_embedding, cand_embedding)
            norm_ref = np.linalg.norm(ref_embedding)
            norm_cand = np.linalg.norm(cand_embedding)

            if norm_ref == 0 or norm_cand == 0:
                similarities.append(0.0)
            else:
                similarity = dot_product / (norm_ref * norm_cand)
                normalized = (similarity + 1) / 2
                similarities.append(float(normalized))

        return similarities

    except Exception as e:
        print(f"Batch neural similarity calculation failed: {e}")
        # Fallback to text similarity
        return [_calculate_text_similarity(reference_text, text) for text in candidate_texts]


def suggest_substitutes(
    sku: str,
    product_data: List[Dict],
    k: int = 3,
    use_advanced_matching: bool = True,
    use_neural_embeddings: bool = False,
) -> List[Dict]:
    """
    Improved substitution logic with multi-factor scoring.

    Scoring factors (standard):
    1. Category match (40%)
    2. Size/pack match (25%)
    3. Name similarity (20%)
    4. Attribute similarity (15%)

    Scoring factors (neural embeddings):
    1. Category match (35%)
    2. Semantic similarity (35% - BERT-based neural embeddings)
    3. Size/pack match (20%)
    4. Attribute similarity (10%)

    Args:
        sku: Product SKU to find substitutes for
        product_data: List of all products
        k: Number of substitutes to return
        use_advanced_matching: Use advanced similarity algorithms
        use_neural_embeddings: Use neural BERT embeddings for semantic matching

    Returns:
        List of substitute products with suitability scores
    """
    if not product_data:
        return []

    # Find reference product
    ref = next((p for p in product_data if str(p.get("sku")) == str(sku)), None)
    if ref is None:
        # No exact match, just return first k as fallback
        return [
            {
                "sku": str(p.get("sku")),
                "name": p.get("name", "Unknown product"),
                "suitability": 0.5,
            }
            for p in product_data[:k]
        ]

    ref_cat = ref.get("category")
    ref_size = ref.get("unit_size") or ref.get("pack_size")
    ref_name = ref.get("name", "")

    # Extract numeric attributes
    ref_attrs = _extract_numeric_attributes(ref) if use_advanced_matching else {}

    # For neural embeddings, pre-compute all candidate texts for batch processing
    if use_neural_embeddings and EMBEDDINGS_AVAILABLE:
        candidate_products = [p for p in product_data if str(p.get("sku")) != str(sku)]
        candidate_names = [p.get("name", "") for p in candidate_products]

        # Batch calculate neural similarities (much faster than one-by-one)
        neural_similarities = _batch_calculate_neural_similarity(ref_name, candidate_names)
    else:
        candidate_products = None
        neural_similarities = []

    candidates = []
    neural_sim_idx = 0

    for p in product_data:
        if str(p.get("sku")) == str(sku):
            continue

        cat = p.get("category")
        size = p.get("unit_size") or p.get("pack_size")
        name = p.get("name", "")

        if use_advanced_matching:
            # Advanced multi-factor scoring

            # 1. Category match
            if ref_cat and cat == ref_cat:
                category_score = 1.0
            elif ref_cat and cat:
                # Partial match (both have categories but different)
                category_score = 0.3
            else:
                category_score = 0.1

            # 2. Size match
            if ref_size and size:
                if ref_size == size:
                    size_score = 1.0
                else:
                    # Try to compare numerically
                    try:
                        ref_num = float(re.search(r'(\d+(?:\.\d+)?)', str(ref_size)).group(1))
                        cand_num = float(re.search(r'(\d+(?:\.\d+)?)', str(size)).group(1))
                        diff_ratio = abs(ref_num - cand_num) / ref_num
                        size_score = max(0, 1 - diff_ratio)
                    except:
                        size_score = 0.5
            else:
                size_score = 0.5

            # 3. Name/Semantic similarity
            if use_neural_embeddings and EMBEDDINGS_AVAILABLE and neural_similarities:
                # Use pre-computed neural similarity
                name_score = neural_similarities[neural_sim_idx]
                neural_sim_idx += 1
            else:
                # Use traditional text similarity
                name_score = _calculate_text_similarity(ref_name, name)

            # 4. Attribute similarity
            cand_attrs = _extract_numeric_attributes(p)
            attr_score = _calculate_attribute_similarity(ref_attrs, cand_attrs)

            # Weighted composite score
            if use_neural_embeddings and EMBEDDINGS_AVAILABLE:
                # Neural embeddings mode: boost semantic similarity weight
                suitability = (
                    category_score * 0.35 +
                    name_score * 0.35 +     # Semantic similarity (neural)
                    size_score * 0.20 +
                    attr_score * 0.10
                )
            else:
                # Standard mode
                suitability = (
                    category_score * 0.40 +
                    size_score * 0.25 +
                    name_score * 0.20 +
                    attr_score * 0.15
                )
        else:
            # Simple scoring (backwards compatible)
            if ref_cat and cat == ref_cat and ref_size and size == ref_size:
                suitability = 1.0
            elif ref_cat and cat == ref_cat:
                suitability = 0.8
            else:
                suitability = 0.5

        candidates.append(
            {
                "sku": str(p.get("sku")),
                "name": p.get("name", "Unknown product"),
                "suitability": float(suitability),
                "category": cat,
                "size": size,
            }
        )

    # Sort by suitability and return top k
    candidates.sort(key=lambda x: x["suitability"], reverse=True)

    # Remove category and size from final output (for backwards compatibility)
    results = []
    for c in candidates[:k]:
        results.append({
            "sku": c["sku"],
            "name": c["name"],
            "suitability": c["suitability"]
        })

    return results


def suggest_substitutes_with_inventory(
    sku: str,
    product_data: List[Dict],
    inventory_data: Optional[Dict[str, float]] = None,
    k: int = 3,
) -> List[Dict]:
    """
    Advanced substitution with inventory availability check.

    Args:
        sku: Product SKU to find substitutes for
        product_data: List of all products
        inventory_data: Dict mapping SKU -> available quantity
        k: Number of substitutes to return

    Returns:
        List of substitute products, prioritizing available stock
    """
    # Get base substitutes
    substitutes = suggest_substitutes(sku, product_data, k=k*2)  # Get more candidates

    if inventory_data:
        # Boost suitability score for products with available inventory
        for sub in substitutes:
            sub_sku = sub["sku"]
            available_qty = inventory_data.get(sub_sku, 0)

            if available_qty > 0:
                # Boost score based on availability
                availability_boost = min(0.2, available_qty / 100)  # Cap boost at 0.2
                sub["suitability"] = min(1.0, sub["suitability"] + availability_boost)
                sub["available_qty"] = available_qty
            else:
                # Penalize out-of-stock items
                sub["suitability"] *= 0.7
                sub["available_qty"] = 0

        # Re-sort after adjusting scores
        substitutes.sort(key=lambda x: x["suitability"], reverse=True)

    return substitutes[:k]
