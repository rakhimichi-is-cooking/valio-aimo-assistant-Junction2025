"""
Phase 2: Build Product Graph for GNN (Updated Version)

This script combines:
1. Extracted product features (names, categories, substitutions) from JSON
2. Sales data patterns (co-purchase, demand correlations) from CSV
3. Pre-defined substitution edges from experts

Output: Complete product graph for GNN forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import pickle
import json
from scipy.stats import pearsonr
from tqdm import tqdm

print("=" * 80)
print("BUILDING PRODUCT GRAPH FOR GNN (V2)")
print("=" * 80)
print("\nCombining:")
print("  - Product features from JSON (names, categories, 261 substitutions)")
print("  - Sales patterns from CSV (co-purchase, demand correlations)")
print()

# ============================================================================
# STEP 1: Load Pre-Extracted Product Catalog
# ============================================================================
print("Step 1: Loading product catalog...")
catalog_path = Path("data/product_features/product_catalog.json")

if not catalog_path.exists():
    print(f"\n❌ Product catalog not found: {catalog_path}")
    print("Please run extract_product_features.py first!")
    exit(1)

with open(catalog_path, 'r', encoding='utf-8') as f:
    product_catalog = json.load(f)

print(f"✅ Loaded {len(product_catalog):,} products from catalog")

# Create GTIN → product mapping
gtin_to_product = {p['gtin']: p for p in product_catalog if p.get('gtin')}
print(f"✅ Created GTIN index for {len(gtin_to_product):,} products")

# ============================================================================
# STEP 2: Load Pre-Defined Substitution Edges
# ============================================================================
print("\nStep 2: Loading pre-defined substitution edges...")
sub_edges_path = Path("data/product_features/substitution_edges.csv")

if sub_edges_path.exists():
    substitution_df = pd.read_csv(sub_edges_path)
    print(f"✅ Loaded {len(substitution_df):,} substitution edges")
else:
    print(f"⚠️  No substitution edges found")
    substitution_df = pd.DataFrame()

# ============================================================================
# STEP 3: Load Sales Data
# ============================================================================
print("\nStep 3: Loading sales data...")
sales_path = Path("data/valio_aimo_sales_and_deliveries_junction_2025.csv")

if not sales_path.exists():
    print(f"\n❌ Sales data not found: {sales_path}")
    print("Cannot build co-purchase and correlation edges without sales data.")
    print("\nSkipping to graph assembly with substitution edges only...")
    sales_df = None
else:
    sales_df = pd.read_csv(sales_path)
    print(f"✅ Loaded {len(sales_df):,} sales records")

    # Parse dates
    sales_df['order_created_date'] = pd.to_datetime(
        sales_df['order_created_date'],
        format='%Y-%m-%d',
        errors='coerce'
    )
    sales_df = sales_df.dropna(subset=['order_created_date'])
    print(f"✅ Date range: {sales_df['order_created_date'].min()} to {sales_df['order_created_date'].max()}")

# ============================================================================
# STEP 4: Co-Purchase Pattern Detection (FIXED VERSION)
# ============================================================================
if sales_df is not None:
    print("\n" + "=" * 80)
    print("Step 4: Detecting co-purchase patterns")
    print("=" * 80)
    print("\nAnalyzing order baskets to find products ordered together...")

    # FIXED: First, build a mapping of product → set of unique orders
    print("Building product-to-orders mapping...")
    product_orders = defaultdict(set)

    for _, row in tqdm(sales_df.iterrows(), total=len(sales_df), desc="Mapping products to orders"):
        product_code = str(row['product_code'])  # Convert to string for JSON compatibility
        order_id = f"{row['customer_number']}_{row['order_created_date']}"
        product_orders[product_code].add(order_id)

    print(f"✅ Found {len(product_orders):,} unique products in sales data")

    # FIXED: Now calculate CORRECT Jaccard similarity
    # Jaccard = |orders_with_both| / |orders_with_A ∪ orders_with_B|
    print("\nCalculating co-purchase Jaccard scores...")
    co_purchase_scores = {}

    # Get top 500 products by order frequency (to avoid combinatorial explosion)
    top_products = sorted(
        product_orders.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:500]

    print(f"Using top {len(top_products)} products by order frequency")

    for i, (prod_a, orders_a) in enumerate(tqdm(top_products, desc="Computing Jaccard scores")):
        # FIXED: Don't overwrite if product already has reverse edges stored
        if prod_a not in co_purchase_scores:
            co_purchase_scores[prod_a] = {}

        for j, (prod_b, orders_b) in enumerate(top_products):
            # Skip self-loops
            if prod_a == prod_b:
                continue

            # Skip if we already calculated the reverse direction
            if j <= i:
                continue

            # Calculate Jaccard similarity: intersection / union
            orders_both = orders_a & orders_b  # Set intersection
            orders_union = orders_a | orders_b  # Set union

            count_both = len(orders_both)
            count_union = len(orders_union)

            if count_union > 0:
                score = count_both / count_union

                # Only store significant co-purchases (>1% Jaccard)
                if score > 0.01:
                    co_purchase_scores[prod_a][prod_b] = float(score)

                    # Store symmetric edge
                    if prod_b not in co_purchase_scores:
                        co_purchase_scores[prod_b] = {}
                    co_purchase_scores[prod_b][prod_a] = float(score)

    total_edges = sum(len(v) for v in co_purchase_scores.values())
    print(f"✅ Calculated {total_edges:,} co-purchase edges (with Jaccard > 0.01)")
else:
    co_purchase_scores = {}

# ============================================================================
# STEP 5: Demand Correlation Detection (FIXED VERSION)
# ============================================================================
if sales_df is not None:
    print("\n" + "=" * 80)
    print("Step 5: Detecting demand correlations over time")
    print("=" * 80)
    print("\nAnalyzing time-series correlations...")
    print("  - Positive correlation: Products with similar demand patterns")
    print("  - Negative correlation: Substitutes or competitive products")

    # FIXED: Use string product codes and only products with sales data
    print("\nBuilding daily demand time-series for top 100 products...")

    # Convert product_code to string first
    sales_df['product_code'] = sales_df['product_code'].astype(str)

    top_products = sales_df['product_code'].value_counts().head(100).index.tolist()

    # Create pivot table: rows=dates, columns=products, values=daily demand
    demand_pivot = sales_df[sales_df['product_code'].isin(top_products)].pivot_table(
        index='order_created_date',
        columns='product_code',
        values='order_qty',
        aggfunc='sum',
        fill_value=0
    )

    print(f"✅ Built demand matrix: {demand_pivot.shape[0]} days × {demand_pivot.shape[1]} products")

    # Calculate pairwise correlations
    print("Calculating pairwise demand correlations...")
    demand_correlations = {}

    for i, prod_a in enumerate(tqdm(top_products, desc="Computing correlations")):
        if prod_a not in demand_pivot.columns:
            continue

        # FIXED: Don't overwrite if product already has reverse edges stored
        if prod_a not in demand_correlations:
            demand_correlations[prod_a] = {}

        for prod_b in top_products[i+1:]:
            if prod_b not in demand_pivot.columns:
                continue

            # Pearson correlation
            series_a = demand_pivot[prod_a].values
            series_b = demand_pivot[prod_b].values

            if len(series_a) > 10 and np.std(series_a) > 0 and np.std(series_b) > 0:
                corr, p_value = pearsonr(series_a, series_b)

                # Only store significant correlations (|corr| > 0.3, p < 0.05)
                if abs(corr) > 0.3 and p_value < 0.05:
                    demand_correlations[prod_a][prod_b] = float(corr)
                    # Store symmetric
                    if prod_b not in demand_correlations:
                        demand_correlations[prod_b] = {}
                    demand_correlations[prod_b][prod_a] = float(corr)

    print(f"✅ Found {sum(len(v) for v in demand_correlations.values()):,} significant demand correlations")
else:
    demand_correlations = {}

# ============================================================================
# STEP 6: Build Product Graph
# ============================================================================
print("\n" + "=" * 80)
print("Step 6: Building product graph structure")
print("=" * 80)

# Graph structure: dict of dicts
# product_graph[gtin_a][gtin_b] = {'edge_type': type, 'weight': weight, ...}
product_graph = defaultdict(lambda: defaultdict(dict))

# Add substitution edges (from experts)
num_sub_edges = 0
if not substitution_df.empty:
    print("\nAdding pre-defined substitution edges...")
    for _, row in substitution_df.iterrows():
        source_gtin = row['source_gtin']
        target_product = row['target_product']

        # Find target GTIN (need to lookup in product catalog)
        # For now, store with product code
        if source_gtin:
            product_graph[source_gtin][target_product] = {
                'edge_type': 'substitution',
                'weight': 1.0,
                'source': 'expert'
            }
            num_sub_edges += 1

print(f"✅ Added {num_sub_edges:,} substitution edges")

# Add co-purchase edges
num_copurchase_edges = 0
if co_purchase_scores:
    print("\nAdding co-purchase edges (threshold: score > 0.01)...")
    for prod_a in co_purchase_scores:
        for prod_b, score in co_purchase_scores[prod_a].items():
            if score > 0.01:  # Minimum threshold
                # Use product_code for now (can map to GTIN later if needed)
                if prod_b not in product_graph[prod_a]:
                    product_graph[prod_a][prod_b] = {}
                product_graph[prod_a][prod_b]['co_purchase'] = score
                num_copurchase_edges += 1

print(f"✅ Added {num_copurchase_edges:,} co-purchase edges")

# Add demand correlation edges
num_corr_edges = 0
if demand_correlations:
    print("\nAdding demand correlation edges...")
    for prod_a in demand_correlations:
        for prod_b, corr in demand_correlations[prod_a].items():
            if prod_b not in product_graph[prod_a]:
                product_graph[prod_a][prod_b] = {}
            product_graph[prod_a][prod_b]['demand_corr'] = corr
            num_corr_edges += 1

print(f"✅ Added {num_corr_edges:,} demand correlation edges")

# Convert to regular dict
product_graph = {k: dict(v) for k, v in product_graph.items()}

# Calculate graph statistics
num_nodes = len(product_graph)
num_edges = sum(len(v) for v in product_graph.values())
avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

print(f"\n📊 Product Graph Statistics:")
print(f"   Nodes (products): {num_nodes:,}")
print(f"   Edges (relationships): {num_edges:,}")
print(f"   Average degree: {avg_degree:.2f}")

# ============================================================================
# STEP 7: Save Graph Artifacts
# ============================================================================
print("\n" + "=" * 80)
print("Step 7: Saving graph artifacts")
print("=" * 80)

output_dir = Path("data/product_graph")
output_dir.mkdir(exist_ok=True)

# Save product graph (for GNN)
with open(output_dir / "product_graph.pkl", 'wb') as f:
    pickle.dump(product_graph, f)
print(f"✅ Saved: {output_dir / 'product_graph.pkl'}")

# Save co-purchase scores
if co_purchase_scores:
    with open(output_dir / "co_purchase_scores.json", 'w') as f:
        json.dump(co_purchase_scores, f, indent=2)
    print(f"✅ Saved: {output_dir / 'co_purchase_scores.json'}")

# Save demand correlations
if demand_correlations:
    with open(output_dir / "demand_correlations.json", 'w') as f:
        json.dump(demand_correlations, f, indent=2)
    print(f"✅ Saved: {output_dir / 'demand_correlations.json'}")

# Save summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'num_products_in_catalog': len(product_catalog),
    'num_products_with_gtin': len(gtin_to_product),
    'graph_stats': {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'num_substitution_edges': num_sub_edges,
        'num_copurchase_edges': num_copurchase_edges,
        'num_correlation_edges': num_corr_edges
    },
    'top_10_products_by_degree': sorted(
        [(k, len(v)) for k, v in product_graph.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]
}

with open(output_dir / "graph_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✅ Saved: {output_dir / 'graph_summary.json'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PRODUCT GRAPH BUILD COMPLETE")
print("=" * 80)

print(f"\n📊 Final Graph:")
print(f"   Nodes: {num_nodes:,}")
print(f"   Edges: {num_edges:,}")
print(f"   - Substitution edges: {num_sub_edges:,} (expert-defined)")
print(f"   - Co-purchase edges: {num_copurchase_edges:,} (from sales data)")
print(f"   - Correlation edges: {num_corr_edges:,} (from time-series)")

print(f"\n🧠 Ready for GNN:")
print(f"   ✅ Product catalog with {len(product_catalog):,} products")
print(f"   ✅ Product graph with {num_edges:,} edges")
print(f"   ✅ Node features: names, categories, vendors")
print(f"   ✅ Edge features: co-purchase scores, correlations, substitutions")

print(f"\n📈 Next Steps:")
print(f"   1. Implement GNN forecaster (GraphSAGE or GCN)")
print(f"   2. Use product graph for message passing")
print(f"   3. Predict demand with cross-product network effects")

print("\n" + "=" * 80)
