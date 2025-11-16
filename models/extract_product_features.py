"""
Extract product features for GNN from JSON data

This script:
1. Extracts product names from synkkaData.names
2. Creates a clean product catalog with GTIN, name, category, vendor
3. Extracts substitution relationships (seed edges for GNN)
4. Saves features for product graph building
"""

import json
from pathlib import Path
import pandas as pd
from collections import defaultdict

print("=" * 80)
print("EXTRACTING PRODUCT FEATURES FOR GNN")
print("=" * 80)

# ============================================================================
# LOAD JSON DATA
# ============================================================================
print("\nStep 1: Loading product JSON...")
json_path = Path("data/valio_aimo_product_data_junction_2025.json")

with open(json_path, 'r', encoding='utf-8') as f:
    products = json.load(f)

print(f"✅ Loaded {len(products):,} products")

# ============================================================================
# EXTRACT PRODUCT NAMES FROM synkkaData
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: Extracting product names from synkkaData")
print("=" * 80)

product_catalog = []
missing_names = 0

for product in products:
    gtin = product.get('salesUnitGtin')
    category = product.get('category')
    vendor = product.get('vendorName')
    country = product.get('countryOfOrigin')
    temp_condition = product.get('temperatureCondition')

    # Extract product name from synkkaData.names
    synkka_data = product.get('synkkaData', {})
    names = synkka_data.get('names', [])

    # Prefer English name, fallback to Finnish, then Swedish
    product_name = None
    name_by_lang = {}

    for name_obj in names:
        lang = name_obj.get('language', '').lower()
        value = name_obj.get('value', '').strip()
        if value:
            name_by_lang[lang] = value

    # Priority: en > fi > sv > any available
    if 'en' in name_by_lang:
        product_name = name_by_lang['en']
    elif 'fi' in name_by_lang:
        product_name = name_by_lang['fi']
    elif 'sv' in name_by_lang:
        product_name = name_by_lang['sv']
    elif name_by_lang:
        product_name = list(name_by_lang.values())[0]

    if not product_name:
        missing_names += 1
        product_name = f"Product {gtin or 'Unknown'}"

    product_catalog.append({
        'gtin': gtin,
        'name': product_name,
        'category': category,
        'vendor': vendor,
        'country': country,
        'temp_condition': temp_condition,
        'name_en': name_by_lang.get('en', ''),
        'name_fi': name_by_lang.get('fi', ''),
        'name_sv': name_by_lang.get('sv', '')
    })

print(f"\n✅ Extracted {len(product_catalog):,} product names")
print(f"   Products with names: {len(product_catalog) - missing_names:,}")
print(f"   Products without names: {missing_names:,}")

# Sample names
print(f"\n📋 Sample product names:")
for i, p in enumerate(product_catalog[:5], 1):
    print(f"  {i}. {p['name']} (GTIN: {p['gtin']})")

# ============================================================================
# EXTRACT SUBSTITUTION RELATIONSHIPS
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: Extracting substitution relationships (GNN seed edges)")
print("=" * 80)

substitution_edges = []

for product in products:
    gtin = product.get('salesUnitGtin')
    subs = product.get('substitutions', [])

    if subs:
        for sub_record in subs:
            source_product = sub_record.get('product')
            target_product = sub_record.get('substituteProduct')
            customers = sub_record.get('customers', [])
            districts = sub_record.get('salesDistricts', [])

            if source_product and target_product:
                substitution_edges.append({
                    'source_gtin': gtin,
                    'source_product': source_product,
                    'target_product': target_product,
                    'num_customers': len(customers) if customers else 0,
                    'num_districts': len(districts) if districts else 0,
                    'edge_type': 'substitution'
                })

print(f"\n✅ Found {len(substitution_edges):,} substitution relationships")

if substitution_edges:
    print(f"\n📋 Sample substitution edges:")
    for i, edge in enumerate(substitution_edges[:5], 1):
        print(f"  {i}. {edge['source_product']} → {edge['target_product']}")

# ============================================================================
# CATEGORY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: Category analysis")
print("=" * 80)

category_counts = defaultdict(int)
for p in product_catalog:
    if p['category']:
        category_counts[p['category']] += 1

print(f"\n📊 Category distribution:")
print(f"   Unique categories: {len(category_counts):,}")
print(f"\n   Top 10 categories by product count:")

for i, (cat, count) in enumerate(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    print(f"     {i}. Category {cat}: {count:,} products")

# ============================================================================
# VENDOR ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: Vendor analysis")
print("=" * 80)

vendor_counts = defaultdict(int)
for p in product_catalog:
    if p['vendor']:
        vendor_counts[p['vendor']] += 1

print(f"\n📊 Vendor distribution:")
print(f"   Unique vendors: {len(vendor_counts):,}")
print(f"\n   Top 10 vendors by product count:")

for i, (vendor, count) in enumerate(sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)[:10], 1):
    vendor_short = vendor[:50] + "..." if len(vendor) > 50 else vendor
    print(f"     {i}. {vendor_short}: {count:,} products")

# ============================================================================
# SAVE EXTRACTED FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("Step 6: Saving extracted features")
print("=" * 80)

output_dir = Path("data/product_features")
output_dir.mkdir(exist_ok=True)

# Save product catalog as CSV (for easy inspection)
catalog_df = pd.DataFrame(product_catalog)
catalog_df.to_csv(output_dir / "product_catalog.csv", index=False, encoding='utf-8')
print(f"✅ Saved: {output_dir / 'product_catalog.csv'}")
print(f"   {len(catalog_df):,} products with names, categories, vendors")

# Save product catalog as JSON (preserves types)
with open(output_dir / "product_catalog.json", 'w', encoding='utf-8') as f:
    json.dump(product_catalog, f, indent=2, ensure_ascii=False)
print(f"✅ Saved: {output_dir / 'product_catalog.json'}")

# Save substitution edges
if substitution_edges:
    edges_df = pd.DataFrame(substitution_edges)
    edges_df.to_csv(output_dir / "substitution_edges.csv", index=False)
    print(f"✅ Saved: {output_dir / 'substitution_edges.csv'}")
    print(f"   {len(edges_df):,} pre-defined substitution edges")
else:
    print(f"⚠️  No substitution edges found")

# Save summary statistics
summary = {
    'total_products': len(product_catalog),
    'products_with_names': len(product_catalog) - missing_names,
    'products_with_gtin': sum(1 for p in product_catalog if p['gtin']),
    'unique_categories': len(category_counts),
    'unique_vendors': len(vendor_counts),
    'substitution_edges': len(substitution_edges),
    'top_5_categories': [
        {'category': cat, 'count': count}
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ],
    'top_5_vendors': [
        {'vendor': vendor, 'count': count}
        for vendor, count in sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    ]
}

with open(output_dir / "feature_summary.json", 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"✅ Saved: {output_dir / 'feature_summary.json'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 80)

print(f"\n📊 Extracted features for GNN:")
print(f"   ✅ Product catalog: {len(product_catalog):,} products")
print(f"   ✅ Product names: {len(product_catalog) - missing_names:,} ({(len(product_catalog) - missing_names)/len(product_catalog)*100:.1f}%)")
print(f"   ✅ GTINs for CSV matching: {sum(1 for p in product_catalog if p['gtin']):,}")
print(f"   ✅ Categories: {len(category_counts):,} unique")
print(f"   ✅ Vendors: {len(vendor_counts):,} unique")
print(f"   ✅ Substitution edges: {len(substitution_edges):,}")

print(f"\n🧠 GNN Node Features Ready:")
print(f"   - Product names (for BERT embeddings)")
print(f"   - Category (for grouping)")
print(f"   - Vendor (for supplier analysis)")
print(f"   - Country of origin")
print(f"   - Temperature condition")

print(f"\n🔗 GNN Edge Features Ready:")
print(f"   - Pre-defined substitutions ({len(substitution_edges):,} edges)")
print(f"   - Ready to add co-purchase edges from CSV")
print(f"   - Ready to add demand correlation edges from CSV")

print(f"\n📈 Next Steps:")
print(f"   1. Match product catalog with CSV sales data (via GTIN)")
print(f"   2. Build co-purchase and demand correlation edges")
print(f"   3. Combine all edges into product graph")
print(f"   4. Feed to GNN for demand forecasting")

print("\n" + "=" * 80)
