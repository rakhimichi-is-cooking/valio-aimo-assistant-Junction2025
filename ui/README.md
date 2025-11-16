# Valio AI - Minimalist UI

Pure Python/Streamlit dashboard with Apple/Claude.ai inspired minimalist design.

## Quick Start

```bash
# From project root
streamlit run ui/app.py
```

## Features

### Dashboard View
- **3 Metric Cards**: Products Normal, Critical (7d), At Risk (21d)
- **Demand Overview Chart**: Clean line chart showing 90-day aggregate demand
- **Product List**: Top products requiring attention, sorted by risk score
- **AI Assistant**: Input box to ask questions (triggers chat modal)

### Design System

**Colors:**
- Background: #FAFAF9 (warm white)
- Text: #1A1A1A (primary), #6B6B6B (secondary), #9B9B9B (tertiary)
- Status: #D14343 (critical), #E8A84D (warning), #4A9B8E (normal)
- Accent: #5B5BD6 (purple, Claude-inspired)

**Typography:**
- Font: Inter (Google Fonts)
- Scale: 48px (big numbers) → 20px (headers) → 16px (body) → 14px (secondary) → 12px (tertiary)
- Line heights: 1.2 (headings), 1.6 (body)

**Spacing:**
- 8px grid system: 8px, 16px, 24px, 32px, 48px
- Generous whitespace
- Minimal borders (1px, #E5E5E3)

**Charts:**
- Plotly with custom minimal config
- White background, single black line
- Horizontal grid only (#F5F5F3)
- No mode bar, clean tooltips

## File Structure

```
ui/
├── app.py                 # Main Streamlit application
├── styles.css             # Minimalist CSS theme
├── components/            # Reusable UI components (future)
│   └── __init__.py
├── views/                 # Different views (future)
│   └── __init__.py
└── utils/                 # Utilities (future)
    └── __init__.py
```

## Current Implementation

**app.py** contains a simplified, working dashboard with:
- Metric cards (HTML/CSS)
- Demand chart (Plotly)
- Product list (basic)
- AI input box (placeholder)

## Next Steps

1. **Integrate Real Data**:
   - Connect to backend API (`http://127.0.0.1:8000/shortages`)
   - Load actual sales data from CSV
   - Use GNN forecaster for predictions

2. **Add Product Detail View**:
   - Click product → show detailed analysis
   - Historical demand chart (area fill)
   - Forecast chart (purple line + confidence interval)
   - Network graph (NetworkX + Plotly)
   - Metrics cards (2-column layout)
   - Suggested substitutes

3. **Add AI Chat Modal**:
   - Full-screen overlay with blur background
   - Conversation interface
   - "Show reasoning" expandable panel
   - LM Studio integration

4. **Polish**:
   - Loading states (skeleton screens)
   - Error handling
   - Responsive breakpoints
   - Performance optimization

## Design Principles

1. **Minimalism**: No gradients, no saturated colors, no decoration
2. **Typography-first**: Text hierarchy drives design
3. **Whitespace**: Generous spacing, breathable layout
4. **Precision**: Clean charts, exact numbers, no clutter
5. **Subtle interactions**: 200ms transitions, hover states only

## Dependencies

Already in `requirements.txt`:
- streamlit
- pandas
- numpy
- plotly

## Tips

- **CSS customization**: Edit `ui/styles.css` for design changes
- **Color updates**: Modify `:root` variables in CSS
- **Chart styling**: See Plotly config in `app.py`
- **Metric cards**: Use HTML with CSS classes for consistency

---

Built for Valio Aimo Hackathon 2025 | Pure Python, No Mock Data
