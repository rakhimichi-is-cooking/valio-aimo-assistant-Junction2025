# Supply Chain Dashboard - Proper Plan

## 🎯 What We're Actually Doing

**Core Purpose**: Predict and prevent product shortages before they happen

**Our Data**:
- Historical sales/deliveries (what was ordered vs what was delivered)
- Shortage risk predictions (1-day, 7-day, 21-day horizons)
- External factors (Finland holidays, weather, demand modifiers)
- Product relationships (substitutes, correlations)

**Risk Logic**:
- Risk = 1.0 - (forecast / average)
- High risk = forecast demand is BELOW average = potential shortage coming
- Critical: risk > 0.4 (40% below normal)
- At Risk: risk > 0.15 (15% below normal)

---

## 📊 What Supply Chain Dashboards Should Show

### Industry Standard Metrics:
1. **Service Level / Fill Rate** - % of orders fulfilled completely
2. **Inventory Health** - Current stock levels, days of supply
3. **Demand Forecast Accuracy** - How well predictions match reality
4. **Lead Time** - Time from order to delivery
5. **Stockout Risk** - Products likely to run out
6. **Backorder Status** - Unfulfilled orders
7. **Trend Analysis** - Demand patterns over time

### Our Focus (Based on Available Data):
1. **Shortage Risk Monitoring** - Which products will run out and when
2. **Fulfillment Performance** - Order vs delivery gap
3. **Demand Trends** - Historical patterns
4. **External Impact** - How holidays/events affect demand
5. **Product Health** - Which items are problematic

---

## 🎨 Dashboard Layout Plan (Final)

### TOP SECTION - At-a-Glance Status
```
┌─────────────────────────────────────────────────────┐
│ Supply Chain Status - [Date]                        │
│ [AI Summary: "X products at risk, Y% fulfillment"]  │
└─────────────────────────────────────────────────────┘

┌──────────┬──────────┬──────────┬──────────┐
│ Overall  │ Tomorrow │ 7 Days   │ 21 Days  │
│ Fill     │ At Risk  │ At Risk  │ At Risk  │
│ 94.2%    │ 3        │ 8        │ 15       │
└──────────┴──────────┴──────────┴──────────┘
```

**Why**: Quick health check - is everything OK or not?

---

### MIDDLE SECTION - Trends & Predictions

**Left Side (60%)**
```
Historical Demand & Forecast
┌─────────────────────────────────────────┐
│ [Line chart]                            │
│ - Last 60 days: actual demand           │
│ - Next 7 days: forecast (shaded)        │
│ - Markers for holidays/events           │
└─────────────────────────────────────────┘

Fulfillment Performance
┌─────────────────────────────────────────┐
│ [Bar chart]                             │
│ Ordered vs Delivered by week            │
│ Shows gap = potential shortages         │
└─────────────────────────────────────────┘
```

**Right Side (40%)**
```
Shortage Risk by Horizon
┌──────────────────────┐
│ [Stacked bar]        │
│ Tomorrow | 7d | 21d  │
│ Critical vs At Risk  │
└──────────────────────┘

Top Risk Products
┌──────────────────────┐
│ Product A  | 0.82    │
│ Product B  | 0.75    │
│ Product C  | 0.68    │
│ [Mini bars]          │
└──────────────────────┘
```

**Why**: Shows trends (what happened) AND predictions (what's coming)

---

### BOTTOM SECTION - Detailed Product List

```
Products Requiring Attention (Sorted by Risk)
┌─────────┬────────────────┬──────┬──────────┬─────────┬────────┐
│ Status  │ Product        │ Risk │ Horizon  │ Avg Dem │ Trend  │
├─────────┼────────────────┼──────┼──────────┼─────────┼────────┤
│ 🔴 Crit │ Milk 3.5%      │ 0.82 │ Tomorrow │ 450     │ -25%   │
│ 🔴 Crit │ Yogurt Berry   │ 0.75 │ 7 Days   │ 280     │ -18%   │
│ 🟡 Risk │ Cheese Slice   │ 0.42 │ 7 Days   │ 320     │ -12%   │
└─────────┴────────────────┴──────┴──────────┴─────────┴────────┘
```

**Why**: Actionable - tells you WHAT to fix, WHEN, and HOW bad it is

---

## 🔑 Key Metrics Explained

### 1. Fill Rate / Fulfillment %
**What**: (Delivered Qty / Ordered Qty) × 100
**Why**: Industry standard - 95%+ is good
**Show**: Big number at top

### 2. Shortage Risk Score
**What**: 1.0 - (forecast / average)
**Why**: Predicts problems before they happen
**Show**: Color-coded (red > 0.4, yellow > 0.15)

### 3. Demand Trend
**What**: Recent average vs historical average
**Why**: Shows if demand is rising/falling
**Show**: % change, line chart

### 4. Time Horizon
**What**: When will shortage happen (1d, 7d, 21d)
**Why**: Determines urgency of action
**Show**: Separate columns/cards

### 5. Average Demand
**What**: Mean daily order quantity
**Why**: Context for understanding scale
**Show**: In table, as baseline

---

## ❌ What NOT to Show

1. **Risk Distribution Histogram** - Too abstract, not actionable
2. **External Factors Line Chart** - Cluttered, confusing
3. **Duplicate metrics** - Don't show same data multiple ways
4. **Too many time horizons** - Focus on today, this week, this month
5. **Overly technical metrics** - Keep it business-focused

---

## ✅ Final Dashboard Structure

```
┌─────────────────────────────────────────────────────────────┐
│ HEADER: Status + Date + AI Summary (1 line)                │
├─────────────────────────────────────────────────────────────┤
│ METRICS: Fill Rate | Critical Today | At Risk 7d | At Risk 21d │
├──────────────────────────────┬──────────────────────────────┤
│ MAIN CHART:                  │ SIDE PANEL:                  │
│ - Historical demand (60d)    │ - Risk by horizon (bars)     │
│ - Forecast (7d ahead)        │ - Top 5 critical products    │
│ - Holiday markers            │ - Key external factors       │
├──────────────────────────────┴──────────────────────────────┤
│ FULFILLMENT CHART: Ordered vs Delivered (weekly bars)      │
├─────────────────────────────────────────────────────────────┤
│ PRODUCT TABLE: Top 20 at-risk products with actions        │
└─────────────────────────────────────────────────────────────┘
```

**Everything fits on ONE screen, every metric is actionable.**

---

## 📋 Implementation Checklist

1. ✅ Calculate overall fill rate from sales data
2. ✅ Show historical + forecast in same chart (with future shaded)
3. ✅ Add ordered vs delivered comparison chart
4. ✅ Simplify metric cards to most important only
5. ✅ Make product table sortable and filterable
6. ✅ Add holiday markers on timeline
7. ✅ Show "what to do" for each critical product
8. ✅ Keep color coding consistent (red/yellow/green)

---

## 🎯 Success Criteria

**User should be able to answer in 5 seconds:**
- Is everything OK? (Fill rate + critical count)
- What needs attention TODAY? (Critical products)
- What's the trend? (Demand chart)
- Why is this happening? (External factors)
- What should I do? (Action column in table)

**Data should make sense:**
- High risk = shortage likely = need to order more
- Fill rate < 100% = some orders not fulfilled
- Trend down = demand falling = higher risk
- Holidays = demand changes = adjust accordingly

