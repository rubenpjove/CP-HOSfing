import pandas as pd
from graphviz import Digraph
import re
import plotly.express as px
from collections import defaultdict
import numpy as np
import os

# Auxiliary functions

unknown_label=["<mUnk>","<MUnk>"]

def _truncate_after_unknown(df, cols, unknown_label=unknown_label):
    out = df[cols].copy()
    
    # For each level, check if any parent has only unknown children
    for i in range(len(cols)):
        col = cols[i]
        
        # Group by parent values and check children
        if i == 0:
            # For first level, check if all values are unknown
            level_values = out[col].dropna().unique()
            non_null_values = [v for v in level_values if pd.notna(v)]
            if non_null_values and all(v in unknown_label for v in non_null_values):
                # All values at this level are unknown, truncate here
                out.loc[out[col].isin(unknown_label), col] = None
                # Set all subsequent columns to None for these rows
                for j in range(i+1, len(cols)):
                    out.loc[out[col].isna(), cols[j]] = None
                break
        else:
            # For deeper levels, check each parent group
            parent_cols = cols[:i]
            child_col = col
            
            # Group by parent values and check if any group has only unknown children
            for parent_vals, group in out.groupby(parent_cols):
                # Check if any parent value is NaN
                if isinstance(parent_vals, tuple):
                    if any(pd.isna(val) for val in parent_vals):
                        continue
                else:
                    if pd.isna(parent_vals):
                        continue
                    
                # Get children values for this parent group
                children = group[child_col].dropna().unique()
                non_null_children = [v for v in children if pd.notna(v)]
                
                # If all children are unknown, truncate here
                if non_null_children and all(v in unknown_label for v in non_null_children):
                    # Create mask for this parent group
                    mask = True
                    for j, parent_col in enumerate(parent_cols):
                        if isinstance(parent_vals, tuple):
                            mask = mask & (out[parent_col] == parent_vals[j])
                        else:
                            mask = mask & (out[parent_col] == parent_vals)
                    
                    # Set unknown children to None and truncate subsequent levels
                    out.loc[mask & out[child_col].isin(unknown_label), child_col] = None
                    for j in range(i+1, len(cols)):
                        out.loc[mask & out[child_col].isna(), cols[j]] = None
    
    return out

#################################################################################
# Interactive plot functions
#################################################################################

def circle_plot(df, cols, ignore_unknown=False):
    if ignore_unknown:
        df = _truncate_after_unknown(df, cols)
    clean = df[cols].fillna(unknown_label[0])  # Use first unknown label as string for non-ignored cases
    counts = clean.value_counts().reset_index(name='count')
    fig = px.sunburst(counts, path=cols, values='count', color=cols[0])
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    return fig

def square_plot(df, cols, ignore_unknown=False):
    if ignore_unknown:
        df = _truncate_after_unknown(df, cols)
    clean = df[cols].fillna(unknown_label[0])  # Use first unknown label as string for non-ignored cases
    counts = clean.value_counts().reset_index(name='count')
    fig = px.treemap(counts, path=cols, values='count', color=cols[0])
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    return fig

def stacked_plot(df, cols, ignore_unknown=False):
    if ignore_unknown:
        df = _truncate_after_unknown(df, cols)
    clean = df[cols].fillna(unknown_label[0])  # Use first unknown label as string for non-ignored cases
    counts = clean.value_counts().reset_index(name='count')
    fig = px.icicle(counts, path=cols, values='count', color=cols[0])
    fig.update_layout(margin=dict(t=30, l=0, r=0, b=0))
    return fig

#################################################################################
# Text tree functions
#################################################################################

def _nested_defaultdict():
    return defaultdict(_nested_defaultdict)

def _sum_counts(node):
    total = node.get('_count', 0)
    for k, v in node.items():
        if k != '_count':
            total += _sum_counts(v)
    return total

def text_tree(df, cols, ignore_unknown=False):
    if ignore_unknown:
        df = _truncate_after_unknown(df, cols)
    
    tree = _nested_defaultdict()
    for row in df[cols].itertuples(index=False):
        node = tree
        for key in row:
            is_unknown = pd.isna(key) or key in unknown_label
            normalized_key = unknown_label[0] if is_unknown else key  # Use first unknown label as string
            if ignore_unknown and is_unknown:
                node = node[normalized_key]
                node['_count'] = node.get('_count', 0) + 1
                break
            node = node[normalized_key]
        else:
            node['_count'] = node.get('_count', 0) + 1
    return tree

def _print_tree(node, depth=0, label=None):
    indent = "  " * depth
    if label is not None:
        print(f"{indent}{label}  ({_sum_counts(node)})")
    for k, v in node.items():
        if k == '_count':
            continue
        _print_tree(v, depth + (0 if label is None else 1), k)

def print_text_tree(node):
    _print_tree(node)

# usage example (kept as reference; do not execute at import time)
# example_tree = build_tree_stop_unknown(df, targets, unknown_label='Unknown')
# print_tree(example_tree)


#################################################################################
# Graphviz tree functions
#################################################################################

def graph_tree(df, cols, ignore_unknown=False, max_children_per_node=30):
    g = Digraph('os_tree', format='png')
    g.attr(rankdir='LR', fontsize='10')

    def _sanitize_id(raw: str) -> str:
        # Graphviz node IDs cannot safely contain characters like ':', '|', quotes, etc.
        # Replace any non-alphanumeric/underscore with '_', and ensure it doesn't start with a digit
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(raw))
        if safe and safe[0].isdigit():
            safe = f"n_{safe}"
        return safe

    if ignore_unknown:
        trunc = _truncate_after_unknown(df, cols)
        counts = trunc.value_counts(dropna=False).reset_index(name='count')
        use_dropna = True
    else:
        counts = df[cols].fillna(unknown_label[0]).value_counts().reset_index(name='count')  # Use first unknown label as string
        use_dropna = False

    root_id = 'ROOT'
    total_count = int(counts['count'].sum())
    g.node(root_id, f"All ({total_count})")

    for depth in range(len(cols)):
        parents = cols[:depth]
        this_cols = cols[:depth+1]

        if depth > 0:
            if use_dropna:
                parent_groups = counts.groupby(parents, dropna=True)['count'].sum().reset_index()
            else:
                parent_groups = counts.groupby(parents)['count'].sum().reset_index()
            for _, row in parent_groups.iterrows():
                parent_vals = row[parents].tolist() if isinstance(parents, list) and len(parents) > 1 else [row[p] for p in parents]
                parent_id_raw = '|'.join(map(str, parent_vals))
                parent_id = _sanitize_id(parent_id_raw)
                parent_lbl = parent_id_raw.replace('|', ' / ')
                g.node(parent_id, f"{parent_lbl} ({int(row['count'])})")

        if use_dropna:
            child_groups = counts.groupby(this_cols, dropna=True)['count'].sum().reset_index()
        else:
            child_groups = counts.groupby(this_cols)['count'].sum().reset_index()

        if depth == 0:
            if use_dropna:
                sub = child_groups[child_groups[cols[0]].notna()]
            else:
                sub = child_groups
            sub = sub.sort_values('count', ascending=False).head(max_children_per_node)
            for _, r in sub.iterrows():
                child_val = r[cols[0]]
                child_id = _sanitize_id(str(child_val))
                g.node(child_id, f"{child_val} ({int(r['count'])})")
                g.edge(root_id, child_id)
        else:
            if use_dropna:
                for parent_vals, subdf in child_groups.groupby(parents, dropna=True):
                    if not isinstance(parent_vals, tuple):
                        parent_vals = (parent_vals,)
                    parent_id_raw = '|'.join(map(str, parent_vals))
                    parent_id = _sanitize_id(parent_id_raw)
                    sub = subdf[subdf[cols[depth]].notna()]
                    sub = sub.sort_values('count', ascending=False).head(max_children_per_node)
                    for _, r in sub.iterrows():
                        child_vals = [r[c] for c in this_cols]
                        child_id_raw = '|'.join(map(str, child_vals))
                        child_id = _sanitize_id(child_id_raw)
                        child_lbl = f"{child_vals[-1]} ({int(r['count'])})"
                        g.node(child_id, child_lbl)
                        g.edge(parent_id, child_id)
            else:
                for parent_vals, subdf in child_groups.groupby(parents):
                    if not isinstance(parent_vals, tuple):
                        parent_vals = (parent_vals,)
                    parent_id_raw = '|'.join(map(str, parent_vals))
                    parent_id = _sanitize_id(parent_id_raw)
                    sub = subdf.sort_values('count', ascending=False).head(max_children_per_node)
                    for _, r in sub.iterrows():
                        child_vals = [r[c] for c in this_cols]
                        child_id_raw = '|'.join(map(str, child_vals))
                        child_id = _sanitize_id(child_id_raw)
                        child_lbl = f"{child_vals[-1]} ({int(r['count'])})"
                        g.node(child_id, child_lbl)
                        g.edge(parent_id, child_id)

    return g


# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------

def _sum_counts_for_write(node):
    total = node.get('_count', 0)
    for k, v in node.items():
        if k != '_count':
            total += _sum_counts_for_write(v)
    return total


def _write_text_tree(node, fh, depth=0, label=None):
    indent = "  " * depth
    if label is not None:
        fh.write(f"{indent}{label}  ({_sum_counts_for_write(node)})\n")
    for k, v in node.items():
        if k == '_count':
            continue
        _write_text_tree(v, fh, depth + (0 if label is None else 1), k)


def export_label_hierarchy_artifacts(df, out_dir, cols=None):
    import os

    if cols is None:
        cols = ["OS family", "OS major", "OS minor"]

    os.makedirs(out_dir, exist_ok=True)

    # Text trees
    tree_no_ignore = text_tree(df, cols, ignore_unknown=False)
    with open(os.path.join(out_dir, "labels_text_tree.txt"), "w", encoding="utf-8") as fh:
        _write_text_tree(tree_no_ignore, fh)

    tree_ignore = text_tree(df, cols, ignore_unknown=True)
    with open(os.path.join(out_dir, "labels_text_tree_ignore_unknown.txt"), "w", encoding="utf-8") as fh:
        _write_text_tree(tree_ignore, fh)

    # Graph trees (PNG)
    try:
        g_no_ignore = graph_tree(df, cols, ignore_unknown=False)
        g_no_ignore.render(os.path.join(out_dir, "labels_graph"), format='png', cleanup=True)
    except Exception as e:
        print(f"[WARN] Failed to render labels_graph PNG: {e}")
    try:
        g_ignore = graph_tree(df, cols, ignore_unknown=True)
        g_ignore.render(os.path.join(out_dir, "labels_graph_ignore_unknown"), format='png', cleanup=True)
    except Exception as e:
        print(f"[WARN] Failed to render labels_graph_ignore_unknown PNG: {e}")

    # Per-level value_counts
    for col in cols:
        vc = df[col].value_counts(dropna=False)
        out_path = os.path.join(out_dir, f"value_counts__{col.replace(' ', '_')}.csv")
        vc.to_frame(name="count").to_csv(out_path)
