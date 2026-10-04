"""
Vercel Serverless Function: SVM Retraining API
"""

import io
import json
import base64
from http.server import BaseHTTPRequestHandler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def fig_to_base64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def run_svm(params):
    n_samples = int(params.get('n_samples', 500))
    noise = float(params.get('noise', 0.25))
    c_val = float(params.get('c_val', 1.0))
    gamma = params.get('gamma', 'scale')
    poly_degree = int(params.get('poly_degree', 3))
    test_size = float(params.get('test_size', 0.30))
    random_state = int(params.get('random_state', 42))

    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        gamma_val = float(gamma)
    except (ValueError, TypeError):
        gamma_val = gamma if gamma in ['scale', 'auto'] else 'scale'

    kernels = {
        'Linear': SVC(kernel='linear', C=c_val, gamma=gamma_val, random_state=random_state),
        'Polynomial (d=' + str(poly_degree) + ')': SVC(kernel='poly', C=c_val, degree=poly_degree, gamma=gamma_val, random_state=random_state),
        'RBF (Gaussian)': SVC(kernel='rbf', C=c_val, gamma=gamma_val, random_state=random_state)
    }

    metrics = {}
    models = {}

    for name, model in kernels.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        n_sv = int(model.n_support_.sum())

        metrics[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'support_vectors': n_sv
        }
        models[name] = model

    bg_dark = '#0F172A'
    panel_bg = '#1E293B'
    text_light = '#F8FAFC'
    text_muted = '#94A3B8'
    border_col = '#334155'

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), facecolor=bg_dark)
    fig.subplots_adjust(top=0.75, bottom=0.12, left=0.05, right=0.96, wspace=0.22)
    fig.suptitle(f'Live SVM Decision Boundaries & Margins (N={n_samples}, σ={noise:.2f}, C={c_val}, γ={gamma})',
                 fontsize=13.5, fontweight='bold', color=text_light, y=0.93)

    h = 0.035
    x_min, x_max = X_train_scaled[:, 0].min() - 0.6, X_train_scaled[:, 0].max() + 0.6
    y_min, y_max = X_train_scaled[:, 1].min() - 0.6, X_train_scaled[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    cmap_surface = ListedColormap(['#1E293B', '#1E3A5F'])
    color_class0 = '#F97316'
    color_class1 = '#38BDF8'

    for ax, (name, model) in zip(axes, models.items()):
        ax.set_facecolor(panel_bg)
        Z_dec = model.decision_function(mesh_points).reshape(xx.shape)
        Z_pred = model.predict(mesh_points).reshape(xx.shape)

        ax.contourf(xx, yy, Z_pred, alpha=0.65, cmap=cmap_surface)
        ax.contour(xx, yy, Z_dec, levels=[-1.0], colors='#CBD5E1', linestyles=':', linewidths=1.2, alpha=0.75)
        ax.contour(xx, yy, Z_dec, levels=[0.0], colors='#FFFFFF', linestyles='-', linewidths=1.8)
        ax.contour(xx, yy, Z_dec, levels=[1.0], colors='#CBD5E1', linestyles=':', linewidths=1.2, alpha=0.75)

        ax.scatter(X_train_scaled[y_train == 0, 0], X_train_scaled[y_train == 0, 1],
                   c=color_class0, edgecolors='#0F172A', s=24, linewidths=0.5, alpha=0.9)
        ax.scatter(X_train_scaled[y_train == 1, 0], X_train_scaled[y_train == 1, 1],
                   c=color_class1, edgecolors='#0F172A', s=24, linewidths=0.5, alpha=0.9)

        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=60, facecolors='none', edgecolors='#F8FAFC',
                   linewidths=1.2, alpha=0.85)

        ax.set_title(f"{name}\nAcc: {metrics[name]['accuracy']*100:.1f}% | F1: {metrics[name]['f1']:.3f} | SVs: {metrics[name]['support_vectors']}",
                     fontsize=11, fontweight='bold', color=text_light, pad=8)
        ax.set_xlabel('Feature 1', fontsize=8.5, color=text_muted)
        ax.set_ylabel('Feature 2', fontsize=8.5, color=text_muted)
        ax.tick_params(colors=text_muted, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(border_col)
        ax.grid(True, linestyle='--', alpha=0.12, color='#64748B')

    boundaries_img = fig_to_base64(fig, dpi=140)

    fig2, ax2 = plt.subplots(figsize=(8.5, 3.8), facecolor=bg_dark)
    ax2.set_facecolor(panel_bg)

    kernel_names = list(metrics.keys())
    metric_keys = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    x = np.arange(len(kernel_names))
    width = 0.18
    palette = ['#38BDF8', '#34D399', '#FB923C', '#A78BFA']

    for i, (m_key, m_label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [metrics[k][m_key] for k in kernel_names]
        bars = ax2.bar(x + i * width, vals, width, label=m_label,
                       color=palette[i], alpha=0.92, edgecolor=bg_dark, linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f'{v:.2f}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color=text_light)

    ax2.set_title('Kernel Performance Profiles (Test Set Holdout)', fontsize=11.5, fontweight='bold', color=text_light, pad=10)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(kernel_names, fontsize=9, fontweight='bold', color=text_light)
    ax2.tick_params(colors=text_muted)
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis='y', alpha=0.15, color='#64748B', linestyle='--')
    ax2.legend(loc='lower right', facecolor='#0F172A', edgecolor=border_col,
               labelcolor=text_light, fontsize=8, framealpha=0.9)
    for spine in ax2.spines.values():
        spine.set_color(border_col)

    metrics_img = fig_to_base64(fig2, dpi=140)

    rbf_key = next((k for k in metrics if 'RBF' in k), 'RBF (Gaussian)')
    summary_text = (f"Trained 3 SVM models on {n_samples} make_moons samples (Noise={noise:.2f}, C={c_val}, γ={gamma}). "
                    f"RBF achieved {metrics[rbf_key]['accuracy']*100:.1f}% accuracy with {metrics[rbf_key]['support_vectors']} support vectors.")

    return {
        'status': 'success',
        'metrics': metrics,
        'boundaries_chart': boundaries_img,
        'metrics_chart': metrics_img,
        'summary': summary_text,
        'params': {
            'n_samples': n_samples,
            'noise': noise,
            'c_val': c_val,
            'poly_degree': poly_degree,
            'gamma': gamma
        },
        'environment': 'Vercel Serverless Function'
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        try:
            params = json.loads(post_data) if post_data else {}
        except Exception:
            params = {}

        try:
            res = run_svm(params)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(res).encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'error', 'error': str(e)}).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
