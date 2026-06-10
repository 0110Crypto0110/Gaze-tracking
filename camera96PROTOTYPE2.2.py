"""VERSAO ATUAL
Anotaçoes:
   
    (cuidados ao treinar o modelo)
        *limitar quantidade de pontos;
        *limitar quantidade e diversidade de variaçoes;
        *manter constancia no treinamento de ponto e diversidade para melhor convergir.

    (Ultima atulizaçao):
        *smoothpursuit( perseguiçao ) como soluçao a interpolaçao de pontos existentes;

    (Proxima atualizaçao):
        *avaliar persistencia, do treinamento de perseguiçao.

    IMPORTANTE:
        *Metrica imprecisa e engessada, pois ha variaçao de distancia durante uso, treinamento, etc; mudar em proximas implementaçoes.
            -USER_DISTANCE_CM = 60.0

        *Apos coletar as metricas, ainda preciso escolher e relacionar elas, alem de separar por inteçao(perseguiçao,precisao, como o erro evolui durante uma sessao de treinamento continuo, avaliar esquecimento catastrofico...)
             
            ou seja, definir as combinaçoes ideais de metricas para cada parte proposta no artigo.

"""
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from filterpy.kalman import UnscentedKalmanFilter as UKFFilterpy
from filterpy.kalman import MerweScaledSigmaPoints
import joblib
import csv
import os
import time
from collections import deque, namedtuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import random
import psutil
import uuid
import math

# ================= CONFIGURAÇÕES =================
GAZE_CSV = "gaze_datav00006.csv"
GAZE_MODEL_FILE = "gaze_modelv00006.pth"
GAZE_SCALER_FILE = "gaze_scalerv00006.pkl"
METRICS_CSV = "metricas00006.csv"

EXPECTED_GAZE_INPUT_DIM = 87

# ================= CONTROLE DAS 11 FEATURES UPV =================
USE_UPV = False  # Alternado pela tecla 'u'

# ================= CONVERGÊNCIA =================
USE_CONVERGENCE_MECHANIC = True
CONVERGENCE_MAX_DIST = 250
CONVERGENCE_RESET_FRAMES = 5

# ================= MÉTRICAS DE DISTÂNCIA =================
USER_DISTANCE_CM = 60.0 
SCREEN_WIDTH_CM = 53.0

# ================= CONFIGURAÇÕES CÂMERA =================
camera_index = 0
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    print("❌ Não foi possível acessar a câmera.")
    exit()

screen_w, screen_h = pyautogui.size()
aspect = screen_w / screen_h
pixels_per_cm = screen_w / SCREEN_WIDTH_CM

# ================= MEDIAPIPE =================
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# ================= ÍNDICES GLOBAIS =================
RIGHT_PUPIL_IDX = [469, 470, 471, 472]
LEFT_PUPIL_IDX = [474, 475, 476, 477]
RIGHT_UPPER_IDX = [159, 160, 161]
RIGHT_LOWER_IDX = [145, 144, 153]
LEFT_UPPER_IDX = [386, 387, 388]
LEFT_LOWER_IDX = [374, 380, 381]
RIGHT_EYEBROW_IDX = [336, 285, 295, 282, 283, 276, 383, 300, 293, 334, 296]
LEFT_EYEBROW_IDX = [70, 55, 65, 52, 53, 46, 156, 105, 66, 107, 63]
RIGHT_EYELID_IDX = [155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 157, 173]
LEFT_EYELID_IDX = [398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]
RIGHT_IRIS_IDX = [468]
LEFT_IRIS_IDX = [473]

RIGHT_EYE_ALL = RIGHT_EYELID_IDX + RIGHT_IRIS_IDX + RIGHT_PUPIL_IDX
LEFT_EYE_ALL = LEFT_EYELID_IDX + LEFT_IRIS_IDX + LEFT_PUPIL_IDX

# ================= VARIÁVEIS GLOBAIS =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaze_model = None
optimizer = None
criterion = nn.MSELoss()
model_trained = False
pose_scalers = {}
focus_points = []
id_to_node = {}

# Filtros
ukf_initialized = False
class UnscentedKalmanFilter:
    def __init__(self, dim_state=4, dim_meas=2):
        self.dim_x = dim_state
        self.dim_z = dim_meas
        points = MerweScaledSigmaPoints(dim_state, alpha=0.001, beta=2.0, kappa=0)
        def fx(x, dt):
            F = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
            return F @ x
        def hx(x):
            return x[:2]
        self._ukf = UKFFilterpy(dim_x=dim_state, dim_z=dim_meas,
                                dt=1/30.0, fx=fx, hx=hx, points=points)
        self._ukf.Q = np.eye(dim_state)
        self._ukf.R = np.eye(dim_meas)
    @property
    def x(self): return self._ukf.x
    @x.setter
    def x(self, val): self._ukf.x[:] = val
    @property
    def P(self): return self._ukf.P
    @P.setter
    def P(self, val): self._ukf.P[:] = val
    @property
    def Q(self): return self._ukf.Q
    @Q.setter
    def Q(self, val): self._ukf.Q = val
    @property
    def R(self): return self._ukf.R
    @R.setter
    def R(self, val): self._ukf.R = val
    def predict(self):
        self._ukf.predict()
        return self._ukf.x.copy()
    def update(self, z):
        self._ukf.update(z)
        return self._ukf.x.copy()

ukf = UnscentedKalmanFilter(dim_state=4, dim_meas=2)
ukf.Q *= 0.12
ukf.R *= 0.5

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()
        dt = t - self.t_prev
        if dt <= 1e-6:
            return self.x_prev.copy()
        edx = (x - self.x_prev) / dt
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx_prev)
        alpha_d = 1.0 / (1.0 + self.d_cutoff / cutoff * dt)
        dx = alpha_d * edx + (1 - alpha_d) * self.dx_prev
        alpha = 1.0 / (1.0 + self.min_cutoff / cutoff * dt)
        x_filtered = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_filtered.copy()
        self.dx_prev = dx.copy()
        self.t_prev = t
        return x_filtered

one_euro = OneEuroFilter(min_cutoff=2.5, beta=0.08, d_cutoff=1.5)
ema_point = None
ema_alpha_base = 0.30
velocity_scale = 0.0005

landmark_buffer = deque(maxlen=3)
alpha_lm = 0.6
alpha_bright = 0.7
Landmark = namedtuple('Landmark', ['x', 'y', 'z'])

# ================= QUADTREE =================
class QuadtreeGaze:
    def __init__(self, x, y, width, height, level=0, max_level=8):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        self.level = level
        self.max_level = max_level
        self.children = None
        self.center = (x + width//2, y + height//2)
        self.region_id = None
        self.is_leaf = True

    def subdivide(self):
        if self.level >= self.max_level:
            return
        half_w, half_h = self.w // 2, self.h // 2
        nw = QuadtreeGaze(self.x, self.y, half_w, half_h, self.level+1, self.max_level)
        ne = QuadtreeGaze(self.x + half_w, self.y, half_w, half_h, self.level+1, self.max_level)
        sw = QuadtreeGaze(self.x, self.y + half_h, half_w, half_h, self.level+1, self.max_level)
        se = QuadtreeGaze(self.x + half_w, self.y + half_h, half_w, half_h, self.level+1, self.max_level)
        self.children = [nw, ne, sw, se]
        self.is_leaf = False

    def merge(self):
        if self.is_leaf:
            return
        if all(child.is_leaf for child in self.children):
            self.children = None
            self.is_leaf = True

    def get_all_points(self):
        if self.is_leaf:
            return [(self.center[0], self.center[1], self.region_id)]
        pts = []
        for child in self.children:
            pts.extend(child.get_all_points())
        return pts

    def find_leaf_containing_point(self, px, py):
        if not (self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h):
            return None
        if self.is_leaf:
            return self
        for child in self.children:
            leaf = child.find_leaf_containing_point(px, py)
            if leaf:
                return leaf
        return None

    def find_parent_of_point(self, px, py, parent=None):
        if not (self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h):
            return None, None
        if self.is_leaf:
            return parent, self
        for child in self.children:
            p, c = child.find_parent_of_point(px, py, parent=self)
            if c is not None:
                return p, c
        return None, None

def build_initial_quadtree():
    root = QuadtreeGaze(0, 0, screen_w, screen_h, max_level=8)
    def subdivide_to_level(node, target_level):
        if node.level < target_level and node.level < node.max_level:
            node.subdivide()
            for child in node.children:
                subdivide_to_level(child, target_level)
    subdivide_to_level(root, 2)
    assign_region_ids(root)
    return root

def assign_region_ids(node, next_id=0):
    if node.is_leaf:
        node.region_id = next_id
        return next_id + 1
    for child in node.children:
        next_id = assign_region_ids(child, next_id)
    return next_id

def get_all_focus_points_and_map(root):
    points_data = root.get_all_points()
    points = [(x, y) for (x, y, _) in points_data]
    id_map = {}
    for (x, y, rid) in points_data:
        node = root.find_leaf_containing_point(x, y)
        if node:
            id_map[rid] = node
    return points, id_map

def draw_quadtree_structure(frame, node):
    """Desenha linhas conectando o centro de um nó pai aos centros dos filhos,
       visualizando a hierarquia da quadtree."""
    if node.is_leaf or node.children is None:
        return
    for child in node.children:
        cv2.line(frame,
                 (int(node.center[0]), int(node.center[1])),
                 (int(child.center[0]), int(child.center[1])),
                 (200, 200, 200), 1)
        draw_quadtree_structure(frame, child)

# ================= REDE NEURAL COM HOOKS =================
class GazeNet(nn.Module):
    def __init__(self, input_dim=87):
        super(GazeNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 2)
        self.activations = {}
        self._register_hooks()

    def _register_hooks(self):
        def hook_fn(name):
            def fn(module, input, output):
                self.activations[name] = output.detach()
            return fn
        self.fc1.register_forward_hook(hook_fn('fc1'))
        self.fc2.register_forward_hook(hook_fn('fc2'))
        self.fc3.register_forward_hook(hook_fn('fc3'))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

    def get_activation_metrics(self):
        metrics = {}
        for name in ['fc1', 'fc2', 'fc3']:
            act = self.activations.get(name)
            if act is not None:
                metrics[f'{name}_mean'] = act.mean().item()
                metrics[f'{name}_std'] = act.std().item()
                metrics[f'{name}_active_ratio'] = (act > 0).float().mean().item()
                metrics[f'{name}_l2'] = act.norm(p=2).item()
            else:
                metrics[f'{name}_mean'] = float('nan')
                metrics[f'{name}_std'] = float('nan')
                metrics[f'{name}_active_ratio'] = float('nan')
                metrics[f'{name}_l2'] = float('nan')
        return metrics

def convert_old_state_dict(state_dict):
    mapping = {
        'net.0.weight': 'fc1.weight', 'net.0.bias': 'fc1.bias',
        'net.2.weight': 'fc2.weight', 'net.2.bias': 'fc2.bias',
        'net.4.weight': 'fc3.weight', 'net.4.bias': 'fc3.bias',
        'net.6.weight': 'fc4.weight', 'net.6.bias': 'fc4.bias',
    }
    new_dict = {}
    for k, v in state_dict.items():
        if k in mapping:
            new_dict[mapping[k]] = v
        else:
            new_dict[k] = v
    return new_dict

# ================= REPLAY BUFFER =================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, target):
        self.buffer.append((state, target))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

replay_buffer = ReplayBuffer(capacity=2000)

# ================= EWC AVANÇADO =================
fisher_short = {}
permanent_fishers = []
region_stats = {}
theta_slow = {}
online_steps_count = 0

class AdaptiveHyperparams:
    def __init__(self, base_lamda=80.0, base_alpha=0.52):
        self.base_lamda = base_lamda
        self.base_alpha = base_alpha
        self.recent_losses = deque(maxlen=5)
    def update_and_get(self, current_loss):
        self.recent_losses.append(current_loss)
        if len(self.recent_losses) < 2:
            return self.base_lamda, self.base_alpha
        avg_loss = sum(self.recent_losses) / len(self.recent_losses)
        ratio = avg_loss / (current_loss + 1e-6)
        adaptive_lamda = self.base_lamda * ratio
        adaptive_lamda = max(10.0, min(adaptive_lamda, 3000.0))
        return adaptive_lamda, self.base_alpha

adaptive_controller = AdaptiveHyperparams()

def get_pose_zone(yaw, pitch):
    if yaw < -20: yaw_zone = 0
    elif yaw > 20: yaw_zone = 2
    else: yaw_zone = 1
    if pitch < -15: pitch_zone = 0
    elif pitch > 15: pitch_zone = 2
    else: pitch_zone = 1
    return (yaw_zone, pitch_zone)

def get_region_id_from_quadtree(gaze_point, quadtree_root, radius=80):
    node = quadtree_root.find_leaf_containing_point(gaze_point[0], gaze_point[1])
    if node is not None:
        return node.region_id
    best_dist = radius
    best_id = None
    leaves = []
    def collect_leaves(n):
        if n.is_leaf:
            leaves.append((n.center[0], n.center[1], n.region_id))
        else:
            for c in n.children:
                collect_leaves(c)
    collect_leaves(quadtree_root)
    for fx, fy, rid in leaves:
        dist = np.hypot(gaze_point[0]-fx, gaze_point[1]-fy)
        if dist < best_dist:
            best_dist = dist
            best_id = rid
    return best_id

def get_effective_fisher():
    combined = {}
    for name, val in fisher_short.items():
        combined[name] = val.clone()
    for perm in permanent_fishers:
        if perm is None: continue
        for name, val in perm.items():
            if name in combined:
                combined[name] += val
            else:
                combined[name] = val.clone()
    return combined

def update_ewc_and_consolidate(model, region_id, error_px, confidence, quadtree_root):
    global fisher_short, permanent_fishers, region_stats, focus_points, id_to_node
    if confidence < 0.4 or region_id is None:
        return
    if region_id not in region_stats:
        region_stats[region_id] = {
            'errors': deque(maxlen=50),
            'total_samples': 0,
            'promoted': False,
            'perm_fisher_idx': None,
            'last_eval': time.time(),
            'consecutive_bad': 0,
            'samples_raw': [],
            'samples_scaled': [],
            'targets': []
        }
    stats = region_stats[region_id]
    stats['errors'].append(error_px)
    stats['total_samples'] += 1
    avg_error = np.mean(stats['errors'])

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_sq = param.grad.data ** 2
            if name not in fisher_short:
                fisher_short[name] = torch.zeros_like(param.data)
            fisher_short[name] = (0.99 * fisher_short[name] +
                                  (1 - 0.99) * grad_sq)

    now = time.time()
    if stats['promoted'] and (now - stats['last_eval']) > (100 / 30.0):
        stats['last_eval'] = now
        if avg_error > 45.0 and len(stats['errors']) >= 50:
            if stats['perm_fisher_idx'] is not None:
                permanent_fishers[stats['perm_fisher_idx']] = None
                stats['promoted'] = False
                stats['perm_fisher_idx'] = None
                stats['total_samples'] = 0
                stats['errors'].clear()
                print(f"⚠️ Região {region_id} REBAIXADA")

    if not stats['promoted'] and stats['total_samples'] >= 20:
        if avg_error <= 30.0:
            perm = {}
            for name, val in fisher_short.items():
                perm[name] = val.clone()
            permanent_fishers.append(perm)
            stats['promoted'] = True
            stats['perm_fisher_idx'] = len(permanent_fishers) - 1
            print(f"🌟 Região {region_id} PROMOVIDA! Erro médio: {avg_error:.1f}px")

def lookahead_layerwise(model, alpha_base=0.52):
    effective_fisher = get_effective_fisher()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in effective_fisher:
                fisher_val = effective_fisher[name].mean().item()
                alpha = alpha_base / (1.0 + fisher_val)
                alpha = max(0.01, min(0.95, alpha))
            else:
                alpha = alpha_base
            if name in theta_slow:
                theta_slow[name] += alpha * (param.data - theta_slow[name])
            else:
                theta_slow[name] = param.data.clone()

def ewc_loss_with_fisher(model, base_loss, fisher_dict, lamda):
    penalty = 0.0
    if not fisher_dict or not theta_slow:
        return base_loss, 0.0
    for name, param in model.named_parameters():
        if name in fisher_dict and name in theta_slow:
            penalty += torch.sum(fisher_dict[name] * (param - theta_slow[name]) ** 2)
    total_loss = base_loss + (lamda / 2) * penalty
    return total_loss, penalty.item()

def online_train_step(input_vec, target_vec, gaze_point_pixels, confidence, yaw, pitch, quadtree_root):
    global gaze_model, optimizer, online_steps_count
    global fisher_short, permanent_fishers, region_stats, theta_slow
    global screen_w, screen_h, focus_points, pose_scalers, id_to_node

    train_metrics = {}
    t_start = time.perf_counter()

    if gaze_model is None:
        return train_metrics

    zone = get_pose_zone(yaw, pitch)
    if zone not in pose_scalers:
        pose_scalers[zone] = StandardScaler()
    scaler = pose_scalers[zone]

    if not hasattr(scaler, 'mean_'):
        scaler.partial_fit(input_vec)
        inp_scaled = input_vec.copy()
    else:
        inp_scaled = scaler.transform(input_vec)

    gaze_model.train()
    x_tensor = torch.tensor(inp_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor([target_vec], dtype=torch.float32).to(device)

    pred = gaze_model(x_tensor)
    base_loss = criterion(pred, y_tensor)
    current_loss_val = base_loss.item()

    target_px = np.array(target_vec) * np.array([screen_w, screen_h])
    pred_px = pred.cpu().detach().numpy()[0] * np.array([screen_w, screen_h])
    error_px = np.linalg.norm(target_px - pred_px)
    region_id = get_region_id_from_quadtree(gaze_point_pixels, quadtree_root)

    weights_before = {n: p.clone().detach() for n, p in gaze_model.named_parameters()}

    optimizer.zero_grad()
    base_loss.backward(retain_graph=True)

    grad_norms = {}
    for name, param in gaze_model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = 0.0

    update_ewc_and_consolidate(gaze_model, region_id, error_px, confidence, quadtree_root)

    effective_fisher = get_effective_fisher()
    current_lamda, current_alpha = adaptive_controller.update_and_get(current_loss_val)
    total_loss, ewc_penalty = ewc_loss_with_fisher(gaze_model, base_loss, effective_fisher, current_lamda)
    total_loss.backward()
    optimizer.step()

    weights_after = {n: p.clone().detach() for n, p in gaze_model.named_parameters()}
    weight_changes = {}
    for name in weights_before:
        diff = (weights_after[name] - weights_before[name]).norm().item()
        weight_changes[name] = diff

    replay_buffer.push(x_tensor.squeeze(0), y_tensor.squeeze(0))
    replay_size = len(replay_buffer)

    if replay_size >= 16:
        batch = replay_buffer.sample(16)
        batch_x = torch.stack([item[0] for item in batch]).to(device)
        batch_y = torch.stack([item[1] for item in batch]).to(device)
        optimizer.zero_grad()
        pred_batch = gaze_model(batch_x)
        loss_batch = criterion(pred_batch, batch_y)
        loss_batch.backward()
        optimizer.step()

    online_steps_count += 1
    lookahead_performed = False
    if online_steps_count % 10 == 0 and theta_slow:
        lookahead_layerwise(gaze_model, adaptive_controller.base_alpha)
        lookahead_performed = True

    if online_steps_count % 1000 == 0:
        permanent_fishers = [p for p in permanent_fishers if p is not None]

    train_metrics['train_loss'] = current_loss_val
    train_metrics['train_time_ms'] = (time.perf_counter() - t_start) * 1000
    train_metrics['region_id'] = region_id if region_id is not None else -1
    train_metrics['ewc_loss'] = ewc_penalty
    train_metrics['lambda_ewc'] = current_lamda
    fisher_tensors = list(fisher_short.values())
    if fisher_tensors:
        all_fisher = torch.cat([f.flatten() for f in fisher_tensors])
        train_metrics['fisher_mean'] = all_fisher.mean().item()
        train_metrics['fisher_max'] = all_fisher.max().item()
    else:
        train_metrics['fisher_mean'] = 0.0
        train_metrics['fisher_max'] = 0.0
    train_metrics['num_promoted_tasks'] = len([p for p in permanent_fishers if p is not None])
    train_metrics['lookahead_alpha'] = adaptive_controller.base_alpha if lookahead_performed else 0.0
    if lookahead_performed and theta_slow:
        dist_sum = 0.0
        count = 0
        for name, param in gaze_model.named_parameters():
            if name in theta_slow:
                dist_sum += (param.data - theta_slow[name]).norm().item()
                count += 1
        train_metrics['lookahead_distance'] = dist_sum / max(1, count)
    else:
        train_metrics['lookahead_distance'] = 0.0
    train_metrics['replay_buffer_size'] = replay_size
    train_metrics['quadtree_leaves'] = len(focus_points)
    train_metrics['quadtree_avg_depth'] = 0
    for name in ['fc1.weight', 'fc2.weight', 'fc3.weight', 'fc4.weight']:
        train_metrics[f'weight_norm_{name.replace(".","_")}'] = weights_before[name].norm().item()
        train_metrics[f'grad_norm_{name.replace(".","_")}'] = grad_norms.get(name, 0.0)
        train_metrics[f'weight_change_{name.replace(".","_")}'] = weight_changes.get(name, 0.0)
    train_metrics['error_px'] = error_px
    return train_metrics

# ================= FUNÇÕES AUXILIARES =================
def save_data_csv(filename, data):
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data[0] + data[1])

def load_data_csv(filename):
    if not os.path.isfile(filename):
        return np.empty((0,)), np.empty((0,))
    with open(filename, mode='r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) == 0:
        return np.empty((0,)), np.empty((0,))
    X, y = [], []
    expected_input = EXPECTED_GAZE_INPUT_DIM
    output_len = 2
    for row in rows:
        try:
            row_floats = [float(x) for x in row]
            if len(row_floats) < output_len:
                continue
            inp = row_floats[:-output_len]
            out = row_floats[-output_len:]
            if len(out) != output_len:
                continue
            if len(inp) < expected_input:
                inp += [0.0] * (expected_input - len(inp))
            elif len(inp) > expected_input:
                inp = inp[:expected_input]
            X.append(inp)
            y.append(out)
        except ValueError:
            pass
    if len(X) == 0:
        return np.empty((0,)), np.empty((0,))
    return np.array(X), np.array(y)

def save_gaze_models(model, scalers_dict):
    try:
        if os.path.isfile(GAZE_MODEL_FILE):
            os.replace(GAZE_MODEL_FILE, GAZE_MODEL_FILE + '.bak')
        if os.path.isfile(GAZE_SCALER_FILE):
            os.replace(GAZE_SCALER_FILE, GAZE_SCALER_FILE + '.bak')
        torch.save(model.state_dict(), GAZE_MODEL_FILE)
        joblib.dump(scalers_dict, GAZE_SCALER_FILE)
        print("💾 Modelo PyTorch e dicionário de scalers salvos.")
    except Exception as e:
        print("❌ Erro ao salvar modelos:", e)

def auto_load_models():
    global gaze_model, optimizer, model_trained, pose_scalers
    global theta_slow, fisher_short, permanent_fishers, region_stats
    if os.path.isfile(GAZE_MODEL_FILE) and os.path.isfile(GAZE_SCALER_FILE):
        try:
            loaded = joblib.load(GAZE_SCALER_FILE)
            if isinstance(loaded, dict):
                pose_scalers = loaded
            else:
                print("⚠️ Arquivo de scaler antigo detectado.")
                pose_scalers = {}
            gaze_model = GazeNet(input_dim=EXPECTED_GAZE_INPUT_DIM).to(device)
            state_dict = torch.load(GAZE_MODEL_FILE, map_location=device)
            state_dict = convert_old_state_dict(state_dict)
            gaze_model.load_state_dict(state_dict)
            optimizer = optim.Adam(gaze_model.parameters(), lr=0.001)
            model_trained = True
            theta_slow = {n: p.data.clone() for n, p in gaze_model.named_parameters()}
            fisher_short = {}
            permanent_fishers = []
            region_stats = {}
            print("▶️ Modelo carregado com âncoras EWC.")
        except Exception as e:
            print(f"⚠️ Falha ao carregar: {e}")
    else:
        print("Nenhum modelo encontrado.")

def train_model_from_csv():
    global gaze_model, optimizer, model_trained, pose_scalers
    global theta_slow, fisher_short, permanent_fishers, region_stats
    X, y = load_data_csv(GAZE_CSV)
    if X.shape[0] == 0:
        print("❌ Nenhum dado no CSV.")
        return
    print("Construindo scalers por pose...")
    pose_scalers = {}
    for i in range(X.shape[0]):
        yaw, pitch = X[i, 0], X[i, 1]
        zone = get_pose_zone(yaw, pitch)
        if zone not in pose_scalers:
            pose_scalers[zone] = StandardScaler()
        pose_scalers[zone].partial_fit(X[i:i+1, :])
    X_scaled_list = []
    for i in range(X.shape[0]):
        yaw, pitch = X[i, 0], X[i, 1]
        zone = get_pose_zone(yaw, pitch)
        x_norm = pose_scalers[zone].transform(X[i:i+1, :])
        X_scaled_list.append(x_norm[0])
    X_scaled = np.array(X_scaled_list)
    input_dim = X.shape[1]
    gaze_model = GazeNet(input_dim=input_dim).to(device)
    optimizer = optim.Adam(gaze_model.parameters(), lr=0.001)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    gaze_model.train()
    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = gaze_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    model_trained = True
    save_gaze_models(gaze_model, pose_scalers)
    theta_slow = {n: p.data.clone() for n, p in gaze_model.named_parameters()}
    fisher_short = {}
    permanent_fishers = []
    region_stats = {}
    print(f"✅ Modelo treinado com {X.shape[0]} amostras.")

def calculate_confidence_and_ear(face_landmarks, w, h):
    try:
        r_upper_y = np.mean([face_landmarks[i].y * h for i in RIGHT_UPPER_IDX if i < len(face_landmarks)])
        r_lower_y = np.mean([face_landmarks[i].y * h for i in RIGHT_LOWER_IDX if i < len(face_landmarks)])
        r_openness = abs(r_lower_y - r_upper_y)
        l_upper_y = np.mean([face_landmarks[i].y * h for i in LEFT_UPPER_IDX if i < len(face_landmarks)])
        l_lower_y = np.mean([face_landmarks[i].y * h for i in LEFT_LOWER_IDX if i < len(face_landmarks)])
        l_openness = abs(l_lower_y - l_upper_y)
        avg_openness = (r_openness + l_openness) / 2.0
        confidence = np.clip((avg_openness - 4.0) / 4.0, 0.0, 1.0)
        return avg_openness, confidence
    except:
        return 0.0, 0.0

def draw_reference_points(frame, focus_points, current_focus_index, predicted_point=None, pink_point=None):
    for i, (x, y) in enumerate(focus_points):
        color = (0, 255, 0)
        if i == len(focus_points) - 1:
            color = (0, 0, 255)
        cv2.circle(frame, (x, y), 5, color, -1)
        if i == current_focus_index:
            cv2.circle(frame, (x, y), 20, (255, 255, 0), 2)
    if predicted_point is not None:
        gx, gy = int(predicted_point[0]), int(predicted_point[1])
        cv2.circle(frame, (gx, gy), 20, (0, 165, 255), 2)
        for fx, fy in focus_points:
            fx, fy = int(fx), int(fy)
            cv2.line(frame, (fx, fy), (gx, gy), (0, 0, 255), 1)
            error = np.linalg.norm(np.array([fx, fy]) - np.array([gx, gy]))
            cv2.putText(frame, f"Err: {error:.0f}", (fx + 10, fy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    if pink_point is not None:
        px, py = int(pink_point[0]), int(pink_point[1])
        cv2.circle(frame, (px, py), 14, (255, 0, 255), 2)  # rosa
        cv2.circle(frame, (px, py), 6, (255, 0, 255), -1)

def calculate_face_angles(landmarks, img_width, img_height):
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    nose_x, nose_y = int(nose_tip.x * img_width), int(nose_tip.y * img_height)
    left_x, left_y = int(left_eye.x * img_width), int(left_eye.y * img_height)
    right_x, right_y = int(right_eye.x * img_width), int(right_eye.y * img_height)
    yaw = np.arctan2(right_y - left_y, right_x - left_x) * (180.0 / np.pi)
    pitch = np.arctan2(nose_y - (left_y + right_y) / 2, img_height) * (180.0 / np.pi)
    roll = np.arctan2(nose_x - (left_x + right_x) / 2, img_width) * (180.0 / np.pi)
    return yaw, pitch, roll

def get_inner_eye_bounding_boxes(landmarks, frame_w, frame_h):
    def get_inner_bbox(eyelid_idx):
        all_idx = [i for i in eyelid_idx if i < len(landmarks)]
        if not all_idx:
            return None
        x_coords = [landmarks[i].x * frame_w for i in all_idx]
        y_coords = [landmarks[i].y * frame_h for i in all_idx]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        margin_x = (x_max - x_min) * 0.3
        margin_y = (y_max - y_min) * 0.3
        return int(x_min - margin_x), int(y_min - margin_y), int(x_max + margin_x), int(y_max + margin_y)
    right_box = get_inner_bbox(RIGHT_EYELID_IDX)
    left_box = get_inner_bbox(LEFT_EYELID_IDX)
    return (right_box, left_box, RIGHT_EYELID_IDX, RIGHT_IRIS_IDX, RIGHT_PUPIL_IDX,
            LEFT_EYELID_IDX, LEFT_IRIS_IDX, LEFT_PUPIL_IDX)

def draw_eye_landmarks(frame, landmarks, eyelid_idx, iris_idx, pupil_idx, frame_w, frame_h, offset_x=0, offset_y=0):
    for idx in eyelid_idx:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * frame_w) - offset_x
            y = int(landmarks[idx].y * frame_h) - offset_y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    for idx in iris_idx:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * frame_w) - offset_x
            y = int(landmarks[idx].y * frame_h) - offset_y
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
    for idx in pupil_idx:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * frame_w) - offset_x
            y = int(landmarks[idx].y * frame_h) - offset_y
            color = (128, 0, 128) if idx in RIGHT_PUPIL_IDX else (255, 0, 255)
            cv2.circle(frame, (x, y), 3, color, -1)

def apply_anchor_correction(current_point, focus_points):
    global last_anchor_time
    if not ANCHOR_MODE_ENABLED:
        return current_point, False, float('nan')
    now = time.time()
    if now - last_anchor_time < ANCHOR_COOLDOWN:
        return current_point, False, float('nan')
    closest_dist = float('inf')
    closest_pt = None
    for fx, fy in focus_points:
        dist = np.linalg.norm(np.array(current_point) - np.array([fx, fy]))
        if dist < closest_dist:
            closest_dist = dist
            closest_pt = (fx, fy)
    if closest_dist < ANCHOR_RADIUS_THRESHOLD:
        strength = max(0, (ANCHOR_RADIUS_THRESHOLD - closest_dist) / ANCHOR_RADIUS_THRESHOLD)
        strength = strength ** 2
        new_x = current_point[0] * (1 - strength) + closest_pt[0] * strength
        new_y = current_point[1] * (1 - strength) + closest_pt[1] * strength
        last_anchor_time = now
        return (new_x, new_y), True, closest_dist
    return current_point, False, closest_dist

# ================= MOTOR DE CONVERGÊNCIA HÍBRIDO (FINAL) =================
DEAD_ZONE_RADIUS_BASE = 30.0
DEAD_ZONE_RADIUS_MIN = 8.0
DEAD_ZONE_VEL_SCALE = 0.15
PERSISTENCE_BOOST_RAMP_FRAMES = 5
PREDICTION_GAIN = 0.3
ALPHA_BASE = 0.15
ALPHA_MAX = 0.98                     # centro mais rápido
ALPHA_VEL_SCALE = 0.006              # resposta mais forte à velocidade
DISTANCE_THRESHOLD_FOR_MAX_ALPHA = 150.0
PERSISTENCE_BOOST_FRAMES = 8
RAW_VEL_THRESHOLD = 15.0
SACCADE_RECOVERY_FRAMES = 3

VEL_BLEND_REF = 110.0                # blend decai mais cedo → menos atraso
BLEND_FIXATION_MAX = 0.98
POST_SACCADE_BLEND_FRAMES = 2        # frames com blend alto após sacada

class StabilizationEngine:
    def __init__(self, screen_size, use_convergence=True):
        self.w, self.h = screen_size
        self.use_convergence = use_convergence
        if self.use_convergence:
            self.convergence_center = None
            self.stability_blend = 0.0
            self.stabilization_level = 0.0
            self.lock_center = None
            self._was_locked = False
            self._drift_accum = 0.0
            self._prev_raw = None
            self._persistent_far_count = 0
            self._boost_alpha = ALPHA_BASE
            self._boost_ramp = 0.0
            self._saccade_recovery_counter = 0
            self._post_saccade_counter = 0   # contador de estabilização pós‑sacada

    def compute_stability_blend(self, velocity):
        if velocity < 1e-6:
            return BLEND_FIXATION_MAX
        blend = math.exp(-velocity / VEL_BLEND_REF)
        return min(BLEND_FIXATION_MAX, blend)

    def update(self, raw_point, filtered_point=None, velocity=0.0, now=None,
               is_saccade=False, pred_vel=None):
        if now is None:
            now = time.time()
        drift_value = 0.0
        if self.use_convergence and filtered_point is not None:
            if self.convergence_center is None:
                self.convergence_center = np.array(raw_point, dtype=float)

            raw_arr = np.array(raw_point)
            filt_arr = np.array(filtered_point)

            # ---------- TRATAMENTO DE SACADA ----------
            if is_saccade:
                self._saccade_recovery_counter = SACCADE_RECOVERY_FRAMES
                self._post_saccade_counter = 0   # reseta o pós‑sacada

            if self._saccade_recovery_counter > 0:
                self._saccade_recovery_counter -= 1
                # Saída = ponto bruto (resposta imediata)
                self.stability_blend = 0.0
                self.stabilization_level = 0.0
                output_point = tuple(raw_arr)
                self.lock_center = tuple(self.convergence_center)
                is_locked = False
                if self._was_locked:
                    drift_value = self._drift_accum
                    self._drift_accum = 0.0
                self._was_locked = False
                self._prev_raw = raw_arr.copy()
                # Ao final da recuperação, prepara o pós‑sacada
                if self._saccade_recovery_counter == 0:
                    self._post_saccade_counter = POST_SACCADE_BLEND_FRAMES
                    # Reposiciona o centro no ponto BRUTO (sem inércia de filtros)
                    self.convergence_center = raw_arr.copy()
                return output_point, is_locked, self.lock_center, drift_value

            # ---------- MODO NORMAL (FIXAÇÃO / PERSEGUIÇÃO) ----------
            # Se estamos nos frames pós‑sacada, forçamos blend alto para estabilizar
            if self._post_saccade_counter > 0:
                self._post_saccade_counter -= 1
                self.stability_blend = 0.95   # quase toda a estabilidade
            else:
                self.stability_blend = self.compute_stability_blend(velocity)
            self.stabilization_level = self.stability_blend

            vel_mag = velocity
            current_dead_zone = max(DEAD_ZONE_RADIUS_MIN,
                                    DEAD_ZONE_RADIUS_BASE - vel_mag * DEAD_ZONE_VEL_SCALE)
            to_center = filt_arr - self.convergence_center
            distance = np.linalg.norm(to_center)

            # Alpha adaptativo (mais agressivo na perseguição)
            vel_alpha = min(ALPHA_MAX, ALPHA_BASE + velocity * ALPHA_VEL_SCALE)
            dist_factor = np.clip((distance - current_dead_zone) /
                                  (DISTANCE_THRESHOLD_FOR_MAX_ALPHA - current_dead_zone), 0.0, 1.0)
            alpha = vel_alpha + (ALPHA_MAX - vel_alpha) * dist_factor
            alpha = np.clip(alpha, ALPHA_BASE, ALPHA_MAX)

            # Boost de persistência (mantido)
            if distance > current_dead_zone * 2.0:
                self._persistent_far_count += 1
            else:
                self._persistent_far_count = max(0, self._persistent_far_count - 1)
            if self._persistent_far_count >= PERSISTENCE_BOOST_FRAMES:
                self._boost_ramp = min(1.0, self._boost_ramp + 1.0 / PERSISTENCE_BOOST_RAMP_FRAMES)
                alpha = min(0.98, alpha + self._boost_ramp * (1.0 - alpha))
                if self._boost_ramp >= 1.0:
                    self._persistent_far_count = 0
                    self._boost_ramp = 0.0
            else:
                self._boost_ramp = max(0.0, self._boost_ramp - 1.0 / PERSISTENCE_BOOST_RAMP_FRAMES)

            # Atualização do centro
            if self._prev_raw is not None:
                raw_vel = filt_arr - self._prev_raw
            else:
                raw_vel = np.zeros(2)
            moving_away = (np.linalg.norm(raw_vel) > RAW_VEL_THRESHOLD and
                           np.dot(raw_vel, to_center) > 0)

            if moving_away:
                if pred_vel is not None:
                    predicted_center = self.convergence_center + pred_vel * PREDICTION_GAIN
                else:
                    predicted_center = self.convergence_center.copy()
                self.convergence_center = (1.0 - alpha) * predicted_center + alpha * filt_arr
            else:
                if distance > 1e-6:
                    step = alpha * distance
                    direction = to_center / distance
                    self.convergence_center += direction * min(step, distance)

            # Saída final
            output_point = tuple(
                self.stability_blend * self.convergence_center +
                (1.0 - self.stability_blend) * filt_arr
            )
            self.lock_center = tuple(self.convergence_center)

            is_locked = (self.stability_blend > 0.95 and distance < current_dead_zone)

            if is_locked and self._prev_raw is not None:
                self._drift_accum += np.linalg.norm(filt_arr - self._prev_raw)
            if self._was_locked and not is_locked:
                drift_value = self._drift_accum
                self._drift_accum = 0.0
            self._was_locked = is_locked
            self._prev_raw = filt_arr.copy()

            return output_point, is_locked, self.lock_center, drift_value
        else:
            return raw_point, False, None, 0.0
                
# ================= VISUALIZADOR DE ESTABILIZAÇÃO =================
STABILIZATION_RING_BASE_RADIUS = 15
STABILIZATION_RING_MAX_RADIUS = 240
STABILIZATION_RING_EXPAND_SPEED = 0.15
STABILIZATION_RING_CONTRACT_SPEED = 0.08

class StabilizationVisualizer:
    def __init__(self):
        self.current_outer_radius = STABILIZATION_RING_BASE_RADIUS * 4
        self.current_middle_radius = STABILIZATION_RING_BASE_RADIUS * 4
        self.target_outer_radius = STABILIZATION_RING_BASE_RADIUS * 4
        self.target_middle_radius = STABILIZATION_RING_BASE_RADIUS * 4
        self.outer_alpha = 0.3
        self.middle_alpha = 0.5
        self.inner_alpha = 0.8
        self.smooth_velocity = 0.0
        self.stability_level = 0.0
        self.trail_points = deque(maxlen=15)
        self.lock_center = None
        self.lock_boundary = 60
        self.estimate_outside_boundary = False
        self.previously_locked = False

    def update(self, current_point, velocity, is_locked, stabilization_level, dt, lock_center=None, distance=None):
        self.smooth_velocity = 0.7 * self.smooth_velocity + 0.3 * velocity
        self.stability_level = 0.8 * self.stability_level + 0.2 * stabilization_level
        self.trail_points.append(current_point)
        if lock_center is not None:
            self.lock_center = lock_center
        if distance is not None:
            self.target_middle_radius = max(2.0, distance)
        else:
            self.lock_boundary = 40 + (1.0 - self.stability_level) * 80
            self.target_middle_radius = self.lock_boundary
        vel_norm = np.clip(self.smooth_velocity / 25.0, 0.0, 2.0)
        self.target_outer_radius = STABILIZATION_RING_BASE_RADIUS * 4 + \
            (STABILIZATION_RING_MAX_RADIUS - STABILIZATION_RING_BASE_RADIUS * 4) * (vel_norm ** 1.5)
        self.outer_alpha = 0.3 * (1.0 - self.stability_level * 0.7)
        self.middle_alpha = 0.5 + self.stability_level * 0.5
        self.inner_alpha = 0.5 + self.stability_level * 0.4
        expand_speed = STABILIZATION_RING_EXPAND_SPEED if self.target_outer_radius > self.current_outer_radius else STABILIZATION_RING_CONTRACT_SPEED
        self.current_outer_radius += (self.target_outer_radius - self.current_outer_radius) * expand_speed
        self.current_middle_radius += (self.target_middle_radius - self.current_middle_radius) * expand_speed * 1.5

    def draw(self, frame, center_x, center_y, is_locked=False, estimate_point=None):
        cx, cy = int(center_x), int(center_y)
        if self.smooth_velocity > 25.0 * 4:
            outer_color = (100,100,100)
        else:
            vel_factor = min(self.smooth_velocity / (25.0 * 4), 1.0)
            gray = int(180 - vel_factor * 80)
            outer_color = (gray, gray, gray)
        overlay = frame.copy()
        if self.current_outer_radius > 5:
            cv2.circle(overlay, (cx, cy), int(self.current_outer_radius), outer_color, 2, cv2.LINE_AA)
            cv2.circle(overlay, (cx, cy), int(self.current_outer_radius*0.98),
                       (*outer_color, int(255*self.outer_alpha*0.08)), -1)
        cv2.addWeighted(overlay, self.outer_alpha, frame, 1-self.outer_alpha, 0, frame)
        if self.current_middle_radius > 3:
            overlay = frame.copy()
            middle_color = (0, 255, 255)
            cv2.circle(overlay, (cx, cy), int(self.current_middle_radius), middle_color, -1)
            border_thickness = 2 if is_locked else 1
            cv2.circle(overlay, (cx, cy), int(self.current_middle_radius), middle_color, border_thickness, cv2.LINE_AA)
            cv2.addWeighted(overlay, self.middle_alpha, frame, 1-self.middle_alpha, 0, frame)
        inner_color = (int(100*(1-self.stability_level)), int(100*(1-self.stability_level) + 200*self.stability_level),
                       int(100*(1-self.stability_level) + 255*self.stability_level))
        if estimate_point is not None:
            ex, ey = int(estimate_point[0]), int(estimate_point[1])
            overlay = frame.copy()
            cv2.circle(overlay, (ex,ey), STABILIZATION_RING_BASE_RADIUS, inner_color, -1)
            cv2.circle(overlay, (ex,ey), STABILIZATION_RING_BASE_RADIUS, inner_color, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, self.inner_alpha, frame, 1-self.inner_alpha, 0, frame)
        cv2.circle(frame, (cx,cy), 4, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx,cy), 6, (0,255,0) if is_locked else inner_color, 1, cv2.LINE_AA)
        if is_locked:
            cv2.putText(frame, "LOCKED", (cx-25, cy-int(self.current_outer_radius)-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        dead_zone_radius = int(DEAD_ZONE_RADIUS_BASE)
        if dead_zone_radius > 0:
            cv2.circle(frame, (cx, cy), dead_zone_radius, (200,200,200), 1, cv2.LINE_AA)
            for angle in range(0,360,30):
                x1 = int(cx + dead_zone_radius * np.cos(np.radians(angle)))
                y1 = int(cy + dead_zone_radius * np.sin(np.radians(angle)))
                x2 = int(cx + (dead_zone_radius-3) * np.cos(np.radians(angle+10)))
                y2 = int(cy + (dead_zone_radius-3) * np.sin(np.radians(angle+10)))
                cv2.line(frame, (x1,y1), (x2,y2), (200,200,200), 1)
        return frame

# ================= LOGGER DE MÉTRICAS =================
class MetricsLogger:
    def __init__(self, filename):
        self.filename = filename
        self.fieldnames = [
            'timestamp', 'frame_id', 'event_type', 'key_pressed',
            'raw_x', 'raw_y', 'filtered_x', 'filtered_y', 'predicted_x', 'predicted_y',
            'target_x', 'target_y',
            'error_px','error_raw_px', 'error_filtered_px', 'error_angular_deg',
            'jitter_px', 'velocity_filtered_px_per_frame', 'smoothness_curvature', 'drift_accumulated',
            'ear_val', 'confidence',
            'ukf_vx', 'ukf_vy', 'ukf_innovation_x', 'ukf_innovation_y',
            'convergence_level', 'is_locked', 'anchor_distance',
            'is_saccade', 'eye_velocity_raw',
            'fps', 'cpu_percent', 'ram_mb', 'vram_mb', 'inference_latency_ms',
            'fc1_mean', 'fc1_std', 'fc1_active_ratio', 'fc1_l2',
            'fc2_mean', 'fc2_std', 'fc2_active_ratio', 'fc2_l2',
            'fc3_mean', 'fc3_std', 'fc3_active_ratio', 'fc3_l2',
            'weight_norm_fc1_weight', 'weight_norm_fc2_weight', 'weight_norm_fc3_weight', 'weight_norm_fc4_weight',
            'grad_norm_fc1_weight', 'grad_norm_fc2_weight', 'grad_norm_fc3_weight', 'grad_norm_fc4_weight',
            'weight_change_fc1_weight', 'weight_change_fc2_weight', 'weight_change_fc3_weight', 'weight_change_fc4_weight',
            'train_loss', 'train_time_ms', 'region_id',
            'ewc_loss', 'lambda_ewc', 'fisher_mean', 'fisher_max', 'num_promoted_tasks',
            'lookahead_alpha', 'lookahead_distance',
            'replay_buffer_size', 'quadtree_leaves', 'quadtree_avg_depth',
            'upv_ipd_norm', 'upv_intercantal_norm', 'upv_face_ratio',
            'upv_reye_width', 'upv_leye_width', 'upv_reye_height', 'upv_leye_height',
            'upv_roffset_x', 'upv_roffset_y', 'upv_loffset_x', 'upv_loffset_y',
            'upv_distance_to_ref'
        ]
        self._init_file()

    def _init_file(self):
        if not os.path.isfile(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, metrics_dict):
        for key in self.fieldnames:
            if key not in metrics_dict:
                metrics_dict[key] = float('nan')
        with open(self.filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics_dict)

metrics_logger = MetricsLogger(METRICS_CSV)
execution_id = str(uuid.uuid4())[:8]
frame_counter = 0
current_target_px = None
current_target_norm = None
ANCHOR_MODE_ENABLED = True
ANCHOR_COOLDOWN = 3.0
ANCHOR_RADIUS_THRESHOLD = 120.0
last_anchor_time = 0.0

# ================= CONFIGURAÇÕES DO REPLAY ADAPTATIVO =================
REPLAY_AUGMENT_ENABLED = True
REPLAY_MIN_SAMPLES_PER_REGION = 20
REPLAY_COOLDOWN_SECONDS = 2.0
REPLAY_STEPS = 20
REPLAY_MIX_CLEAN = 0.2

NOISE_BASE_STD = 0.005
NOISE_SCALE = 0.002
NOISE_ERROR_THRESHOLD = 5.0
NOISE_MAX_STD = 0.05

last_replay_time = {}

# ================= CONFIGURAÇÕES DE FIXAÇÃO =================
FIXATION_RADIUS = 50
FIXATION_MIN_FRAMES = 15
FIXATION_SAMPLE_LIMIT = 30
FIXATION_CONFIDENCE_MIN = 0.5

# ================= MODO DE PERSEGUIÇÃO AUTOMÁTICA ROSA =================
AUTOPURSUIT_MODE = False
AUTOPURSUIT_SPEED = 60  # frames per leg (ida ou volta)
autopursuit_t = 0.0       # 0.0 -> base, 1.0 -> target, 2.0 -> de volta à base
autopursuit_base_idx = 0
autopursuit_target_idx = 1
autopursuit_direction = 1  # 1 = indo para o alvo, -1 = voltando

def update_autopursuit():
    global autopursuit_t, autopursuit_direction, autopursuit_base_idx, autopursuit_target_idx, AUTOPURSUIT_MODE
    if not AUTOPURSUIT_MODE or len(focus_points) < 2:
        return None
    # incrementa t
    dt = 1.0 / AUTOPURSUIT_SPEED
    autopursuit_t += dt * autopursuit_direction
    # inverte direção se chegou ao fim
    if autopursuit_t > 1.0:
        autopursuit_t = 1.0
        autopursuit_direction = -1
    elif autopursuit_t < 0.0:
        autopursuit_t = 0.0
        # completou ida e volta, avança para o próximo alvo
        autopursuit_target_idx = (autopursuit_target_idx + 1) % len(focus_points)
        if autopursuit_target_idx == autopursuit_base_idx:
            autopursuit_target_idx = (autopursuit_target_idx + 1) % len(focus_points)
        autopursuit_direction = 1
    # interpola entre base e target
    base_pt = focus_points[autopursuit_base_idx]
    target_pt = focus_points[autopursuit_target_idx]
    pink_x = base_pt[0] + (target_pt[0] - base_pt[0]) * autopursuit_t
    pink_y = base_pt[1] + (target_pt[1] - base_pt[1]) * autopursuit_t
    return (pink_x, pink_y)

# ================= FUNÇÕES DO REPLAY ADAPTATIVO =================
def get_region_avg_error(region_id):
    stats = region_stats.get(region_id)
    if stats and len(stats['errors']) > 0:
        return np.mean(stats['errors'])
    return None

def adaptive_noise_std(avg_error):
    if avg_error is None:
        return NOISE_BASE_STD
    extra = NOISE_SCALE * max(0, avg_error - NOISE_ERROR_THRESHOLD)
    return min(NOISE_BASE_STD + extra, NOISE_MAX_STD)

def replay_region_with_adaptive_noise(region_id):
    global last_replay_time, gaze_model, optimizer

    now = time.time()
    if region_id in last_replay_time and (now - last_replay_time[region_id]) < REPLAY_COOLDOWN_SECONDS:
        return

    stats = region_stats.get(region_id)
    if stats is None:
        return
    samples_raw = stats.get('samples_raw', [])
    targets = stats.get('targets', [])
    if len(samples_raw) < REPLAY_MIN_SAMPLES_PER_REGION or len(samples_raw) != len(targets):
        return

    samples_np = np.array(samples_raw)
    targets_np = np.array(targets)
    avg_error = get_region_avg_error(region_id)
    current_std = adaptive_noise_std(avg_error)
    n = len(samples_np)

    print(f"🔄 Replay adaptativo na região {region_id}: {REPLAY_STEPS} passos, "
          f"ruído σ={current_std:.4f} (erro médio={avg_error:.1f}px se disponível)")

    for _ in range(REPLAY_STEPS):
        idx = np.random.randint(0, n)
        sample = samples_np[idx].copy()
        target = targets_np[idx].copy()

        if np.random.rand() > REPLAY_MIX_CLEAN:
            noise = np.random.normal(0, current_std, size=sample.shape)
            sample += noise

        yaw, pitch = sample[0], sample[1]
        input_vec = sample.reshape(1, -1)

        zone = get_pose_zone(yaw, pitch)
        if zone in pose_scalers and hasattr(pose_scalers[zone], 'mean_'):
            input_scaled = pose_scalers[zone].transform(input_vec)
        else:
            input_scaled = input_vec

        target_px = np.array([target[0] * screen_w, target[1] * screen_h])
        node = id_to_node.get(region_id)
        gaze_point = np.array(node.center) if node else target_px

        online_train_step(input_scaled, target.tolist(), gaze_point,
                          1.0, yaw, pitch, quadtree_root)

    last_replay_time[region_id] = now

# ================= MONITOR DE FIXAÇÃO =================
class FixationMonitor:
    def __init__(self):
        self.fixation_counter = 0
        self.is_fixating = False
        self.current_region_id = None

    def update(self, is_locked, lock_center, focus_point, confidence, region_id):
        if not is_locked or lock_center is None or focus_point is None:
            self.reset()
            return

        dist = np.linalg.norm(np.array(lock_center) - np.array(focus_point))
        if dist < FIXATION_RADIUS and confidence >= FIXATION_CONFIDENCE_MIN:
            self.fixation_counter += 1
            if self.fixation_counter >= FIXATION_MIN_FRAMES:
                if not self.is_fixating:
                    self.is_fixating = True
                self.current_region_id = region_id
        else:
            self.reset()

    def should_trigger_replay(self):
        return self.is_fixating and self.current_region_id is not None

    def get_region_id(self):
        return self.current_region_id

    def reset(self):
        self.fixation_counter = 0
        self.is_fixating = False
        self.current_region_id = None

# ================= EXTRAÇÃO DE UPV =================
def extract_face_geometry(landmarks, image_shape):
    h, w = image_shape[:2]
    def to_pixel(lm):
        return np.array([lm.x * w, lm.y * h])
    right_pupil = np.mean([to_pixel(landmarks[i]) for i in RIGHT_PUPIL_IDX], axis=0)
    left_pupil  = np.mean([to_pixel(landmarks[i]) for i in LEFT_PUPIL_IDX], axis=0)
    ipd = np.linalg.norm(left_pupil - right_pupil)
    right_inner = to_pixel(landmarks[133])
    left_inner  = to_pixel(landmarks[362])
    intercantal_dist = np.linalg.norm(left_inner - right_inner)
    left_tragus  = to_pixel(landmarks[234])
    right_tragus = to_pixel(landmarks[454])
    face_width = np.linalg.norm(left_tragus - right_tragus)
    glabella = to_pixel(landmarks[9])
    menton   = to_pixel(landmarks[152])
    face_height = np.linalg.norm(glabella - menton)
    nose_tip = to_pixel(landmarks[4])
    ipd_norm = ipd / (face_width + 1e-6)
    intercantal_norm = intercantal_dist / (ipd + 1e-6)
    face_ratio = face_width / (face_height + 1e-6)
    right_offset = (right_pupil - nose_tip) / (ipd + 1e-6)
    left_offset  = (left_pupil  - nose_tip) / (ipd + 1e-6)
    right_outer = to_pixel(landmarks[33])
    left_outer  = to_pixel(landmarks[263])
    right_eye_width = np.linalg.norm(right_outer - right_inner)
    left_eye_width  = np.linalg.norm(left_outer - left_inner)
    right_upper = np.mean([to_pixel(landmarks[i]) for i in RIGHT_UPPER_IDX], axis=0)
    right_lower = np.mean([to_pixel(landmarks[i]) for i in RIGHT_LOWER_IDX], axis=0)
    left_upper  = np.mean([to_pixel(landmarks[i]) for i in LEFT_UPPER_IDX], axis=0)
    left_lower  = np.mean([to_pixel(landmarks[i]) for i in LEFT_LOWER_IDX], axis=0)
    right_eye_height = np.linalg.norm(right_upper - right_lower)
    left_eye_height  = np.linalg.norm(left_upper - left_lower)
    right_eye_width_norm  = right_eye_width  / (ipd + 1e-6)
    left_eye_width_norm   = left_eye_width   / (ipd + 1e-6)
    right_eye_height_norm = right_eye_height / (ipd + 1e-6)
    left_eye_height_norm  = left_eye_height  / (ipd + 1e-6)
    features = np.array([
        ipd_norm, intercantal_norm, face_ratio,
        right_eye_width_norm, left_eye_width_norm,
        right_eye_height_norm, left_eye_height_norm,
        right_offset[0], right_offset[1],
        left_offset[0], left_offset[1]
    ], dtype=np.float32)
    return features

# ================= CARREGAMENTO INICIAL =================
auto_load_models()

# ================= LOOP PRINCIPAL =================
with FaceLandmarker.create_from_options(options) as landmarker:
    cv2.namedWindow("Gaze Tracker", cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    quadtree_root = build_initial_quadtree()
    focus_points, id_to_node = get_all_focus_points_and_map(quadtree_root)
    current_focus_index = 0

    stab_engine = StabilizationEngine((screen_w, screen_h), use_convergence=USE_CONVERGENCE_MECHANIC)
    stab_visualizer = StabilizationVisualizer()
    fixation_monitor = FixationMonitor()

    prev_raw = None
    ref_upv = None

    prev_bright_mean_R = 0.0
    prev_bright_var_R = 0.0
    prev_bright_mean_L = 0.0
    prev_bright_var_L = 0.0

    pos_history = deque(maxlen=30)
    last_predicted = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        t0 = time.perf_counter()
        predicted_point = None
        raw_estimate = None
        h, w, _ = frame.shape
        face_landmarks = None
        input_vec_np = None
        frame_metrics = {
            'timestamp': time.time(),
            'frame_id': frame_counter,
            'event_type': 'frame',
            'key_pressed': -1
        }

        # Processamento da autoperseguição rosa
        pink_point = update_autopursuit()

        if results.face_landmarks:
            current_landmarks = results.face_landmarks[0]
            landmark_buffer.append(current_landmarks)
            if len(landmark_buffer) == landmark_buffer.maxlen:
                smoothed_landmarks = [Landmark(0.0, 0.0, 0.0) for _ in range(len(current_landmarks))]
                for i in range(len(current_landmarks)):
                    xs = [lm[i].x for lm in landmark_buffer]
                    ys = [lm[i].y for lm in landmark_buffer]
                    zs = [lm[i].z for lm in landmark_buffer]
                    smoothed_landmarks[i] = Landmark(
                        alpha_lm * xs[-1] + (1 - alpha_lm) * np.mean(xs[:-1]),
                        alpha_lm * ys[-1] + (1 - alpha_lm) * np.mean(ys[:-1]),
                        alpha_lm * zs[-1] + (1 - alpha_lm) * np.mean(zs[:-1])
                    )
                face_landmarks = smoothed_landmarks
            else:
                face_landmarks = [Landmark(l.x, l.y, l.z) for l in current_landmarks]

            if USE_UPV:
                face_geom = extract_face_geometry(face_landmarks, (h, w))
                upv_keys = ['upv_ipd_norm', 'upv_intercantal_norm', 'upv_face_ratio',
                            'upv_reye_width', 'upv_leye_width', 'upv_reye_height', 'upv_leye_height',
                            'upv_roffset_x', 'upv_roffset_y', 'upv_loffset_x', 'upv_loffset_y']
                for i, k in enumerate(upv_keys):
                    frame_metrics[k] = float(face_geom[i])
                if ref_upv is None:
                    ref_upv = face_geom.copy()
                frame_metrics['upv_distance_to_ref'] = np.linalg.norm(face_geom - ref_upv)
            else:
                for k in ['upv_ipd_norm', 'upv_intercantal_norm', 'upv_face_ratio',
                          'upv_reye_width', 'upv_leye_width', 'upv_reye_height', 'upv_leye_height',
                          'upv_roffset_x', 'upv_roffset_y', 'upv_loffset_x', 'upv_loffset_y',
                          'upv_distance_to_ref']:
                    frame_metrics[k] = float('nan')

            p33 = np.array([face_landmarks[33].x * w, face_landmarks[33].y * h])
            p263 = np.array([face_landmarks[263].x * w, face_landmarks[263].y * h])
            face_scale = max(np.linalg.norm(p33 - p263), 1.0)

            def normalize_eye(landmarks, indices, scale, w, h):
                pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices if i < len(landmarks)])
                if len(pts) == 0:
                    return []
                center = np.mean(pts, axis=0)
                normalized = (pts - center) / scale
                return normalized.flatten().tolist()

            right_eye_norm = normalize_eye(face_landmarks, RIGHT_EYE_ALL, face_scale, w, h)
            left_eye_norm = normalize_eye(face_landmarks, LEFT_EYE_ALL, face_scale, w, h)

            yaw, pitch, roll = calculate_face_angles(face_landmarks, w, h)

            z_mean_R = np.mean([face_landmarks[i].z for i in RIGHT_EYELID_IDX if i < len(face_landmarks)])
            z_mean_L = np.mean([face_landmarks[i].z for i in LEFT_EYELID_IDX if i < len(face_landmarks)])

            (right_eye_box, left_eye_box, right_eyelid_idx, right_iris_idx, right_pupil_idx,
             left_eyelid_idx, left_iris_idx, left_pupil_idx) = get_inner_eye_bounding_boxes(face_landmarks, w, h)

            bright_mean_R, bright_var_R, bright_mean_L, bright_var_L = 0, 0, 0, 0
            eyes = {"Right": (right_eye_box, right_eyelid_idx, right_iris_idx, right_pupil_idx),
                    "Left": (left_eye_box, left_eyelid_idx, left_iris_idx, left_pupil_idx)}
            for name, (box, eyelid_idx, iris_idx, pupil_idx) in eyes.items():
                if box:
                    x_min, y_min, x_max, y_max = box
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)
                    eye_region = frame[y_min:y_max, x_min:x_max]
                    if eye_region.size > 0:
                        eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                        eye_gray = cv2.medianBlur(eye_gray, 3)
                        bright_mean = np.mean(eye_gray)
                        bright_var = np.var(eye_gray)
                        if name == "Right":
                            bright_mean_R = alpha_bright * bright_mean + (1 - alpha_bright) * prev_bright_mean_R
                            bright_var_R = alpha_bright * bright_var + (1 - alpha_bright) * prev_bright_var_R
                            prev_bright_mean_R, prev_bright_var_R = bright_mean_R, bright_var_R
                        else:
                            bright_mean_L = alpha_bright * bright_mean + (1 - alpha_bright) * prev_bright_mean_L
                            bright_var_L = alpha_bright * bright_var + (1 - alpha_bright) * prev_bright_var_L
                            prev_bright_mean_L, prev_bright_var_L = bright_mean_L, bright_var_L
                        draw_eye_landmarks(eye_region, face_landmarks, eyelid_idx, iris_idx, pupil_idx, w, h, x_min, y_min)
                        cv2.imshow(name, cv2.resize(eye_region, (300, 300)))

            input_vec = [yaw, pitch, roll, aspect, face_scale] + right_eye_norm + left_eye_norm + \
                        [bright_mean_R, bright_var_R, z_mean_R, bright_mean_L, bright_var_L, z_mean_L]

            if len(input_vec) < EXPECTED_GAZE_INPUT_DIM:
                input_vec += [0.0] * (EXPECTED_GAZE_INPUT_DIM - len(input_vec))
            elif len(input_vec) > EXPECTED_GAZE_INPUT_DIM:
                input_vec = input_vec[:EXPECTED_GAZE_INPUT_DIM]

            input_vec_np = np.array(input_vec).reshape(1, -1)

            if model_trained:
                try:
                    ear_val, confidence = calculate_confidence_and_ear(face_landmarks, w, h)

                    zone = get_pose_zone(yaw, pitch)
                    if zone in pose_scalers and hasattr(pose_scalers[zone], 'mean_'):
                        input_scaled = pose_scalers[zone].transform(input_vec_np)
                    else:
                        input_scaled = input_vec_np

                    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)
                    t_inf = time.perf_counter()
                    gaze_model.eval()
                    with torch.no_grad():
                        pred_norm = gaze_model(input_tensor).cpu().numpy()[0]
                    inference_latency = (time.perf_counter() - t_inf) * 1000
                    raw_estimate = [pred_norm[0] * screen_w, pred_norm[1] * screen_h]
                    measurement = np.array(raw_estimate)

                    act_metrics = gaze_model.get_activation_metrics()
                    frame_metrics.update(act_metrics)

                    if not ukf_initialized:
                        ukf.x = np.array([raw_estimate[0], raw_estimate[1], 0, 0])
                        ukf_initialized = True
                        filtered_point = (raw_estimate[0], raw_estimate[1])
                        innovation = np.array([0.0, 0.0])
                    else:
                        innovation = measurement - ukf.x[:2]
                        ukf.R = np.eye(2) * 0.5
                        ukf.Q = np.eye(4) * 0.12
                        ukf.predict()
                        corrected_state = ukf.update(measurement)
                        filtered_point = (corrected_state[0], corrected_state[1])

                    now = time.time()
                    filtered_point = tuple(one_euro(np.array(filtered_point), now))
                    current_velocity = np.linalg.norm(ukf.x[2:])
                    adaptive_alpha = ema_alpha_base / (1.0 + velocity_scale * current_velocity)
                    if ema_point is None:
                        ema_point = filtered_point
                    else:
                        ema_point = (
                            adaptive_alpha * filtered_point[0] + (1 - adaptive_alpha) * ema_point[0],
                            adaptive_alpha * filtered_point[1] + (1 - adaptive_alpha) * ema_point[1]
                        )
                    filtered_pre_stab = ema_point

                    if prev_raw is not None:
                        vel_raw = np.linalg.norm(np.array(raw_estimate) - prev_raw)
                        is_saccade = vel_raw > 200
                    else:
                        is_saccade = False
                        vel_raw = 0.0

                    # --- RESET DOS FILTROS NA SACADA ---
                    if is_saccade:
                        # UKF: re-inicializa estado com posição bruta e velocidade zero
                        ukf.x = np.array([raw_estimate[0], raw_estimate[1], 0.0, 0.0])
                        # OneEuro: reset interno
                        one_euro.x_prev = np.array(raw_estimate)
                        one_euro.dx_prev = np.zeros(2)
                        one_euro.t_prev = time.time()
                        # EMA: reset (ema_point é global, não precisa de 'global' aqui)
                        ema_point = tuple(raw_estimate)
                    
                    # ------------------------------------

                    prev_raw = np.array(raw_estimate)

                    if USE_CONVERGENCE_MECHANIC:
                        ukf_vel_vec = np.array([ukf.x[2], ukf.x[3]]) if ukf_initialized else None
                        stab_point, is_locked, lock_center, drift_out = stab_engine.update(
                            raw_estimate, filtered_pre_stab, current_velocity, now,
                            is_saccade=is_saccade,
                            pred_vel=ukf_vel_vec
                        )
                    else:
                        stab_point = filtered_pre_stab
                        is_locked = False
                        lock_center = None
                        drift_out = 0.0

                    stab_point, anchored, anchor_dist = apply_anchor_correction(stab_point, focus_points)
                    if anchored:
                        cv2.circle(frame, (int(stab_point[0]), int(stab_point[1])), 12, (0, 255, 255), 2)

                    predicted_point = stab_point
                    pos_history.append(predicted_point)

                    frame_metrics['raw_x'] = raw_estimate[0]
                    frame_metrics['raw_y'] = raw_estimate[1]
                    frame_metrics['filtered_x'] = filtered_pre_stab[0]
                    frame_metrics['filtered_y'] = filtered_pre_stab[1]
                    frame_metrics['predicted_x'] = predicted_point[0]
                    frame_metrics['predicted_y'] = predicted_point[1]
                    frame_metrics['error_raw_px'] = float('nan')
                    frame_metrics['error_filtered_px'] = float('nan')
                    frame_metrics['error_angular_deg'] = float('nan')
                    if current_target_px is not None:
                        err_raw = np.linalg.norm(np.array(raw_estimate) - current_target_px)
                        err_filt = np.linalg.norm(np.array(predicted_point) - current_target_px)
                        frame_metrics['error_raw_px'] = err_raw
                        frame_metrics['error_filtered_px'] = err_filt
                        if pixels_per_cm > 0:
                            frame_metrics['error_angular_deg'] = np.degrees(np.arctan2(err_filt / pixels_per_cm, USER_DISTANCE_CM))
                    if len(pos_history) >= 2:
                        arr = np.array(pos_history)
                        center = np.mean(arr, axis=0)
                        dists = np.linalg.norm(arr - center, axis=1)
                        frame_metrics['jitter_px'] = np.std(dists)
                    else:
                        frame_metrics['jitter_px'] = 0.0
                    if last_predicted is not None:
                        frame_metrics['velocity_filtered_px_per_frame'] = np.linalg.norm(np.array(predicted_point) - last_predicted)
                    else:
                        frame_metrics['velocity_filtered_px_per_frame'] = 0.0
                    frame_metrics['smoothness_curvature'] = float('nan')
                    if len(pos_history) >= 3:
                        p0, p1, p2 = pos_history[-3], pos_history[-2], pos_history[-1]
                        v1 = np.array(p1) - np.array(p0)
                        v2 = np.array(p2) - np.array(p1)
                        norm_v1 = np.linalg.norm(v1)
                        norm_v2 = np.linalg.norm(v2)
                        if norm_v1 > 0 and norm_v2 > 0:
                            cosang = np.dot(v1, v2) / (norm_v1 * norm_v2)
                            cosang = np.clip(cosang, -1.0, 1.0)
                            frame_metrics['smoothness_curvature'] = np.arccos(cosang)
                    frame_metrics['drift_accumulated'] = drift_out
                    frame_metrics['ear_val'] = ear_val
                    frame_metrics['confidence'] = confidence
                    frame_metrics['ukf_vx'] = ukf.x[2]
                    frame_metrics['ukf_vy'] = ukf.x[3]
                    frame_metrics['ukf_innovation_x'] = innovation[0]
                    frame_metrics['ukf_innovation_y'] = innovation[1]
                    frame_metrics['convergence_level'] = stab_engine.stabilization_level
                    frame_metrics['is_locked'] = 1 if is_locked else 0
                    frame_metrics['anchor_distance'] = anchor_dist
                    frame_metrics['is_saccade'] = 1 if is_saccade else 0
                    frame_metrics['eye_velocity_raw'] = vel_raw
                    frame_metrics['fps'] = 1.0 / max(1e-6, time.perf_counter() - t0)
                    frame_metrics['inference_latency_ms'] = inference_latency
                    frame_metrics['cpu_percent'] = psutil.cpu_percent()
                    frame_metrics['ram_mb'] = psutil.Process().memory_info().rss / (1024*1024)
                    frame_metrics['vram_mb'] = torch.cuda.memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0.0
                    for name in ['fc1_weight', 'fc2_weight', 'fc3_weight', 'fc4_weight']:
                        frame_metrics[f'weight_norm_{name}'] = float('nan')
                        frame_metrics[f'grad_norm_{name}'] = float('nan')
                        frame_metrics[f'weight_change_{name}'] = float('nan')
                    for k in ['train_loss','train_time_ms','region_id','ewc_loss','lambda_ewc','fisher_mean','fisher_max','num_promoted_tasks','lookahead_alpha','lookahead_distance','replay_buffer_size','quadtree_leaves','quadtree_avg_depth']:
                        frame_metrics[k] = float('nan')

                    last_predicted = np.array(predicted_point)

                    # Se o modo de perseguição automática está ativo e temos um ponto rosa, captura automática
                    if AUTOPURSUIT_MODE and pink_point is not None and input_vec_np is not None:
                        out_norm = [pink_point[0] / screen_w, pink_point[1] / screen_h]
                        current_target_px = np.array(pink_point)
                        current_target_norm = out_norm
                        # Salva no CSV
                        save_data_csv(GAZE_CSV, (input_vec, out_norm))
                        # Treina online se confiança suficiente
                        if confidence > 0.6:
                            train_metrics = online_train_step(input_vec_np, out_norm, pink_point, confidence, yaw, pitch, quadtree_root)
                            if train_metrics:
                                train_metrics['timestamp'] = time.time()
                                train_metrics['frame_id'] = frame_counter
                                train_metrics['event_type'] = 'auto_train'
                                train_metrics['key_pressed'] = ord('p')
                                metrics_logger.log(train_metrics)

                    if USE_CONVERGENCE_MECHANIC:
                        focus_pt = focus_points[current_focus_index]
                        region_id_focus = get_region_id_from_quadtree(focus_pt, quadtree_root)
                        fixation_monitor.update(is_locked, lock_center, focus_pt,
                                                confidence, region_id_focus)
                        if REPLAY_AUGMENT_ENABLED and fixation_monitor.should_trigger_replay():
                            rid = fixation_monitor.get_region_id()
                            if rid is not None:
                                replay_region_with_adaptive_noise(rid)

                    if USE_CONVERGENCE_MECHANIC:
                        dist = np.linalg.norm(np.array(raw_estimate) - stab_engine.convergence_center) if stab_engine.convergence_center is not None else 0
                        stab_visualizer.update(stab_point, current_velocity, is_locked,
                                               stab_engine.stabilization_level, 1/30.0,
                                               lock_center, distance=dist)
                        frame = stab_visualizer.draw(frame, stab_point[0], stab_point[1],
                                                     is_locked, estimate_point=raw_estimate)
                    else:
                        cv2.circle(frame, (int(raw_estimate[0]), int(raw_estimate[1])), 4, (0,165,255), -1)
                        cv2.circle(frame, (int(filtered_pre_stab[0]), int(filtered_pre_stab[1])), 7, (0,255,255), 2)

                except Exception as e:
                    print(f"Erro no processamento de Gaze: {e}")

            FACEMESH_CONNECTIONS = [
                (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
                (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
                (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
                (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
                (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
                (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
                (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
                (154, 155), (155, 133), (133, 33),
                (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
                (381, 382), (382, 362), (362, 263),
                (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
                (314, 405), (405, 321), (321, 375), (375, 291), (291, 409), (409, 270),
                (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40), (40, 185),
                (185, 61),
            ]
            for start_idx, end_idx in FACEMESH_CONNECTIONS:
                if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                    x1 = int(face_landmarks[start_idx].x * w)
                    y1 = int(face_landmarks[start_idx].y * h)
                    x2 = int(face_landmarks[end_idx].x * w)
                    y2 = int(face_landmarks[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        else:
            if ukf_initialized:
                predicted_state = ukf.predict()
                predicted_point = (predicted_state[0], predicted_state[1])

        # Desenha os pontos de referência, ponto rosa e estrutura da quadtree
        draw_reference_points(frame, focus_points, current_focus_index, predicted_point, pink_point)
        draw_quadtree_structure(frame, quadtree_root)

        upv_status = "ON" if USE_UPV else "OFF"
        replay_status = "ON" if REPLAY_AUGMENT_ENABLED else "OFF"
        auto_status = "AUTO PURSUIT" if AUTOPURSUIT_MODE else "MANUAL"
        cv2.putText(frame, f"UPV: {upv_status} | Replay: {replay_status} | {auto_status} | Folhas: {len(focus_points)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"EWC lambda: {adaptive_controller.base_lamda:.1f}   Lookahead alpha: {adaptive_controller.base_alpha:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Teclas: c=foco | t=treino CSV | v=capt+treino | a=ancoragem | 1/2=quad | u=UPV | g=Replay | p=Auto Pursuit | +/-=lambda | [/]=alpha | z/x/b=estab | ESC=sair",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Gaze Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        frame_metrics['key_pressed'] = key if key != 255 else -1
        metrics_logger.log(frame_metrics)
        frame_counter += 1

        if key == 27:
            break
        elif key == ord('c') and not AUTOPURSUIT_MODE:
            current_focus_index = (current_focus_index + 1) % len(focus_points)
        elif key == ord('t'):
            train_model_from_csv()
        elif key == ord('v') and not AUTOPURSUIT_MODE and results.face_landmarks:
            out_norm = [focus_points[current_focus_index][0] / screen_w, focus_points[current_focus_index][1] / screen_h]
            current_target_px = np.array(focus_points[current_focus_index])
            current_target_norm = out_norm
            save_data_csv(GAZE_CSV, (input_vec, out_norm))
            ear_val, confidence = calculate_confidence_and_ear(face_landmarks, w, h)
            if confidence > 0.6 and predicted_point is not None:
                train_metrics = online_train_step(input_vec_np, out_norm, predicted_point, confidence, yaw, pitch, quadtree_root)
                region_id_capt = get_region_id_from_quadtree(predicted_point, quadtree_root)
                if region_id_capt is not None:
                    stats_capt = region_stats.get(region_id_capt)
                    if stats_capt is not None:
                        if 'samples_raw' not in stats_capt:
                            stats_capt['samples_raw'] = []
                            stats_capt['samples_scaled'] = []
                            stats_capt['targets'] = []
                        stats_capt['samples_raw'].append(input_vec_np[0].copy())
                        stats_capt['samples_scaled'].append(input_scaled[0].copy())
                        stats_capt['targets'].append(out_norm)
                if train_metrics:
                    train_metrics['timestamp'] = time.time()
                    train_metrics['frame_id'] = frame_counter
                    train_metrics['event_type'] = 'train'
                    train_metrics['key_pressed'] = ord('v')
                    metrics_logger.log(train_metrics)
                print(f"🎯 Capturado e treinado online! Confiança: {confidence:.2f}")
            else:
                print(f"⚠️ Capturado, mas ignorado no treino online (confiança: {confidence:.2f})")
        elif key == ord('p'):
            AUTOPURSUIT_MODE = not AUTOPURSUIT_MODE
            if AUTOPURSUIT_MODE:
                # Inicializa o estado da perseguição com o foco atual como base
                autopursuit_base_idx = current_focus_index
                autopursuit_target_idx = (current_focus_index + 1) % len(focus_points)
                autopursuit_t = 0.0
                autopursuit_direction = 1
                print("🟣 Modo de Perseguição Automática ATIVADO")
            else:
                print("🟣 Modo de Perseguição Automática DESATIVADO")
        elif key == ord('z'):
            if USE_CONVERGENCE_MECHANIC:
                stab_engine.convergence_center = np.array(raw_estimate) if raw_estimate is not None else None
                stab_engine.stability_blend = 0.0
        elif key == ord('x'):
            if USE_CONVERGENCE_MECHANIC:
                stab_engine.stability_blend = max(stab_engine.stability_blend - 0.2, 0.0)
        elif key == ord('b'):
            if USE_CONVERGENCE_MECHANIC:
                stab_engine.stability_blend = 1.0
        elif key == ord('a'):
            ANCHOR_MODE_ENABLED = not ANCHOR_MODE_ENABLED
        elif key == ord('g'):
            REPLAY_AUGMENT_ENABLED = not REPLAY_AUGMENT_ENABLED
            status = "ATIVADO" if REPLAY_AUGMENT_ENABLED else "DESATIVADO"
            print(f"Replay Adaptativo {status}")
        elif key == ord('+') or key == ord('='):
            adaptive_controller.base_lamda *= 1.2
            adaptive_controller.base_lamda = min(3000.0, adaptive_controller.base_lamda)
        elif key == ord('-') or key == ord('_'):
            adaptive_controller.base_lamda *= 0.8
            adaptive_controller.base_lamda = max(10.0, adaptive_controller.base_lamda)
        elif key == ord('['):
            adaptive_controller.base_alpha *= 0.9
            adaptive_controller.base_alpha = max(0.05, adaptive_controller.base_alpha)
        elif key == ord(']'):
            adaptive_controller.base_alpha *= 1.1
            adaptive_controller.base_alpha = min(0.9, adaptive_controller.base_alpha)
        elif key == ord('u'):
            USE_UPV = not USE_UPV
            ref_upv = None
            status = "ATIVADO" if USE_UPV else "DESATIVADO"
            print(f"UPV {status}")
        elif key == ord('1'):
            fx, fy = focus_points[current_focus_index]
            parent, leaf = quadtree_root.find_parent_of_point(fx, fy)
            if parent is not None:
                parent.merge()
                assign_region_ids(quadtree_root)
                focus_points, id_to_node = get_all_focus_points_and_map(quadtree_root)
                print(f"Quadtree merge na região ({fx}, {fy}). Folhas: {len(focus_points)}")
            else:
                print("Não é possível fazer merge (raiz ou região já é folha).")
        elif key == ord('2'):
            fx, fy = focus_points[current_focus_index]
            leaf = quadtree_root.find_leaf_containing_point(fx, fy)
            if leaf and leaf.is_leaf:
                leaf.subdivide()
                assign_region_ids(quadtree_root)
                focus_points, id_to_node = get_all_focus_points_and_map(quadtree_root)
                print(f"Quadtree subdividida na região ({fx}, {fy}). Folhas: {len(focus_points)}")
            else:
                print("Não foi possível subdividir (já no nível máximo?).")

cap.release()
cv2.destroyAllWindows()