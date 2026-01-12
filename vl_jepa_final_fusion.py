# ============================================================
# VL-JEPA: FINAL FUSION (Drills + Sleep + Stability)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import random
import os
import shutil
import time
import difflib

# ----------------------------
# 1. Config
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LOG_DIR = "final_fusion_logs"
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
IMG_RES = 200 
print(f"✅ Hardware: {DEVICE}")

def log_section(title):
    print(f"\n{'='*60}")
    print(f" 🧠 {title.upper()}")
    print(f"{'='*60}")

# ----------------------------
# 2. Physics & Colors
# ----------------------------
SHAPE_SIDES = {
    "circle": 0.0, "triangle": 3.0, "square": 4.0, "diamond": 4.0,
    "pentagon": 5.0, "hexagon": 6.0, "octagon": 8.0, "star": 10.0
}
SIZE_AREA = { "small": 0.15, "medium": 0.40, "large": 0.80 }
COLORS = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "green": (0, 255, 0), "white": (255, 255, 255), "black": (60, 60, 60),
    "teal": (0, 128, 128), "purple": (128, 0, 128), "orange": (255, 165, 0),
    "cyan": (0, 255, 255), "pink": (255, 192, 203), "grey": (100, 100, 100),
    "violet": (238, 130, 238), "brown": (165, 42, 42)
}

def draw_tensor(shape, color, size_name, save_name=None):
    img = Image.new("RGB", (IMG_RES, IMG_RES), "black")
    draw = ImageDraw.Draw(img)
    rgb = COLORS.get(color, (100, 100, 100))
    cx, cy = IMG_RES // 2, IMG_RES // 2
    if size_name == "small": r = random.randint(25, 35)
    elif size_name == "large": r = random.randint(80, 90)
    else: r = random.randint(50, 60)
    
    points = []
    if shape == "circle": draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=rgb)
    elif shape == "square": draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=rgb)
    elif shape == "triangle": points = [cx, cy-r, cx-r, cy+r, cx+r, cy+r]
    elif shape == "diamond": points = [cx, cy-r, cx+r, cy, cx, cy+r, cx-r, cy]
    elif shape == "star":
        for i in range(10):
            angle = i * np.pi/5; d = r if i%2==0 else r//3 
            points.extend([cx+d*np.cos(angle-np.pi/2), cy+d*np.sin(angle-np.pi/2)])
    elif shape == "hexagon":
        for i in range(6):
            angle = i * np.pi/3; points.extend([cx+r*np.cos(angle-np.pi/2), cy+r*np.sin(angle-np.pi/2)])
    elif shape == "octagon":
        for i in range(8):
            angle = i * np.pi/4; points.extend([cx+r*np.cos(angle-np.pi/2), cy+r*np.sin(angle-np.pi/2)])
    elif shape == "pentagon":
        for i in range(5):
            angle = i * 2*np.pi/5; points.extend([cx+r*np.cos(angle-np.pi/2), cy+r*np.sin(angle-np.pi/2)])
            
    if points: draw.polygon(points, fill=rgb)
    if save_name: img.save(f"{LOG_DIR}/{save_name}.png")
    
    arr = np.array(img).astype(np.float32) / 255.0
    sides_gt = torch.tensor([SHAPE_SIDES.get(shape, 0.0)], dtype=torch.float32)
    area_gt = torch.tensor([SIZE_AREA.get(size_name, 0.4)], dtype=torch.float32)
    return torch.tensor(arr).permute(2, 0, 1), sides_gt, area_gt, img

# ----------------------------
# 3. Architecture
# ----------------------------
class HierarchicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            # Deeper layer for fine details (Triangle vs Diamond)
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
    def forward(self, x): return self.net(x)

class OmniJEPA(nn.Module):
    def __init__(self, n_colors, n_shapes, n_sizes):
        super().__init__()
        self.vision = HierarchicalCNN()
        self.head_c = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 32))
        self.head_s = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 32))
        self.head_z = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 32))
        
        # Logic
        self.logic_sides = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.logic_area = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        
        # Fixed Embeddings (Orthogonal)
        self.txt_c = nn.Embedding(n_colors, 32)
        self.txt_s = nn.Embedding(n_shapes, 32)
        self.txt_z = nn.Embedding(n_sizes, 32)
        self.txt_c.weight.data = torch.eye(n_colors, 32)
        self.txt_s.weight.data = torch.eye(n_shapes, 32)
        self.txt_z.weight.data = torch.eye(n_sizes, 32)
        self.txt_c.weight.requires_grad = False
        self.txt_s.weight.requires_grad = False
        self.txt_z.weight.requires_grad = False

    def forward(self, x):
        feat = self.vision(x)
        pc, ps, pz = self.head_c(feat), self.head_s(feat), self.head_z(feat)
        p_sides, p_area = self.logic_sides(feat), self.logic_area(feat)
        return feat, pc, ps, pz, p_sides, p_area

# ----------------------------
# 4. Memory & Training
# ----------------------------
class MemoryBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, batch):
        if len(self.buffer) >= self.capacity: self.buffer.pop(0)
        detached = [t.detach().cpu() for t in batch]
        self.buffer.append(detached)
        
    def sample(self, batch_size=4):
        if not self.buffer: return None
        indices = np.random.choice(len(self.buffer), min(len(self.buffer), batch_size), replace=False)
        return [[t.to(DEVICE) for t in self.buffer[i]] for i in indices]

memory = MemoryBuffer()

def train_step(model, opt, batch, logic_mode="locked", focus_weight=1.0):
    img, tc, ts, tz, t_sides, t_area = batch
    
    # MIX REPLAY (Interleaved Batching)
    replay = memory.sample(batch_size=2)
    if replay:
        for b in replay:
            img = torch.cat([img, b[0]], dim=0)
            tc = torch.cat([tc, b[1]], dim=0)
            ts = torch.cat([ts, b[2]], dim=0)
            tz = torch.cat([tz, b[3]], dim=0)
            t_sides = torch.cat([t_sides, b[4]], dim=0)
            t_area = torch.cat([t_area, b[5]], dim=0)

    # Gentle LR for stability
    for g in opt.param_groups: g['lr'] = 0.0005 

    opt.zero_grad()
    _, pc, ps, pz, p_sides, p_area = model(img)
    
    loss_sem = F.mse_loss(pc, model.txt_c(tc)) + F.mse_loss(ps, model.txt_s(ts)) + F.mse_loss(pz, model.txt_z(tz))
    loss_phy = F.mse_loss(p_sides, t_sides) + F.mse_loss(p_area, t_area)
    
    # Maintenance Rehearsal (Always train physics slightly)
    loss = loss_sem + (loss_phy * 0.2) 
    
    if logic_mode == "remedial" or logic_mode == "bootcamp": 
        loss += (loss_phy * 2.0)
        
    loss.backward()
    opt.step()
    return loss_sem.item(), loss_phy.item()

# ----------------------------
# 5. Core Systems
# ----------------------------
def run_exam(model, c2i, s2i, z2i, target):
    print(f"\n📝 EXAM: {target.upper()}")
    card = Image.new("RGB", (IMG_RES*5, IMG_RES*5), "black")
    pass_count = 0
    mode = "shape"
    if target in z2i: mode = "size"
    elif target in c2i: mode = "color"

    with torch.no_grad():
        for i in range(25):
            c = target if mode=="color" else random.choice(list(c2i.keys()))
            s = target if mode=="shape" else random.choice(list(s2i.keys()))
            z = target if mode=="size" else random.choice(list(z2i.keys()))
            
            t, gt_sides, _, pil_img = draw_tensor(s, c, z)
            t = t.unsqueeze(0).to(DEVICE)
            _, pc, ps, pz, p_sides, p_area = model(t)
            
            pred_s = list(s2i.keys())[F.cosine_similarity(ps, model.txt_s.weight).argmax()]
            pred_z = list(z2i.keys())[F.cosine_similarity(pz, model.txt_z.weight).argmax()]
            pred_c = list(c2i.keys())[F.cosine_similarity(pc, model.txt_c.weight).argmax()]
            
            passed = False
            if mode == "shape": 
                tol = 0.5 + (gt_sides.item() * 0.15)
                logic_pass = abs(p_sides.item() - gt_sides.item()) < (tol * 1.5) 
                passed = (pred_s == target and logic_pass)
            elif mode == "size": passed = (pred_z == target)
            elif mode == "color": passed = (pred_c == target)
            if passed: pass_count += 1
            card.paste(pil_img, ((i%5)*IMG_RES, (i//5)*IMG_RES))
            
    fn = f"REPORT_{target}_{int(time.time())}.png"
    card.save(f"{LOG_DIR}/{fn}")
    print(f"   • Score: {pass_count}/25")
    return pass_count/25.0

def parse_concept_string(text, c_list, s_list, z_list):
    words = text.lower().replace(",", "").split()
    concepts = []
    s, c, z = "circle", "white", "medium"
    for w in words:
        if w in c_list: c = w; concepts.append(w)
        if w in s_list: s = w; concepts.append(w)
        if w in z_list: z = w; concepts.append(w)
    return s, c, z, concepts

def probe_mind(model, c2i, s2i, z2i, t):
    model.eval()
    with torch.no_grad():
        feat, pc, ps, pz, p_sides, p_area = model(t)
        pred_s = list(s2i.keys())[F.cosine_similarity(ps, model.txt_s.weight).argmax()]
        pred_c = list(c2i.keys())[F.cosine_similarity(pc, model.txt_c.weight).argmax()]
        pred_z = list(z2i.keys())[F.cosine_similarity(pz, model.txt_z.weight).argmax()]
        
        log_section("SYSTEM 1")
        print(f"   • I see a: '{pred_z} {pred_c} {pred_s}'")
        
        log_section("SYSTEM 2")
        print(f"   • Geometry: {p_sides.item():.2f} sides")
        
        expected = SHAPE_SIDES.get(pred_s, 0)
        tol = 0.8 + (expected * 0.2)
        if abs(p_sides.item() - expected) > tol:
            print(f"   🚨 REJECTION! Logic overrides Intuition.")
        else:
            print(f"   ✅ ACCEPTED.")

def compare(model, text_a, text_b, c2i, s2i, z2i):
    s1, c1, z1, _ = parse_concept_string(text_a, list(c2i.keys()), list(s2i.keys()), list(z2i.keys()))
    s2, c2, z2, _ = parse_concept_string(text_b, list(c2i.keys()), list(s2i.keys()), list(z2i.keys()))
    
    print(f"\n⚖️  COMPARING: '{text_a}' vs '{text_b}'")
    t1, _, _, _ = draw_tensor(s1, c1, z1); t2, _, _, _ = draw_tensor(s2, c2, z2)
    model.eval()
    with torch.no_grad():
        _, _, ps1, pc1, _, _ = model(t1.unsqueeze(0).to(DEVICE))
        _, _, ps2, pc2, _, _ = model(t2.unsqueeze(0).to(DEVICE))
        print(f"   • Shape Dist: {F.pairwise_distance(ps1, ps2).item():.4f}")
        print(f"   • Color Dist: {F.pairwise_distance(pc1, pc2).item():.4f}")

# ----------------------------
# 6. Drill & Solidify
# ----------------------------
def run_drill(model, opt, c1, c2, c2i, s2i, z2i):
    print(f"\n⚔️  DRILL SESSION: '{c1}' vs '{c2}'")
    mode = "shape"
    if c1 in c2i: mode = "color"
    elif c1 in z2i: mode = "size"
    
    distance = 0.0
    rounds = 0
    
    while distance < 0.8 and rounds < 25:
        rounds += 1
        model.train()
        
        img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
        for _ in range(16):
            target = c1 if random.random() < 0.5 else c2
            s = target if mode=="shape" else random.choice(list(s2i.keys()))
            c = target if mode=="color" else random.choice(list(c2i.keys()))
            z = target if mode=="size" else random.choice(list(z2i.keys()))
            im, si, ar, _ = draw_tensor(s, c, z)
            img.append(im); tc.append(c2i[c]); ts.append(s2i[s]); tz.append(z2i[z]); 
            tsides.append(si); tarea.append(ar)
            
        batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
        train_step(model, opt, batch, logic_mode="locked", focus_weight=1.5)
        
        model.eval()
        with torch.no_grad():
            t1, _, _, _ = draw_tensor(c1 if mode=="shape" else "circle", "red", "medium")
            t2, _, _, _ = draw_tensor(c2 if mode=="shape" else "circle", "red", "medium")
            f1, _, ps1, pc1, _, _ = model(t1.unsqueeze(0).to(DEVICE))
            f2, _, ps2, pc2, _, _ = model(t2.unsqueeze(0).to(DEVICE))
            if mode == "shape": distance = F.pairwise_distance(ps1, ps2).item()
            elif mode == "color": distance = F.pairwise_distance(pc1, pc2).item()
            
        if rounds % 5 == 0: print(f"   Round {rounds}: Distance = {distance:.4f}")

    if distance > 0.8: print(f"   ✅ SUCCESS: (Dist: {distance:.4f})")
    else: print(f"   ⚠️  TIMED OUT: (Dist: {distance:.4f})")

def solidify_memory(model, opt, known_concepts, c_list, s_list, z_list, c2i, s2i, z2i):
    print(f"\n🧠 SOLIDIFYING MEMORY (Deep Sleep)...")
    if not known_concepts: print("   (Empty mind.)"); return

    # 1. Refresh Buffer
    print(f"   ...Recalling: {list(known_concepts)[:5]}...")
    for target in known_concepts:
        img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
        mode = "shape"
        if target in c_list: mode = "color"
        elif target in z_list: mode = "size"
        
        for _ in range(8):
            s = target if mode=="shape" else random.choice(s_list)
            c = target if mode=="color" else random.choice(c_list)
            z = target if mode=="size" else random.choice(z_list)
            im, si, ar, _ = draw_tensor(s, c, z)
            img.append(im); tc.append(c2i[c]); ts.append(s2i[s]); tz.append(z2i[z]); 
            tsides.append(si); tarea.append(ar)
        
        batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
        memory.add(batch)

    # 2. Train on memories
    print(f"   ...Dreaming (100 cycles)")
    for _ in range(100):
        replay = memory.sample(batch_size=8)
        if replay:
            for b in replay: train_step(model, opt, b, logic_mode="locked", focus_weight=0.5)
    print("   ✨ Memory Solidified.")

# ----------------------------
# 7. Main Execution
# ----------------------------
if __name__ == "__main__":
    c_list = list(COLORS.keys()); s_list = list(SHAPE_SIDES.keys()); z_list = list(SIZE_AREA.keys())
    c2i = {c:i for i,c in enumerate(c_list)}; s2i = {s:i for i,s in enumerate(s_list)}; z2i = {z:i for i,z in enumerate(z_list)}
    model = OmniJEPA(len(c_list), len(s_list), len(z_list)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    known_concepts = set()

    print("\n👶 PHASE 1: INFANCY")
    for i in range(1001): 
        img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
        for _ in range(32):
            c = random.choice(c_list); s = random.choice(s_list); z = random.choice(z_list)
            im, si, ar, _ = draw_tensor(s, c, z)
            img.append(im); tc.append(c2i[c]); ts.append(0); tz.append(0); tsides.append(si); tarea.append(ar)
        batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
        _, loss_phy = train_step(model, opt, batch, logic_mode="bootcamp")
        if i % 200 == 0: print(f"   Step {i}: Physics Loss={loss_phy:.4f}")

    print("\n❄️  LOCKING SYSTEM 2.")
    
    def learn_concept(target):
        all_concepts = s_list + c_list + z_list
        matches = difflib.get_close_matches(target, all_concepts)
        if matches: target = matches[0]
        else: print(f"   ❌ Unknown '{target}'"); return False

        print(f"\n📚 LEARNING: {target}")
        mode = "shape"
        if target in c_list: mode = "color"
        elif target in z_list: mode = "size"

        attempts = 0; max_attempts = 600
        while attempts < max_attempts:
            attempts += 1
            model.train()
            img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
            for _ in range(16):
                s = target if mode=="shape" else random.choice(s_list)
                c = target if mode=="color" else random.choice(c_list)
                z = target if mode=="size" else random.choice(z_list)
                im, si, ar, _ = draw_tensor(s, c, z)
                img.append(im); tc.append(c2i[c]); ts.append(s2i[s]); tz.append(z2i[z]); 
                tsides.append(si); tarea.append(ar)
            
            batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
            if attempts % 10 == 0: memory.add(batch)
            train_step(model, opt, batch, logic_mode="locked")
            
            if attempts % 100 == 0:
                score = run_exam(model, c2i, s2i, z2i, target)
                if score >= 0.8:
                    print(f"   ✅ MASTERY CONFIRMED.")
                    known_concepts.add(target)
                    return True
        return False

    def ensure_knowledge(concepts):
        for c in concepts:
            if c not in known_concepts:
                print(f"   🚧 Learning '{c}' first...")
                learn_concept(c)

    def explain(model, text):
        s, c, z, concepts = parse_concept_string(text, list(c2i.keys()), list(s2i.keys()), list(z2i.keys()))
        ensure_knowledge(concepts)
        print(f"\n🤔 EXPLAINING: '{text}'")
        t, _, _, _ = draw_tensor(s, c, z, save_name="EXPLAIN_QUERY")
        probe_mind(model, c2i, s2i, z2i, t.unsqueeze(0).to(DEVICE))

    # 2. CURRICULUM
    print("\n🏫 PHASE 2: PRIMARY SCHOOL")
    for t in ["circle", "square", "triangle", "red", "blue"]: learn_concept(t)

    print("\n🤖 FINAL FUSION ONLINE.")
    print("   Commands: 'drill', 'solidify', 'explain', 'compare', 'auto', 'quit'")
    
    while True:
        try:
            i = input("\nYou: ").lower().strip()
            if not i: continue
            if i == "quit": break
            
            lines = i.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                print(f"   ▶ {line}")
                
                if line == "solidify":
                    solidify_memory(model, opt, known_concepts, c_list, s_list, z_list, c2i, s2i, z2i)
                    continue

                if line.startswith("drill:"):
                    parts = line.split("drill:")[1].split("vs")
                    c1 = parts[0].strip(); c2 = parts[1].strip()
                    ensure_knowledge([c1, c2])
                    run_drill(model, opt, c1, c2, c2i, s2i, z2i)
                    continue

                if line == "auto":
                    unknowns = [x for x in s_list + c_list if x not in known_concepts]
                    if not unknowns: unknowns = s_list
                    for t in random.sample(unknowns, min(3, len(unknowns))): learn_concept(t)
                    continue

                if line.startswith("learn:"): learn_concept(line.split(":")[1].strip()); continue
                if line.startswith("exam:"): run_exam(model, c2i, s2i, z2i, line.split(":")[1].strip()); continue
                if line.startswith("compare:"): 
                    parts = line.split("compare:")[1].split("to")
                    compare(model, parts[0].strip(), parts[1].strip(), c2i, s2i, z2i)
                    continue
                if line.startswith("explain:"): explain(model, line.split(":")[1].strip()); continue

                parts = line.split()
                if len(parts) >= 4 and parts[0] == "show":
                    t, _, _, _ = draw_tensor(parts[3], parts[2], parts[1])
                    probe_mind(model, c2i, s2i, z2i, t.unsqueeze(0).to(DEVICE))

        except KeyboardInterrupt: break