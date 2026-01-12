# ============================================================
# VL-JEPA: CURRICULUM FINAL (Robust Input + Fixed Logic)
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
# 1. Config & Hardware
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
LOG_DIR = "curriculum_logs"
if os.path.exists(LOG_DIR): shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
IMG_RES = 200 
PATCH_SIZE = 20 
print(f"✅ Hardware: {DEVICE}")

def log_section(title):
    print(f"\n{'='*60}")
    print(f" 🧠 {title.upper()}")
    print(f"{'='*60}")

# ----------------------------
# 2. Physics, Globals & Curriculum
# ----------------------------
SHAPE_SIDES = {
    "circle": 0.0, "triangle": 3.0, "square": 4.0, "diamond": 4.0,
    "pentagon": 5.0, "hexagon": 6.0, "octagon": 8.0, "star": 10.0, 
    "decagon": 10.0, "line": 1.0
}
SIZE_AREA = { "small": 0.15, "medium": 0.40, "large": 0.80 }
COLORS = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "green": (0, 255, 0), "white": (255, 255, 255), "black": (60, 60, 60),
    "teal": (0, 128, 128), "purple": (128, 0, 128), "orange": (255, 165, 0),
    "cyan": (0, 255, 255), "pink": (255, 192, 203), "grey": (100, 100, 100),
    "violet": (238, 130, 238), "brown": (165, 42, 42),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0)
}

CURRICULUM = {
    "purple": ["red", "blue"],
    "orange": ["red", "yellow"],
    "green":  ["blue", "yellow"],
    "brown":  ["red", "green"], 
    "pink":   ["red", "white"],
    "grey":   ["black", "white"],
    "star":     ["triangle"],
    "hexagon":  ["triangle"],
    "octagon":  ["square"],
    "diamond":  ["triangle"],
    "decagon": ["pentagon"],  # A decagon is like a complex pentagon
    "gold": ["yellow", "orange"] # Gold requires understanding Yellow/Orange
}

C_LIST = list(COLORS.keys())
S_LIST = list(SHAPE_SIDES.keys())
Z_LIST = list(SIZE_AREA.keys())
C2I = {c:i for i,c in enumerate(C_LIST)}
S2I = {s:i for i,s in enumerate(S_LIST)}
Z2I = {z:i for i,z in enumerate(Z_LIST)}

known_concepts = set()

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
    elif shape == "line":
        draw.line([cx-r, cy, cx+r, cy], fill=rgb, width=5)
    elif shape == "decagon":
        for i in range(10):
            angle = i * 2 * np.pi / 10
            points.extend([cx + r * np.cos(angle), cy + r * np.sin(angle)])       
    if points: draw.polygon(points, fill=rgb)
    if save_name: img.save(f"{LOG_DIR}/{save_name}.png")
    
    arr = np.array(img).astype(np.float32) / 255.0
    sides_gt = torch.tensor([SHAPE_SIDES.get(shape, 0.0)], dtype=torch.float32)
    area_gt = torch.tensor([SIZE_AREA.get(size_name, 0.4)], dtype=torch.float32)
    t = torch.tensor(arr).permute(2, 0, 1).contiguous()
    return t, sides_gt, area_gt, img

# ----------------------------
# 3. Architecture
# ----------------------------
class SimpleViT(nn.Module):
    def __init__(self, img_size=IMG_RES, patch_size=PATCH_SIZE, dim=128, depth=4, heads=4):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(num_patches + 1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*2, batch_first=False)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        if not x.is_contiguous(): x = x.contiguous()
        x = self.patch_embed(x) 
        x = x.flatten(2).permute(2, 0, 1).contiguous()
        n, b, dim = x.shape
        cls = self.cls_token.expand(1, b, -1)
        x = torch.cat((cls, x), dim=0)
        x = x + self.pos_embed
        x = self.transformer(x)
        return x[0].contiguous()

class OmniJEPA(nn.Module):
    def __init__(self, n_colors, n_shapes, n_sizes):
        super().__init__()
        self.vision = SimpleViT(dim=128)
        self.head_c = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 32))
        self.head_s = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 32))
        self.head_z = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 32))
        self.logic_sides = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        self.logic_area = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
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
# 4. Global Memory
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

def train_step(model, opt, batch, logic_mode="locked", focus_mode="all"):
    img, tc, ts, tz, t_sides, t_area = batch
    replay = memory.sample(batch_size=2)
    if replay:
        for b in replay:
            img = torch.cat([img, b[0]], dim=0)
            tc = torch.cat([tc, b[1]], dim=0)
            ts = torch.cat([ts, b[2]], dim=0)
            tz = torch.cat([tz, b[3]], dim=0)
            t_sides = torch.cat([t_sides, b[4]], dim=0)
            t_area = torch.cat([t_area, b[5]], dim=0)

    if not img.is_contiguous(): img = img.contiguous()
    lr = 0.0003
    if focus_mode != "all": lr = 0.0005 
    for g in opt.param_groups: g['lr'] = lr

    opt.zero_grad()
    _, pc, ps, pz, p_sides, p_area = model(img)
    
    loss_c = F.mse_loss(pc, model.txt_c(tc))
    loss_s = F.mse_loss(ps, model.txt_s(ts))
    loss_z = F.mse_loss(pz, model.txt_z(tz))
    loss_phy = F.mse_loss(p_sides, t_sides) + F.mse_loss(p_area, t_area)
    
    if focus_mode == "color": loss = (loss_c * 15.0) + loss_s + loss_z + (loss_phy * 0.1)
    elif focus_mode == "shape": loss = loss_c + (loss_s * 15.0) + loss_z + (loss_phy * 0.1)
    else: loss = loss_c + loss_s + loss_z + (loss_phy * 0.2)

    if logic_mode == "remedial": loss += (loss_phy * 2.0)
    loss.backward()
    opt.step()
    return loss_c.item(), loss_phy.item()

# ----------------------------
# 5. Brain Functions
# ----------------------------
def run_exam(model, target):
    print(f"\n📝 EXAM: {target.upper()}")
    card = Image.new("RGB", (IMG_RES*5, IMG_RES*5), "black")
    pass_count = 0
    mode = "shape"
    if target in Z2I: mode = "size"
    elif target in C2I: mode = "color"

    with torch.no_grad():
        for i in range(25):
            c = target if mode=="color" else random.choice(list(C2I.keys()))
            s = target if mode=="shape" else random.choice(list(S2I.keys()))
            z = target if mode=="size" else random.choice(list(Z2I.keys()))
            t, gt_sides, _, pil_img = draw_tensor(s, c, z)
            t = t.unsqueeze(0).to(DEVICE)
            _, pc, ps, pz, p_sides, p_area = model(t)
            
            pred_s = list(S2I.keys())[F.cosine_similarity(ps, model.txt_s.weight).argmax()]
            pred_z = list(Z2I.keys())[F.cosine_similarity(pz, model.txt_z.weight).argmax()]
            pred_c = list(C2I.keys())[F.cosine_similarity(pc, model.txt_c.weight).argmax()]
            
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

def learn_concept(model, opt, target):
    all_concepts = S_LIST + C_LIST + Z_LIST
    matches = difflib.get_close_matches(target, all_concepts)
    if matches: target = matches[0]
    else: print(f"   ❌ Unknown '{target}'"); return False

    print(f"\n📚 LEARNING: {target}")
    mode = "shape"
    if target in C_LIST: mode = "color"
    elif target in Z_LIST: mode = "size"

    attempts = 0; max_attempts = 600
    while attempts < max_attempts:
        attempts += 1
        model.train()
        img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
        for _ in range(16):
            s = target if mode=="shape" else random.choice(S_LIST)
            c = target if mode=="color" else random.choice(C_LIST)
            z = target if mode=="size" else random.choice(Z_LIST)
            im, si, ar, _ = draw_tensor(s, c, z)
            img.append(im); tc.append(C2I[c]); ts.append(S2I[s]); tz.append(Z2I[z]); 
            tsides.append(si); tarea.append(ar)
        
        batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
        if attempts % 10 == 0: memory.add(batch)
        
        focus = "color" if mode == "color" else "shape" if mode == "shape" else "all"
        train_step(model, opt, batch, logic_mode="locked", focus_mode=focus)
        
        if attempts % 100 == 0:
            score = run_exam(model, target)
            if score >= 0.8:
                print(f"   ✅ MASTERY CONFIRMED.")
                known_concepts.add(target)
                return True
    return False

def ensure_knowledge(model, opt, concepts):
    for c in concepts:
        if c not in known_concepts:
            print(f"   🚧 Learning '{c}' first...")
            learn_concept(model, opt, c)

def probe_mind(model, t):
    model.eval()
    with torch.no_grad():
        if not t.is_contiguous(): t = t.contiguous()
        feat, pc, ps, pz, p_sides, p_area = model(t)
        pred_s = list(S2I.keys())[F.cosine_similarity(ps, model.txt_s.weight).argmax()]
        pred_c = list(C2I.keys())[F.cosine_similarity(pc, model.txt_c.weight).argmax()]
        pred_z = list(Z2I.keys())[F.cosine_similarity(pz, model.txt_z.weight).argmax()]
        
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

def parse_concept_string(text):
    words = text.lower().replace(",", "").split()
    concepts = []
    s, c, z = "circle", "white", "medium"
    for w in words:
        if w in C_LIST: c = w; concepts.append(w)
        if w in S_LIST: s = w; concepts.append(w)
        if w in Z_LIST: z = w; concepts.append(w)
    return s, c, z, concepts

def explain(model, opt, text):
    s, c, z, concepts = parse_concept_string(text)
    ensure_knowledge(model, opt, concepts)
    print(f"\n🤔 EXPLAINING: '{text}'")
    t, _, _, _ = draw_tensor(s, c, z, save_name="EXPLAIN_QUERY")
    probe_mind(model, t.unsqueeze(0).to(DEVICE))

def compare(model, text_a, text_b):
    s1, c1, z1, _ = parse_concept_string(text_a)
    s2, c2, z2, _ = parse_concept_string(text_b)
    print(f"\n⚖️  COMPARING: '{text_a}' vs '{text_b}'")
    t1, _, _, _ = draw_tensor(s1, c1, z1); t2, _, _, _ = draw_tensor(s2, c2, z2)
    model.eval()
    with torch.no_grad():
        if not t1.is_contiguous(): t1 = t1.contiguous()
        if not t2.is_contiguous(): t2 = t2.contiguous()
        _, _, ps1, pc1, _, _ = model(t1.unsqueeze(0).to(DEVICE))
        _, _, ps2, pc2, _, _ = model(t2.unsqueeze(0).to(DEVICE))
        print(f"   • Shape Dist: {F.pairwise_distance(ps1, ps2).item():.4f}")
        print(f"   • Color Dist: {F.pairwise_distance(pc1, pc2).item():.4f}")

def run_drill(model, opt, c1, c2):
    print(f"\n⚔️  DRILL SESSION: '{c1}' vs '{c2}'")
    mode = "shape"
    if c1 in C2I: mode = "color"
    elif c1 in Z2I: mode = "size"
    focus = "color" if mode == "color" else "shape"
    distance = 0.0
    rounds = 0
    while distance < 0.8 and rounds < 25:
        rounds += 1
        model.train()
        img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
        for _ in range(16):
            target = c1 if random.random() < 0.5 else c2
            s = target if mode=="shape" else random.choice(list(S2I.keys()))
            c = target if mode=="color" else random.choice(list(C2I.keys()))
            z = target if mode=="size" else random.choice(list(Z2I.keys()))
            im, si, ar, _ = draw_tensor(s, c, z)
            img.append(im); tc.append(C2I[c]); ts.append(S2I[s]); tz.append(Z2I[z]); 
            tsides.append(si); tarea.append(ar)
        batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
        train_step(model, opt, batch, logic_mode="locked", focus_mode=focus)
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

def run_bootcamp(model, opt, topic="color"):
    print(f"\n🥾 BOOTCAMP: {topic.upper()}")
    targets = C_LIST if topic == "color" else S_LIST
    for i in range(len(targets)-1):
        t1 = targets[i]; t2 = targets[i+1]
        ensure_knowledge(model, opt, [t1, t2])
        run_drill(model, opt, t1, t2)
    solidify_memory(model, opt, known_concepts)

def solidify_memory(model, opt, known_concepts):
    print(f"\n🧠 SOLIDIFYING MEMORY (Deep Sleep)...")
    if not known_concepts: print("   (Empty mind.)"); return
    print(f"   ...Recalling: {list(known_concepts)[:5]}...")
    for target in known_concepts:
        img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
        mode = "shape"
        if target in C_LIST: mode = "color"
        elif target in Z_LIST: mode = "size"
        for _ in range(8):
            s = target if mode=="shape" else random.choice(S_LIST)
            c = target if mode=="color" else random.choice(C_LIST)
            z = target if mode=="size" else random.choice(Z_LIST)
            im, si, ar, _ = draw_tensor(s, c, z)
            img.append(im); tc.append(C2I[c]); ts.append(S2I[s]); tz.append(Z2I[z]); 
            tsides.append(si); tarea.append(ar)
        batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
        memory.add(batch)
    print(f"   ...Dreaming (100 cycles)")
    for _ in range(100):
        replay = memory.sample(batch_size=8)
        if replay:
            for b in replay: train_step(model, opt, b, logic_mode="locked", focus_mode="all")
    print("   ✨ Memory Solidified.")

def consult_teacher(known):
    print("\n🤔 AGENT CONSULTING CURRICULUM...")
    unlockable = []
    all_concepts = set(S_LIST + C_LIST)
    remaining = all_concepts - known
    
    for concept in remaining:
        if concept in CURRICULUM:
            prereqs = CURRICULUM[concept]
            if all(p in known for p in prereqs):
                unlockable.append((concept, prereqs))
        else:
            unlockable.append((concept, []))

    if not unlockable:
        print("   🤖 I have mastered everything in my universe!")
        return None

    derived = [u for u in unlockable if u[1]]
    if derived:
        target, parents = random.choice(derived)
        parents_str = " and ".join(parents)
        print(f"   🤖 Master! I know {parents_str}.")
        print(f"   🥺 May I attempt to learn '{target}' next?")
        return target
    else:
        target = random.choice(unlockable)[0]
        print(f"   🤖 I need to build my foundation.")
        print(f"   🥺 Please teach me a primary concept like '{target}'.")
        return target

# ----------------------------
# 6. Main Execution
# ----------------------------
if __name__ == "__main__":
    model = OmniJEPA(len(C_LIST), len(S_LIST), len(Z_LIST)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n👶 PHASE 1: INFANCY")
    for i in range(1001): 
        img, tc, ts, tz, tsides, tarea = [], [], [], [], [], []
        for _ in range(32):
            c = random.choice(C_LIST); s = random.choice(S_LIST); z = random.choice(Z_LIST)
            im, si, ar, _ = draw_tensor(s, c, z)
            img.append(im); tc.append(C2I[c]); ts.append(0); tz.append(0); tsides.append(si); tarea.append(ar)
        batch = (torch.stack(img).to(DEVICE), torch.tensor(tc).to(DEVICE), torch.tensor(ts).to(DEVICE), torch.tensor(tz).to(DEVICE), torch.stack(tsides).to(DEVICE), torch.stack(tarea).to(DEVICE))
        _, loss_phy = train_step(model, opt, batch, logic_mode="bootcamp")
        if i % 200 == 0: print(f"   Step {i}: Physics Loss={loss_phy:.4f}")

    print("\n❄️  LOCKING SYSTEM 2.")

    # 2. CURRICULUM (Primaries Only)
    print("\n🏫 PHASE 2: PRIMARY SCHOOL")
    for t in ["circle", "square", "triangle", "red", "blue", "yellow", "white", "black"]: 
        learn_concept(model, opt, t)

    print("\n🤖 VIT CURRICULUM EDITION ONLINE.")
    print("   Commands: 'auto 10', 'consult', 'bootcamp color', 'solidify', 'explain'")
    
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
                
                if line == "consult":
                    suggestion = consult_teacher(known_concepts)
                    if suggestion:
                        confirm = input(f"   Teacher (You): Approve '{suggestion}'? (y/n): ")
                        if confirm.lower() == 'y':
                            learn_concept(model, opt, suggestion)
                            solidify_memory(model, opt, known_concepts)
                    continue

                if line == "solidify": solidify_memory(model, opt, known_concepts); continue
                if line == "bootcamp color": run_bootcamp(model, opt, "color"); continue
                if line == "bootcamp shape": run_bootcamp(model, opt, "shape"); continue
                
                if line.startswith("drill:"):
                    parts = line.split("drill:")[1].split("vs")
                    c1 = parts[0].strip(); c2 = parts[1].strip()
                    ensure_knowledge(model, opt, [c1, c2])
                    run_drill(model, opt, c1, c2)
                    continue

                if line.startswith("auto"):
                    parts = line.split()
                    mode = "count"
                    limit = 3

                    if len(parts) > 1:
                        arg = parts[1]
                        try:
                            if arg.endswith("m"):
                                mode = "time"
                                limit = int(arg.replace("m", "")) * 60
                            elif arg.endswith("h"):
                                mode = "time"
                                limit = int(arg.replace("h", "")) * 3600
                            else:
                                mode = "count"
                                limit = int(arg)
                        except ValueError:
                            print(f"   ⚠️ Invalid argument '{arg}', defaulting to 3 concepts.")
                            mode = "count"; limit = 3

                    print(f"   🤖 AUTO-PILOT: Target = {limit} {'seconds' if mode=='time' else 'concepts'}")
                    start_time = time.time()
                    concepts_learned = 0
                    
                    while True:
                        if mode == "count" and concepts_learned >= limit: 
                            print("   🛑 Limit reached."); break
                        if mode == "time" and (time.time() - start_time) > limit: 
                            print("   🛑 Time limit reached."); break

                        unknowns = [x for x in S_LIST + C_LIST + Z_LIST if x not in known_concepts]
                        candidates = []
                        for u in unknowns:
                            if u in CURRICULUM:
                                if all(p in known_concepts for p in CURRICULUM[u]):
                                    candidates.append(u) 
                            else:
                                candidates.append(u)
                        
                        if not candidates: candidates = unknowns
                        if not candidates: print("   ✨ Universe Mastered."); break

                        target = random.choice(candidates)
                        learn_concept(model, opt, target)
                        concepts_learned += 1

                        if concepts_learned % 3 == 0:
                            print(f"   💤 Power Nap...")
                            solidify_memory(model, opt, known_concepts)
                    
                    solidify_memory(model, opt, known_concepts) 
                    continue

                if line.startswith("learn:"): learn_concept(model, opt, line.split(":")[1].strip()); continue
                if line.startswith("exam:"): run_exam(model, line.split(":")[1].strip()); continue
                if line.startswith("compare:"): 
                    parts = line.split("compare:")[1].split("to")
                    compare(model, parts[0].strip(), parts[1].strip())
                    continue
                if line.startswith("explain:"): explain(model, opt, line.split(":")[1].strip()); continue
                
                parts = line.split()
                if len(parts) >= 4 and parts[0] == "show":
                    t, _, _, _ = draw_tensor(parts[3], parts[2], parts[1])
                    probe_mind(model, t.unsqueeze(0).to(DEVICE))

        except KeyboardInterrupt: break