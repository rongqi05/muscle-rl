"""肌肉分组映射：左右侧与上下肢，源文件权威标签 + 关节跨越交叉验证。

两条独立证据链：

**A. 权威分组（决定实验缩放范围）**
官方模型把肌肉按解剖区域分文件组织：``Muscle/Muscle_{Leg,Arm,Arm_Hand,Torso}_{r,l}.xml``。
这给出 ``limb_group ∈ {lower, upper, torso}``。

**B. 几何/功能验证（用于审计，不决定分组）**
对每个肌肉执行器，沿其 tendon 的 wrap 对象（site / cylinder / sphere）找到跨越的 rigid body，
再在运动学树上求「最低公共祖先 → 各 body」路径上的关节集合，得到该肌肉**实际跨越的关节**。
按关节名归类为 lower / upper / torso，得到 ``crossed_limbs``。

两者不一致时不会被静默忽略：``agrees_with_geometry=False`` 的条目会被列出。
典型情况如背阔肌（官方归 torso，但功能上跨越肩关节）——这类条目被标注而非丢弃。

用法::

    PYTHONPATH=. python scripts/export_muscle_map.py
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from hemirl import paths

# ---------------------------------------------------------------- body 归类

LOWER_BODY_TOKENS: Tuple[str, ...] = ("femur", "tibia", "talus", "calcn", "toes", "patella")
UPPER_BODY_TOKENS: Tuple[str, ...] = (
    "clavicle",
    "clavphant",
    "scapula",
    "scapphant",
    "humphant",
    "humerus",
    "ulna",
    "radius",
    "proximal_row",
    "hand",
)
TORSO_BODY_TOKENS: Tuple[str, ...] = (
    "pelvis",
    "sacrum",
    "abdomen",
    "lumbar",
    "thoracic",
    "sternum",
    "head_neck",
    "rib",
)

# ---------------------------------------------------------------- 关节归类

LOWER_JOINT_TOKENS: Tuple[str, ...] = ("hip_", "knee_", "ankle_", "subtalar_", "mtp_")
UPPER_JOINT_TOKENS: Tuple[str, ...] = (
    "sternoclavicular",
    "acromioclavicular",
    "unrotscap",
    "shoulder",
    "unrothum",
    "elv_angle",
    "elbow_",
    "pro_sup",
    "wrist_hand",
)
#: 腕关节在模型里直接叫 `flexion_r` / `deviation_r`，单独列出
UPPER_BARE_JOINT_NAMES: Tuple[str, ...] = ("flexion", "deviation")
TORSO_JOINT_TOKENS: Tuple[str, ...] = ("pelvis_", "l5_s1", "t12_l1", "t1_head_neck")

SIDE_SUFFIX = re.compile(r"_(r|l|R|L)$")

LIMB_ORDER = ("lower", "upper", "torso")

_FILE_LABEL: Dict[str, str] = {
    "Muscle_Leg_r": "lower",
    "Muscle_Leg_l": "lower",
    "Muscle_Arm_r": "upper",
    "Muscle_Arm_l": "upper",
    "Muscle_Arm_Hand_r": "upper",
    "Muscle_Arm_Hand_l": "upper",
    "Muscle_Torso": "torso",
}


def strip_side(name: str) -> str:
    """去掉结尾的 `_r`/`_l`/`_R`/`_L`。"""
    return SIDE_SUFFIX.sub("", name)


def body_limb(body_name: str) -> str:
    """把 rigid body 名归入 lower / upper / torso / unknown。"""
    base = strip_side(body_name).lower()
    for tok in LOWER_BODY_TOKENS:
        if base.startswith(tok):
            return "lower"
    for tok in UPPER_BODY_TOKENS:
        if base.startswith(tok):
            return "upper"
    for tok in TORSO_BODY_TOKENS:
        if base.startswith(tok):
            return "torso"
    return "unknown"


def body_side(body_name: str) -> str:
    """把 rigid body 名归入 L / R / M（中线）。"""
    m = SIDE_SUFFIX.search(body_name)
    return m.group(1).upper() if m else "M"


def joint_limb(joint_name: str) -> str:
    """把关节名归入 lower / upper / torso / unknown。"""
    low = joint_name.lower()
    for tok in LOWER_JOINT_TOKENS:
        if low.startswith(tok):
            return "lower"
    for tok in UPPER_JOINT_TOKENS:
        if low.startswith(tok):
            return "upper"
    for tok in UPPER_BARE_JOINT_NAMES:
        if low == tok or low.startswith(tok + "_"):
            return "upper"
    for tok in TORSO_JOINT_TOKENS:
        if low.startswith(tok):
            return "torso"
    return "unknown"


# ---------------------------------------------------------------- 数据结构


@dataclass
class MuscleEntry:
    """单个肌肉执行器的解剖归属。"""

    index: int
    name: str
    tendon: str
    limb_group: str  # 权威分组（官方源文件）
    source_file: Optional[str]
    side: str  # L / R / M
    side_source: str  # name / geom / midline / geom-ambiguous
    crossed_limbs: List[str]  # 由跨越关节推出
    crossed_joints: List[str]
    bodies: List[str]
    sites: List[str]
    agrees_with_geometry: bool
    note: str = ""

    @property
    def group_key(self) -> str:
        return f"{self.side}/{self.limb_group}"

    def as_row(self) -> Dict[str, object]:
        d = asdict(self)
        d["crossed_limbs"] = "|".join(self.crossed_limbs)
        d["crossed_joints"] = "|".join(self.crossed_joints)
        d["bodies"] = "|".join(self.bodies)
        d["sites"] = "|".join(self.sites)
        d["group_key"] = self.group_key
        return d


@dataclass
class MuscleMap:
    """整模型映射与统计。"""

    entries: List[MuscleEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def indices_for(self, side: str, limb: str) -> List[int]:
        """给定侧别与部位，返回执行器索引。``side ∈ {'L','R','M'}``。"""
        return [e.index for e in self.entries if e.side == side and e.limb_group == limb]

    def by_group(self, side: str, limb: str) -> List[MuscleEntry]:
        return [e for e in self.entries if e.side == side and e.limb_group == limb]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.entries:
            out[e.group_key] = out.get(e.group_key, 0) + 1
        return dict(sorted(out.items()))

    def disagreements(self) -> List[MuscleEntry]:
        return [e for e in self.entries if not e.agrees_with_geometry]

    def to_json(self) -> dict:
        no_joint = [e for e in self.entries if not e.crossed_joints]
        extra_limb = [
            e for e in self.entries
            if e.crossed_limbs and e.limb_group in e.crossed_limbs and len(e.crossed_limbs) > 1
        ]
        return {
            "n": len(self.entries),
            "counts": self.counts(),
            "n_no_movable_joint": len(no_joint),
            "no_movable_joint_examples": [e.name for e in no_joint[:10]],
            "n_crosses_other_limb_too": len(extra_limb),
            "n_geometry_disagreements": len(self.disagreements()),
            "geometry_disagreements": [
                {
                    "name": e.name,
                    "limb_group": e.limb_group,
                    "crossed_limbs": e.crossed_limbs,
                    "note": e.note,
                }
                for e in self.disagreements()
            ],
        }

    def write_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [e.as_row() for e in self.entries]
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path


# ---------------------------------------------------------------- 源文件标签


def actuator_source_labels(muscle_dir: Path) -> Dict[str, Tuple[str, str, str]]:
    """解析官方 `Muscle/*.xml`，返回 {执行器名: (limb_group, side, 文件名)}。"""
    out: Dict[str, Tuple[str, str, str]] = {}
    muscle_dir = Path(muscle_dir)
    if not muscle_dir.is_dir():
        return out
    for xml_file in sorted(muscle_dir.glob("*.xml")):
        limb = _FILE_LABEL.get(xml_file.stem)
        if limb is None:
            continue
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue
        for elem in root.iter("general"):
            name = elem.get("name")
            if not name:
                continue
            m = SIDE_SUFFIX.search(name)
            side = m.group(1).upper() if m else "M"
            out[name] = (limb, side, xml_file.name)
    return out


# ---------------------------------------------------------------- 几何解析


def _tendon_bodies_sites(model, tendon_id: int) -> Tuple[List[int], List[int]]:
    """返回该 tendon 的 wrap 对象落到的 (body_ids, site_ids)。"""
    import mujoco

    adr = int(model.tendon_adr[tendon_id])
    num = int(model.tendon_num[tendon_id])
    bodies: List[int] = []
    sites: List[int] = []
    for w in range(adr, adr + num):
        wtype = int(model.wrap_type[w])
        objid = int(model.wrap_objid[w])
        if wtype == int(mujoco.mjtWrap.mjWRAP_SITE):
            sites.append(objid)
            bodies.append(int(model.site_bodyid[objid]))
        elif wtype in (
            int(mujoco.mjtWrap.mjWRAP_SPHERE),
            int(mujoco.mjtWrap.mjWRAP_CYLINDER),
        ):
            bodies.append(int(model.geom_bodyid[objid]))
    return bodies, sites


def _body_joints(model, body_id: int) -> List[int]:
    adr = int(model.body_jntadr[body_id])
    num = int(model.body_jntnum[body_id])
    return list(range(adr, adr + num))


def _path_to_root(model, body_id: int) -> List[int]:
    path = []
    b = body_id
    while b != 0:
        path.append(b)
        b = int(model.body_parentid[b])
    return path


def _lowest_common_ancestor(model, bodies: Sequence[int]) -> int:
    if not bodies:
        return 0
    paths = [set(_path_to_root(model, b)) for b in bodies]
    common = set.intersection(*paths) if paths else set()
    if not common:
        return 0
    ref = _path_to_root(model, bodies[0])
    for b in ref:
        if b in common:
            return b
    return 0


def crossed_joints_of(model, tendon_id: int) -> Tuple[List[int], List[int], List[int]]:
    """返回 (crossed_joint_ids, body_ids, site_ids)。

    跨越关节 = 最低公共祖先之下，各 wrap body 到 LCA 路径上的关节之并。
    """
    bodies, sites = _tendon_bodies_sites(model, tendon_id)
    if not bodies:
        return [], [], []
    bodies = sorted(set(bodies))
    lca = _lowest_common_ancestor(model, bodies)
    joints: Set[int] = set()
    for b in bodies:
        for x in _path_to_root(model, b):
            if x == lca:
                break
            joints.update(_body_joints(model, x))
    return sorted(joints), bodies, sites


# ---------------------------------------------------------------- 构建


def build_map(model, mshuman_root: Optional[Path] = None, strict: bool = False) -> MuscleMap:
    """由编译后的模型建立肌肉分组映射。

    Args:
        model: 已编译的 MjModel（MS-Human-700 主模型）。
        mshuman_root: 官方模型仓库根目录（含 `Muscle/`）；默认 `external/MS-Human-700`。
        strict: True 时若存在 geometry 与源文件不一致的条目则抛错。
    """
    import mujoco

    root = Path(mshuman_root) if mshuman_root is not None else paths.MSHUMAN_ROOT
    labels = actuator_source_labels(root / "Muscle")

    result = MuscleMap()
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act_{a}"
        tendon_name = ""
        bodies_names: List[str] = []
        sites_names: List[str] = []
        joint_names: List[str] = []

        if int(model.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_TENDON):
            t = int(model.actuator_trnid[a, 0])
            tendon_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TENDON, t) or f"tendon_{t}"
            jids, bids, sids = crossed_joints_of(model, t)
            bodies_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}" for b in bids
            ]
            sites_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or f"site_{s}" for s in sids
            ]
            joint_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}" for j in jids
            ]

        crossed_limbs: List[str] = []
        for jn in joint_names:
            lb = joint_limb(jn)
            if lb in LIMB_ORDER and lb not in crossed_limbs:
                crossed_limbs.append(lb)
        crossed_limbs = [x for x in LIMB_ORDER if x in crossed_limbs]

        lab = labels.get(name)
        if lab is None:
            limb_group, src_file = "unknown", None
        else:
            limb_group, _, src_file = lab

        ms = SIDE_SUFFIX.search(name)
        if ms:
            side, side_source = ms.group(1).upper(), "name"
        else:
            side_set = {body_side(b) for b in bodies_names} - {"M"}
            if len(side_set) == 1:
                side, side_source = side_set.pop(), "geom"
            elif not side_set:
                side, side_source = "M", "midline"
            else:
                side, side_source = "M", "geom-ambiguous"

        # 校验规则（避免过度严格造成假警报）：
        #  * 该简化模型脊柱只保留 L5_S1 / T12_L1 / T1_head_neck 等少数活动关节，
        #    腰椎体、胸椎体之间为刚性连接，因此多裂肌、腹斜肌等**确实不跨越可动关节**。
        #    这种情况记为「几何不可判定」，而不是不一致。
        #  * 权威分组 ∈ 跨越关节集合 → 一致。
        #  * 跨关节集合非空但不含权威分组 → 真正的不一致，需要人工复核。
        agrees = True
        notes: List[str] = []
        if limb_group == "unknown":
            agrees = False
            notes.append("无源文件标签")
        elif not crossed_limbs:
            notes.append("未跨越可动关节（该体段在此模型中为刚性连接）")
        elif limb_group not in crossed_limbs:
            agrees = False
            notes.append(f"跨关节({crossed_limbs}) 不含权威分组 {limb_group}")
        else:
            extra = [x for x in crossed_limbs if x != limb_group]
            if extra:
                notes.append(f"额外跨越 {extra}")

        result.entries.append(
            MuscleEntry(
                index=a,
                name=name,
                tendon=tendon_name,
                limb_group=limb_group,
                source_file=src_file,
                side=side,
                side_source=side_source,
                crossed_limbs=crossed_limbs,
                crossed_joints=joint_names,
                bodies=bodies_names,
                sites=sites_names,
                agrees_with_geometry=agrees,
                note="; ".join(notes),
            )
        )

    if strict:
        bad = [e.name for e in result.entries if not e.agrees_with_geometry]
        if bad:
            raise ValueError(f"肌群映射存在不一致条目 ({len(bad)}): {bad[:20]}")
    return result


def validate_group_consistency(m: MuscleMap) -> List[str]:
    """返回可疑条目说明（空列表 = 全部通过基础校验）。

    注意：**不**把「未跨越任何可动关节」当作问题。该简化模型的脊柱只保留
    ``L5_S1`` / ``T12_L1`` / ``T1_head_neck`` 三组活动关节，腰椎体与胸椎体之间为
    刚性连接，因此多裂肌（MF_*）、腹斜肌（IO*）等确实不跨越可动关节
    （共 192 块）。这类肌肉的缩放对关节力矩没有影响，属于模型的真实属性。
    该数量由 :meth:`MuscleMap.to_json` 的 ``n_no_movable_joint`` 单独报告。
    """
    problems: List[str] = []
    for e in m.entries:
        if e.limb_group == "unknown":
            problems.append(f"{e.name}: 无权威分组")
        if e.side == "M" and e.limb_group in ("lower", "upper"):
            problems.append(f"{e.name}: 肢体肌肉但侧别未确定")
        if not e.tendon:
            problems.append(f"{e.name}: 没有 tendon 传输（非肌肉驱动）")
    return problems


def report_no_movable_joint(m: MuscleMap) -> List[str]:
    """列出未跨越可动关节的肌肉（信息性，不是错误）。"""
    return [e.name for e in m.entries if not e.crossed_joints]


__all__ = [
    "MuscleEntry",
    "MuscleMap",
    "build_map",
    "body_limb",
    "body_side",
    "joint_limb",
    "actuator_source_labels",
    "validate_group_consistency",
    "report_no_movable_joint",
    "crossed_joints_of",
]
