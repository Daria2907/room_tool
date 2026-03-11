bl_info = {
    "name": "Room Tool",
    "version": (1, 4),
    "blender": (3, 0, 0),
    "location": "3D View ▸ N-Panel ▸ Room Tool  |  Shift+R",
    "description": "Click to draw rooms. Connected rooms share aligned door openings.",
    "category": "Mesh",
}

import bpy, bmesh, gpu, json, time
from bpy_extras import view3d_utils
from mathutils import Vector
from gpu_extras.batch import batch_for_shader

_OPPOSITE = {'S': 'N', 'N': 'S', 'E': 'W', 'W': 'E'}

# Tracks every active draw-handler token so ROOM_OT_clear_overlays can
# nuke them all even if the owning operator crashed without calling
# _remove_draw_handle.
_DRAW_HANDLES: set = set()


# ═════════════════════════════════════════════════════════════════════════════
# Property groups
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_PG_floor(bpy.types.PropertyGroup):
    """One saved floor level (name + Z offset)."""
    z_offset: bpy.props.FloatProperty(name="Z Offset", default=0.0)


class ROOM_PG_door_preset(bpy.types.PropertyGroup):
    """One saved door-size preset (name inherited from PropertyGroup)."""
    door_width : bpy.props.FloatProperty(
        name="Width",  default=0.9, min=0.1, max=10.0, unit="LENGTH")
    door_height: bpy.props.FloatProperty(
        name="Height", default=2.0, min=0.1, max=15.0, unit="LENGTH")


class ROOM_PG_window_preset(bpy.types.PropertyGroup):
    """One saved window-size preset (name inherited from PropertyGroup)."""
    window_width : bpy.props.FloatProperty(
        name="Width",       default=1.0, min=0.1, max=10.0, unit="LENGTH")
    window_height: bpy.props.FloatProperty(
        name="Height",      default=1.2, min=0.1, max=15.0, unit="LENGTH")
    v_offset     : bpy.props.FloatProperty(
        name="Sill Height", default=0.9, min=0.0, max=10.0, unit="LENGTH")


class ROOM_PG_registry_entry(bpy.types.PropertyGroup):
    """One room's registry data persisted in the scene for addon-reload survival."""
    x1: bpy.props.FloatProperty()
    y1: bpy.props.FloatProperty()
    x2: bpy.props.FloatProperty()
    y2: bpy.props.FloatProperty()
    t:  bpy.props.FloatProperty(default=0.125)
    z:  bpy.props.FloatProperty(default=0.0)
    # New: flat lists of door/window dicts serialised as JSON
    doors_json:   bpy.props.StringProperty(default="[]")
    windows_json: bpy.props.StringProperty(default="[]")
    # Legacy fields kept for backward-compat when loading old .blend files
    door_walls:   bpy.props.StringProperty()
    no_walls:     bpy.props.StringProperty()
    door_anchors: bpy.props.StringProperty()
    door_dims:    bpy.props.StringProperty(default="{}")
    obj_name:     bpy.props.StringProperty()
    door_width:   bpy.props.FloatProperty(default=0.9)
    door_height:  bpy.props.FloatProperty(default=2.0)
    plinth_bottom_enabled : bpy.props.BoolProperty(default=True)
    plinth_top_enabled    : bpy.props.BoolProperty(default=True)


def _cb_mat(self, context):
    """PointerProperty update: re-apply materials to all existing rooms (no geometry rebuild)."""
    _apply_materials_all_rooms(context.scene.room_settings)


def _cb_tiling(self, context):
    """FloatProperty update: recompute UV on all existing rooms (no geometry rebuild)."""
    _apply_uvs_all_rooms(context.scene.room_settings)


def _cb_rebuild(self, context):
    """BoolProperty/FloatProperty update: full geometry rebuild on all existing rooms."""
    s = context.scene.room_settings
    for reg in ROOM_OT_draw._room_list:
        _rebuild_room_mesh(reg, s)


class ROOM_PG_settings(bpy.types.PropertyGroup):
    z_foundation : bpy.props.FloatProperty(
        name="Foundation Z",
        description="Z coordinate of the room floor (foundation alignment)",
        default=0.0, step=10, precision=3)
    wall_height : bpy.props.FloatProperty(
        name="Wall Height", default=2.4, min=0.1, max=30.0, unit="LENGTH")
    wall_thickness : bpy.props.FloatProperty(
        name="Wall Thickness", default=0.125, min=0.005, max=2.0, unit="LENGTH")
    add_ceiling : bpy.props.BoolProperty(name="Add Ceiling", default=True)
    add_floor   : bpy.props.BoolProperty(name="Add Floor",   default=True)
    add_door    : bpy.props.BoolProperty(name="Add Door Frame", default=True)
    door_width  : bpy.props.FloatProperty(
        name="Door Width",  default=0.9, min=0.1, max=10.0, unit="LENGTH")
    door_height : bpy.props.FloatProperty(
        name="Door Height", default=2.0, min=0.1, max=15.0, unit="LENGTH")
    door_margin : bpy.props.FloatProperty(
        name="Door / Window Margin",
        description="Minimum distance between opening edge and wall corner / other openings",
        default=0.15, min=0.0, max=2.0, unit="LENGTH")
    window_width  : bpy.props.FloatProperty(
        name="Window Width",  default=1.0, min=0.1, max=10.0, unit="LENGTH")
    window_height : bpy.props.FloatProperty(
        name="Window Height", default=1.2, min=0.1, max=15.0, unit="LENGTH")
    window_v_offset: bpy.props.FloatProperty(
        name="Sill Height",   default=0.9, min=0.0, max=10.0, unit="LENGTH")
    window_array_count : bpy.props.IntProperty(
        name="Count",         default=1, min=1, max=20)
    window_array_gap   : bpy.props.FloatProperty(
        name="Gap Between",   default=0.2, min=0.0, max=5.0, unit="LENGTH")
    mat_walls        : bpy.props.PointerProperty(type=bpy.types.Material, name="Walls",           update=_cb_mat)
    mat_walls_tiling : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    mat_floor        : bpy.props.PointerProperty(type=bpy.types.Material, name="Floor",           update=_cb_mat)
    mat_floor_tiling : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    mat_ceiling      : bpy.props.PointerProperty(type=bpy.types.Material, name="Ceiling",         update=_cb_mat)
    mat_ceiling_tiling      : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    mat_door_frame          : bpy.props.PointerProperty(type=bpy.types.Material, name="Door Frames",   update=_cb_mat)
    mat_door_frame_tiling   : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    mat_window_frame        : bpy.props.PointerProperty(type=bpy.types.Material, name="Window Frames", update=_cb_mat)
    mat_window_frame_tiling : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    add_plinth_bottom : bpy.props.BoolProperty(
        name="Bottom Plinth",
        description="Add a skirting board / baseboard along the bottom of the walls",
        default=False, update=_cb_rebuild)
    add_plinth_top : bpy.props.BoolProperty(
        name="Top Plinth",
        description="Add a cornice / crown moulding along the top of the walls",
        default=False, update=_cb_rebuild)
    plinth_bottom_height : bpy.props.FloatProperty(
        name="Height", default=0.1, min=0.005, max=2.0, unit="LENGTH",
        description="Height of the bottom plinth strip", update=_cb_rebuild)
    plinth_bottom_thickness : bpy.props.FloatProperty(
        name="Thickness", default=0.02, min=0.002, max=0.5, unit="LENGTH",
        description="How far the bottom plinth protrudes from the wall surface", update=_cb_rebuild)
    mat_plinth_bottom        : bpy.props.PointerProperty(type=bpy.types.Material, name="Bot Plinth", update=_cb_mat)
    mat_plinth_bottom_tiling : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    plinth_top_height : bpy.props.FloatProperty(
        name="Height", default=0.1, min=0.005, max=2.0, unit="LENGTH",
        description="Height of the top plinth strip", update=_cb_rebuild)
    plinth_top_thickness : bpy.props.FloatProperty(
        name="Thickness", default=0.02, min=0.002, max=0.5, unit="LENGTH",
        description="How far the top plinth protrudes from the wall surface", update=_cb_rebuild)
    mat_plinth_top        : bpy.props.PointerProperty(type=bpy.types.Material, name="Top Plinth", update=_cb_mat)
    mat_plinth_top_tiling : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    pivot_mode : bpy.props.EnumProperty(
        name="Room Pivot",
        description="Where to place the origin of each room object",
        items=[
            ('WORLD_ORIGIN', "World Origin", "Origin at (0, 0, 0)"),
            ('FLOOR_CENTER', "Floor Center", "Origin at center of room floor"),
            ('GEOMETRY_CENTER', "Geometry Center", "Origin at bounding-box center of the mesh"),
        ],
        default='WORLD_ORIGIN')


# ═════════════════════════════════════════════════════════════════════════════
# Collection helpers
# ═════════════════════════════════════════════════════════════════════════════
def _get_or_create_col(name, parent):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        parent.children.link(col)
    elif col not in list(parent.children):
        try:
            parent.children.link(col)
        except Exception:
            pass
    return col


def _room_target_collection(context, room_num):
    floors = context.scene.room_floors
    active = context.scene.room_active_floor
    if 0 <= active < len(floors):
        fl        = floors[active]
        floor_col = _get_or_create_col(fl.name, context.scene.collection)
        room_col  = _get_or_create_col(f"Room.{room_num:03}", floor_col)
        return room_col
    return context.collection


# ═════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═════════════════════════════════════════════════════════════════════════════
_CAT_WALL       = 0
_CAT_FLOOR      = 1
_CAT_CEILING    = 2
_CAT_DOOR_FRAME = 3
_CAT_WIN_FRAME  = 4
_CAT_PLINTH_BOTTOM = 5
_CAT_PLINTH_TOP    = 6
_CAT_NAMES = ('walls', 'floor', 'ceiling', 'door_frames', 'window_frames', 'plinth_bottom', 'plinth_top')

def _face4(bm, v0, v1, v2, v3):
    bm.faces.new([bm.verts.new(co) for co in (v0, v1, v2, v3)])


def _fill_room(bm, x1, y1, x2, y2, z, s, doors=(), no_walls=(), windows=(),
               add_plinth_bottom=None, add_plinth_top=None):
    """
    Populate *bm* with room geometry as an interior shell.
    doors:    list of {"wc": str, "anchor": float, "w": float, "h": float}
              Multiple doors per wall are supported.
    windows:  list of {"wc": str, "anchor": float, "v_offset": float, "w": float, "h": float}
    no_walls: wall chars to skip entirely.
    add_plinth_bottom/top: if not None, overrides the global s.add_plinth_* flag for this room.
    """
    t  = s.wall_thickness
    h  = s.wall_height
    zt = z + h
    fd = t * 0.5    # tunnel reveal depth = half wall thickness

    # Pre-compute strip split positions used for ceiling/floor and opposite-wall seams.
    # Include BOTH door and window edge positions so ceiling/floor quads always share
    # vertices with wall faces that were split by _face_with_holes, eliminating
    # T-junctions (the "vertices that go nowhere" on wall top/bottom edges).
    _ey = []   # Y splits for E/W openings (doors + windows)
    _ex = []   # X splits for N/S openings (doors + windows)
    for d in doors:
            anchor = d["anchor"]
            dw_i   = d["w"]
            if d["wc"] in ('E', 'W'):
                _ey += [max(y1, anchor - dw_i * 0.5), min(y2, anchor + dw_i * 0.5)]
            else:
                _ex += [max(x1, anchor - dw_i * 0.5), min(x2, anchor + dw_i * 0.5)]
    for w in windows:
            anchor = w["anchor"]
            ww_i   = w["w"]
            if w["wc"] in ('E', 'W'):
                _ey += [max(y1, anchor - ww_i * 0.5), min(y2, anchor + ww_i * 0.5)]
            else:
                _ex += [max(x1, anchor - ww_i * 0.5), min(x2, anchor + ww_i * 0.5)]
    _ey = [v for v in sorted(set(_ey)) if y1 < v < y2]
    _ex = [v for v in sorted(set(_ex)) if x1 < v < x2]

    _dh_fallback = min(s.door_height, h - t)

    # ── inner helpers ────────────────────────────────────────────────────────
    def _wall_door_segs(wc, span_lo, span_hi, max_span):
        """Sorted list of (dl, dr, dh) for all doors on wall wc."""
        result = []
        for d in doors:
            if d["wc"] != wc:
                continue
            dh_i = min(d["h"], h - t)
            dw_i = min(d["w"], max_span * 0.85)
            cx   = d["anchor"]
            dl_i = max(span_lo, cx - dw_i * 0.5)
            dr_i = min(span_hi, cx + dw_i * 0.5)
            if dr_i > dl_i:
                result.append((dl_i, dr_i, dh_i))
        result.sort(key=lambda seg: seg[0])
        return result

    def _solid_strip_levels(segs):
        """
        Yield (xa, xb_or_ya_yb, levels) for each solid strip between/around
        door segments.  levels is the list of Z coordinates that subdivide
        the strip so door above-faces share vertices cleanly.
        Works for both N/S (X-axis span) and E/W (Y-axis span) walls.
        """
        positions = [segs[0][0] - 1]   # placeholder; overwritten below
        dhs       = []
        # Build flat list: [span_lo, dl0, dr0, dl1, dr1, ..., span_hi]
        # and parallel dh list:      [None, dh0, dh0, dh1, dh1, ..., None]
        return _gen_strips(segs)

    def _gen_strips(segs):
        # segs = [(lo, hi, dh), ...] sorted by lo
        # Interleave: span_lo, lo0, hi0, lo1, hi1, ..., span_hi — we yield the gaps.
        for i in range(len(segs) + 1):
            xa = segs[i-1][1] if i > 0 else None   # right edge of previous door (or None)
            xb = segs[i][0]   if i < len(segs) else None   # left edge of next door (or None)
            ldh = segs[i-1][2] if i > 0            else None
            rdh = segs[i][2]   if i < len(segs)    else None
            # xa / xb are the extents of the strip RELATIVE to the segment list;
            # we return them along with the levels so the caller can use them.
            heights = sorted({dh for dh in (ldh, rdh) if dh is not None})
            levels  = [z] + [z + dh for dh in heights] + [zt]
            yield xa, xb, levels

    def _jamb_extra_levels(segs, j):
        """
        Return (lev_l, lev_r) — lists of extra Z heights to split door j's
        left and right jambs respectively, to avoid T-junctions where an
        adjacent between-door solid strip has an intermediate level that the
        jamb face doesn't share.
        left jamb: if left neighbour dh < this door's dh, strip has split at left_dh
        right jamb: if right neighbour dh < this door's dh, strip has split at right_dh
        """
        dh_j  = segs[j][2]
        lev_l = ([z + segs[j-1][2]] if j > 0 and segs[j-1][2] < dh_j else [])
        lev_r = ([z + segs[j+1][2]] if j < len(segs)-1 and segs[j+1][2] < dh_j else [])
        return lev_l, lev_r

    def _wall_win_segs(wc, span_lo, span_hi):
        """Sorted list of (wl, wr, wz_lo, wz_hi) for all windows on wall wc."""
        result = []
        for w in windows:
            if w["wc"] != wc:
                continue
            ww = min(w["w"], (span_hi - span_lo) * 0.85)
            cx = w["anchor"]
            wl = max(span_lo, cx - ww * 0.5)
            wr = min(span_hi, cx + ww * 0.5)
            if wr <= wl:
                continue
            wz_lo = z + max(0.0, w.get("v_offset", 0.9))
            wz_hi = min(zt, wz_lo + w["h"])
            if wz_hi > wz_lo:
                result.append((wl, wr, wz_lo, wz_hi))
        result.sort(key=lambda seg: seg[0])
        return result

    def _face_with_holes(emit, h_lo, h_hi, z_lo, z_hi, holes):
        """Emit quads for [h_lo,h_hi]×[z_lo,z_hi] with rectangular holes cut out.
        emit(ha,hb,za,zb): per-wall quad emitter (handles coordinate mapping).
        holes: list of (hl, hr, hz_lo, hz_hi).
        """
        clipped = []
        for hl, hr, hz_lo, hz_hi in holes:
            cl, cr = max(hl, h_lo), min(hr, h_hi)
            if cr > cl + 1e-6:
                clipped.append((cl, cr, hz_lo, hz_hi))
        if not clipped:
            if h_hi - h_lo > 1e-6 and z_hi - z_lo > 1e-6:
                emit(h_lo, h_hi, z_lo, z_hi)
            return
        h_pts = sorted(set([h_lo] + [hl for hl,_,_,_ in clipped]
                                   + [hr for _,hr,_,_ in clipped] + [h_hi]))
        for i in range(len(h_pts) - 1):
            ha, hb = h_pts[i], h_pts[i + 1]
            if hb - ha < 1e-6:
                continue
            col = [(hz_lo, hz_hi) for hl, hr, hz_lo, hz_hi in clipped
                   if hl <= ha + 1e-6 and hr >= hb - 1e-6]
            if not col:
                emit(ha, hb, z_lo, z_hi)
            else:
                z_pts = sorted(set([z_lo] + [hz for h in col for hz in h] + [z_hi]))
                for j in range(len(z_pts) - 1):
                    za, zb = z_pts[j], z_pts[j + 1]
                    if zb - za < 1e-6:
                        continue
                    if not any(hz_lo <= za + 1e-6 and hz_hi >= zb - 1e-6 for hz_lo, hz_hi in col):
                        emit(ha, hb, za, zb)

    # ── Face category layer + tagged emitter ─────────────────────────────────
    # UV is applied after pivot shift via _apply_cube_uv_to_mesh (uses local coords)
    cat_layer   = bm.faces.layers.int.new("room_cat")
    _cur_cat    = [_CAT_WALL]
    _cur_tiling = [s.mat_walls_tiling]   # kept for compatibility; UV handled post-pivot
    def _f4(v0, v1, v2, v3):
        face = bm.faces.new([bm.verts.new(co) for co in (v0, v1, v2, v3)])
        face[cat_layer] = _cur_cat[0]

    # ── Per-wall emitters and window hole lists ───────────────────────────────
    def _emit_s(ha, hb, za, zb):
        _f4((hb,y1,za),(ha,y1,za),(ha,y1,zb),(hb,y1,zb))
    _s_holes = _wall_win_segs('S', x1, x2)

    def _emit_n(ha, hb, za, zb):
        _f4((ha,y2,za),(hb,y2,za),(hb,y2,zb),(ha,y2,zb))
    _n_holes = _wall_win_segs('N', x1, x2)

    def _emit_w(ha, hb, za, zb):
        _f4((x1,ha,za),(x1,hb,za),(x1,hb,zb),(x1,ha,zb))
    _w_holes = _wall_win_segs('W', y1, y2)

    def _emit_e(ha, hb, za, zb):
        _f4((x2,hb,za),(x2,ha,za),(x2,ha,zb),(x2,hb,zb))
    _e_holes = _wall_win_segs('E', y1, y2)

    # ── South inner panel (at y=y1, reveals go –Y) ───────────────────────────
    if 'S' not in no_walls:
        segs_s = _wall_door_segs('S', x1, x2, x2 - x1)
        if segs_s:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            strip_iter = list(_gen_strips(segs_s))
            lo_x = x1
            for idx, (_, xb, levels) in enumerate(strip_iter):
                hi_x = segs_s[idx][0] if idx < len(segs_s) else x2
                xa_s, xb_s = lo_x, hi_x
                if xa_s < xb_s - 1e-6:
                    for k in range(len(levels) - 1):
                        _face_with_holes(_emit_s, xa_s, xb_s, levels[k], levels[k+1], _s_holes)
                if idx < len(segs_s):
                    lo_x = segs_s[idx][1]
            _cur_cat[0] = _CAT_DOOR_FRAME
            _cur_tiling[0] = s.mat_door_frame_tiling
            for j, (dl_i, dr_i, dh_i) in enumerate(segs_s):
                lev_l, lev_r = _jamb_extra_levels(segs_s, j)
                _f4((dr_i,y1,z+dh_i),(dl_i,y1,z+dh_i),(dl_i,y1,zt),(dr_i,y1,zt))
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((dl_i,y1,ll[k]),(dl_i,y1-fd,ll[k]),(dl_i,y1-fd,ll[k+1]),(dl_i,y1,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((dr_i,y1-fd,rl[k]),(dr_i,y1,rl[k]),(dr_i,y1,rl[k+1]),(dr_i,y1-fd,rl[k+1]))
                _f4((dr_i,y1,z+dh_i),(dl_i,y1,z+dh_i),(dl_i,y1-fd,z+dh_i),(dr_i,y1-fd,z+dh_i))
                _f4((dl_i,y1-fd,z),(dr_i,y1-fd,z),(dr_i,y1,z),(dl_i,y1,z))
        else:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            if _ex:
                xs = [x1] + _ex + [x2]
                for i in range(len(xs) - 1):
                    _face_with_holes(_emit_s, xs[i], xs[i+1], z,              z+_dh_fallback, _s_holes)
                    _face_with_holes(_emit_s, xs[i], xs[i+1], z+_dh_fallback, zt,             _s_holes)
            else:
                _face_with_holes(_emit_s, x1, x2, z,              z+_dh_fallback, _s_holes)
                _face_with_holes(_emit_s, x1, x2, z+_dh_fallback, zt,             _s_holes)
    _cur_cat[0] = _CAT_WIN_FRAME
    _cur_tiling[0] = s.mat_window_frame_tiling
    for (wl, wr, wz_lo, wz_hi) in _wall_win_segs('S', x1, x2):
        _f4((wl,y1,wz_lo),(wl,y1,wz_hi),(wl,y1-fd,wz_hi),(wl,y1-fd,wz_lo))  # left jamb
        _f4((wr,y1-fd,wz_lo),(wr,y1-fd,wz_hi),(wr,y1,wz_hi),(wr,y1,wz_lo))  # right jamb
        _f4((wl,y1,wz_lo),(wl,y1-fd,wz_lo),(wr,y1-fd,wz_lo),(wr,y1,wz_lo))  # sill
        _f4((wl,y1,wz_hi),(wr,y1,wz_hi),(wr,y1-fd,wz_hi),(wl,y1-fd,wz_hi))  # lintel

    # ── North inner panel (at y=y2, reveals go +Y) ───────────────────────────
    if 'N' not in no_walls:
        segs_n = _wall_door_segs('N', x1, x2, x2 - x1)
        if segs_n:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            lo_x = x1
            for idx, (_, xb, levels) in enumerate(list(_gen_strips(segs_n))):
                hi_x = segs_n[idx][0] if idx < len(segs_n) else x2
                xa_s, xb_s = lo_x, hi_x
                if xa_s < xb_s - 1e-6:
                    for k in range(len(levels) - 1):
                        _face_with_holes(_emit_n, xa_s, xb_s, levels[k], levels[k+1], _n_holes)
                if idx < len(segs_n):
                    lo_x = segs_n[idx][1]
            _cur_cat[0] = _CAT_DOOR_FRAME
            _cur_tiling[0] = s.mat_door_frame_tiling
            for j, (dl_i, dr_i, dh_i) in enumerate(segs_n):
                lev_l, lev_r = _jamb_extra_levels(segs_n, j)
                _f4((dl_i,y2,z+dh_i),(dr_i,y2,z+dh_i),(dr_i,y2,zt),(dl_i,y2,zt))
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((dl_i,y2+fd,ll[k]),(dl_i,y2,ll[k]),(dl_i,y2,ll[k+1]),(dl_i,y2+fd,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((dr_i,y2,rl[k]),(dr_i,y2+fd,rl[k]),(dr_i,y2+fd,rl[k+1]),(dr_i,y2,rl[k+1]))
                _f4((dl_i,y2,z+dh_i),(dr_i,y2,z+dh_i),(dr_i,y2+fd,z+dh_i),(dl_i,y2+fd,z+dh_i))
                _f4((dl_i,y2,z),(dr_i,y2,z),(dr_i,y2+fd,z),(dl_i,y2+fd,z))
        else:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            if _ex:
                xs = [x1] + _ex + [x2]
                for i in range(len(xs) - 1):
                    _face_with_holes(_emit_n, xs[i], xs[i+1], z,              z+_dh_fallback, _n_holes)
                    _face_with_holes(_emit_n, xs[i], xs[i+1], z+_dh_fallback, zt,             _n_holes)
            else:
                _face_with_holes(_emit_n, x1, x2, z,              z+_dh_fallback, _n_holes)
                _face_with_holes(_emit_n, x1, x2, z+_dh_fallback, zt,             _n_holes)
    _cur_cat[0] = _CAT_WIN_FRAME
    _cur_tiling[0] = s.mat_window_frame_tiling
    for (wl, wr, wz_lo, wz_hi) in _wall_win_segs('N', x1, x2):
        _f4((wl,y2+fd,wz_lo),(wl,y2+fd,wz_hi),(wl,y2,wz_hi),(wl,y2,wz_lo))  # left jamb
        _f4((wr,y2,wz_lo),(wr,y2,wz_hi),(wr,y2+fd,wz_hi),(wr,y2+fd,wz_lo))  # right jamb
        _f4((wl,y2,wz_lo),(wr,y2,wz_lo),(wr,y2+fd,wz_lo),(wl,y2+fd,wz_lo))  # sill
        _f4((wl,y2+fd,wz_hi),(wr,y2+fd,wz_hi),(wr,y2,wz_hi),(wl,y2,wz_hi))  # lintel

    # ── West inner panel (at x=x1, reveals go –X) ────────────────────────────
    if 'W' not in no_walls:
        segs_w = _wall_door_segs('W', y1, y2, y2 - y1)
        if segs_w:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            lo_y = y1
            for idx, (_, yb, levels) in enumerate(list(_gen_strips(segs_w))):
                hi_y = segs_w[idx][0] if idx < len(segs_w) else y2
                ya_s, yb_s = lo_y, hi_y
                if ya_s < yb_s - 1e-6:
                    for k in range(len(levels) - 1):
                        _face_with_holes(_emit_w, ya_s, yb_s, levels[k], levels[k+1], _w_holes)
                if idx < len(segs_w):
                    lo_y = segs_w[idx][1]
            _cur_cat[0] = _CAT_DOOR_FRAME
            _cur_tiling[0] = s.mat_door_frame_tiling
            for j, (db_i, df_i, dh_i) in enumerate(segs_w):
                lev_l, lev_r = _jamb_extra_levels(segs_w, j)
                _f4((x1,db_i,z+dh_i),(x1,df_i,z+dh_i),(x1,df_i,zt),(x1,db_i,zt))
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((x1,db_i,ll[k]),(x1-fd,db_i,ll[k]),(x1-fd,db_i,ll[k+1]),(x1,db_i,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((x1-fd,df_i,rl[k]),(x1,df_i,rl[k]),(x1,df_i,rl[k+1]),(x1-fd,df_i,rl[k+1]))
                _f4((x1,db_i,z+dh_i),(x1-fd,db_i,z+dh_i),(x1-fd,df_i,z+dh_i),(x1,df_i,z+dh_i))
                _f4((x1-fd,db_i,z),(x1,db_i,z),(x1,df_i,z),(x1-fd,df_i,z))
        else:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            if _ey:
                ys = [y1] + _ey + [y2]
                for i in range(len(ys) - 1):
                    _face_with_holes(_emit_w, ys[i], ys[i+1], z,              z+_dh_fallback, _w_holes)
                    _face_with_holes(_emit_w, ys[i], ys[i+1], z+_dh_fallback, zt,             _w_holes)
            else:
                _face_with_holes(_emit_w, y1, y2, z,              z+_dh_fallback, _w_holes)
                _face_with_holes(_emit_w, y1, y2, z+_dh_fallback, zt,             _w_holes)
    _cur_cat[0] = _CAT_WIN_FRAME
    _cur_tiling[0] = s.mat_window_frame_tiling
    for (wb, wf, wz_lo, wz_hi) in _wall_win_segs('W', y1, y2):
        _f4((x1,wb,wz_lo),(x1-fd,wb,wz_lo),(x1-fd,wb,wz_hi),(x1,wb,wz_hi))  # back jamb
        _f4((x1,wf,wz_hi),(x1-fd,wf,wz_hi),(x1-fd,wf,wz_lo),(x1,wf,wz_lo))  # front jamb
        _f4((x1,wb,wz_lo),(x1,wf,wz_lo),(x1-fd,wf,wz_lo),(x1-fd,wb,wz_lo))  # sill
        _f4((x1-fd,wb,wz_hi),(x1-fd,wf,wz_hi),(x1,wf,wz_hi),(x1,wb,wz_hi))  # lintel

    # ── East inner panel (at x=x2, reveals go +X) ────────────────────────────
    if 'E' not in no_walls:
        segs_e = _wall_door_segs('E', y1, y2, y2 - y1)
        if segs_e:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            lo_y = y1
            for idx, (_, yb, levels) in enumerate(list(_gen_strips(segs_e))):
                hi_y = segs_e[idx][0] if idx < len(segs_e) else y2
                ya_s, yb_s = lo_y, hi_y
                if ya_s < yb_s - 1e-6:
                    for k in range(len(levels) - 1):
                        _face_with_holes(_emit_e, ya_s, yb_s, levels[k], levels[k+1], _e_holes)
                if idx < len(segs_e):
                    lo_y = segs_e[idx][1]
            _cur_cat[0] = _CAT_DOOR_FRAME
            _cur_tiling[0] = s.mat_door_frame_tiling
            for j, (db_i, df_i, dh_i) in enumerate(segs_e):
                lev_l, lev_r = _jamb_extra_levels(segs_e, j)
                _f4((x2,df_i,z+dh_i),(x2,db_i,z+dh_i),(x2,db_i,zt),(x2,df_i,zt))
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((x2+fd,db_i,ll[k]),(x2,db_i,ll[k]),(x2,db_i,ll[k+1]),(x2+fd,db_i,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((x2,df_i,rl[k]),(x2+fd,df_i,rl[k]),(x2+fd,df_i,rl[k+1]),(x2,df_i,rl[k+1]))
                _f4((x2+fd,db_i,z+dh_i),(x2,db_i,z+dh_i),(x2,df_i,z+dh_i),(x2+fd,df_i,z+dh_i))
                _f4((x2,db_i,z),(x2+fd,db_i,z),(x2+fd,df_i,z),(x2,df_i,z))
        else:
            _cur_cat[0] = _CAT_WALL
            _cur_tiling[0] = s.mat_walls_tiling
            if _ey:
                ys = [y1] + _ey + [y2]
                for i in range(len(ys) - 1):
                    _face_with_holes(_emit_e, ys[i], ys[i+1], z,              z+_dh_fallback, _e_holes)
                    _face_with_holes(_emit_e, ys[i], ys[i+1], z+_dh_fallback, zt,             _e_holes)
            else:
                _face_with_holes(_emit_e, y1, y2, z,              z+_dh_fallback, _e_holes)
                _face_with_holes(_emit_e, y1, y2, z+_dh_fallback, zt,             _e_holes)
    _cur_cat[0] = _CAT_WIN_FRAME
    _cur_tiling[0] = s.mat_window_frame_tiling
    for (wb, wf, wz_lo, wz_hi) in _wall_win_segs('E', y1, y2):
        _f4((x2+fd,wb,wz_lo),(x2,wb,wz_lo),(x2,wb,wz_hi),(x2+fd,wb,wz_hi))  # back jamb
        _f4((x2,wf,wz_lo),(x2+fd,wf,wz_lo),(x2+fd,wf,wz_hi),(x2,wf,wz_hi))  # front jamb
        _f4((x2+fd,wb,wz_lo),(x2,wb,wz_lo),(x2,wf,wz_lo),(x2+fd,wf,wz_lo))  # sill
        _f4((x2,wb,wz_hi),(x2+fd,wb,wz_hi),(x2+fd,wf,wz_hi),(x2,wf,wz_hi))  # lintel

    # ── Ceiling & Floor ────────────────────────────────────────────────────────
    if s.add_ceiling:
        _cur_cat[0] = _CAT_CEILING
        _cur_tiling[0] = s.mat_ceiling_tiling
        if _ex or _ey:
            xs = ([x1] + _ex + [x2]) if _ex else [x1, x2]
            ys = ([y1] + _ey + [y2]) if _ey else [y1, y2]
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    _f4((xs[i],ys[j],zt),(xs[i],ys[j+1],zt),
                               (xs[i+1],ys[j+1],zt),(xs[i+1],ys[j],zt))
        else:
            _f4((x1,y1,zt),(x1,y2,zt),(x2,y2,zt),(x2,y1,zt))

    if s.add_floor:
        _cur_cat[0] = _CAT_FLOOR
        _cur_tiling[0] = s.mat_floor_tiling
        if _ex or _ey:
            xs = ([x1] + _ex + [x2]) if _ex else [x1, x2]
            ys = ([y1] + _ey + [y2]) if _ey else [y1, y2]
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    _f4((xs[i],ys[j],z),(xs[i+1],ys[j],z),
                               (xs[i+1],ys[j+1],z),(xs[i],ys[j+1],z))
        else:
            _f4((x1,y1,z),(x2,y1,z),(x2,y2,z),(x1,y2,z))

    # ── Plinth / Skirting / Cornice ──────────────────────────────────────────
    _add_pb = add_plinth_bottom if add_plinth_bottom is not None else getattr(s, 'add_plinth_bottom', False)
    _add_pt = add_plinth_top    if add_plinth_top    is not None else getattr(s, 'add_plinth_top',    False)
    if _add_pb or _add_pt:
        _pt_b = max(getattr(s, 'plinth_bottom_thickness', 0.02), 1e-4)
        _ph_b = min(max(getattr(s, 'plinth_bottom_height', 0.10), 1e-4), h * 0.45)
        _pt_t = max(getattr(s, 'plinth_top_thickness', 0.02), 1e-4)
        _ph_t = min(max(getattr(s, 'plinth_top_height',    0.10), 1e-4), h * 0.45)

        # Per-side corner clips: N/S bottom/top plinths are clipped only by their
        # own thickness at corners where an E/W plinth of the same kind exists.
        _pcw_b = _pt_b if ('W' not in no_walls and _add_pb) else 0.0
        _pce_b = _pt_b if ('E' not in no_walls and _add_pb) else 0.0
        _pcw_t = _pt_t if ('W' not in no_walls and _add_pt) else 0.0
        _pce_t = _pt_t if ('E' not in no_walls and _add_pt) else 0.0

        def _plinth_gaps(wc, span_lo, span_hi):
            """Solid spans of [span_lo, span_hi] with door footprints removed."""
            if span_lo >= span_hi - 1e-6:
                return []
            dsegs = _wall_door_segs(wc, span_lo, span_hi, span_hi - span_lo)
            if not dsegs:
                return [(span_lo, span_hi)]
            result = []
            lo = span_lo
            for dl, dr, _ in dsegs:
                if dl > lo + 1e-6:
                    result.append((lo, dl))
                lo = dr
            if lo < span_hi - 1e-6:
                result.append((lo, span_hi))
            return result

        # South wall — inner face at y=y1, plinth protrudes toward +Y
        if 'S' not in no_walls:
            if _add_pb:
                _cur_cat[0] = _CAT_PLINTH_BOTTOM
                for _xa, _xb in _plinth_gaps('S', x1 + _pcw_b, x2 - _pce_b):
                    _f4((_xb, y1+_pt_b, z),       (_xa, y1+_pt_b, z),       (_xa, y1+_pt_b, z+_ph_b),    (_xb, y1+_pt_b, z+_ph_b))   # front  (+Y)
                    _f4((_xa, y1,       z+_ph_b),  (_xb, y1,       z+_ph_b), (_xb, y1+_pt_b, z+_ph_b),   (_xa, y1+_pt_b, z+_ph_b))   # top    (+Z)
            if _add_pt:
                _cur_cat[0] = _CAT_PLINTH_TOP
                for _xa, _xb in [(x1 + _pcw_t, x2 - _pce_t)]:
                    _f4((_xb, y1+_pt_t, zt-_ph_t), (_xa, y1+_pt_t, zt-_ph_t), (_xa, y1+_pt_t, zt),       (_xb, y1+_pt_t, zt))        # front  (+Y)
                    _f4((_xa, y1,       zt-_ph_t),  (_xa, y1+_pt_t, zt-_ph_t), (_xb, y1+_pt_t, zt-_ph_t), (_xb, y1,       zt-_ph_t))  # bottom (-Z)

        # North wall — inner face at y=y2, plinth protrudes toward -Y
        if 'N' not in no_walls:
            if _add_pb:
                _cur_cat[0] = _CAT_PLINTH_BOTTOM
                for _xa, _xb in _plinth_gaps('N', x1 + _pcw_b, x2 - _pce_b):
                    _f4((_xa, y2-_pt_b, z),       (_xb, y2-_pt_b, z),       (_xb, y2-_pt_b, z+_ph_b),   (_xa, y2-_pt_b, z+_ph_b))   # front  (-Y)
                    _f4((_xa, y2-_pt_b, z+_ph_b), (_xb, y2-_pt_b, z+_ph_b), (_xb, y2,       z+_ph_b),   (_xa, y2,       z+_ph_b))   # top    (+Z)
            if _add_pt:
                _cur_cat[0] = _CAT_PLINTH_TOP
                for _xa, _xb in [(x1 + _pcw_t, x2 - _pce_t)]:
                    _f4((_xa, y2-_pt_t, zt-_ph_t), (_xb, y2-_pt_t, zt-_ph_t), (_xb, y2-_pt_t, zt),      (_xa, y2-_pt_t, zt))        # front  (-Y)
                    _f4((_xa, y2,       zt-_ph_t),  (_xb, y2,       zt-_ph_t), (_xb, y2-_pt_t, zt-_ph_t), (_xa, y2-_pt_t, zt-_ph_t)) # bottom (-Z)

        # East wall — inner face at x=x2, plinth protrudes toward -X
        if 'E' not in no_walls:
            if _add_pb:
                _cur_cat[0] = _CAT_PLINTH_BOTTOM
                for _ya, _yb in _plinth_gaps('E', y1, y2):
                    _f4((x2-_pt_b, _yb, z),       (x2-_pt_b, _ya, z),       (x2-_pt_b, _ya, z+_ph_b),  (x2-_pt_b, _yb, z+_ph_b))   # front  (-X)
                    _f4((x2-_pt_b, _ya, z+_ph_b), (x2,       _ya, z+_ph_b), (x2,       _yb, z+_ph_b),  (x2-_pt_b, _yb, z+_ph_b))   # top    (+Z)
            if _add_pt:
                _cur_cat[0] = _CAT_PLINTH_TOP
                for _ya, _yb in [(y1, y2)]:
                    _f4((x2-_pt_t, _yb, zt-_ph_t), (x2-_pt_t, _ya, zt-_ph_t), (x2-_pt_t, _ya, zt),     (x2-_pt_t, _yb, zt))        # front  (-X)
                    _f4((x2-_pt_t, _yb, zt-_ph_t), (x2,       _yb, zt-_ph_t), (x2,       _ya, zt-_ph_t), (x2-_pt_t, _ya, zt-_ph_t)) # bottom (-Z)

        # West wall — inner face at x=x1, plinth protrudes toward +X
        if 'W' not in no_walls:
            if _add_pb:
                _cur_cat[0] = _CAT_PLINTH_BOTTOM
                for _ya, _yb in _plinth_gaps('W', y1, y2):
                    _f4((x1+_pt_b, _ya, z),       (x1+_pt_b, _yb, z),       (x1+_pt_b, _yb, z+_ph_b),  (x1+_pt_b, _ya, z+_ph_b))   # front  (+X)
                    _f4((x1,       _ya, z+_ph_b), (x1+_pt_b, _ya, z+_ph_b), (x1+_pt_b, _yb, z+_ph_b),  (x1,       _yb, z+_ph_b))   # top    (+Z)
            if _add_pt:
                _cur_cat[0] = _CAT_PLINTH_TOP
                for _ya, _yb in [(y1, y2)]:
                    _f4((x1+_pt_t, _ya, zt-_ph_t), (x1+_pt_t, _yb, zt-_ph_t), (x1+_pt_t, _yb, zt),     (x1+_pt_t, _ya, zt))        # front  (+X)
                    _f4((x1,       _yb, zt-_ph_t), (x1+_pt_t, _yb, zt-_ph_t), (x1+_pt_t, _ya, zt-_ph_t), (x1,       _ya, zt-_ph_t)) # bottom (-Z)

    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)

    # Build category → vertex index sets for vertex groups
    cat_vert_sets = {i: set() for i in range(len(_CAT_NAMES))}
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()
    for face in bm.faces:
        cat = face[cat_layer]
        for vert in face.verts:
            cat_vert_sets[cat].add(vert.index)
    return cat_vert_sets


def _apply_cube_uv_to_mesh(me, s):
    """Apply cube-projection UV using local vertex coordinates (post-pivot).
    Matches bpy.ops.uv.cube_project(cube_size=1/tiling) per material category."""
    uv_layer = me.uv_layers.new(name="UVMap")
    cat_attr = me.attributes.get("room_cat") if hasattr(me, 'attributes') else None
    tiling_map = [
        s.mat_walls_tiling,
        s.mat_floor_tiling,
        s.mat_ceiling_tiling,
        s.mat_door_frame_tiling,
        s.mat_window_frame_tiling,
        getattr(s, 'mat_plinth_bottom_tiling', 1.0),
        getattr(s, 'mat_plinth_top_tiling',    1.0),
    ]
    verts  = me.vertices
    loops  = me.loops
    for poly in me.polygons:
        cat = cat_attr.data[poly.index].value if cat_attr else 0
        sc  = tiling_map[cat] if 0 <= cat < len(tiling_map) else 1.0
        n   = poly.normal
        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
        for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
            co = verts[loops[li].vertex_index].co
            if az >= ax and az >= ay:
                uv_layer.data[li].uv = (co.x * sc, co.y * sc)
            elif ax >= ay:
                uv_layer.data[li].uv = (co.y * sc, co.z * sc)
            else:
                uv_layer.data[li].uv = (co.x * sc, co.z * sc)


def _apply_materials_all_rooms(s):
    """Update material slots + face material_index on every existing room mesh.
    Uses me.attributes['room_cat'] — no geometry rebuild required."""
    for reg in ROOM_OT_draw._room_list:
        obj_name = reg.get("obj_name", "")
        if obj_name not in bpy.data.objects:
            continue
        me = bpy.data.objects[obj_name].data
        while me.materials:
            me.materials.pop(index=0)
        mat_w  = getattr(s, 'mat_walls',        None)
        mat_f  = getattr(s, 'mat_floor',        None)
        mat_c  = getattr(s, 'mat_ceiling',      None)
        mat_d  = getattr(s, 'mat_door_frame',   None) or mat_w
        mat_wf = getattr(s, 'mat_window_frame', None) or mat_w
        mat_pb = getattr(s, 'mat_plinth_bottom', None) or mat_w
        mat_pt = getattr(s, 'mat_plinth_top',    None) or mat_w
        mat_list, slot_map = [], {}
        for cat, mat in enumerate((mat_w, mat_f, mat_c, mat_d, mat_wf, mat_pb, mat_pt)):
            if mat is None:
                slot_map[cat] = None
            elif mat in mat_list:
                slot_map[cat] = mat_list.index(mat)
            else:
                slot_map[cat] = len(mat_list)
                mat_list.append(mat)
        for mat in mat_list:
            me.materials.append(mat)
        cat_attr = me.attributes.get("room_cat") if hasattr(me, 'attributes') else None
        if cat_attr and any(v is not None for v in slot_map.values()):
            for poly in me.polygons:
                si = slot_map.get(cat_attr.data[poly.index].value)
                if si is not None:
                    poly.material_index = si
        me.update()


def _apply_uvs_all_rooms(s):
    """Recompute the UVMap on every existing room mesh using current tiling values."""
    for reg in ROOM_OT_draw._room_list:
        obj_name = reg.get("obj_name", "")
        if obj_name not in bpy.data.objects:
            continue
        me = bpy.data.objects[obj_name].data
        uv = me.uv_layers.get("UVMap")
        if uv:
            me.uv_layers.remove(uv)
        _apply_cube_uv_to_mesh(me, s)


def _make_room_obj(name, x1, y1, x2, y2, s, doors=(), no_walls=(), windows=(), collection=None):
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    cat_vert_sets = _fill_room(bm, x1, y1, x2, y2, s.z_foundation, s, doors, no_walls, windows)
    _setup_room_materials(me, s, bm)
    bm.to_mesh(me)
    bm.free()
    me.update()
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.collection).objects.link(obj)
    _setup_room_vertex_groups(obj, cat_vert_sets)
    _apply_room_pivot(obj, x1, y1, x2, y2, s.z_foundation, s.pivot_mode)
    _apply_cube_uv_to_mesh(me, s)
    return obj


def _rebuild_room_mesh(reg, s):
    """Regenerate the mesh of an existing room in-place from its registry entry."""
    obj_name = reg.get("obj_name", "")
    if obj_name not in bpy.data.objects:
        return
    obj = bpy.data.objects[obj_name]
    obj.location = (0.0, 0.0, 0.0)   # reset before regenerating at world coords
    me  = obj.data
    bm  = bmesh.new()
    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    z   = reg.get("z", s.z_foundation)
    cat_vert_sets = _fill_room(bm, x1, y1, x2, y2, z, s,
               doors=reg.get("doors", []),
               no_walls=tuple(reg.get("no_walls", [])),
               windows=reg.get("windows", []),
               add_plinth_bottom=s.add_plinth_bottom and reg.get('plinth_bottom_enabled', True),
               add_plinth_top   =s.add_plinth_top    and reg.get('plinth_top_enabled',    True))
    _setup_room_materials(me, s, bm)
    bm.to_mesh(me)
    bm.free()
    me.update()
    _setup_room_vertex_groups(obj, cat_vert_sets)
    _apply_room_pivot(obj, x1, y1, x2, y2, z, s.pivot_mode)
    _apply_cube_uv_to_mesh(me, s)


def _apply_room_pivot(obj, x1, y1, x2, y2, z, mode):
    """Shift mesh vertices and set obj.location so the origin matches mode."""
    if mode == 'WORLD_ORIGIN':
        obj.location = (0.0, 0.0, 0.0)
        return
    if mode == 'FLOOR_CENTER':
        pivot = Vector(((x1 + x2) * 0.5, (y1 + y2) * 0.5, z))
    else:  # GEOMETRY_CENTER
        verts = obj.data.vertices
        if not verts:
            obj.location = (0.0, 0.0, 0.0)
            return
        xs = [v.co.x for v in verts]
        ys = [v.co.y for v in verts]
        zs = [v.co.z for v in verts]
        pivot = Vector((
            (min(xs) + max(xs)) * 0.5,
            (min(ys) + max(ys)) * 0.5,
            (min(zs) + max(zs)) * 0.5,
        ))
    for v in obj.data.vertices:
        v.co -= pivot
    obj.data.update()
    obj.location = pivot


def _setup_room_materials(me, s, bm):
    """Clear mesh material slots and re-populate from settings; assign face material_index.
    Door frames and window frames fall back to the wall material when not explicitly set."""
    while me.materials:
        me.materials.pop(index=0)
    mat_w  = getattr(s, 'mat_walls',        None)
    mat_f  = getattr(s, 'mat_floor',        None)
    mat_c  = getattr(s, 'mat_ceiling',      None)
    mat_d  = getattr(s, 'mat_door_frame',   None) or mat_w
    mat_wf = getattr(s, 'mat_window_frame', None) or mat_w
    mat_pb = getattr(s, 'mat_plinth_bottom', None) or mat_w
    mat_pt = getattr(s, 'mat_plinth_top',    None) or mat_w
    # Build deduplicated slot list so shared materials use one slot
    mat_list  = []
    slot_map  = {}
    for cat, mat in enumerate((mat_w, mat_f, mat_c, mat_d, mat_wf, mat_pb, mat_pt)):
        if mat is None:
            slot_map[cat] = None
        elif mat in mat_list:
            slot_map[cat] = mat_list.index(mat)
        else:
            slot_map[cat] = len(mat_list)
            mat_list.append(mat)
    for mat in mat_list:
        me.materials.append(mat)
    if any(v is not None for v in slot_map.values()):
        cat_layer = bm.faces.layers.int.get("room_cat")
        if cat_layer:
            for face in bm.faces:
                si = slot_map.get(face[cat_layer])
                if si is not None:
                    face.material_index = si


def _setup_room_vertex_groups(obj, cat_vert_sets):
    """Create named vertex groups from category → vertex index mapping."""
    for name in _CAT_NAMES:
        vg = obj.vertex_groups.get(name)
        if vg:
            obj.vertex_groups.remove(vg)
    for cat, indices in cat_vert_sets.items():
        if indices:
            vg = obj.vertex_groups.new(name=_CAT_NAMES[cat])
            vg.add(list(indices), 1.0, 'REPLACE')


# ═════════════════════════════════════════════════════════════════════════════
# Ray / snap utilities
# ═════════════════════════════════════════════════════════════════════════════
def _ray_to_z(context, event, z):
    r, rv3d = context.region, context.region_data
    co  = event.mouse_region_x, event.mouse_region_y
    org = view3d_utils.region_2d_to_origin_3d(r, rv3d, co)
    d   = view3d_utils.region_2d_to_vector_3d(r, rv3d, co)
    if abs(d.z) < 1e-6:
        return None
    t = (z - org.z) / d.z
    return None if t < 0 else org + t * d


def _ray_to_wall_z(context, event, wall_char, wall_coord):
    """Project the mouse onto a wall's vertical plane and return the Z coordinate.
    wall_char : 'N'/'S' → XZ plane at y=wall_coord, normal (0,1,0)
                'E'/'W' → YZ plane at x=wall_coord, normal (1,0,0)
    Returns float Z, or None if the ray is parallel to the wall.
    """
    from mathutils.geometry import intersect_line_plane
    r, rv3d = context.region, context.region_data
    co      = (event.mouse_region_x, event.mouse_region_y)
    origin    = view3d_utils.region_2d_to_origin_3d(r, rv3d, co)
    direction = view3d_utils.region_2d_to_vector_3d(r, rv3d, co)
    if wall_char in ('E', 'W'):
        plane_co = Vector((wall_coord, 0.0, 0.0))
        plane_no = Vector((1.0, 0.0, 0.0))
    else:
        plane_co = Vector((0.0, wall_coord, 0.0))
        plane_no = Vector((0.0, 1.0, 0.0))
    pt = intersect_line_plane(origin, origin + direction, plane_co, plane_no)
    return pt.z if pt is not None else None


_SNAP_DIST = 0.5   # metres


def _room_is_usable(r):
    """Return True iff the room's mesh object exists AND is visible in the
    active view layer.  Hidden / excluded rooms are NOT treated as obstacles."""
    obj = bpy.data.objects.get(r.get("obj_name", ""))
    if obj is None:
        return False
    try:
        return obj.visible_get()
    except Exception:
        return True   # fall back to allowing it if visibility check fails


def _wall_snap_ext(pt, rooms, t_fallback, current_z=None):
    """
    Find the nearest OUTER wall face within _SNAP_DIST, approached from the OUTSIDE.
    Returns (snap_2d, (nx,ny), room_idx, wall_char) or None.
    current_z: when set, only considers rooms on the same floor (within 1 mm).
    """
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
        if not _room_is_usable(r):
            continue
        if current_z is not None and abs(r.get("z", 0.0) - current_z) > 0.001:
            continue
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        t = r.get("t", t_fallback)
        checks = [
            ('E', x2+t, abs(px-(x2+t)), py,  y1-t, y2+t, ( 1, 0)),
            ('W', x1-t, abs(px-(x1-t)), py,  y1-t, y2+t, (-1, 0)),
            ('N', y2+t, abs(py-(y2+t)), px,  x1-t, x2+t, ( 0, 1)),
            ('S', y1-t, abs(py-(y1-t)), px,  x1-t, x2+t, ( 0,-1)),
        ]
        for wc, face_pos, dist, along, lo, hi, normal in checks:
            if wc == 'E' and px < x2:  continue
            if wc == 'W' and px > x1:  continue
            if wc == 'N' and py < y2:  continue
            if wc == 'S' and py > y1:  continue
            if dist < best_d and lo <= along <= hi:
                if wc == 'E':
                    sp = Vector((x2 + t, max(y1, min(y2, py))))
                elif wc == 'W':
                    sp = Vector((x1 - t, max(y1, min(y2, py))))
                elif wc == 'N':
                    sp = Vector((max(x1, min(x2, px)), y2 + t))
                else:
                    sp = Vector((max(x1, min(x2, px)), y1 - t))
                best_d = dist
                best   = (sp, normal, i, wc)
    return best


def _wall_snap_any(pt, rooms, t_fallback, current_z=None):
    """
    Find the nearest wall face (from any direction) within _SNAP_DIST.
    Returns (room_idx, wall_char, anchor) or None.
    anchor = cursor position clamped along the wall span.
    current_z: when set, only considers rooms on the same floor (within 1 mm).
    """
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
        if not _room_is_usable(r):
            continue
        if current_z is not None and abs(r.get("z", 0.0) - current_z) > 0.001:
            continue
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        t = r.get("t", t_fallback)
        checks = [
            ('S', abs(py - y1), px, x1, x2, max(x1, min(x2, px))),
            ('N', abs(py - y2), px, x1, x2, max(x1, min(x2, px))),
            ('W', abs(px - x1), py, y1, y2, max(y1, min(y2, py))),
            ('E', abs(px - x2), py, y1, y2, max(y1, min(y2, py))),
        ]
        for wc, dist, along, lo, hi, anchor in checks:
            if dist < best_d and lo - t <= along <= hi + t:
                best_d = dist
                best   = (i, wc, anchor)
    return best


def _door_snap(pt, rooms, s, t_fallback):
    """Return (room_idx, door_idx) of the nearest door frame, or None.
    Iterates over all doors in reg['doors']; cursor must be near the wall
    and within the door-width span."""
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        t = r.get("t", t_fallback)
        for di, door in enumerate(r.get("doors", [])):
            wc     = door["wc"]
            dw     = door["w"]
            anchor = door["anchor"]
            if wc == 'E':
                dist  = abs(px - x2)
                along = py
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            elif wc == 'W':
                dist  = abs(px - x1)
                along = py
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            elif wc == 'N':
                dist  = abs(py - y2)
                along = px
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            else:  # S
                dist  = abs(py - y1)
                along = px
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            if dist < best_d and lo <= along <= hi:
                best_d = dist
                best   = (i, di)
    return best


def _clamp_anchor(anchor, reg, wall_char, door_width, margin):
    """Clamp door centre to wall-edge clearance only (used in draw-mode DS phase).
    Does NOT reject even if it would overlap an existing door."""
    if wall_char in ('E', 'W'):
        lo = reg["y1"] + door_width * 0.5 + margin
        hi = reg["y2"] - door_width * 0.5 - margin
    else:
        lo = reg["x1"] + door_width * 0.5 + margin
        hi = reg["x2"] - door_width * 0.5 - margin
    if lo > hi:
        return ((reg["y1"] + reg["y2"]) * 0.5 if wall_char in ('E', 'W')
                else (reg["x1"] + reg["x2"]) * 0.5)
    return max(lo, min(hi, anchor))


def _valid_anchor(anchor, reg, wall_char, door_width, margin, skip_idx=None):
    """Return clamped anchor if position is valid, None if it must be rejected.

    Rejects when:
    - Wall is too narrow to fit the door (with margin on both sides), OR
    - The (clamped) position would be closer than *margin* to any existing
      door on the same wall (gap between door edges < margin).

    skip_idx: door index in reg['doors'] to ignore (used when sliding).
    """
    if wall_char in ('E', 'W'):
        lo = reg["y1"] + door_width * 0.5 + margin
        hi = reg["y2"] - door_width * 0.5 - margin
    else:
        lo = reg["x1"] + door_width * 0.5 + margin
        hi = reg["x2"] - door_width * 0.5 - margin
    if lo > hi:
        return None   # wall too narrow
    anchor = max(lo, min(hi, anchor))   # clamp to wall edges
    for idx, d in enumerate(reg.get("doors", [])):
        if d["wc"] != wall_char:
            continue
        if skip_idx is not None and idx == skip_idx:
            continue
        gap = abs(anchor - d["anchor"]) - (door_width + d["w"]) * 0.5
        if gap < margin:
            return None   # overlap or too close
    # Also reject if the door would overlap any window on the same wall
    for w in reg.get("windows", []):
        if w["wc"] != wall_char:
            continue
        gap = abs(anchor - w["anchor"]) - (door_width + w["w"]) * 0.5
        if gap < margin:
            return None   # overlaps or too close to a window
    return anchor


def _max_array_count(reg, wc, ww, gap, margin):
    """Maximum windows that physically fit on the wall interior."""
    span_lo, span_hi = ((reg["y1"], reg["y2"]) if wc in ('E', 'W')
                        else (reg["x1"], reg["x2"]))
    available = span_hi - span_lo - 2 * margin
    if available <= 0 or ww <= 0:
        return 0
    return max(0, int((available + gap) / (ww + gap)))


def _array_anchors(center, count, ww, gap, reg, wc, margin):
    """Return list of `count` anchor positions for a window array centered at `center`.
    Clamps the center so the whole array stays within wall span + margin.
    """
    total = count * ww + (count - 1) * gap
    span_lo, span_hi = ((reg["y1"], reg["y2"]) if wc in ('E', 'W')
                        else (reg["x1"], reg["x2"]))
    c_lo = span_lo + total / 2 + margin
    c_hi = span_hi - total / 2 - margin
    c = ((span_lo + span_hi) / 2 if c_lo > c_hi
         else max(c_lo, min(c_hi, center)))
    return [c - total / 2 + i * (ww + gap) + ww / 2 for i in range(count)]


def _wall_adj_zones(reg, wc, all_rooms, t):
    """Return [(lo, hi)] boundary-point pairs for adjacent room edges on this wall.

    Each adjacent room contributes two degenerate (point) zones — one at the
    start and one at the end of the shared overlap span.  _valid_spans treats
    each zone as a buffer of ww/2+margin on both sides, so a window is blocked
    only when it would *straddle* a boundary.  A window placed fully inside the
    shared span, or fully outside it, is always valid.
    """
    zones = []
    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    # Outer face of this room's wall (matches _find_partner_wall logic)
    if   wc == 'E': a_pos = x2 + t
    elif wc == 'W': a_pos = x1 - t
    elif wc == 'N': a_pos = y2 + t
    else:           a_pos = y1 - t
    for other in all_rooms:
        if other is reg:
            continue
        t_o = other.get("t", t)
        thr = 2 * max(t, t_o)
        ox1, oy1, ox2, oy2 = other["x1"], other["y1"], other["x2"], other["y2"]
        # Outer face of the other room's opposite wall
        if   wc == 'E': b_pos = ox1 - t_o
        elif wc == 'W': b_pos = ox2 + t_o
        elif wc == 'N': b_pos = oy1 - t_o
        else:           b_pos = oy2 + t_o
        if abs(a_pos - b_pos) > thr:
            continue
        if wc in ('E', 'W'):
            lo, hi = max(y1, oy1), min(y2, oy2)
        else:
            lo, hi = max(x1, ox1), min(x2, ox2)
        if lo < hi:
            # Two degenerate (point) zones mark the zone boundaries.
            # Windows can be fully inside or fully outside but not straddling.
            zones.append((lo, lo))  # lower boundary of shared span
            zones.append((hi, hi))  # upper boundary of shared span
    return zones


def _valid_spans(lo, hi, zones, ww, margin):
    """Sub-spans of [lo,hi] where a window of ww+margin clears all blocked zones."""
    segs = [(lo, hi)]
    for z_lo, z_hi in sorted(zones):
        exc_lo = z_lo - ww * 0.5 - margin
        exc_hi = z_hi + ww * 0.5 + margin
        new_segs = []
        for sl, sr in segs:
            if exc_lo > sl:
                new_segs.append((sl, min(sr, exc_lo)))
            if exc_hi < sr:
                new_segs.append((max(sl, exc_hi), sr))
        segs = [(a, b) for a, b in new_segs if a < b]
    return segs


def _clamp_window_anchor(anchor, reg, wall_char, win_width, margin, zones=()):
    """Clamp window centre to wall-edge clearance only (soft clamp, no rejection)."""
    if wall_char in ('E', 'W'):
        lo = reg["y1"] + win_width * 0.5 + margin
        hi = reg["y2"] - win_width * 0.5 - margin
    else:
        lo = reg["x1"] + win_width * 0.5 + margin
        hi = reg["x2"] - win_width * 0.5 - margin
    if lo > hi:
        return ((reg["y1"] + reg["y2"]) * 0.5 if wall_char in ('E', 'W')
                else (reg["x1"] + reg["x2"]) * 0.5)
    anchor = max(lo, min(hi, anchor))
    if zones:
        sub = _valid_spans(lo, hi, zones, win_width, margin)
        if sub:
            anchor = min((abs(max(sl, min(sr, anchor)) - anchor),
                          max(sl, min(sr, anchor))) for sl, sr in sub)[1]
    return anchor


def _valid_window_anchor(anchor, reg, wall_char, win_width, margin,
                         skip_idx=None, zones=()):
    """Return clamped anchor if valid, None if wall too narrow or overlapping.
    skip_idx: window index in reg['windows'] to ignore (used when sliding).
    zones: [(lo, hi)] blocked intervals from adjacent rooms.
    """
    if wall_char in ('E', 'W'):
        lo = reg["y1"] + win_width * 0.5 + margin
        hi = reg["y2"] - win_width * 0.5 - margin
    else:
        lo = reg["x1"] + win_width * 0.5 + margin
        hi = reg["x2"] - win_width * 0.5 - margin
    if lo > hi:
        return None
    anchor = max(lo, min(hi, anchor))
    if zones:
        sub = _valid_spans(lo, hi, zones, win_width, margin)
        if not sub:
            return None
        anchor = min((abs(max(sl, min(sr, anchor)) - anchor),
                      max(sl, min(sr, anchor))) for sl, sr in sub)[1]
    for idx, w in enumerate(reg.get("windows", [])):
        if w["wc"] != wall_char:
            continue
        if skip_idx is not None and idx == skip_idx:
            continue
        gap = abs(anchor - w["anchor"]) - (win_width + w["w"]) * 0.5
        if gap < margin:
            return None
    for d in reg.get("doors", []):
        if d["wc"] != wall_char:
            continue
        gap = abs(anchor - d["anchor"]) - (win_width + d["w"]) * 0.5
        if gap < margin:
            return None
    return anchor


def _window_frame_verts(r, win, room_z, ww, wh):
    """Return 4 CCW 3-D corners of the window opening face for GPU highlighting."""
    wc     = win["wc"]
    anchor = win["anchor"]
    x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
    z0 = room_z + win.get("v_offset", 0.9)
    z1 = z0 + wh
    if wc == 'E':
        lo = max(y1, anchor - ww * 0.5)
        hi = min(y2, anchor + ww * 0.5)
        return [(x2, lo, z0), (x2, hi, z0), (x2, hi, z1), (x2, lo, z1)]
    elif wc == 'W':
        lo = max(y1, anchor - ww * 0.5)
        hi = min(y2, anchor + ww * 0.5)
        return [(x1, hi, z0), (x1, lo, z0), (x1, lo, z1), (x1, hi, z1)]
    elif wc == 'N':
        lo = max(x1, anchor - ww * 0.5)
        hi = min(x2, anchor + ww * 0.5)
        return [(lo, y2, z0), (hi, y2, z0), (hi, y2, z1), (lo, y2, z1)]
    else:  # S
        lo = max(x1, anchor - ww * 0.5)
        hi = min(x2, anchor + ww * 0.5)
        return [(hi, y1, z0), (lo, y1, z0), (lo, y1, z1), (hi, y1, z1)]


def _window_snap(pt, rooms, s, t_fallback):
    """Return (room_idx, win_idx) of the nearest window frame, or None."""
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        t = r.get("t", t_fallback)
        for wi, win in enumerate(r.get("windows", [])):
            wc     = win["wc"]
            ww     = win["w"]
            anchor = win["anchor"]
            if wc == 'E':
                dist  = abs(px - x2);  along = py
            elif wc == 'W':
                dist  = abs(px - x1);  along = py
            elif wc == 'N':
                dist  = abs(py - y2);  along = px
            else:
                dist  = abs(py - y1);  along = px
            lo = anchor - ww * 0.5 - t
            hi = anchor + ww * 0.5 + t
            if dist < best_d and lo <= along <= hi:
                best_d = dist
                best   = (i, wi)
    return best


def _door_frame_verts(r, door, z, dw, dh):
    """Return 4 CCW 3-D corners of the door opening face for GPU highlighting.
    door: {"wc": str, "anchor": float, "w": float, "h": float}
    """
    wc     = door["wc"]
    anchor = door["anchor"]
    x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
    z0, z1 = z, z + dh
    if wc == 'E':
        lo = max(y1, anchor - dw * 0.5)
        hi = min(y2, anchor + dw * 0.5)
        return [(x2, lo, z0), (x2, hi, z0), (x2, hi, z1), (x2, lo, z1)]
    elif wc == 'W':
        lo = max(y1, anchor - dw * 0.5)
        hi = min(y2, anchor + dw * 0.5)
        return [(x1, hi, z0), (x1, lo, z0), (x1, lo, z1), (x1, hi, z1)]
    elif wc == 'N':
        lo = max(x1, anchor - dw * 0.5)
        hi = min(x2, anchor + dw * 0.5)
        return [(lo, y2, z0), (hi, y2, z0), (hi, y2, z1), (lo, y2, z1)]
    else:  # S
        lo = max(x1, anchor - dw * 0.5)
        hi = min(x2, anchor + dw * 0.5)
        return [(hi, y1, z0), (lo, y1, z0), (lo, y1, z1), (hi, y1, z1)]


def _solid_x_overlap(nx1, nx2, r, wall_char, s):
    """Return True if [nx1,nx2] touches any SOLID section of r's N or S wall.
    Returns False only if the overlap is entirely inside a door-frame opening."""
    rx1, rx2 = r["x1"], r["x2"]
    ox1, ox2 = max(nx1, rx1), min(nx2, rx2)
    if ox1 > ox2:
        return False
    for d in r.get("doors", []):
        if d["wc"] != wall_char:
            continue
        dl = max(rx1, d["anchor"] - d["w"] * 0.5)
        dr = min(rx2, d["anchor"] + d["w"] * 0.5)
        if ox1 >= dl and ox2 <= dr:
            return False   # entirely within this door opening
    return True


def _solid_y_overlap(ny1, ny2, r, wall_char, s):
    """Return True if [ny1,ny2] touches any SOLID section of r's E or W wall.
    Returns False only if the overlap is entirely inside a door-frame opening."""
    ry1, ry2 = r["y1"], r["y2"]
    oy1, oy2 = max(ny1, ry1), min(ny2, ry2)
    if oy1 > oy2:
        return False
    for d in r.get("doors", []):
        if d["wc"] != wall_char:
            continue
        db = max(ry1, d["anchor"] - d["w"] * 0.5)
        df = min(ry2, d["anchor"] + d["w"] * 0.5)
        if oy1 >= db and oy2 <= df:
            return False
    return True


def _clamp_y_for_rooms(new_y, fixed_y, nx1, nx2, rooms, skip_idx, s, current_z=None):
    t      = s.wall_thickness
    margin = s.door_margin
    if new_y == fixed_y:
        return new_y
    going_south = new_y < fixed_y
    for i, r in enumerate(rooms):
        if i == skip_idx:
            continue
        if not _room_is_usable(r):
            continue
        if current_z is not None and abs(r.get("z", 0.0) - current_z) > 0.001:
            continue
        rx1, ry1, rx2, ry2 = r["x1"], r["y1"], r["x2"], r["y2"]
        if nx2 < rx1 or nx1 > rx2:
            continue
        if going_south:
            if ry2 <= fixed_y:
                if _solid_x_overlap(nx1, nx2, r, 'N', s):
                    new_y = max(new_y, ry2 + t)
                else:
                    new_y = max(new_y, ry2)
        else:
            if ry1 >= fixed_y:
                if _solid_x_overlap(nx1, nx2, r, 'S', s):
                    new_y = min(new_y, ry1 - t)
                else:
                    new_y = min(new_y, ry1)

    # Window no-go zones: free N/S edge must not bisect windows on adjacent E/W walls
    for i, r in enumerate(rooms):
        if i == skip_idx:
            continue
        if not _room_is_usable(r):
            continue
        if current_z is not None and abs(r.get("z", 0.0) - current_z) > 0.001:
            continue
        rx1, ry1, rx2, ry2 = r["x1"], r["y1"], r["x2"], r["y2"]
        t_r = r.get("t", t)
        tol = t + t_r
        shared_w = abs(nx2 - rx1) < tol   # new room's east edge ≈ room's west wall
        shared_e = abs(nx1 - rx2) < tol   # new room's west edge ≈ room's east wall
        if not shared_w and not shared_e:
            continue
        walls = ([('W', None)] if shared_w else []) + ([('E', None)] if shared_e else [])
        for wc, _ in walls:
            for w in r.get("windows", []):
                if w["wc"] != wc:
                    continue
                ww  = w["w"]
                cy  = w["anchor"]
                wb  = max(ry1, cy - ww * 0.5) - margin
                wf  = min(ry2, cy + ww * 0.5) + margin
                if wb < new_y < wf:
                    if going_south:
                        new_y = max(new_y, wf)   # snap north of window zone
                    else:
                        new_y = min(new_y, wb)   # snap south of window zone
    return new_y


def _clamp_x_for_rooms(new_x, fixed_x, ny1, ny2, rooms, skip_idx, s, current_z=None):
    t      = s.wall_thickness
    margin = s.door_margin
    if new_x == fixed_x:
        return new_x
    going_west = new_x < fixed_x
    for i, r in enumerate(rooms):
        if i == skip_idx:
            continue
        if not _room_is_usable(r):
            continue
        if current_z is not None and abs(r.get("z", 0.0) - current_z) > 0.001:
            continue
        rx1, ry1, rx2, ry2 = r["x1"], r["y1"], r["x2"], r["y2"]
        if ny2 < ry1 or ny1 > ry2:
            continue
        if going_west:
            if rx2 <= fixed_x:
                if _solid_y_overlap(ny1, ny2, r, 'E', s):
                    new_x = max(new_x, rx2 + t)
                else:
                    new_x = max(new_x, rx2)
        else:
            if rx1 >= fixed_x:
                if _solid_y_overlap(ny1, ny2, r, 'W', s):
                    new_x = min(new_x, rx1 - t)
                else:
                    new_x = min(new_x, rx1)

    # Window no-go zones: free E/W edge must not bisect windows on adjacent N/S walls
    for i, r in enumerate(rooms):
        if i == skip_idx:
            continue
        if not _room_is_usable(r):
            continue
        if current_z is not None and abs(r.get("z", 0.0) - current_z) > 0.001:
            continue
        rx1, ry1, rx2, ry2 = r["x1"], r["y1"], r["x2"], r["y2"]
        t_r = r.get("t", t)
        tol = t + t_r
        shared_s = abs(ny2 - ry1) < tol   # new room's north edge ≈ room's south wall
        shared_n = abs(ny1 - ry2) < tol   # new room's south edge ≈ room's north wall
        if not shared_s and not shared_n:
            continue
        walls = ([('S', None)] if shared_s else []) + ([('N', None)] if shared_n else [])
        for wc, _ in walls:
            for w in r.get("windows", []):
                if w["wc"] != wc:
                    continue
                ww  = w["w"]
                cx  = w["anchor"]
                wl  = max(rx1, cx - ww * 0.5) - margin
                wr  = min(rx2, cx + ww * 0.5) + margin
                if wl < new_x < wr:
                    if going_west:
                        new_x = max(new_x, wr)   # snap east of window zone
                    else:
                        new_x = min(new_x, wl)   # snap west of window zone
    return new_x


def _enforce_snap_window_span(new_coord, fixed_coord, snap_info, rooms, s):
    """Ensure the free edge (dragged along the snapped wall) reaches past every
    window zone on that wall, so no window is left outside the new room's span."""
    if snap_info is None:
        return new_coord
    _normal, wc, room_idx = snap_info
    if room_idx < 0 or room_idx >= len(rooms):
        return new_coord
    r      = rooms[room_idx]
    margin = s.door_margin
    going_neg = new_coord < fixed_coord   # True = going south / west
    if wc in ('E', 'W'):                  # windows have Y-axis anchors
        r_lo, r_hi = r["y1"], r["y2"]
        for w in r.get("windows", []):
            if w["wc"] != wc:
                continue
            cy, ww = w["anchor"], w["w"]
            wb = max(r_lo, cy - ww * 0.5) - margin
            wf = min(r_hi, cy + ww * 0.5) + margin
            if going_neg:
                new_coord = min(new_coord, wb)   # drag south: must reach wb or further
            else:
                new_coord = max(new_coord, wf)   # drag north: must reach wf or further
    else:                                  # N/S walls — windows have X-axis anchors
        r_lo, r_hi = r["x1"], r["x2"]
        for w in r.get("windows", []):
            if w["wc"] != wc:
                continue
            cx, ww = w["anchor"], w["w"]
            wl = max(r_lo, cx - ww * 0.5) - margin
            wr = min(r_hi, cx + ww * 0.5) + margin
            if going_neg:
                new_coord = min(new_coord, wl)   # drag west: must reach wl or further
            else:
                new_coord = max(new_coord, wr)   # drag east: must reach wr or further
    return new_coord


def _find_partner_wall(rooms, room_idx, wall_char, t_fallback):
    """
    Find a room whose opposite wall's outer face is within 2×t of room_idx's
    wall outer face, with overlapping perpendicular span.
    Returns (partner_idx, opp_wall_char) or None.
    """
    r_A = rooms[room_idx]
    t_A = r_A.get("t", t_fallback)
    opp = _OPPOSITE[wall_char]

    if wall_char == 'E':   a_pos = r_A["x2"] + t_A
    elif wall_char == 'W': a_pos = r_A["x1"] - t_A
    elif wall_char == 'N': a_pos = r_A["y2"] + t_A
    else:                  a_pos = r_A["y1"] - t_A

    for j, r_B in enumerate(rooms):
        if j == room_idx:
            continue
        t_B = r_B.get("t", t_fallback)
        thr = 2 * max(t_A, t_B)

        if opp == 'E':   b_pos = r_B["x2"] + t_B
        elif opp == 'W': b_pos = r_B["x1"] - t_B
        elif opp == 'N': b_pos = r_B["y2"] + t_B
        else:            b_pos = r_B["y1"] - t_B

        if abs(a_pos - b_pos) > thr:
            continue

        if wall_char in ('E', 'W'):
            if min(r_A["y2"], r_B["y2"]) > max(r_A["y1"], r_B["y1"]):
                return j, opp
        else:
            if min(r_A["x2"], r_B["x2"]) > max(r_A["x1"], r_B["x1"]):
                return j, opp
    return None


def _find_partner_door(rooms, room_idx, door_idx, t_fallback):
    """Find the matching door on the partner room's opposite wall.
    Matches by anchor position (within 1 mm).
    Returns (partner_room_idx, partner_door_idx) or None.
    """
    reg   = rooms[room_idx]
    doors = reg.get("doors", [])
    if door_idx >= len(doors):
        return None
    door = doors[door_idx]
    wc   = door["wc"]
    res  = _find_partner_wall(rooms, room_idx, wc, t_fallback)
    if res is None:
        return None
    p_idx, p_wc    = res
    target_anchor  = door["anchor"]
    p_reg          = rooms[p_idx]
    for di, d in enumerate(p_reg.get("doors", [])):
        if d["wc"] == p_wc and abs(d["anchor"] - target_anchor) < 0.001:
            return p_idx, di
    return None


def _find_partner_window(rooms, room_idx, win_idx, t_fallback):
    """Find the matching window on the partner room's opposite wall.
    Matches by wall char, anchor and v_offset (within 1 mm).
    Returns (partner_room_idx, partner_win_idx) or None.
    """
    reg = rooms[room_idx]
    wins = reg.get("windows", [])
    if win_idx >= len(wins):
        return None
    win = wins[win_idx]
    wc  = win["wc"]
    res = _find_partner_wall(rooms, room_idx, wc, t_fallback)
    if res is None:
        return None
    p_idx, p_wc   = res
    target_anchor  = win["anchor"]
    target_voff    = win.get("v_offset", 0.9)
    p_reg          = rooms[p_idx]
    for wi, w in enumerate(p_reg.get("windows", [])):
        if (w["wc"] == p_wc
                and abs(w["anchor"] - target_anchor) < 0.001
                and abs(w.get("v_offset", 0.9) - target_voff) < 0.001):
            return p_idx, wi
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Registry persistence helpers
# ═════════════════════════════════════════════════════════════════════════════
def _reg_to_entry(reg, entry):
    entry.x1 = reg["x1"]
    entry.y1 = reg["y1"]
    entry.x2 = reg["x2"]
    entry.y2 = reg["y2"]
    entry.t  = reg.get("t", 0.125)
    entry.z  = reg.get("z", 0.0)
    entry.doors_json   = json.dumps(reg.get("doors",   []))
    entry.windows_json = json.dumps(reg.get("windows", []))
    entry.no_walls     = ",".join(reg.get("no_walls", []))
    entry.obj_name   = reg.get("obj_name", "")
    entry.plinth_bottom_enabled = reg.get('plinth_bottom_enabled', True)
    entry.plinth_top_enabled    = reg.get('plinth_top_enabled',    True)
    # Legacy fields — kept so old .blend files don't error on load
    entry.door_walls   = ""
    entry.door_anchors = "{}"
    entry.door_dims    = "{}"
    entry.door_width   = 0.9
    entry.door_height  = 2.0


def _entry_to_reg(entry):
    # Deserialise new format if present, otherwise reconstruct from legacy fields
    doors_raw = entry.doors_json if hasattr(entry, "doors_json") else ""
    if doors_raw and doors_raw != "[]":
        doors = json.loads(doors_raw)
    else:
        # Backward compat: rebuild from old door_walls / door_anchors / door_dims
        doors = []
        for wc in (entry.door_walls or "").split(","):
            if not wc:
                continue
            anchors = json.loads(entry.door_anchors) if entry.door_anchors else {}
            dims    = json.loads(entry.door_dims)    if entry.door_dims    else {}
            anchor  = anchors.get(wc, 0.0)
            d       = dims.get(wc, {})
            doors.append({
                "wc":     wc,
                "anchor": anchor,
                "w":      d.get("w", entry.door_width),
                "h":      d.get("h", entry.door_height),
            })
    try:
        windows = json.loads(getattr(entry, "windows_json", None) or "[]")
    except Exception:
        windows = []
    return {
        "x1": entry.x1, "y1": entry.y1,
        "x2": entry.x2, "y2": entry.y2,
        "t":  entry.t,
        "z":  entry.z,
        "doors":    doors,
        "windows":  windows,
        "no_walls": [w for w in entry.no_walls.split(",") if w],
        "obj_name": entry.obj_name,
        "plinth_bottom_enabled": getattr(entry, 'plinth_bottom_enabled', True),
        "plinth_top_enabled":    getattr(entry, 'plinth_top_enabled',    True),
    }


def _sync_to_scene(context):
    entries = getattr(context.scene, "room_registry", None)
    if entries is None:
        return
    entries.clear()
    for reg in ROOM_OT_draw._room_list:
        _reg_to_entry(reg, entries.add())


def _sync_from_scene(context):
    """Rebuild _room_list from the scene registry.
    Always runs a cleanup pass first to drop deleted / invisible rooms,
    then repopulates any entries that are missing (e.g. after file reload)."""
    # Remove stale entries: deleted OR hidden objects
    ROOM_OT_draw._room_list[:] = [
        r for r in ROOM_OT_draw._room_list
        if _room_is_usable(r)
    ]
    # Re-add any scene-registry entries whose objects now exist and are visible
    # but aren't tracked yet (e.g. file reload, undo bringing an object back).
    existing_names = {r.get("obj_name", "") for r in ROOM_OT_draw._room_list}
    entries = getattr(context.scene, "room_registry", None)
    if not entries:
        return
    for entry in entries:
        if entry.obj_name not in existing_names and entry.obj_name in bpy.data.objects:
            obj = bpy.data.objects[entry.obj_name]
            try:
                visible = obj.visible_get()
            except Exception:
                visible = True
            if visible:
                ROOM_OT_draw._room_list.append(_entry_to_reg(entry))


# ═════════════════════════════════════════════════════════════════════════════
# GPU highlight helpers
# ═════════════════════════════════════════════════════════════════════════════
def _wall_face_verts(r, wall_char, z, h, t_fallback):
    x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
    rt = r.get("t", t_fallback)
    z0, z1 = z, z + h
    if wall_char == 'E':
        return [(x2+rt,y1-rt,z0),(x2+rt,y2+rt,z0),(x2+rt,y2+rt,z1),(x2+rt,y1-rt,z1)]
    elif wall_char == 'W':
        return [(x1-rt,y2+rt,z0),(x1-rt,y1-rt,z0),(x1-rt,y1-rt,z1),(x1-rt,y2+rt,z1)]
    elif wall_char == 'N':
        return [(x1-rt,y2+rt,z0),(x2+rt,y2+rt,z0),(x2+rt,y2+rt,z1),(x1-rt,y2+rt,z1)]
    else:
        return [(x2+rt,y1-rt,z0),(x1-rt,y1-rt,z0),(x1-rt,y1-rt,z1),(x2+rt,y1-rt,z1)]


# ═════════════════════════════════════════════════════════════════════════════
# Modal draw operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_draw(bpy.types.Operator):
    bl_idname  = "room.draw"
    bl_label   = "Draw Room"
    bl_options = {"REGISTER", "UNDO"}

    _room_list   = []
    _addon_kmaps = []

    # ── GPU draw callback ──────────────────────────────────────────────────
    def _draw_cb(self, context):
        if not self._hovered:
            return
        sp, _normal, room_idx, wall_char = self._hovered
        rooms = ROOM_OT_draw._room_list
        if room_idx >= len(rooms):
            return
        s     = context.scene.room_settings
        r     = rooms[room_idx]
        # Guard: clear hover if the room object was deleted or hidden
        _obj = bpy.data.objects.get(r.get("obj_name", ""))
        if _obj is None or not _obj.visible_get():
            self._hovered = None
            return
        verts = _wall_face_verts(r, wall_char, s.z_foundation,
                                 s.wall_height, s.wall_thickness)
        idx_fill = [(0,1,2),(0,2,3)]
        idx_line = [(0,1),(1,2),(2,3),(3,0)]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')

        # Orange wall highlight — always shown
        b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
        shader.uniform_float("color", (1.0, 0.55, 0.0, 0.28))
        b.draw(shader)
        gpu.state.line_width_set(2.5)
        b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
        shader.uniform_float("color", (1.0, 0.78, 0.0, 0.95))
        b2.draw(shader)

        if s.add_door:
            dw = s.door_width
            dh = s.door_height
            active  = context.scene.room_active_door_preset
            presets = context.scene.room_door_presets
            if 0 <= active < len(presets):
                dp = presets[active]
                dw, dh = dp.door_width, dp.door_height
            raw_anchor  = sp.y if wall_char in ('E', 'W') else sp.x
            anchor      = _valid_anchor(raw_anchor, r, wall_char, dw, s.door_margin)
            is_blocked  = (anchor is None)
            draw_anchor = _clamp_anchor(raw_anchor, r, wall_char, dw, s.door_margin)
            ghost_door  = {"wc": wall_char, "anchor": draw_anchor, "w": dw, "h": dh}
            d_verts = _door_frame_verts(r, ghost_door, r.get("z", s.z_foundation), dw, dh)
            # Door ghost — green = valid, red = blocked
            b3 = batch_for_shader(shader, 'TRIS', {"pos": d_verts}, indices=idx_fill)
            shader.uniform_float("color",
                (1.0, 0.08, 0.05, 0.35) if is_blocked else (0.05, 0.85, 0.2, 0.32))
            b3.draw(shader)
            gpu.state.line_width_set(2.0)
            b4 = batch_for_shader(shader, 'LINES', {"pos": d_verts}, indices=idx_line)
            shader.uniform_float("color",
                (1.0, 0.12, 0.05, 1.0) if is_blocked else (0.1, 1.0, 0.3, 0.9))
            b4.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def _add_draw_handle(self, context):
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_VIEW')
        _DRAW_HANDLES.add(self._draw_handle)

    def _remove_draw_handle(self):
        if self._draw_handle:
            _DRAW_HANDLES.discard(self._draw_handle)
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            except Exception:
                pass
            self._draw_handle = None

    # ── preview helpers ────────────────────────────────────────────────────
    def _create_preview(self, context):
        me  = bpy.data.meshes.new("__room_prev__")
        obj = bpy.data.objects.new("__room_prev__", me)
        context.collection.objects.link(obj)
        obj.display_type = "WIRE"
        obj.hide_select  = True
        self._prev = obj

    def _update_preview_mesh(self, context):
        if self._prev is None or self._start is None or self._end is None:
            return
        s  = context.scene.room_settings
        x1 = min(self._start.x, self._end.x)
        y1 = min(self._start.y, self._end.y)
        x2 = max(self._start.x, self._end.x)
        y2 = max(self._start.y, self._end.y)
        x2 = max(x2, x1 + 0.01)
        y2 = max(y2, y1 + 0.01)
        me = self._prev.data
        me.clear_geometry()
        preview_doors = _preview_doors(self._snap_info)
        nw = _snap_no_walls(self._snap_info)
        bm = bmesh.new()
        _fill_room(bm, x1, y1, x2, y2, s.z_foundation, s, preview_doors, nw)
        bm.to_mesh(me)
        bm.free()
        me.update()

    def _delete_preview(self):
        if self._prev is not None:
            try:
                me = self._prev.data
                bpy.data.objects.remove(self._prev, do_unlink=True)
                bpy.data.meshes.remove(me)
            except Exception:
                pass
            self._prev = None

    def _msg(self, context, text):
        if context.area:
            context.area.header_text_set(text)

    def _idle_msg(self, context):
        door_state = "ON" if context.scene.room_settings.add_door else "OFF"
        self._msg(context,
            f"Door [{door_state}]  ·  LMB – draw room  |  Hover wall to snap-connect"
            "  ·  D – toggle door  |  Enter/RMB – exit")

    def _reset_placement(self, context):
        """Abort current placement, return to hover phase."""
        # Revert door added to existing room during DS phase
        if self._ds_room_idx is not None:
            rooms = ROOM_OT_draw._room_list
            if 0 <= self._ds_room_idx < len(rooms):
                ex_reg = rooms[self._ds_room_idx]
                doors  = ex_reg.get("doors", [])
                if (self._ds_door_idx is not None and
                        0 <= self._ds_door_idx < len(doors)):
                    doors.pop(self._ds_door_idx)
                _rebuild_room_mesh(ex_reg, context.scene.room_settings)
        self._ds_room_idx   = None
        self._ds_wall_char  = None
        self._ds_door_idx   = None
        self._delete_preview()
        self._phase       = 0
        self._phase1_end  = None
        self._start       = None
        self._end         = None
        self._snap_info   = None
        self._axis1_char  = 'X'
        self._hovered     = None
        context.area.tag_redraw()
        self._idle_msg(context)

    # ── lifecycle ──────────────────────────────────────────────────────────
    def invoke(self, context, event):
        if context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "Must be used inside the 3D Viewport")
            return {"CANCELLED"}
        _sync_from_scene(context)
        self._start        = None
        self._end          = None
        self._prev         = None
        self._hovered      = None
        self._snap_info    = None
        self._draw_handle  = None
        self._session_rooms = []
        self._phase        = 0
        self._axis1_char   = 'X'
        self._phase1_end   = None
        self._ds_room_idx  = None
        self._ds_wall_char = None
        self._ds_door_idx  = None
        self._add_draw_handle(context)
        context.window_manager.modal_handler_add(self)
        self._idle_msg(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        s = context.scene.room_settings
        z = s.z_foundation

        # ── navigation pass-through ────────────────────────────────────────
        if event.type in {'MIDDLEMOUSE',
                'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3',
                'NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7',
                'NUMPAD_8','NUMPAD_9','NUMPAD_DECIMAL','NUMPAD_PERIOD',
                'F', 'TILDE'}:
            return {'PASS_THROUGH'}

        # When mouse is over the N-panel or any non-viewport region, let all
        # events pass through so the user can adjust settings mid-draw.
        # ESC/RMB are exempt so they can still cancel the modal from anywhere.
        if context.region.type != 'WINDOW':
            if event.type not in {'ESC', 'RIGHTMOUSE'}:
                return {'PASS_THROUGH'}

        # Scroll — cycle door presets during DS phase, navigation otherwise
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
            if self._phase == 'DS':
                presets = context.scene.room_door_presets
                active  = context.scene.room_active_door_preset
                if len(presets) > 1:
                    delta      = 1 if event.type == 'WHEELUPMOUSE' else -1
                    new_active = (active + delta) % len(presets)
                    context.scene.room_active_door_preset = new_active
                    dp   = presets[new_active]
                    ex_reg = ROOM_OT_draw._room_list[self._ds_room_idx]
                    di     = self._ds_door_idx
                    if di is not None and di < len(ex_reg.get("doors", [])):
                        ex_reg["doors"][di]["w"] = dp.door_width
                        ex_reg["doors"][di]["h"] = dp.door_height
                        _rebuild_room_mesh(ex_reg, s)
                        partner = _find_partner_wall(
                            ROOM_OT_draw._room_list, self._ds_room_idx,
                            self._ds_wall_char, s.wall_thickness)
                        if partner is not None:
                            p_idx, p_wc = partner
                            p_reg = ROOM_OT_draw._room_list[p_idx]
                            # Find the partner door by anchor
                            cur_anchor = ex_reg["doors"][di]["anchor"]
                            for pd in p_reg.get("doors", []):
                                if pd["wc"] == p_wc and abs(pd["anchor"] - cur_anchor) < 0.001:
                                    pd["w"] = dp.door_width
                                    pd["h"] = dp.door_height
                                    break
                            _rebuild_room_mesh(p_reg, s)
                        _sync_to_scene(context)
                        context.area.header_text_set(
                            f"Preset: {dp.name} ({dp.door_width:.2f}×{dp.door_height:.2f}m)  |  "
                            "Scroll to cycle · Click to confirm  |  RMB – cancel")
                        context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # ── Ctrl+Z — undo last placed room ────────────────────────────────
        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            if self._phase == 0 and self._session_rooms:
                obj, reg, ex_reg, prev_ds_door_idx = self._session_rooms.pop()
                try:
                    me = obj.data
                    bpy.data.objects.remove(obj, do_unlink=True)
                    bpy.data.meshes.remove(me)
                except Exception:
                    pass
                try:
                    ROOM_OT_draw._room_list.remove(reg)
                except ValueError:
                    pass
                # Revert the DS door that was added to the existing room
                if ex_reg is not None and prev_ds_door_idx is not None:
                    doors = ex_reg.get("doors", [])
                    if 0 <= prev_ds_door_idx < len(doors):
                        doors.pop(prev_ds_door_idx)
                    _rebuild_room_mesh(ex_reg, s)
                self._hovered = None
                _sync_to_scene(context)
                context.area.tag_redraw()
                self._msg(context,
                    f"Undid last room  ({len(self._session_rooms)} this session)  |  "
                    "LMB – draw room  |  Enter/RMB – exit")
            elif self._phase == 0:
                self._msg(context, "Nothing to undo this session")
            return {'RUNNING_MODAL'}

        # ── D — toggle door frame ──────────────────────────────────────────
        if event.type == 'D' and event.value == 'PRESS' and self._phase == 0:
            s.add_door = not s.add_door
            self._idle_msg(context)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # ── Alt+S — snap current side to existing wall edge ────────────────
        if (event.type == 'S' and event.alt and event.value == 'PRESS'
                and self._phase in (1, 2)
                and self._snap_info is not None):
            _, wc, room_idx = self._snap_info
            rooms = ROOM_OT_draw._room_list
            if 0 <= room_idx < len(rooms):
                ex = rooms[room_idx]
                if wc in ('E', 'W'):
                    ey1, ey2 = ex["y1"], ex["y2"]
                    target_y = (ey1 if abs(self._end.y - ey1) <= abs(self._end.y - ey2)
                                else ey2)
                    self._end = Vector((self._end.x, target_y, z))
                else:
                    ex1, ex2 = ex["x1"], ex["x2"]
                    target_x = (ex1 if abs(self._end.x - ex1) <= abs(self._end.x - ex2)
                                else ex2)
                    self._end = Vector((target_x, self._end.y, z))
                self._update_preview_mesh(context)
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # ── exit / cancel ──────────────────────────────────────────────────
        if event.type in {"RIGHTMOUSE", "ESC"} and event.value == 'PRESS':
            if self._phase != 0:
                self._reset_placement(context)
                return {'RUNNING_MODAL'}
            self._delete_preview()
            self._remove_draw_handle()
            self._msg(context, None)
            return {"CANCELLED"}

        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == 'PRESS':
            if self._phase != 0:
                self._reset_placement(context)
                return {'RUNNING_MODAL'}
            self._delete_preview()
            self._remove_draw_handle()
            self._msg(context, None)
            return {"FINISHED"}

        # ── mouse move ────────────────────────────────────────────────────
        if event.type == "MOUSEMOVE":
            pt = _ray_to_z(context, event, z)

            if self._phase == 0:
                if pt:
                    snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list, s.wall_thickness, current_z=z)
                    self._hovered = snap
                    if snap:
                        if s.add_door:
                            sp, _n, ri, wc = snap
                            dw = s.door_width
                            active  = context.scene.room_active_door_preset
                            presets = context.scene.room_door_presets
                            if 0 <= active < len(presets):
                                dw = presets[active].door_width
                            raw = sp.y if wc in ('E', 'W') else sp.x
                            if _valid_anchor(raw, ROOM_OT_draw._room_list[ri],
                                             wc, dw, s.door_margin) is None:
                                self._msg(context,
                                    "Cannot place door here — window is blocking  ·  "
                                    "D – toggle door  |  Enter/RMB – exit")
                            else:
                                self._msg(context,
                                    "Click to connect (door [ON])  ·  "
                                    "D – toggle door  |  Enter/RMB – exit")
                        else:
                            self._msg(context,
                                "Click to connect (door [OFF])  ·  "
                                "D – toggle door  |  Enter/RMB – exit")
                    else:
                        self._idle_msg(context)
                else:
                    # Cursor left the viewport (e.g. over N-panel) — clear wall ghost
                    if self._hovered is not None:
                        self._hovered = None
                context.area.tag_redraw()
                return {'PASS_THROUGH'}

            if pt is None:
                return {'RUNNING_MODAL', 'PASS_THROUGH'}

            if self._phase == 'DS':
                ex_reg = ROOM_OT_draw._room_list[self._ds_room_idx]
                wc     = self._ds_wall_char
                di     = self._ds_door_idx
                doors  = ex_reg.get("doors", [])
                if di is not None and di < len(doors):
                    dw = doors[di]["w"]
                    raw = pt.y if wc in ('E', 'W') else pt.x
                    anchor = _valid_anchor(raw, ex_reg, wc, dw, s.door_margin, skip_idx=di)
                    if anchor is not None:
                        doors[di]["anchor"] = anchor
                        if wc in ('E', 'W'):
                            self._start.y = anchor
                        else:
                            self._start.x = anchor
                        _rebuild_room_mesh(ex_reg, s)
                        self._end = self._start.copy()
                        context.area.tag_redraw()

            elif self._phase == 1:
                rooms_p1 = ROOM_OT_draw._room_list
                sidx_p1  = self._snap_info[2] if self._snap_info else -1
                if self._axis1_char == 'X':
                    nx = _clamp_x_for_rooms(pt.x, self._start.x,
                                            self._start.y, self._start.y,
                                            rooms_p1, sidx_p1, s, current_z=z)
                    if self._snap_info is not None:
                        nx = _enforce_snap_window_span(nx, self._start.x,
                                                       self._snap_info, rooms_p1, s)
                    self._end = Vector((nx, self._start.y, z))
                else:
                    ny = _clamp_y_for_rooms(pt.y, self._start.y,
                                            self._start.x, self._start.x,
                                            rooms_p1, sidx_p1, s, current_z=z)
                    if self._snap_info is not None:
                        ny = _enforce_snap_window_span(ny, self._start.y,
                                                       self._snap_info, rooms_p1, s)
                    self._end = Vector((self._start.x, ny, z))
                self._update_preview_mesh(context)

            elif self._phase == 2:
                if self._snap_info is not None:
                    rooms_p2 = ROOM_OT_draw._room_list
                    sidx_p2  = self._snap_info[2]
                    if self._axis1_char == 'X':
                        nx = _clamp_x_for_rooms(pt.x, self._start.x,
                                                self._start.y, self._start.y,
                                                rooms_p2, sidx_p2, s, current_z=z)
                        nx = _enforce_snap_window_span(nx, self._start.x,
                                                       self._snap_info, rooms_p2, s)
                        self._end = Vector((nx, self._start.y, z))
                    else:
                        ny = _clamp_y_for_rooms(pt.y, self._start.y,
                                                self._start.x, self._start.x,
                                                rooms_p2, sidx_p2, s, current_z=z)
                        ny = _enforce_snap_window_span(ny, self._start.y,
                                                       self._snap_info, rooms_p2, s)
                        self._end = Vector((self._start.x, ny, z))
                else:
                    rooms = ROOM_OT_draw._room_list
                    if self._axis1_char == 'X':
                        nx1 = min(self._start.x, self._phase1_end.x)
                        nx2 = max(self._start.x, self._phase1_end.x)
                        ny  = _clamp_y_for_rooms(pt.y, self._start.y,
                                                 nx1, nx2, rooms, -1, s, current_z=z)
                        self._end = Vector((self._phase1_end.x, ny, z))
                    else:
                        ny1 = min(self._start.y, self._phase1_end.y)
                        ny2 = max(self._start.y, self._phase1_end.y)
                        nx  = _clamp_x_for_rooms(pt.x, self._start.x,
                                                 ny1, ny2, rooms, -1, s, current_z=z)
                        self._end = Vector((nx, self._phase1_end.y, z))
                    _apply_snap_constraint(self._end, self._start, self._snap_info)
                self._update_preview_mesh(context)

            elif self._phase == 3:
                rooms    = ROOM_OT_draw._room_list
                snap_idx = self._snap_info[2] if self._snap_info else -1
                if self._axis1_char == 'X':
                    nx1 = min(self._start.x, self._end.x)
                    nx2 = max(self._start.x, self._end.x)
                    ny  = _clamp_y_for_rooms(pt.y, self._start.y,
                                             nx1, nx2, rooms, snap_idx, s, current_z=z)
                    self._end = Vector((self._end.x, ny, z))
                else:
                    ny1 = min(self._start.y, self._end.y)
                    ny2 = max(self._start.y, self._end.y)
                    nx  = _clamp_x_for_rooms(pt.x, self._start.x,
                                             ny1, ny2, rooms, snap_idx, s, current_z=z)
                    self._end = Vector((nx, self._end.y, z))
                _apply_snap_constraint(self._end, self._start, self._snap_info)
                self._update_preview_mesh(context)

            return {'RUNNING_MODAL', 'PASS_THROUGH'}

        # ── left mouse ────────────────────────────────────────────────────
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            in_region = (0 <= event.mouse_region_x < context.region.width and
                         0 <= event.mouse_region_y < context.region.height)
            if not in_region:
                return {'PASS_THROUGH'}
            pt = _ray_to_z(context, event, z)
            if pt is None:
                return {'RUNNING_MODAL'}

            # ── Click 1: lock start point ──────────────────────────────
            if self._phase == 0:
                snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list, s.wall_thickness, current_z=z)
                if snap:
                    sp, normal, room_idx, wc = snap
                    self._start      = Vector((sp.x, sp.y, z))
                    self._snap_info  = (normal, wc, room_idx)
                    self._axis1_char = 'Y' if wc in ('E', 'W') else 'X'
                    self._hovered    = None
                    if s.add_door:
                        # DS phase: add a new door to the existing room at the snap position
                        ex_reg = ROOM_OT_draw._room_list[room_idx]
                        self._ds_room_idx  = room_idx
                        self._ds_wall_char = wc
                        # Use active preset dims if available, else fall back to global settings
                        ds_dw = s.door_width
                        ds_dh = s.door_height
                        _active_p  = context.scene.room_active_door_preset
                        _presets_p = context.scene.room_door_presets
                        if 0 <= _active_p < len(_presets_p):
                            _dp = _presets_p[_active_p]
                            ds_dw, ds_dh = _dp.door_width, _dp.door_height
                        raw_anchor = sp.y if wc in ('E', 'W') else sp.x
                        anchor = _valid_anchor(raw_anchor, ex_reg, wc, ds_dw, s.door_margin)
                        if anchor is None:
                            # Window (or wall too narrow) blocks door — skip door, go to phase 1
                            self._ds_room_idx = self._ds_wall_char = self._ds_door_idx = None
                            self._end   = self._start.copy()
                            self._phase = 1
                            self._create_preview(context)
                            self._msg(context,
                                "No room for a door here — window is blocking  |  RMB – cancel")
                        else:
                            new_door = {"wc": wc, "anchor": anchor,
                                        "w": ds_dw, "h": ds_dh}
                            ex_reg.setdefault("doors", []).append(new_door)
                            self._ds_door_idx = len(ex_reg["doors"]) - 1
                            _rebuild_room_mesh(ex_reg, s)
                            self._end  = self._start.copy()
                            self._phase = 'DS'
                            context.area.tag_redraw()
                            self._msg(context,
                                "Slide door along wall · Click to confirm position  |  RMB – cancel")
                    else:
                        # No door: skip DS, go straight to drawing the room shape
                        self._ds_room_idx = self._ds_wall_char = self._ds_door_idx = None
                        self._end   = self._start.copy()
                        self._phase = 1
                        self._create_preview(context)
                        self._msg(context,
                            "Move to set first side  ·  Alt+S – snap to wall edge"
                            "  |  Click to confirm  |  RMB – cancel")
                else:
                    self._start      = Vector((pt.x, pt.y, z))
                    self._snap_info  = None
                    self._axis1_char = 'X'
                    self._end   = self._start.copy()
                    self._phase = 1
                    self._create_preview(context)
                    self._msg(context,
                        "Move to set east/west width  |  Click to confirm  |  RMB – cancel")
                return {'RUNNING_MODAL'}

            # ── Click DS: confirm door position → phase 1 ─────────────
            if self._phase == 'DS':
                self._end   = self._start.copy()
                self._phase = 1
                self._create_preview(context)
                self._msg(context,
                    "Move to set first side  ·  Alt+S – snap to wall edge"
                    "  |  Click to confirm  |  RMB – cancel")
                return {'RUNNING_MODAL'}

            # ── Click 2: lock first side ───────────────────────────────
            if self._phase == 1:
                axis_idx = 1 if self._axis1_char == 'Y' else 0
                if abs(self._end[axis_idx] - self._start[axis_idx]) < 0.05:
                    self._msg(context, "Too small — move further first")
                    return {'RUNNING_MODAL'}
                self._phase1_end = self._end.copy()
                if self._snap_info is not None:
                    if self._axis1_char == 'Y':
                        self._start.y = self._phase1_end.y
                    else:
                        self._start.x = self._phase1_end.x
                    self._end   = self._start.copy()
                    self._phase = 2
                    self._msg(context,
                        "Move to set other side  ·  Alt+S – snap to wall edge"
                        "  |  Click to confirm  |  RMB – cancel")
                else:
                    self._phase = 2
                    self._msg(context,
                        "Move to set north/south depth  |  Click to place  |  RMB – cancel")
                return {'RUNNING_MODAL'}

            # ── Click 3: lock second side (snapped) OR place (standalone) ──
            if self._phase == 2:
                if self._snap_info is not None:
                    axis_idx = 1 if self._axis1_char == 'Y' else 0
                    if abs(self._end[axis_idx] - self._start[axis_idx]) < 0.05:
                        self._msg(context, "Too small — move further first")
                        return {'RUNNING_MODAL'}
                    snap_perp = self._start.x if self._axis1_char == 'Y' else self._start.y
                    if self._axis1_char == 'Y':
                        y_lo = min(self._start.y, self._end.y)
                        y_hi = max(self._start.y, self._end.y)
                        self._start = Vector((snap_perp, y_lo, z))
                        self._end   = Vector((snap_perp, y_hi, z))
                    else:
                        x_lo = min(self._start.x, self._end.x)
                        x_hi = max(self._start.x, self._end.x)
                        self._start = Vector((x_lo, snap_perp, z))
                        self._end   = Vector((x_hi, snap_perp, z))
                    self._phase = 3
                    self._msg(context,
                        "Move to set depth  |  Click to place  |  RMB – cancel")
                    return {'RUNNING_MODAL'}

            # ── Click 4 (snapped) / Click 3 (standalone): place room ────
            if self._phase in (2, 3):
                x1 = min(self._start.x, self._end.x)
                y1 = min(self._start.y, self._end.y)
                x2 = max(self._start.x, self._end.x)
                y2 = max(self._start.y, self._end.y)

                if x2 - x1 > 0.05 and y2 - y1 > 0.05:
                    nw              = _snap_no_walls(self._snap_info)
                    preview_doors   = _preview_doors(self._snap_info)
                    preview_windows = _preview_windows(self._snap_info)

                    ex_reg         = None
                    prev_ds_door_idx = None
                    if self._snap_info is not None:
                        _, wc, snap_idx = self._snap_info
                        if 0 <= snap_idx < len(ROOM_OT_draw._room_list):
                            ex_reg           = ROOM_OT_draw._room_list[snap_idx]
                            prev_ds_door_idx = self._ds_door_idx

                    n        = len(ROOM_OT_draw._room_list) + 1
                    room_col = _room_target_collection(context, n)
                    obj      = _make_room_obj(f"Room.{n:03}", x1, y1, x2, y2, s,
                                              preview_doors, nw, preview_windows,
                                              collection=room_col)
                    reg = {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "t":  s.wall_thickness,
                        "z":  s.z_foundation,
                        "doors":    list(preview_doors),
                        "no_walls": list(nw),
                        "windows":  list(preview_windows),
                        "obj_name": obj.name,
                    }
                    ROOM_OT_draw._room_list.append(reg)
                    self._session_rooms.append((obj, reg, ex_reg, prev_ds_door_idx))
                    _sync_to_scene(context)
                    for o in list(context.selected_objects):
                        o.select_set(False)
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    self.report({"INFO"}, f"Created {obj.name}")
                else:
                    self._msg(context, "Room too small — skipped")

                # Clear DS state so _reset_placement doesn't revert the placed door
                self._ds_room_idx = self._ds_wall_char = self._ds_door_idx = None
                self._reset_placement(context)
                self._idle_msg(context)
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        self._delete_preview()
        self._remove_draw_handle()
        self._msg(context, None)


# ── module-level helpers ───────────────────────────────────────────────────────

def _apply_snap_constraint(end, start, snap_info):
    if snap_info is None:
        return
    (nx, ny), _wc, _idx = snap_info
    if nx != 0:
        if nx > 0:
            end.x = max(end.x, start.x + 0.05)
        else:
            end.x = min(end.x, start.x - 0.05)
    else:
        if ny > 0:
            end.y = max(end.y, start.y + 0.05)
        else:
            end.y = min(end.y, start.y - 0.05)


def _preview_doors(snap_info):
    """Return list of door dicts for the new room's connecting wall.
    All existing doors on the snapped wall are always mirrored onto the
    new room's opposite wall, regardless of add_door setting."""
    if snap_info is None:
        return []
    _normal, wc, room_idx = snap_info
    rooms = ROOM_OT_draw._room_list
    if room_idx < 0 or room_idx >= len(rooms):
        return []
    ex  = rooms[room_idx]
    opp = _OPPOSITE[wc]
    return [{"wc": opp, "anchor": d["anchor"], "w": d["w"], "h": d["h"]}
            for d in ex.get("doors", []) if d["wc"] == wc]


def _preview_windows(snap_info):
    """Return list of window dicts for the new room's connecting wall.
    All existing windows on the snapped wall are mirrored onto the
    new room's opposite wall so they appear on both sides."""
    if snap_info is None:
        return []
    _normal, wc, room_idx = snap_info
    rooms = ROOM_OT_draw._room_list
    if room_idx < 0 or room_idx >= len(rooms):
        return []
    ex  = rooms[room_idx]
    opp = _OPPOSITE[wc]
    return [{"wc": opp, "anchor": w["anchor"], "v_offset": w.get("v_offset", 0.9),
             "w": w["w"], "h": w["h"]}
            for w in ex.get("windows", []) if w["wc"] == wc]


def _snap_no_walls(snap_info):
    return ()


@bpy.app.handlers.persistent
def _room_registry_cleanup(scene, depsgraph):
    """Remove rooms whose objects no longer exist (deleted or not yet restored by undo)."""
    ROOM_OT_draw._room_list[:] = [
        r for r in ROOM_OT_draw._room_list
        if r.get("obj_name", "") in bpy.data.objects
    ]


@bpy.app.handlers.persistent
def _room_on_load(*args):
    ROOM_OT_draw._room_list.clear()


# ═════════════════════════════════════════════════════════════════════════════
# Utility operator – clear snap registry
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_clear(bpy.types.Operator):
    bl_idname    = "room.clear_registry"
    bl_label     = "Reset Snap Registry"
    bl_description = ("Clear the internal room list used for wall-snap detection.\n"
                      "Use this if you delete or move rooms manually.")

    def execute(self, context):
        ROOM_OT_draw._room_list.clear()
        self.report({"INFO"}, "Room snap registry cleared")
        return {"FINISHED"}


class ROOM_OT_clear_overlays(bpy.types.Operator):
    bl_idname     = "room.clear_overlays"
    bl_label      = "Clear Phantom Overlays"
    bl_description = ("Remove any stuck GPU overlays left by the room tool.\n"
                      "Use this if orange/green highlights remain after exiting a mode.")

    def execute(self, context):
        removed = 0
        for handle in list(_DRAW_HANDLES):
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
                removed += 1
            except Exception:
                pass
        _DRAW_HANDLES.clear()
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        self.report({'INFO'}, f"Cleared {removed} overlay handle(s)")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════════
# Unified door-frame edit-mode operator
# LMB click        → add door (no drag)
# LMB hold + drag  → slide door along wall
# Double-click     → remove door
# LMB hold + scroll→ cycle preset live
# RMB / Esc        → cancel / exit
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_door_edit(bpy.types.Operator):
    bl_idname  = "room.door_edit"
    bl_label   = "Door Frame Edit Mode"
    bl_description = ("Edit door frames.  "
                      "LMB=add  ·  LMB+drag=slide  ·  Double-click=remove  ·  "
                      "LMB+scroll=change preset  |  RMB/Esc=exit")
    bl_options = {"REGISTER", "UNDO"}

    _DRAG_PX = 8

    # ── GPU draw callback ──────────────────────────────────────────────────
    def _draw_cb(self, context):
        s     = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')
        idx_fill = [(0, 1, 2), (0, 2, 3)]
        idx_line = [(0, 1), (1, 2), (2, 3), (3, 0)]

        if self._hovered_door is not None:
            ri, di = self._hovered_door
            if ri < len(rooms):
                r    = rooms[ri]
                # Guard: skip if room object deleted or hidden
                _obj = bpy.data.objects.get(r.get("obj_name", ""))
                if _obj is None or not _obj.visible_get():
                    self._hovered_door = None
                    gpu.state.blend_set('NONE')
                    gpu.state.depth_test_set('NONE')
                    return
                doors = r.get("doors", [])
                if di < len(doors):
                    door  = doors[di]
                    dw, dh = door["w"], door["h"]
                    verts = _door_frame_verts(r, door, r.get("z", s.z_foundation), dw, dh)
                    b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                    shader.uniform_float("color", (0.05, 0.85, 0.2, 0.30))
                    b.draw(shader)
                    gpu.state.line_width_set(2.5)
                    b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                    shader.uniform_float("color", (0.1, 1.0, 0.3, 1.0))
                    b2.draw(shader)

        elif self._hover_snap is not None:
            ri, wc, anchor = self._hover_snap
            if ri < len(rooms):
                r  = rooms[ri]
                # Guard: skip if room object deleted or hidden
                _obj = bpy.data.objects.get(r.get("obj_name", ""))
                if _obj is None or not _obj.visible_get():
                    self._hover_snap = None
                    gpu.state.blend_set('NONE')
                    gpu.state.depth_test_set('NONE')
                    return
                dw = s.door_width
                dh = s.door_height
                active  = context.scene.room_active_door_preset
                presets = context.scene.room_door_presets
                if 0 <= active < len(presets):
                    dp = presets[active]
                    dw, dh = dp.door_width, dp.door_height
                ghost = dict(r)
                ghost_door = {"wc": wc, "anchor": anchor, "w": dw, "h": dh}
                verts = _door_frame_verts(ghost, ghost_door,
                                          r.get("z", s.z_foundation), dw, dh)
                # Check whether this position is blocked by a window (or too narrow)
                is_blocked = (_valid_anchor(anchor, r, wc, dw, s.door_margin) is None)
                b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                if is_blocked:
                    shader.uniform_float("color", (1.0, 0.08, 0.05, 0.28))
                else:
                    shader.uniform_float("color", (0.05, 0.85, 0.2, 0.25))
                b.draw(shader)
                gpu.state.line_width_set(2.0)
                b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                if is_blocked:
                    shader.uniform_float("color", (1.0, 0.12, 0.05, 1.0))
                else:
                    shader.uniform_float("color", (0.1, 1.0, 0.3, 0.9))
                b2.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def _add_draw_handle(self, context):
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_VIEW')
        _DRAW_HANDLES.add(self._draw_handle)

    def _remove_draw_handle(self):
        if self._draw_handle:
            _DRAW_HANDLES.discard(self._draw_handle)
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            except Exception:
                pass
            self._draw_handle = None

    # ── helpers ───────────────────────────────────────────────────────────
    def _header(self, context):
        context.area.header_text_set(
            "LMB=add  ·  LMB+drag=slide  ·  Double-click=remove  ·  "
            "LMB+scroll=preset  |  RMB/Esc=exit")

    def _restore_preview(self, context, rooms, s):
        """Revert preset dims saved in _preview_orig back to the doors."""
        if self._preview_orig is None:
            return
        poi = self._preview_orig
        ri, di = poi["room_idx"], poi["door_idx"]
        if ri < len(rooms):
            doors = rooms[ri].get("doors", [])
            if di < len(doors):
                doors[di]["w"] = poi["dims"]["w"]
                doors[di]["h"] = poi["dims"]["h"]
                _rebuild_room_mesh(rooms[ri], s)
        if poi["partner"] is not None:
            p_ri, p_di, p_dims = poi["partner"]
            if p_ri < len(rooms):
                p_doors = rooms[p_ri].get("doors", [])
                if p_di < len(p_doors):
                    p_doors[p_di]["w"] = p_dims["w"]
                    p_doors[p_di]["h"] = p_dims["h"]
                    _rebuild_room_mesh(rooms[p_ri], s)
        _sync_to_scene(context)
        self._preview_orig = None

    def _remove_active_door(self, context, rooms, s):
        """Remove _active_idx/_active_door_idx door (and partner if any)."""
        ri, di = self._active_idx, self._active_door_idx
        if ri is None or di is None:
            return
        # Find partner BEFORE removing (indices shift after pop)
        partner = _find_partner_door(rooms, ri, di, s.wall_thickness)
        reg = rooms[ri]
        if di < len(reg.get("doors", [])):
            reg["doors"].pop(di)
        _rebuild_room_mesh(reg, s)
        if partner is not None:
            p_ri, p_di = partner
            p_reg = rooms[p_ri]
            if p_di < len(p_reg.get("doors", [])):
                p_reg["doors"].pop(p_di)
            _rebuild_room_mesh(p_reg, s)
        _sync_to_scene(context)

    def _go_hover(self, context):
        self._phase          = 'HOVER'
        self._active_idx     = None
        self._active_wc      = None
        self._active_door_idx = None
        self._active_partner = None
        self._orig_anchor    = None
        self._added_in_press = False
        self._preview_orig   = None
        self._press_screen   = None
        self._hovered_door   = None
        self._hover_snap     = None
        self._header(context)
        context.area.tag_redraw()

    def _finish(self, context):
        self._remove_draw_handle()
        context.area.header_text_set(None)
        context.scene.room_door_edit_active = False
        if context.area:
            context.area.tag_redraw()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def invoke(self, context, event):
        if context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "Must be used inside the 3D Viewport")
            return {"CANCELLED"}
        _sync_from_scene(context)
        self._phase           = 'HOVER'
        self._hovered_door    = None
        self._hover_snap      = None
        self._last_press_time = 0.0
        self._last_press_door = None
        self._press_screen    = None
        self._active_idx      = None
        self._active_wc       = None
        self._active_door_idx = None
        self._active_partner  = None
        self._orig_anchor     = None
        self._added_in_press  = False
        self._preview_orig    = None
        self._draw_handle     = None
        self._add_draw_handle(context)
        context.window_manager.modal_handler_add(self)
        context.scene.room_door_edit_active = True
        if context.area:
            context.area.tag_redraw()
        self._header(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        s     = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list

        # ── Navigation pass-through ────────────────────────────────────────
        if event.type in {'MIDDLEMOUSE',
                'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3',
                'NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7',
                'NUMPAD_8','NUMPAD_9','NUMPAD_DECIMAL','NUMPAD_PERIOD',
                'F','TILDE'}:
            return {'PASS_THROUGH'}

        # When mouse is over the N-panel or any non-viewport region, let all
        # events pass through so the user can adjust settings mid-edit.
        # ESC/RMB are exempt so they can still cancel the modal from anywhere.
        if context.region.type != 'WINDOW':
            if event.type not in {'ESC', 'RIGHTMOUSE'}:
                return {'PASS_THROUGH'}

        # ── Scroll — cycle presets while LMB is held ───────────────────────
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
            if (self._phase in ('LMB_DOWN', 'SLIDING') and
                    self._active_idx is not None and self._active_door_idx is not None):
                presets = context.scene.room_door_presets
                active  = context.scene.room_active_door_preset
                if len(presets) > 0:
                    delta      = 1 if event.type == 'WHEELUPMOUSE' else -1
                    new_active = (active + delta) % len(presets)
                    context.scene.room_active_door_preset = new_active
                    dp   = presets[new_active]
                    reg  = rooms[self._active_idx]
                    di   = self._active_door_idx
                    doors = reg.get("doors", [])
                    if di < len(doors):
                        # Save original dims on first scroll
                        if self._preview_orig is None:
                            orig_dims = {"w": doors[di]["w"], "h": doors[di]["h"]}
                            partner_info = None
                            if self._active_partner is not None:
                                p_ri, p_di = self._active_partner
                                p_doors = rooms[p_ri].get("doors", [])
                                if p_di < len(p_doors):
                                    partner_info = (p_ri, p_di,
                                                    {"w": p_doors[p_di]["w"],
                                                     "h": p_doors[p_di]["h"]})
                            self._preview_orig = {
                                "room_idx": self._active_idx,
                                "door_idx": di,
                                "dims":     orig_dims,
                                "partner":  partner_info,
                            }
                        doors[di]["w"] = dp.door_width
                        doors[di]["h"] = dp.door_height
                        _rebuild_room_mesh(reg, s)
                        if self._active_partner is not None:
                            p_ri, p_di = self._active_partner
                            p_doors = rooms[p_ri].get("doors", [])
                            if p_di < len(p_doors):
                                p_doors[p_di]["w"] = dp.door_width
                                p_doors[p_di]["h"] = dp.door_height
                                _rebuild_room_mesh(rooms[p_ri], s)
                        _sync_to_scene(context)
                        context.area.header_text_set(
                            f"Preset: {dp.name} ({dp.door_width:.2f}×{dp.door_height:.2f}m)  |  "
                            "Scroll to cycle  ·  Release LMB to confirm  |  RMB=cancel")
                        context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # ── Exit ──────────────────────────────────────────────────────────
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == 'PRESS':
            if self._phase != 'HOVER':
                _sync_to_scene(context)
            self._finish(context)
            return {"FINISHED"}

        if event.type in {"RIGHTMOUSE", "ESC"} and event.value == 'PRESS':
            if self._phase != 'HOVER':
                self._restore_preview(context, rooms, s)
                if (self._active_idx is not None and
                        self._active_door_idx is not None and
                        self._orig_anchor is not None):
                    reg   = rooms[self._active_idx]
                    doors = reg.get("doors", [])
                    di    = self._active_door_idx
                    if di < len(doors):
                        doors[di]["anchor"] = self._orig_anchor
                        _rebuild_room_mesh(reg, s)
                        if self._active_partner is not None:
                            p_ri, p_di = self._active_partner
                            p_doors = rooms[p_ri].get("doors", [])
                            if p_di < len(p_doors):
                                p_doors[p_di]["anchor"] = self._orig_anchor
                                _rebuild_room_mesh(rooms[p_ri], s)
                if self._added_in_press:
                    self._remove_active_door(context, rooms, s)
                else:
                    _sync_to_scene(context)
                self._go_hover(context)
                return {'RUNNING_MODAL'}
            self._finish(context)
            return {"CANCELLED"}

        # ── Mouse move ────────────────────────────────────────────────────
        if event.type == "MOUSEMOVE":
            pt = _ray_to_z(context, event, s.z_foundation)

            if self._phase == 'HOVER':
                if pt:
                    self._hovered_door = _door_snap(pt, rooms, s, s.wall_thickness)
                    self._hover_snap   = (_wall_snap_any(pt, rooms, s.wall_thickness, current_z=s.z_foundation)
                                         if self._hovered_door is None else None)
                    # Show a warning header when the ghost is in a window-blocked zone
                    if self._hover_snap is not None and self._hovered_door is None:
                        ri, wc, anchor = self._hover_snap
                        if 0 <= ri < len(rooms):
                            dw = s.door_width
                            active  = context.scene.room_active_door_preset
                            presets = context.scene.room_door_presets
                            if 0 <= active < len(presets):
                                dw = presets[active].door_width
                            if _valid_anchor(anchor, rooms[ri], wc, dw, s.door_margin) is None:
                                context.area.header_text_set(
                                    "Cannot place door here — window is blocking")
                            else:
                                self._header(context)
                        else:
                            self._header(context)
                    else:
                        self._header(context)
                else:
                    self._hovered_door = None
                    self._hover_snap   = None
                    self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL', 'PASS_THROUGH'}

            elif self._phase == 'LMB_DOWN':
                if (self._press_screen and
                        (abs(event.mouse_region_x - self._press_screen[0]) > self._DRAG_PX or
                         abs(event.mouse_region_y - self._press_screen[1]) > self._DRAG_PX)):
                    self._phase = 'SLIDING'
                    context.area.header_text_set(
                        "Sliding door  ·  Scroll to change preset  |  "
                        "Release to confirm  ·  RMB=cancel")
                else:
                    return {'RUNNING_MODAL'}

            if (self._phase == 'SLIDING' and pt is not None and
                    self._active_idx is not None and self._active_door_idx is not None):
                reg   = rooms[self._active_idx]
                doors = reg.get("doors", [])
                di    = self._active_door_idx
                if di < len(doors):
                    dw     = doors[di]["w"]
                    raw    = pt.y if self._active_wc in ('E', 'W') else pt.x
                    anchor = _valid_anchor(raw, reg, self._active_wc, dw,
                                          s.door_margin, skip_idx=di)
                    if anchor is not None:
                        doors[di]["anchor"] = anchor
                        _rebuild_room_mesh(reg, s)
                        if self._active_partner is not None:
                            p_ri, p_di = self._active_partner
                            p_doors = rooms[p_ri].get("doors", [])
                            if p_di < len(p_doors):
                                p_doors[p_di]["anchor"] = anchor
                                _rebuild_room_mesh(rooms[p_ri], s)
                        context.area.tag_redraw()

            return {'RUNNING_MODAL'}

        # ── Left mouse ────────────────────────────────────────────────────
        if event.type == "LEFTMOUSE":
            in_region = (0 <= event.mouse_region_x < context.region.width and
                         0 <= event.mouse_region_y < context.region.height)

            if event.value == "PRESS":
                if not in_region:
                    return {'PASS_THROUGH'}

                pt  = _ray_to_z(context, event, s.z_foundation)
                now = time.time()

                # ── Double-click: remove the door pressed last ─────────
                is_double = ((now - self._last_press_time) < 0.3 and
                             self._last_press_door is not None)
                if is_double and pt is not None:
                    current_door = _door_snap(pt, rooms, s, s.wall_thickness)
                    if current_door is not None and current_door == self._last_press_door:
                        ri, di = current_door
                        self._active_idx      = ri
                        self._active_door_idx = di
                        self._active_wc       = rooms[ri]["doors"][di]["wc"]
                        self._active_partner  = _find_partner_door(rooms, ri, di, s.wall_thickness)
                        self._added_in_press  = True   # treat as "newly added" for removal
                        self._remove_active_door(context, rooms, s)
                        self._last_press_time = 0.0
                        self._last_press_door = None
                        self._hovered_door    = None
                        self._go_hover(context)
                        return {'RUNNING_MODAL'}

                # ── Single press ──────────────────────────────────────
                self._last_press_time = now
                self._press_screen    = (event.mouse_region_x, event.mouse_region_y)

                if pt is None:
                    return {'PASS_THROUGH'}

                if self._hovered_door is not None:
                    # Press on existing door → ready to slide / change preset
                    ri, di = self._hovered_door
                    self._last_press_door = (ri, di)
                    self._active_idx      = ri
                    self._active_door_idx = di
                    self._active_wc       = rooms[ri]["doors"][di]["wc"]
                    self._active_partner  = _find_partner_door(rooms, ri, di, s.wall_thickness)
                    door = rooms[ri]["doors"][di]
                    self._orig_anchor     = door["anchor"]
                    self._added_in_press  = False
                    self._preview_orig    = None
                    self._phase           = 'LMB_DOWN'
                    context.area.header_text_set(
                        "Hold+drag=slide  ·  Hold+scroll=change preset  |  "
                        "Release=confirm  ·  RMB=cancel")

                elif self._hover_snap is not None:
                    # Press on wall → add new door
                    ri, wc, raw_anchor = self._hover_snap
                    self._last_press_door = None
                    if 0 <= ri < len(rooms):
                        reg = rooms[ri]
                        dw  = s.door_width
                        dh  = s.door_height
                        active  = context.scene.room_active_door_preset
                        presets = context.scene.room_door_presets
                        if 0 <= active < len(presets):
                            dp = presets[active]
                            dw, dh = dp.door_width, dp.door_height
                        anchor = _valid_anchor(raw_anchor, reg, wc, dw, s.door_margin)
                        if anchor is None:
                            context.area.header_text_set(
                                "No room for a door here — try a different spot")
                            self._last_press_door = None
                            return {'RUNNING_MODAL'}
                        new_door = {"wc": wc, "anchor": anchor, "w": dw, "h": dh}
                        reg.setdefault("doors", []).append(new_door)
                        new_di = len(reg["doors"]) - 1
                        _rebuild_room_mesh(reg, s)
                        partner = _find_partner_wall(rooms, ri, wc, s.wall_thickness)
                        partner_door_idx = None
                        if partner is not None:
                            p_ri, p_wc = partner
                            p_reg = rooms[p_ri]
                            p_door = {"wc": p_wc, "anchor": anchor, "w": dw, "h": dh}
                            p_reg.setdefault("doors", []).append(p_door)
                            partner_door_idx = len(p_reg["doors"]) - 1
                            _rebuild_room_mesh(p_reg, s)
                        self._active_idx      = ri
                        self._active_wc       = wc
                        self._active_door_idx = new_di
                        self._active_partner  = (partner[0], partner_door_idx) if partner and partner_door_idx is not None else None
                        self._orig_anchor     = anchor
                        self._added_in_press  = True
                        self._preview_orig    = None
                        self._phase           = 'LMB_DOWN'
                        _sync_to_scene(context)
                        context.area.tag_redraw()
                        context.area.header_text_set(
                            "Hold+drag=slide  ·  Hold+scroll=change preset  |  "
                            "Release=confirm  ·  RMB=cancel")
                else:
                    self._last_press_door = None
                    return {'PASS_THROUGH'}

                return {'RUNNING_MODAL'}

            elif event.value == "RELEASE":
                if self._phase in ('LMB_DOWN', 'SLIDING'):
                    self._preview_orig = None   # dims confirmed
                    _sync_to_scene(context)
                    self._go_hover(context)
                return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL', 'PASS_THROUGH'}

    def cancel(self, context):
        self._remove_draw_handle()
        context.area.header_text_set(None)
        context.scene.room_door_edit_active = False
        if context.area:
            context.area.tag_redraw()


# ═════════════════════════════════════════════════════════════════════════════
# Window-frame edit operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_window_edit(bpy.types.Operator):
    bl_idname  = "room.window_edit"
    bl_label   = "Window Frame Edit Mode"
    bl_description = ("Edit window frames.  "
                      "LMB=place (click H then V)  ·  LMB+drag=slide  ·  "
                      "Double-click=remove  |  RMB/Esc=exit")
    bl_options = {"REGISTER", "UNDO"}

    _DRAG_PX = 8

    # ── GPU draw callback ──────────────────────────────────────────────────
    def _draw_cb(self, context):
        s     = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')
        idx_fill = [(0, 1, 2), (0, 2, 3)]
        idx_line = [(0, 1), (1, 2), (2, 3), (3, 0)]

        # ── Active window(s) (being placed or slid) — always drawn live ────────
        if (self._phase in ('SLIDING', 'H_POSITION', 'V_POSITION') and
                self._active_idx is not None and self._active_win_idx is not None):
            ri = self._active_idx
            if ri < len(rooms):
                r     = rooms[ri]
                # Guard: skip if room object deleted or hidden
                _obj = bpy.data.objects.get(r.get("obj_name", ""))
                if _obj is None or not _obj.visible_get():
                    self._active_idx = None
                    gpu.state.blend_set('NONE')
                    gpu.state.depth_test_set('NONE')
                    return
                wins  = r.get("windows", [])
                count = getattr(self, '_array_count', 1)
                for k in range(count):
                    wi_k = self._active_win_idx + k
                    if wi_k < len(wins):
                        win    = wins[wi_k]
                        ww, wh = win["w"], win["h"]
                        verts  = _window_frame_verts(r, win, r.get("z", s.z_foundation), ww, wh)
                        b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                        shader.uniform_float("color", (0.05, 0.85, 0.2, 0.30))
                        b.draw(shader)
                        gpu.state.line_width_set(2.5)
                        b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                        shader.uniform_float("color", (0.1, 1.0, 0.3, 1.0))
                        b2.draw(shader)

        elif self._hovered_win is not None:
            ri, wi = self._hovered_win
            if ri < len(rooms):
                r    = rooms[ri]
                # Guard: skip if room object deleted or hidden
                _obj = bpy.data.objects.get(r.get("obj_name", ""))
                if _obj is None or not _obj.visible_get():
                    self._hovered_win = None
                    gpu.state.blend_set('NONE')
                    gpu.state.depth_test_set('NONE')
                    return
                wins = r.get("windows", [])
                if wi < len(wins):
                    win  = wins[wi]
                    ww, wh = win["w"], win["h"]
                    verts = _window_frame_verts(r, win, r.get("z", s.z_foundation), ww, wh)
                    b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                    shader.uniform_float("color", (0.05, 0.85, 0.2, 0.30))
                    b.draw(shader)
                    gpu.state.line_width_set(2.5)
                    b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                    shader.uniform_float("color", (0.1, 1.0, 0.3, 1.0))
                    b2.draw(shader)

        elif self._hover_snap is not None:
            ri, wc, anchor = self._hover_snap
            if ri < len(rooms):
                r  = rooms[ri]
                # Guard: skip if room object deleted or hidden
                _obj = bpy.data.objects.get(r.get("obj_name", ""))
                if _obj is None or not _obj.visible_get():
                    self._hover_snap = None
                    gpu.state.blend_set('NONE')
                    gpu.state.depth_test_set('NONE')
                    return
                ww = s.window_width
                wh = s.window_height
                voff = s.window_v_offset
                active  = context.scene.room_active_window_preset
                presets = context.scene.room_window_presets
                if 0 <= active < len(presets):
                    wp = presets[active]
                    ww, wh, voff = wp.window_width, wp.window_height, wp.v_offset
                count   = s.window_array_count
                gap     = s.window_array_gap
                count   = min(count, max(1, _max_array_count(rooms[ri], wc, ww, gap, s.door_margin)))
                anchors = _array_anchors(anchor, count, ww, gap, rooms[ri], wc, s.door_margin)
                # Green when every anchor is valid, red if any is blocked.
                # Draw at zone-clamped positions so the ghost shows exactly where
                # the window will land (snapped away from the adjacent-room zone).
                _t_cb = r.get("t", s.wall_thickness)
                _zones_cb = _wall_adj_zones(r, wc, rooms, _t_cb)
                display_anchors = [_clamp_window_anchor(anch, r, wc, ww, s.door_margin,
                                                        zones=_zones_cb)
                                   for anch in anchors]
                is_blocked = any(
                    _valid_window_anchor(anch, r, wc, ww, s.door_margin,
                                         zones=_zones_cb) is None
                    for anch in anchors
                )
                for anch in display_anchors:
                    ghost_win = {"wc": wc, "anchor": anch, "v_offset": voff, "w": ww, "h": wh}
                    verts = _window_frame_verts(r, ghost_win, r.get("z", s.z_foundation), ww, wh)
                    b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                    if is_blocked:
                        shader.uniform_float("color", (1.0, 0.08, 0.05, 0.28))
                    else:
                        shader.uniform_float("color", (0.05, 0.85, 0.2, 0.25))
                    b.draw(shader)
                    gpu.state.line_width_set(2.0)
                    b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                    if is_blocked:
                        shader.uniform_float("color", (1.0, 0.12, 0.05, 1.0))
                    else:
                        shader.uniform_float("color", (0.1, 1.0, 0.3, 0.9))
                    b2.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def _add_draw_handle(self, context):
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_VIEW')
        _DRAW_HANDLES.add(self._draw_handle)

    def _remove_draw_handle(self):
        if self._draw_handle:
            _DRAW_HANDLES.discard(self._draw_handle)
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._draw_handle, 'WINDOW')
            except Exception:
                pass
            self._draw_handle = None

    # ── helpers ───────────────────────────────────────────────────────────
    def _header(self, context):
        context.area.header_text_set(
            "LMB=place (click H · click V)  ·  LMB+drag=slide  ·  "
            "Double-click=remove  ·  Scroll=preset  |  RMB/Esc=exit")

    def _remove_active_win(self, context, rooms, s):
        ri, wi = self._active_idx, self._active_win_idx
        if ri is None or wi is None:
            return
        reg  = rooms[ri]
        wins = reg.get("windows", [])
        # Derive array start and count from stored metadata so deletion is
        # correct regardless of how self._array_count was reset by _go_hover().
        _cw   = wins[wi] if wi < len(wins) else {}
        count = _cw.get("array_n", self._array_count)
        wi    = wi - _cw.get("array_idx", 0)   # rewind to first window in array
        for k in reversed(range(count)):
            if wi + k < len(wins):
                wins.pop(wi + k)
        _rebuild_room_mesh(reg, s)
        if self._active_partner is not None:
            p_ri, p_wi = self._active_partner
            p_reg  = rooms[p_ri]
            p_wins = p_reg.get("windows", [])
            _p_cw   = p_wins[p_wi] if p_wi < len(p_wins) else {}
            p_count = _p_cw.get("array_n", count)
            p_wi    = p_wi - _p_cw.get("array_idx", 0)   # rewind
            for k in reversed(range(p_count)):
                if p_wi + k < len(p_wins):
                    p_wins.pop(p_wi + k)
            _rebuild_room_mesh(p_reg, s)
        _sync_to_scene(context)

    def _resize_array(self, context, rooms, s, new_count):
        """Remove the current array and re-insert with new_count windows."""
        ri, wi = self._active_idx, self._active_win_idx
        reg  = rooms[ri]
        wins = reg.get("windows", [])
        if not wins or wi >= len(wins):
            return
        # Read stored array size so resize is correct even after _go_hover reset.
        # wi (= self._active_win_idx) is already the first window of the array
        # (set to first_wi at placement time), so no rewind is needed here.
        _cw_r     = wins[wi]
        old_count = _cw_r.get("array_n", self._array_count)
        proto    = _cw_r.copy()
        ww, wh   = proto["w"], proto["h"]
        voff, wc = proto["v_offset"], proto["wc"]
        gap      = s.window_array_gap
        center   = proto["anchor"] + (old_count - 1) * (ww + gap) / 2
        for k in reversed(range(old_count)):
            if wi + k < len(wins):
                wins.pop(wi + k)
        p_ri = p_wi = None
        if self._active_partner is not None:
            p_ri, p_wi = self._active_partner
            p_wins = rooms[p_ri].get("windows", [])
            for k in reversed(range(old_count)):
                if p_wi + k < len(p_wins):
                    p_wins.pop(p_wi + k)
        anchors = _array_anchors(center, new_count, ww, gap, reg, wc, s.door_margin)
        for k, anch in enumerate(anchors):
            wins.insert(wi + k, {"wc": wc, "anchor": anch,
                                 "v_offset": voff, "w": ww, "h": wh,
                                 "array_n": new_count, "array_idx": k})
        _rebuild_room_mesh(reg, s)
        if p_ri is not None and p_wi is not None:
            p_wc   = _OPPOSITE[wc]
            p_wins = rooms[p_ri].get("windows", [])
            for k, anch in enumerate(anchors):
                p_wins.insert(p_wi + k, {"wc": p_wc, "anchor": anch,
                                         "v_offset": voff, "w": ww, "h": wh,
                                         "array_n": new_count, "array_idx": k})
            _rebuild_room_mesh(rooms[p_ri], s)
        self._array_count    = new_count
        s.window_array_count = new_count
        _sync_to_scene(context)

    def _go_hover(self, context):
        self._phase          = 'HOVER'
        self._active_idx     = None
        self._active_wc      = None
        self._active_win_idx = None
        self._active_partner = None
        self._orig_anchor    = None
        self._added_in_press = False
        self._press_screen   = None
        self._hovered_win    = None
        self._hover_snap     = None
        self._array_count    = 1
        self._header(context)
        context.area.tag_redraw()

    def _finish(self, context):
        self._remove_draw_handle()
        context.area.header_text_set(None)
        context.scene.room_window_edit_active = False
        if context.area:
            context.area.tag_redraw()

    def _get_preset_dims(self, context):
        s       = context.scene.room_settings
        ww, wh, voff = s.window_width, s.window_height, s.window_v_offset
        active  = context.scene.room_active_window_preset
        presets = context.scene.room_window_presets
        if 0 <= active < len(presets):
            wp = presets[active]
            ww, wh, voff = wp.window_width, wp.window_height, wp.v_offset
        return ww, wh, voff

    def _wall_coord_for(self, reg, wc):
        """Return the inner-face coordinate for ray-to-wall projection."""
        if wc == 'S': return reg["y1"]
        if wc == 'N': return reg["y2"]
        if wc == 'W': return reg["x1"]
        return reg["x2"]  # 'E'

    # ── lifecycle ──────────────────────────────────────────────────────────
    def invoke(self, context, event):
        if context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "Must be used inside the 3D Viewport")
            return {"CANCELLED"}
        _sync_from_scene(context)
        self._phase           = 'HOVER'
        self._hovered_win     = None
        self._hover_snap      = None
        self._last_press_time = 0.0
        self._last_press_win  = None
        self._press_screen    = None
        self._active_idx      = None
        self._active_wc       = None
        self._active_win_idx  = None
        self._active_partner  = None
        self._orig_anchor     = None
        self._added_in_press  = False
        self._array_count     = 1
        self._undo_stack      = []   # list of [(ri, [win_dicts, ...]), ...]
        self._draw_handle     = None
        self._add_draw_handle(context)
        context.window_manager.modal_handler_add(self)
        context.scene.room_window_edit_active = True
        if context.area:
            context.area.tag_redraw()
        self._header(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        s     = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list

        # ── Navigation pass-through ────────────────────────────────────────
        if event.type in {'MIDDLEMOUSE',
                'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3',
                'NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7',
                'NUMPAD_8','NUMPAD_9','NUMPAD_DECIMAL','NUMPAD_PERIOD',
                'F','TILDE'}:
            return {'PASS_THROUGH'}

        # When mouse is over the N-panel or any non-viewport region, let all
        # events pass through so the user can adjust settings mid-edit.
        # ESC/RMB are exempt so they can still cancel the modal from anywhere.
        if context.region.type != 'WINDOW':
            if event.type not in {'ESC', 'RIGHTMOUSE'}:
                return {'PASS_THROUGH'}

        # ── Ctrl+Z — undo last window placement / deletion / slide ────────────
        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            if self._undo_stack:
                for ri_u, wins_u in self._undo_stack.pop():
                    if ri_u < len(rooms):
                        rooms[ri_u]["windows"] = wins_u
                        _rebuild_room_mesh(rooms[ri_u], s)
                _sync_to_scene(context)
                self._go_hover(context)
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # ── Scroll ────────────────────────────────────────────────────────────
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':

            # H_POSITION: change array count
            if (self._phase == 'H_POSITION' and
                    self._active_idx is not None and self._active_win_idx is not None):
                delta     = 1 if event.type == 'WHEELUPMOUSE' else -1
                new_count = max(1, self._array_count + delta)
                reg       = rooms[self._active_idx]
                proto     = reg.get("windows", [])[self._active_win_idx]
                max_c     = _max_array_count(reg, proto["wc"], proto["w"],
                                             s.window_array_gap, s.door_margin)
                new_count = min(new_count, max(1, max_c))
                if new_count != self._array_count:
                    # Pre-validate candidate anchors against doors before resizing.
                    # (window-window overlap is not checked here because _resize_array
                    # removes the current array first; only door positions are fixed.)
                    ww_   = proto["w"]
                    wc_   = proto["wc"]
                    ctr   = proto["anchor"] + (self._array_count - 1) * (ww_ + s.window_array_gap) / 2
                    cands = _array_anchors(ctr, new_count, ww_, s.window_array_gap,
                                          reg, wc_, s.door_margin)
                    if any(d["wc"] == wc_ and
                           abs(a - d["anchor"]) - (ww_ + d["w"]) * 0.5 < s.door_margin
                           for a in cands for d in reg.get("doors", [])):
                        return {'RUNNING_MODAL'}   # would overlap a door — reject
                    self._resize_array(context, rooms, s, new_count)
                    hdr = (f"Move left/right  |  Scroll=count ({new_count})  "
                           f"|  Click confirms H  |  RMB – cancel")
                    context.area.header_text_set(hdr)
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # LMB_DOWN / SLIDING / V_POSITION: cycle presets
            if (self._phase in ('LMB_DOWN', 'SLIDING', 'V_POSITION') and
                    self._active_idx is not None and self._active_win_idx is not None):
                presets = context.scene.room_window_presets
                active  = context.scene.room_active_window_preset
                if len(presets) > 0:
                    delta      = 1 if event.type == 'WHEELUPMOUSE' else -1
                    new_active = (active + delta) % len(presets)
                    context.scene.room_active_window_preset = new_active
                    wp    = presets[new_active]
                    reg   = rooms[self._active_idx]
                    wi    = self._active_win_idx
                    wins  = reg.get("windows", [])
                    count = self._array_count
                    for k in range(count):
                        if wi + k < len(wins):
                            wins[wi + k]["w"]        = wp.window_width
                            wins[wi + k]["h"]        = wp.window_height
                            wins[wi + k]["v_offset"] = wp.v_offset
                    _rebuild_room_mesh(reg, s)
                    if self._active_partner is not None:
                        p_ri, p_wi = self._active_partner
                        p_wins = rooms[p_ri].get("windows", [])
                        for k in range(count):
                            if p_wi + k < len(p_wins):
                                p_wins[p_wi + k]["w"]        = wp.window_width
                                p_wins[p_wi + k]["h"]        = wp.window_height
                                p_wins[p_wi + k]["v_offset"] = wp.v_offset
                        _rebuild_room_mesh(rooms[p_ri], s)
                    _sync_to_scene(context)
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            return {'PASS_THROUGH'}

        # ── Exit ──────────────────────────────────────────────────────────
        if event.type in {"RET", "NUMPAD_ENTER"} and event.value == 'PRESS':
            if self._phase != 'HOVER':
                _sync_to_scene(context)
            self._finish(context)
            return {"FINISHED"}

        if event.type in {"RIGHTMOUSE", "ESC"} and event.value == 'PRESS':
            if self._phase in ('H_POSITION', 'V_POSITION'):
                # Cancel: remove unconfirmed window + partner
                self._remove_active_win(context, rooms, s)
                self._go_hover(context)
                return {'RUNNING_MODAL'}
            if self._phase in ('LMB_DOWN', 'SLIDING'):
                # Restore original anchor
                if (self._active_idx is not None and
                        self._active_win_idx is not None and
                        self._orig_anchor is not None):
                    reg  = rooms[self._active_idx]
                    wins = reg.get("windows", [])
                    wi   = self._active_win_idx
                    if wi < len(wins):
                        wins[wi]["anchor"] = self._orig_anchor
                        _rebuild_room_mesh(reg, s)
                        if self._active_partner is not None:
                            p_ri, p_wi = self._active_partner
                            p_wins = rooms[p_ri].get("windows", [])
                            if p_wi < len(p_wins):
                                p_wins[p_wi]["anchor"] = self._orig_anchor
                                _rebuild_room_mesh(rooms[p_ri], s)
                if self._added_in_press:
                    self._remove_active_win(context, rooms, s)
                else:
                    _sync_to_scene(context)
                self._go_hover(context)
                return {'RUNNING_MODAL'}
            self._finish(context)
            return {"CANCELLED"}

        # ── Mouse move ────────────────────────────────────────────────────
        if event.type == "MOUSEMOVE":
            pt = _ray_to_z(context, event, s.z_foundation)

            if self._phase == 'HOVER':
                if pt:
                    self._hovered_win = _window_snap(pt, rooms, s, s.wall_thickness)
                    self._hover_snap  = (_wall_snap_any(pt, rooms, s.wall_thickness, current_z=s.z_foundation)
                                        if self._hovered_win is None else None)
                    # Warning header when hover position is blocked
                    if self._hover_snap is not None and self._hovered_win is None:
                        ri, wc, anchor = self._hover_snap
                        if 0 <= ri < len(rooms):
                            ww     = s.window_width
                            active  = context.scene.room_active_window_preset
                            presets = context.scene.room_window_presets
                            if 0 <= active < len(presets):
                                ww = presets[active].window_width
                            count   = s.window_array_count
                            gap     = s.window_array_gap
                            anchors = _array_anchors(anchor, count, ww, gap,
                                                     rooms[ri], wc, s.door_margin)
                            _t_hv = rooms[ri].get("t", s.wall_thickness)
                            _zones_hv = _wall_adj_zones(rooms[ri], wc, rooms, _t_hv)
                            blocked = any(
                                _valid_window_anchor(anch, rooms[ri], wc, ww, s.door_margin,
                                                     zones=_zones_hv) is None
                                for anch in anchors
                            )
                            if blocked:
                                context.area.header_text_set(
                                    "Cannot place window here — overlapping or too close")
                            else:
                                self._header(context)
                        else:
                            self._header(context)
                    else:
                        self._header(context)
                else:
                    self._hovered_win = None
                    self._hover_snap  = None
                    self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL', 'PASS_THROUGH'}

            elif self._phase == 'LMB_DOWN':
                if (self._press_screen and
                        (abs(event.mouse_region_x - self._press_screen[0]) > self._DRAG_PX or
                         abs(event.mouse_region_y - self._press_screen[1]) > self._DRAG_PX)):
                    # Push undo exactly here — once per actual drag, never for plain clicks
                    ri_sl = self._active_idx
                    if ri_sl is not None and ri_sl < len(rooms):
                        undo_sl = [(ri_sl, [w.copy() for w in rooms[ri_sl].get("windows", [])])]
                        if self._active_partner is not None:
                            p_ri_sl = self._active_partner[0]
                            undo_sl.append((p_ri_sl, [w.copy() for w in rooms[p_ri_sl].get("windows", [])]))
                        self._undo_stack.append(undo_sl)
                    self._phase = 'SLIDING'
                    context.area.header_text_set(
                        "Sliding window  ·  Scroll to change preset  |  "
                        "Release to confirm  ·  RMB=cancel")
                else:
                    return {'RUNNING_MODAL'}

            if (self._phase == 'SLIDING' and pt is not None and
                    self._active_idx is not None and self._active_win_idx is not None):
                reg  = rooms[self._active_idx]
                wins = reg.get("windows", [])
                wi   = self._active_win_idx
                if wi < len(wins):
                    ww     = wins[wi]["w"]
                    raw    = pt.y if self._active_wc in ('E', 'W') else pt.x
                    _t_sl  = reg.get("t", s.wall_thickness)
                    _zones_sl = _wall_adj_zones(reg, self._active_wc, rooms, _t_sl)
                    anchor = _valid_window_anchor(raw, reg, self._active_wc, ww,
                                                  s.door_margin, skip_idx=wi,
                                                  zones=_zones_sl)
                    if anchor is not None:
                        wins[wi]["anchor"] = anchor
                        _rebuild_room_mesh(reg, s)
                        if self._active_partner is not None:
                            p_ri, p_wi = self._active_partner
                            p_wins = rooms[p_ri].get("windows", [])
                            if p_wi < len(p_wins):
                                p_wins[p_wi]["anchor"] = anchor
                                _rebuild_room_mesh(rooms[p_ri], s)
                        context.area.tag_redraw()

            elif (self._phase == 'H_POSITION' and pt is not None and
                    self._active_idx is not None and self._active_win_idx is not None):
                reg  = rooms[self._active_idx]
                wins = reg.get("windows", [])
                wi   = self._active_win_idx
                if wi < len(wins):
                    ww      = wins[wi]["w"]
                    gap     = s.window_array_gap
                    count   = self._array_count
                    raw     = pt.y if self._active_wc in ('E', 'W') else pt.x
                    anchors = _array_anchors(raw, count, ww, gap, reg,
                                            self._active_wc, s.door_margin)
                    # Stop the array at door boundaries — same guard as scroll resize
                    door_overlap = any(
                        d["wc"] == self._active_wc and
                        abs(a - d["anchor"]) - (ww + d["w"]) * 0.5 < s.door_margin
                        for a in anchors for d in reg.get("doors", []))
                    _t_hp = reg.get("t", s.wall_thickness)
                    _zones_hp = _wall_adj_zones(reg, self._active_wc, rooms, _t_hp)
                    zone_overlap = bool(_zones_hp) and any(
                        a - ww * 0.5 - s.door_margin < z_hi and
                        a + ww * 0.5 + s.door_margin > z_lo
                        for a in anchors for z_lo, z_hi in _zones_hp)
                    if not door_overlap and not zone_overlap:
                        for k, anch in enumerate(anchors):
                            if wi + k < len(wins):
                                wins[wi + k]["anchor"] = anch
                        _rebuild_room_mesh(reg, s)
                        if self._active_partner is not None:
                            p_ri, p_wi = self._active_partner
                            p_wins = rooms[p_ri].get("windows", [])
                            for k, anch in enumerate(anchors):
                                if p_wi + k < len(p_wins):
                                    p_wins[p_wi + k]["anchor"] = anch
                            _rebuild_room_mesh(rooms[p_ri], s)
                    context.area.tag_redraw()

            elif (self._phase == 'V_POSITION' and
                    self._active_idx is not None and self._active_win_idx is not None):
                reg  = rooms[self._active_idx]
                wins = reg.get("windows", [])
                wi   = self._active_win_idx
                if wi < len(wins):
                    wc   = self._active_wc
                    wh   = wins[wi]["h"]
                    wcoord = self._wall_coord_for(reg, wc)
                    raw_z  = _ray_to_wall_z(context, event, wc, wcoord)
                    if raw_z is not None:
                        room_z = reg.get("z", s.z_foundation)
                        voff   = raw_z - room_z - wh * 0.5   # centre on cursor
                        voff   = max(0.0, min(s.wall_height - wh - s.wall_thickness, voff))
                        count  = self._array_count
                        for k in range(count):
                            if wi + k < len(wins):
                                wins[wi + k]["v_offset"] = voff
                        _rebuild_room_mesh(reg, s)
                        if self._active_partner is not None:
                            p_ri, p_wi = self._active_partner
                            p_wins = rooms[p_ri].get("windows", [])
                            for k in range(count):
                                if p_wi + k < len(p_wins):
                                    p_wins[p_wi + k]["v_offset"] = voff
                            _rebuild_room_mesh(rooms[p_ri], s)
                        context.area.tag_redraw()

            return {'RUNNING_MODAL'}

        # ── Left mouse ────────────────────────────────────────────────────
        if event.type == "LEFTMOUSE":
            in_region = (0 <= event.mouse_region_x < context.region.width and
                         0 <= event.mouse_region_y < context.region.height)

            if event.value == "PRESS":
                if not in_region:
                    return {'PASS_THROUGH'}

                # ── H_POSITION: first click confirms horizontal, enter V ──
                if self._phase == 'H_POSITION':
                    active  = context.scene.room_active_window_preset
                    presets = context.scene.room_window_presets
                    if 0 <= active < len(presets):
                        # Preset defines v_offset — skip V_POSITION entirely
                        _sync_to_scene(context)
                        self._go_hover(context)
                    else:
                        self._phase = 'V_POSITION'
                        context.area.header_text_set(
                            "Move up/down to set sill height  |  Click to confirm  |  RMB – cancel")
                    return {'RUNNING_MODAL'}

                # ── V_POSITION: second click confirms vertical placement ────
                if self._phase == 'V_POSITION':
                    _sync_to_scene(context)
                    self._go_hover(context)
                    return {'RUNNING_MODAL'}

                pt  = _ray_to_z(context, event, s.z_foundation)
                now = time.time()

                # ── Double-click: remove hovered window ───────────────────
                is_double = ((now - self._last_press_time) < 0.3 and
                             self._last_press_win is not None)
                if is_double and pt is not None:
                    current_win = _window_snap(pt, rooms, s, s.wall_thickness)
                    if current_win is not None and current_win == self._last_press_win:
                        ri, wi = current_win
                        self._active_idx      = ri
                        self._active_win_idx  = wi
                        self._active_wc       = rooms[ri]["windows"][wi]["wc"]
                        self._active_partner  = _find_partner_window(rooms, ri, wi, s.wall_thickness)
                        self._added_in_press  = True
                        undo_entry = [(ri, [w.copy() for w in rooms[ri].get("windows", [])])]
                        if self._active_partner is not None:
                            p_ri_d = self._active_partner[0]
                            undo_entry.append((p_ri_d, [w.copy() for w in rooms[p_ri_d].get("windows", [])]))
                        self._undo_stack.append(undo_entry)
                        self._remove_active_win(context, rooms, s)
                        self._last_press_time = 0.0
                        self._last_press_win  = None
                        self._hovered_win     = None
                        self._go_hover(context)
                        return {'RUNNING_MODAL'}

                # ── Single press ──────────────────────────────────────────
                self._last_press_time = now
                self._press_screen    = (event.mouse_region_x, event.mouse_region_y)

                if pt is None:
                    return {'PASS_THROUGH'}

                if self._hovered_win is not None:
                    # Press on existing window → ready to slide
                    ri, wi = self._hovered_win
                    self._last_press_win  = (ri, wi)
                    self._active_idx      = ri
                    self._active_win_idx  = wi
                    self._active_wc       = rooms[ri]["windows"][wi]["wc"]
                    self._active_partner  = _find_partner_window(rooms, ri, wi, s.wall_thickness)
                    self._orig_anchor     = rooms[ri]["windows"][wi]["anchor"]
                    self._added_in_press  = False
                    self._phase           = 'LMB_DOWN'
                    context.area.header_text_set(
                        "Hold+drag=slide  ·  Hold+scroll=change preset  |  "
                        "Release=confirm  ·  RMB=cancel")

                elif self._hover_snap is not None:
                    # Press on wall → place N windows as array, enter H_POSITION
                    ri, wc, raw_anchor = self._hover_snap
                    self._last_press_win = None
                    if 0 <= ri < len(rooms):
                        reg          = rooms[ri]
                        ww, wh, voff = self._get_preset_dims(context)
                        count        = s.window_array_count
                        gap          = s.window_array_gap
                        count        = min(count, max(1, _max_array_count(reg, wc, ww, gap, s.door_margin)))
                        anchors      = _array_anchors(raw_anchor, count, ww, gap,
                                                      reg, wc, s.door_margin)
                        _t_lmb = reg.get("t", s.wall_thickness)
                        _zones_lmb = _wall_adj_zones(reg, wc, rooms, _t_lmb)
                        # Capture zone-snapped anchors so stored positions never
                        # land inside the adjacent-room blocked zone.
                        clamped_lmb = [_valid_window_anchor(anch, reg, wc, ww, s.door_margin,
                                                            zones=_zones_lmb)
                                       for anch in anchors]
                        if any(c is None for c in clamped_lmb):
                            return {'RUNNING_MODAL'}
                        anchors = clamped_lmb   # use zone-snapped positions
                        # Save state for Ctrl+Z (before any windows are added)
                        undo_entry = [(ri, [w.copy() for w in reg.get("windows", [])])]
                        partner_pre = _find_partner_wall(rooms, ri, wc, s.wall_thickness)
                        if partner_pre is not None:
                            undo_entry.append((partner_pre[0],
                                               [w.copy() for w in rooms[partner_pre[0]].get("windows", [])]))
                        self._undo_stack.append(undo_entry)
                        first_wi     = len(reg.setdefault("windows", []))
                        for k, anch in enumerate(anchors):
                            reg["windows"].append({"wc": wc, "anchor": anch,
                                                   "v_offset": voff, "w": ww, "h": wh,
                                                   "array_n": count, "array_idx": k})
                        _rebuild_room_mesh(reg, s)
                        # Mirror to partner wall
                        partner          = _find_partner_wall(rooms, ri, wc, s.wall_thickness)
                        partner_first_wi = None
                        if partner is not None:
                            p_ri, p_wc = partner
                            p_reg      = rooms[p_ri]
                            p_wins     = p_reg.setdefault("windows", [])
                            partner_first_wi = len(p_wins)
                            for k, anch in enumerate(anchors):
                                p_wins.append({"wc": p_wc, "anchor": anch,
                                               "v_offset": voff, "w": ww, "h": wh,
                                               "array_n": count, "array_idx": k})
                            _rebuild_room_mesh(p_reg, s)
                        self._active_idx     = ri
                        self._active_wc      = wc
                        self._active_win_idx = first_wi
                        self._active_partner = ((partner[0], partner_first_wi)
                                                if partner and partner_first_wi is not None
                                                else None)
                        self._orig_anchor    = anchors[0]
                        self._added_in_press = True
                        self._array_count    = count
                        self._phase          = 'H_POSITION'
                        _sync_to_scene(context)
                        context.area.tag_redraw()
                        context.area.header_text_set(
                            f"Move left/right  |  Scroll=count ({count})  "
                            f"|  Click confirms H  |  RMB – cancel")
                else:
                    self._last_press_win = None
                    return {'PASS_THROUGH'}

                return {'RUNNING_MODAL'}

            elif event.value == "RELEASE":
                if self._phase in ('LMB_DOWN', 'SLIDING'):
                    _sync_to_scene(context)
                    self._go_hover(context)
                return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL', 'PASS_THROUGH'}

    def cancel(self, context):
        self._remove_draw_handle()
        context.area.header_text_set(None)
        context.scene.room_window_edit_active = False
        if context.area:
            context.area.tag_redraw()


# ═════════════════════════════════════════════════════════════════════════════
# Floor operators
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_add_floor(bpy.types.Operator):
    bl_idname     = "room.add_floor"
    bl_label      = "Add Floor"
    bl_description = "Save the current Z Offset as a named floor preset"

    def execute(self, context):
        s      = context.scene.room_settings
        floors = context.scene.room_floors
        n      = len(floors) + 1
        fl          = floors.add()
        fl.name     = f"Floor.{n:02}"
        fl.z_offset = s.z_foundation
        context.scene.room_active_floor = len(floors) - 1
        return {"FINISHED"}


class ROOM_OT_select_floor(bpy.types.Operator):
    bl_idname     = "room.select_floor"
    bl_label      = "Select Floor"
    bl_description = "Switch the active Z Offset to this floor"

    floor_index: bpy.props.IntProperty()

    def execute(self, context):
        floors = context.scene.room_floors
        idx    = self.floor_index
        if 0 <= idx < len(floors):
            context.scene.room_settings.z_foundation = floors[idx].z_offset
            context.scene.room_active_floor           = idx
        return {"FINISHED"}


class ROOM_OT_remove_floor(bpy.types.Operator):
    bl_idname     = "room.remove_floor"
    bl_label      = "Remove Floor"
    bl_description = "Remove this floor preset (does not delete placed rooms)"

    floor_index: bpy.props.IntProperty()

    def execute(self, context):
        floors = context.scene.room_floors
        idx    = self.floor_index
        if 0 <= idx < len(floors):
            floors.remove(idx)
            active = context.scene.room_active_floor
            context.scene.room_active_floor = max(-1, min(active, len(floors) - 1))
        return {"FINISHED"}


# ═════════════════════════════════════════════════════════════════════════════
# Door-preset operators
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_save_door_preset(bpy.types.Operator):
    bl_idname     = "room.save_door_preset"
    bl_label      = "Save Door Preset"
    bl_description = "Save current door dimensions as a named preset"

    def execute(self, context):
        s       = context.scene.room_settings
        presets = context.scene.room_door_presets
        n       = len(presets) + 1
        dp           = presets.add()
        dp.name      = f"Door.{n:02}"
        dp.door_width  = s.door_width
        dp.door_height = s.door_height
        context.scene.room_active_door_preset = len(presets) - 1
        return {"FINISHED"}


class ROOM_OT_select_door_preset(bpy.types.Operator):
    bl_idname     = "room.select_door_preset"
    bl_label      = "Select Door Preset"
    bl_description = "Switch active door preset and copy its dimensions to the settings"

    preset_index: bpy.props.IntProperty()

    def execute(self, context):
        presets = context.scene.room_door_presets
        idx     = self.preset_index
        if 0 <= idx < len(presets):
            if context.scene.room_active_door_preset == idx:
                # Click active preset again → deselect
                context.scene.room_active_door_preset = -1
            else:
                context.scene.room_active_door_preset = idx
                dp = presets[idx]
                context.scene.room_settings.door_width  = dp.door_width
                context.scene.room_settings.door_height = dp.door_height
        return {"FINISHED"}


class ROOM_OT_remove_door_preset(bpy.types.Operator):
    bl_idname     = "room.remove_door_preset"
    bl_label      = "Remove Door Preset"
    bl_description = "Remove this door preset"

    preset_index: bpy.props.IntProperty()

    def execute(self, context):
        presets = context.scene.room_door_presets
        idx     = self.preset_index
        if 0 <= idx < len(presets):
            presets.remove(idx)
            active = context.scene.room_active_door_preset
            context.scene.room_active_door_preset = max(-1, min(active, len(presets) - 1))
        return {"FINISHED"}


# ═════════════════════════════════════════════════════════════════════════════
# Window-preset operators
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_save_window_preset(bpy.types.Operator):
    bl_idname     = "room.save_window_preset"
    bl_label      = "Save Window Preset"
    bl_description = "Save current window dimensions as a named preset"

    def execute(self, context):
        s       = context.scene.room_settings
        presets = context.scene.room_window_presets
        n       = len(presets) + 1
        wp              = presets.add()
        wp.name         = f"Window.{n:02}"
        wp.window_width  = s.window_width
        wp.window_height = s.window_height
        wp.v_offset      = s.window_v_offset
        context.scene.room_active_window_preset = len(presets) - 1
        return {"FINISHED"}


class ROOM_OT_select_window_preset(bpy.types.Operator):
    bl_idname     = "room.select_window_preset"
    bl_label      = "Select Window Preset"
    bl_description = "Switch active window preset and copy its dimensions to the settings"

    preset_index: bpy.props.IntProperty()

    def execute(self, context):
        presets = context.scene.room_window_presets
        idx     = self.preset_index
        if 0 <= idx < len(presets):
            if context.scene.room_active_window_preset == idx:
                # Click active preset again → deselect
                context.scene.room_active_window_preset = -1
            else:
                context.scene.room_active_window_preset = idx
                wp = presets[idx]
                context.scene.room_settings.window_width   = wp.window_width
                context.scene.room_settings.window_height  = wp.window_height
                context.scene.room_settings.window_v_offset = wp.v_offset
        return {"FINISHED"}


class ROOM_OT_remove_window_preset(bpy.types.Operator):
    bl_idname     = "room.remove_window_preset"
    bl_label      = "Remove Window Preset"
    bl_description = "Remove this window preset"

    preset_index: bpy.props.IntProperty()

    def execute(self, context):
        presets = context.scene.room_window_presets
        idx     = self.preset_index
        if 0 <= idx < len(presets):
            presets.remove(idx)
            active = context.scene.room_active_window_preset
            context.scene.room_active_window_preset = max(-1, min(active, len(presets) - 1))
        return {"FINISHED"}


# ─────────────────────────────────────────────────────────────────────────────
# Dimension Sampler (Eyedropper / Pipette)
# ─────────────────────────────────────────────────────────────────────────────
class ROOM_OT_sample_dims_apply(bpy.types.Operator):
    """Apply sampled object dimensions to door or window settings."""
    bl_idname  = "room.sample_dims_apply"
    bl_label   = "Apply Sampled Dimensions"
    bl_options = {'REGISTER', 'UNDO'}

    apply_as: bpy.props.EnumProperty(
        name="Apply As",
        items=[('DOOR', 'Door', ''), ('WINDOW', 'Window', '')],
        default='DOOR',
    )
    sampled_width:    bpy.props.FloatProperty()
    sampled_height:   bpy.props.FloatProperty()
    sampled_z_bottom: bpy.props.FloatProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context):
        L = self.layout
        L.prop(self, "apply_as", expand=True)
        col = L.column(align=True)
        col.label(text=f"Width:  {self.sampled_width:.3f} m")
        col.label(text=f"Height: {self.sampled_height:.3f} m")
        if self.apply_as == 'WINDOW':
            v_off = self.sampled_z_bottom - context.scene.room_settings.z_foundation
            col.label(text=f"Sill:   {max(0.0, v_off):.3f} m")

    def execute(self, context):
        s = context.scene.room_settings
        if self.apply_as == 'DOOR':
            s.door_width  = self.sampled_width
            s.door_height = self.sampled_height
        else:
            s.window_width    = self.sampled_width
            s.window_height   = self.sampled_height
            v_off = self.sampled_z_bottom - s.z_foundation
            s.window_v_offset = max(0.0, v_off)
        return {'FINISHED'}


class ROOM_OT_sample_dims(bpy.types.Operator):
    """Click an object in the viewport to sample its width and height."""
    bl_idname      = "room.sample_dims"
    bl_label       = "Sample Object Dimensions"
    bl_description = "Click any object to read its bounding-box width and height"

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set('EYEDROPPER')
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        # Let viewport navigation pass through
        if event.type in {'MIDDLEMOUSE',
                'NUMPAD_0', 'NUMPAD_1', 'NUMPAD_2', 'NUMPAD_3',
                'NUMPAD_4', 'NUMPAD_5', 'NUMPAD_6', 'NUMPAD_7',
                'NUMPAD_8', 'NUMPAD_9', 'NUMPAD_DECIMAL', 'NUMPAD_PERIOD',
                'F', 'TILDE'}:
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            context.window.cursor_modal_restore()
            # Find the 3D viewport region under the cursor
            mx, my   = event.mouse_x, event.mouse_y
            region   = None
            rv3d     = None
            for area in context.screen.areas:
                if area.type != 'VIEW_3D':
                    continue
                if not (area.x <= mx < area.x + area.width and
                        area.y <= my < area.y + area.height):
                    continue
                for r in area.regions:
                    if (r.type == 'WINDOW' and
                            r.x <= mx < r.x + r.width and
                            r.y <= my < r.y + r.height):
                        region = r
                        rv3d   = area.spaces.active.region_3d
                        break
                break
            if region is None or rv3d is None:
                self.report({'WARNING'}, "Click inside the 3D viewport")
                return {'CANCELLED'}
            coord     = (mx - region.x, my - region.y)
            origin    = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            hit, _, _, _, obj, _ = context.scene.ray_cast(
                context.view_layer.depsgraph, origin, direction)
            if not hit or obj is None:
                self.report({'WARNING'}, "No object hit — click directly on a mesh")
                return {'CANCELLED'}
            # Measure world-space bounding box
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            xs = [c.x for c in corners]
            ys = [c.y for c in corners]
            zs = [c.z for c in corners]
            width  = max(max(xs) - min(xs), max(ys) - min(ys))
            height = max(zs) - min(zs)
            z_bot  = min(zs)
            bpy.ops.room.sample_dims_apply('INVOKE_DEFAULT',
                sampled_width=round(width, 4),
                sampled_height=round(height, 4),
                sampled_z_bottom=round(z_bot, 4))
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.window.cursor_modal_restore()
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_toggle_room_plinth(bpy.types.Operator):
    """Toggle bottom or top plinth on the selected room only."""
    bl_idname  = "room.toggle_room_plinth"
    bl_label   = "Toggle Room Plinth"
    bl_options = {"REGISTER", "UNDO"}

    side: bpy.props.EnumProperty(items=[
        ('BOTTOM', 'Bottom', ''),
        ('TOP',    'Top',    ''),
    ])

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (obj is not None and
                any(r.get('obj_name') == obj.name for r in ROOM_OT_draw._room_list))

    def execute(self, context):
        obj = context.active_object
        s   = context.scene.room_settings
        reg = next((r for r in ROOM_OT_draw._room_list
                    if r.get('obj_name') == obj.name), None)
        if reg is None:
            return {'CANCELLED'}
        key = 'plinth_bottom_enabled' if self.side == 'BOTTOM' else 'plinth_top_enabled'
        reg[key] = not reg.get(key, True)
        _rebuild_room_mesh(reg, s)
        _sync_to_scene(context)
        return {'FINISHED'}


# N-Panel
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_PT_panel(bpy.types.Panel):
    bl_label       = "Room Tool"
    bl_idname      = "ROOM_PT_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Room Tool"

    def draw(self, context):
        L = self.layout
        s = context.scene.room_settings

        row = L.row()
        row.scale_y = 1.5
        row.operator("room.draw", icon="MESH_CUBE", text="Draw Room  (Shift+R)")

        box = L.box()
        col = box.column(align=True)
        col.label(text="Foundation Alignment:", icon="OBJECT_ORIGIN")
        col.prop(s, "z_foundation", text="Z Offset")
        col.operator("room.add_floor", icon="ADD", text="Add Floor")

        floors = context.scene.room_floors
        active = context.scene.room_active_floor
        if floors:
            col.separator()
            for i, fl in enumerate(floors):
                row = col.row(align=True)
                is_active = (i == active)
                op = row.operator(
                    "room.select_floor",
                    text=f"{fl.name}  (z={fl.z_offset:.3f})",
                    icon="LAYER_ACTIVE" if is_active else "LAYER_USED",
                    depress=is_active,
                )
                op.floor_index = i
                rem = row.operator("room.remove_floor", text="", icon="X")
                rem.floor_index = i

        box = L.box()
        col = box.column(align=True)
        col.label(text="Wall Dimensions:", icon="MOD_BUILD")
        col.prop(s, "wall_height")
        col.prop(s, "wall_thickness")

        box = L.box()
        col = box.column(align=True)
        col.label(text="Features:", icon="MODIFIER")
        col.prop(s, "add_ceiling")
        col.prop(s, "add_floor")
        col.prop(s, "add_door")
        col.separator()
        col.label(text="Plinth:")
        row = col.row(align=True)
        row.prop(s, "add_plinth_bottom", toggle=True, icon="TRIA_UP_BAR")
        row.prop(s, "add_plinth_top",    toggle=True, icon="TRIA_DOWN_BAR")
        if s.add_plinth_bottom:
            sub = col.column(align=True)
            sub.label(text="Bottom:")
            sub.prop(s, "plinth_bottom_height")
            sub.prop(s, "plinth_bottom_thickness")
        if s.add_plinth_top:
            sub = col.column(align=True)
            sub.label(text="Top:")
            sub.prop(s, "plinth_top_height")
            sub.prop(s, "plinth_top_thickness")
        col.separator()
        col.label(text="Object Pivot:")
        col.prop(s, "pivot_mode", text="")

        box = L.box()
        col = box.column(align=True)
        col.label(text="Materials  /  Tiling:", icon="MATERIAL")
        for mat_attr, tile_attr, label in (
            ("mat_walls",        "mat_walls_tiling",        "Walls"),
            ("mat_floor",        "mat_floor_tiling",        "Floor"),
            ("mat_ceiling",      "mat_ceiling_tiling",      "Ceiling"),
            ("mat_door_frame",   "mat_door_frame_tiling",   "Door (∅=Walls)"),
            ("mat_window_frame", "mat_window_frame_tiling", "Win (∅=Walls)"),
        ):
            row = col.row(align=True)
            split = row.split(factor=0.65, align=True)
            split.prop(s, mat_attr, text=label)
            split.prop(s, tile_attr, text="")
        if s.add_plinth_bottom:
            row = col.row(align=True)
            split = row.split(factor=0.65, align=True)
            split.prop(s, "mat_plinth_bottom", text="Bot Plinth (∅=Walls)")
            split.prop(s, "mat_plinth_bottom_tiling", text="")
        if s.add_plinth_top:
            row = col.row(align=True)
            split = row.split(factor=0.65, align=True)
            split.prop(s, "mat_plinth_top", text="Top Plinth (∅=Walls)")
            split.prop(s, "mat_plinth_top_tiling", text="")

        # ── Per-room plinth overrides (shown when a room object is selected) ──
        obj = context.active_object
        if obj and (s.add_plinth_bottom or s.add_plinth_top):
            reg = next((r for r in ROOM_OT_draw._room_list
                        if r.get('obj_name') == obj.name), None)
            if reg is not None:
                box = L.box()
                col = box.column(align=True)
                col.label(text=obj.name, icon="MESH_CUBE")
                if s.add_plinth_bottom:
                    pb_on = reg.get('plinth_bottom_enabled', True)
                    op = col.operator("room.toggle_room_plinth",
                                      text=("Bot Plinth: On" if pb_on else "Bot Plinth: Off"),
                                      icon=("CHECKMARK" if pb_on else "X"),
                                      depress=pb_on)
                    op.side = 'BOTTOM'
                if s.add_plinth_top:
                    pt_on = reg.get('plinth_top_enabled', True)
                    op = col.operator("room.toggle_room_plinth",
                                      text=("Top Plinth: On" if pt_on else "Top Plinth: Off"),
                                      icon=("CHECKMARK" if pt_on else "X"),
                                      depress=pt_on)
                    op.side = 'TOP'

        L.separator()
        L.operator("room.clear_registry",  icon="X",           text="Reset Snap Registry")
        L.operator("room.clear_overlays",  icon="GHOST_DISABLED", text="Clear Phantom Overlays")


class ROOM_PT_door_panel(bpy.types.Panel):
    bl_label       = "Doors"
    bl_idname      = "ROOM_PT_door_panel"
    bl_parent_id   = "ROOM_PT_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Room Tool"
    bl_options     = {"DEFAULT_CLOSED"}

    def draw(self, context):
        L = self.layout
        s = context.scene.room_settings

        col = L.column(align=True)
        col.scale_y = 1.3
        is_editing = getattr(context.scene, "room_door_edit_active", False)
        col.operator("room.door_edit", icon="OUTLINER_OB_EMPTY",
                     text="Door Frame Edit Mode  (Ctrl+Shift+D)", depress=is_editing)
        col.separator()

        col = L.column(align=True)
        col.prop(s, "door_width")
        col.prop(s, "door_height")
        col.prop(s, "door_margin")
        col.operator("room.sample_dims", icon="EYEDROPPER", text="Sample from Object")
        col.operator("room.save_door_preset", icon="ADD", text="Save Door Preset")

        presets = context.scene.room_door_presets
        active  = context.scene.room_active_door_preset
        if presets:
            col.separator()
            for i, dp in enumerate(presets):
                is_active = (i == active)
                row = col.row(align=True)
                op = row.operator(
                    "room.select_door_preset",
                    text=f"{dp.name}  {dp.door_width:.2f}\u00d7{dp.door_height:.2f}m",
                    icon="LAYER_ACTIVE" if is_active else "LAYER_USED",
                    depress=is_active,
                )
                op.preset_index = i
                rem = row.operator("room.remove_door_preset", text="", icon="X")
                rem.preset_index = i
                if is_active:
                    col.prop(dp, "name", text="Rename")


class ROOM_PT_window_panel(bpy.types.Panel):
    bl_label       = "Windows"
    bl_idname      = "ROOM_PT_window_panel"
    bl_parent_id   = "ROOM_PT_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Room Tool"
    bl_options     = {"DEFAULT_CLOSED"}

    def draw(self, context):
        L = self.layout
        s = context.scene.room_settings

        col = L.column(align=True)
        col.scale_y = 1.3
        is_editing = getattr(context.scene, "room_window_edit_active", False)
        col.operator("room.window_edit", icon="OUTLINER_OB_EMPTY",
                     text="Window Frame Edit Mode  (Ctrl+Shift+W)", depress=is_editing)
        col.separator()

        col = L.column(align=True)
        col.prop(s, "window_width")
        col.prop(s, "window_height")
        col.prop(s, "window_v_offset")
        col.separator()
        row = col.row(align=True)
        row.prop(s, "window_array_count")
        row.prop(s, "window_array_gap")
        col.operator("room.sample_dims", icon="EYEDROPPER", text="Sample from Object")
        col.operator("room.save_window_preset", icon="ADD", text="Save Window Preset")

        presets = context.scene.room_window_presets
        active  = context.scene.room_active_window_preset
        if presets:
            col.separator()
            for i, wp in enumerate(presets):
                is_active = (i == active)
                row = col.row(align=True)
                op = row.operator(
                    "room.select_window_preset",
                    text=f"{wp.name}  {wp.window_width:.2f}\u00d7{wp.window_height:.2f}m  @{wp.v_offset:.2f}m",
                    icon="LAYER_ACTIVE" if is_active else "LAYER_USED",
                    depress=is_active,
                )
                op.preset_index = i
                rem = row.operator("room.remove_window_preset", text="", icon="X")
                rem.preset_index = i
                if is_active:
                    col.prop(wp, "name", text="Rename")
                    col.prop(wp, "window_width")
                    col.prop(wp, "window_height")
                    col.prop(wp, "v_offset")


# ═════════════════════════════════════════════════════════════════════════════
# Register / Unregister
# ═════════════════════════════════════════════════════════════════════════════
_classes = (
    ROOM_PG_registry_entry,
    ROOM_PG_floor,
    ROOM_PG_door_preset,
    ROOM_PG_window_preset,
    ROOM_PG_settings,
    ROOM_OT_draw,
    ROOM_OT_clear,
    ROOM_OT_clear_overlays,
    ROOM_OT_door_edit,
    ROOM_OT_window_edit,
    ROOM_OT_add_floor,
    ROOM_OT_select_floor,
    ROOM_OT_remove_floor,
    ROOM_OT_save_door_preset,
    ROOM_OT_select_door_preset,
    ROOM_OT_remove_door_preset,
    ROOM_OT_save_window_preset,
    ROOM_OT_select_window_preset,
    ROOM_OT_remove_window_preset,
    ROOM_OT_sample_dims_apply,
    ROOM_OT_sample_dims,
    ROOM_OT_toggle_room_plinth,
    ROOM_PT_panel,
    ROOM_PT_door_panel,
    ROOM_PT_window_panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.room_settings              = bpy.props.PointerProperty(type=ROOM_PG_settings)
    bpy.types.Scene.room_floors               = bpy.props.CollectionProperty(type=ROOM_PG_floor)
    bpy.types.Scene.room_active_floor         = bpy.props.IntProperty(default=-1)
    bpy.types.Scene.room_door_presets         = bpy.props.CollectionProperty(type=ROOM_PG_door_preset)
    bpy.types.Scene.room_active_door_preset   = bpy.props.IntProperty(default=-1)
    bpy.types.Scene.room_window_presets       = bpy.props.CollectionProperty(type=ROOM_PG_window_preset)
    bpy.types.Scene.room_active_window_preset = bpy.props.IntProperty(default=-1)
    bpy.types.Scene.room_registry             = bpy.props.CollectionProperty(type=ROOM_PG_registry_entry)
    bpy.types.Scene.room_door_edit_active     = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.room_window_edit_active   = bpy.props.BoolProperty(default=False)

    for km, kmi in ROOM_OT_draw._addon_kmaps:
        try: km.keymap_items.remove(kmi)
        except Exception: pass
    ROOM_OT_draw._addon_kmaps.clear()

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km  = kc.keymaps.new(name="3D View", space_type="VIEW_3D")
        kmi = km.keymap_items.new("room.draw", "R", "PRESS", shift=True)
        ROOM_OT_draw._addon_kmaps.append((km, kmi))
        kmi2 = km.keymap_items.new("room.door_edit", "D", "PRESS", shift=True, ctrl=True)
        ROOM_OT_draw._addon_kmaps.append((km, kmi2))
        kmi3 = km.keymap_items.new("room.window_edit", "W", "PRESS", shift=True, ctrl=True)
        ROOM_OT_draw._addon_kmaps.append((km, kmi3))

    if _room_registry_cleanup not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_room_registry_cleanup)
    if _room_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_room_on_load)


def unregister():
    # Remove any live draw handles so no phantoms survive addon reload
    for handle in list(_DRAW_HANDLES):
        try:
            bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
        except Exception:
            pass
    _DRAW_HANDLES.clear()

    if _room_registry_cleanup in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(_room_registry_cleanup)
    if _room_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_room_on_load)

    for km, kmi in ROOM_OT_draw._addon_kmaps:
        try: km.keymap_items.remove(kmi)
        except Exception: pass
    ROOM_OT_draw._addon_kmaps.clear()

    for attr in ("room_settings", "room_floors", "room_active_floor",
                 "room_door_presets", "room_active_door_preset",
                 "room_window_presets", "room_active_window_preset",
                 "room_registry", "room_door_edit_active", "room_window_edit_active"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)

    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
