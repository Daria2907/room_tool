bl_info = {
    "name": "Room Tool",
    "version": (1, 3),
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


class ROOM_PG_registry_entry(bpy.types.PropertyGroup):
    """One room's registry data persisted in the scene for addon-reload survival."""
    x1: bpy.props.FloatProperty()
    y1: bpy.props.FloatProperty()
    x2: bpy.props.FloatProperty()
    y2: bpy.props.FloatProperty()
    t:  bpy.props.FloatProperty(default=0.125)
    z:  bpy.props.FloatProperty(default=0.0)
    door_walls:   bpy.props.StringProperty()   # comma-separated e.g. "S,W"
    no_walls:     bpy.props.StringProperty()   # comma-separated
    door_anchors: bpy.props.StringProperty()   # JSON e.g. '{"S": 1.5}'
    door_dims:    bpy.props.StringProperty(default="{}")  # JSON e.g. '{"S": {"w": 1.2, "h": 2.1}}'
    obj_name:     bpy.props.StringProperty()
    door_width:   bpy.props.FloatProperty(default=0.9)
    door_height:  bpy.props.FloatProperty(default=2.0)


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
        name="Door Edge Margin",
        description="Minimum distance from door edge to wall corner",
        default=0.15, min=0.0, max=2.0, unit="LENGTH")


# ═════════════════════════════════════════════════════════════════════════════
# Collection helpers
# ═════════════════════════════════════════════════════════════════════════════
def _get_or_create_col(name, parent):
    """Return collection *name*, creating it under *parent* if needed."""
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
    """Return the collection in which the new room object should be placed."""
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
def _face4(bm, v0, v1, v2, v3):
    """Create a single quad face from 4 corner tuples (CCW winding = front face)."""
    bm.faces.new([bm.verts.new(co) for co in (v0, v1, v2, v3)])


def _fill_room(bm, x1, y1, x2, y2, z, s, door_walls=(), no_walls=(), door_anchors=None,
               door_width=None, door_height=None, door_dims=None):
    """
    Populate *bm* with room geometry as an interior shell.
    door_walls:   wall chars that get a door opening.
    no_walls:     wall chars to skip entirely.
    door_anchors: {wall_char: center_along_wall} to pin door centres.
                  E/W walls use a Y anchor; N/S walls use an X anchor.
    door_dims:    {wall_char: {"w": float, "h": float}} per-door overrides.
    door_width/door_height: room-level fallback; fall back to s.door_width/door_height.
    """
    dd = door_dims or {}
    da = door_anchors or {}
    t  = s.wall_thickness
    h  = s.wall_height
    zt = z + h
    fd = t * 0.5    # tunnel reveal depth = half wall thickness (two frames meet flush)

    # Per-wall door size helpers ─────────────────────────────────────────────
    def _dw(wc):
        """Door opening width for wall wc, clamped to 85 % of room span."""
        raw = dd.get(wc, {}).get("w", door_width if door_width is not None else s.door_width)
        span = (x2 - x1) if wc in ('N', 'S') else (y2 - y1)
        return min(raw, span * 0.85)

    def _dh(wc):
        """Door opening height for wall wc, clamped below ceiling."""
        raw = dd.get(wc, {}).get("h", door_height if door_height is not None else s.door_height)
        return min(raw, h - t)

    # Fallback dh for non-door wall height seams
    _dh_fallback = min(door_height if door_height is not None else s.door_height, h - t)

    # Pre-compute strip split positions used by both walls and ceiling/floor
    _ey = []   # Y splits for E/W door columns
    _ex = []   # X splits for N/S door columns
    if s.add_door:
        for _wc in door_walls:
            _w = _dw(_wc)
            if _w > 0:
                if _wc in ('E', 'W'):
                    _cy = da.get(_wc, (y1 + y2) * 0.5)
                    _ey += [max(y1, _cy - _w * 0.5), min(y2, _cy + _w * 0.5)]
                elif _wc in ('N', 'S'):
                    _cx = da.get(_wc, (x1 + x2) * 0.5)
                    _ex += [max(x1, _cx - _w * 0.5), min(x2, _cx + _w * 0.5)]
    _ey = sorted(set(_ey))
    _ex = sorted(set(_ex))

    # ── South inner panel (at y = y1, reveals go –Y) ──────────────────────────
    if 'S' not in no_walls:
        dw_s = _dw('S'); dh_s = _dh('S')
        if 'S' in door_walls and s.add_door and dw_s > 0:
            cx = da.get('S', (x1 + x2) * 0.5)
            dl = max(x1, cx - dw_s * 0.5)   # clamp: never west of room
            dr = min(x2, cx + dw_s * 0.5)   # clamp: never east of room
            if dl > x1:
                _face4(bm, (dl,y1,z),    (x1,y1,z),    (x1,y1,z+dh_s),    (dl,y1,z+dh_s))
                _face4(bm, (dl,y1,z+dh_s), (x1,y1,z+dh_s), (x1,y1,zt),    (dl,y1,zt))
            if dr < x2:
                _face4(bm, (x2,y1,z),    (dr,y1,z),    (dr,y1,z+dh_s),    (x2,y1,z+dh_s))
                _face4(bm, (x2,y1,z+dh_s), (dr,y1,z+dh_s), (dr,y1,zt),    (x2,y1,zt))
            _face4(bm, (dr,y1,z+dh_s), (dl,y1,z+dh_s), (dl,y1,zt),     (dr,y1,zt))
            _face4(bm, (dl,y1,z),      (dl,y1-fd,z),    (dl,y1-fd,z+dh_s), (dl,y1,z+dh_s))
            _face4(bm, (dr,y1-fd,z),   (dr,y1,z),       (dr,y1,z+dh_s),    (dr,y1-fd,z+dh_s))
            _face4(bm, (dr,y1,z+dh_s), (dl,y1,z+dh_s),  (dl,y1-fd,z+dh_s), (dr,y1-fd,z+dh_s))
            _face4(bm, (dl,y1-fd,z),   (dr,y1-fd,z),    (dr,y1,z),         (dl,y1,z))
        else:
            if _ex:
                xs = [x1] + _ex + [x2]
                for i in range(len(xs) - 1):
                    _face4(bm, (xs[i+1],y1,z),           (xs[i],y1,z),           (xs[i],y1,z+_dh_fallback),   (xs[i+1],y1,z+_dh_fallback))
                    _face4(bm, (xs[i+1],y1,z+_dh_fallback), (xs[i],y1,z+_dh_fallback), (xs[i],y1,zt),         (xs[i+1],y1,zt))
            else:
                _face4(bm, (x2,y1,z),    (x1,y1,z),    (x1,y1,z+_dh_fallback), (x2,y1,z+_dh_fallback))
                _face4(bm, (x2,y1,z+_dh_fallback), (x1,y1,z+_dh_fallback), (x1,y1,zt), (x2,y1,zt))

    # ── North inner panel (at y = y2, reveals go +Y) ──────────────────────────
    if 'N' not in no_walls:
        dw_n = _dw('N'); dh_n = _dh('N')
        if 'N' in door_walls and s.add_door and dw_n > 0:
            cx = da.get('N', (x1 + x2) * 0.5)
            dl = max(x1, cx - dw_n * 0.5)   # clamp: never west of room
            dr = min(x2, cx + dw_n * 0.5)   # clamp: never east of room
            if dl > x1:
                _face4(bm, (x1,y2,z),    (dl,y2,z),    (dl,y2,z+dh_n),    (x1,y2,z+dh_n))
                _face4(bm, (x1,y2,z+dh_n), (dl,y2,z+dh_n), (dl,y2,zt),    (x1,y2,zt))
            if dr < x2:
                _face4(bm, (dr,y2,z),    (x2,y2,z),    (x2,y2,z+dh_n),    (dr,y2,z+dh_n))
                _face4(bm, (dr,y2,z+dh_n), (x2,y2,z+dh_n), (x2,y2,zt),    (dr,y2,zt))
            _face4(bm, (dl,y2,z+dh_n), (dr,y2,z+dh_n), (dr,y2,zt),     (dl,y2,zt))
            _face4(bm, (dl,y2+fd,z),   (dl,y2,z),      (dl,y2,z+dh_n),    (dl,y2+fd,z+dh_n))
            _face4(bm, (dr,y2,z),      (dr,y2+fd,z),   (dr,y2+fd,z+dh_n), (dr,y2,z+dh_n))
            _face4(bm, (dl,y2,z+dh_n), (dr,y2,z+dh_n), (dr,y2+fd,z+dh_n), (dl,y2+fd,z+dh_n))
            _face4(bm, (dl,y2,z),      (dr,y2,z),      (dr,y2+fd,z),      (dl,y2+fd,z))
        else:
            if _ex:
                xs = [x1] + _ex + [x2]
                for i in range(len(xs) - 1):
                    _face4(bm, (xs[i],y2,z),           (xs[i+1],y2,z),           (xs[i+1],y2,z+_dh_fallback), (xs[i],y2,z+_dh_fallback))
                    _face4(bm, (xs[i],y2,z+_dh_fallback), (xs[i+1],y2,z+_dh_fallback), (xs[i+1],y2,zt),       (xs[i],y2,zt))
            else:
                _face4(bm, (x1,y2,z),    (x2,y2,z),    (x2,y2,z+_dh_fallback), (x1,y2,z+_dh_fallback))
                _face4(bm, (x1,y2,z+_dh_fallback), (x2,y2,z+_dh_fallback), (x2,y2,zt), (x1,y2,zt))

    # ── West inner panel (at x = x1, reveals go –X) ───────────────────────────
    if 'W' not in no_walls:
        dw_w = _dw('W'); dh_w = _dh('W')
        if 'W' in door_walls and s.add_door and dw_w > 0:
            cy = da.get('W', (y1 + y2) * 0.5)
            db = max(y1, cy - dw_w * 0.5)   # clamp: never south of room
            df = min(y2, cy + dw_w * 0.5)   # clamp: never north of room
            if db > y1:
                _face4(bm, (x1,y1,z),    (x1,db,z),    (x1,db,z+dh_w),    (x1,y1,z+dh_w))
                _face4(bm, (x1,y1,z+dh_w), (x1,db,z+dh_w), (x1,db,zt),    (x1,y1,zt))
            if df < y2:
                _face4(bm, (x1,df,z),    (x1,y2,z),    (x1,y2,z+dh_w),    (x1,df,z+dh_w))
                _face4(bm, (x1,df,z+dh_w), (x1,y2,z+dh_w), (x1,y2,zt),    (x1,df,zt))
            _face4(bm, (x1,db,z+dh_w), (x1,df,z+dh_w), (x1,df,zt),     (x1,db,zt))
            _face4(bm, (x1,db,z),      (x1-fd,db,z),    (x1-fd,db,z+dh_w), (x1,db,z+dh_w))
            _face4(bm, (x1-fd,df,z),   (x1,df,z),       (x1,df,z+dh_w),    (x1-fd,df,z+dh_w))
            _face4(bm, (x1,db,z+dh_w), (x1-fd,db,z+dh_w), (x1-fd,df,z+dh_w), (x1,df,z+dh_w))
            _face4(bm, (x1-fd,db,z),   (x1,db,z),       (x1,df,z),         (x1-fd,df,z))
        else:
            if _ey:
                ys = [y1] + _ey + [y2]
                for i in range(len(ys) - 1):
                    _face4(bm, (x1,ys[i],z),           (x1,ys[i+1],z),           (x1,ys[i+1],z+_dh_fallback), (x1,ys[i],z+_dh_fallback))
                    _face4(bm, (x1,ys[i],z+_dh_fallback), (x1,ys[i+1],z+_dh_fallback), (x1,ys[i+1],zt),       (x1,ys[i],zt))
            else:
                _face4(bm, (x1,y1,z),    (x1,y2,z),    (x1,y2,z+_dh_fallback), (x1,y1,z+_dh_fallback))
                _face4(bm, (x1,y1,z+_dh_fallback), (x1,y2,z+_dh_fallback), (x1,y2,zt), (x1,y1,zt))

    # ── East inner panel (at x = x2, reveals go +X) ───────────────────────────
    if 'E' not in no_walls:
        dw_e = _dw('E'); dh_e = _dh('E')
        if 'E' in door_walls and s.add_door and dw_e > 0:
            cy = da.get('E', (y1 + y2) * 0.5)
            db = max(y1, cy - dw_e * 0.5)   # clamp: never south of room
            df = min(y2, cy + dw_e * 0.5)   # clamp: never north of room
            if db > y1:
                _face4(bm, (x2,db,z),    (x2,y1,z),    (x2,y1,z+dh_e),    (x2,db,z+dh_e))
                _face4(bm, (x2,db,z+dh_e), (x2,y1,z+dh_e), (x2,y1,zt),    (x2,db,zt))
            if df < y2:
                _face4(bm, (x2,y2,z),    (x2,df,z),    (x2,df,z+dh_e),    (x2,y2,z+dh_e))
                _face4(bm, (x2,y2,z+dh_e), (x2,df,z+dh_e), (x2,df,zt),    (x2,y2,zt))
            _face4(bm, (x2,df,z+dh_e), (x2,db,z+dh_e), (x2,db,zt),     (x2,df,zt))
            _face4(bm, (x2+fd,db,z),   (x2,db,z),      (x2,db,z+dh_e),    (x2+fd,db,z+dh_e))
            _face4(bm, (x2,df,z),      (x2+fd,df,z),   (x2+fd,df,z+dh_e), (x2,df,z+dh_e))
            _face4(bm, (x2+fd,db,z+dh_e), (x2,db,z+dh_e), (x2,df,z+dh_e), (x2+fd,df,z+dh_e))
            _face4(bm, (x2,db,z),      (x2+fd,db,z),   (x2+fd,df,z),      (x2,df,z))
        else:
            if _ey:
                ys = [y1] + _ey + [y2]
                for i in range(len(ys) - 1):
                    _face4(bm, (x2,ys[i+1],z),           (x2,ys[i],z),           (x2,ys[i],z+_dh_fallback),   (x2,ys[i+1],z+_dh_fallback))
                    _face4(bm, (x2,ys[i+1],z+_dh_fallback), (x2,ys[i],z+_dh_fallback), (x2,ys[i],zt),         (x2,ys[i+1],zt))
            else:
                _face4(bm, (x2,y2,z),    (x2,y1,z),    (x2,y1,z+_dh_fallback), (x2,y2,z+_dh_fallback))
                _face4(bm, (x2,y2,z+_dh_fallback), (x2,y1,z+_dh_fallback), (x2,y1,zt), (x2,y2,zt))

    # ── Ceiling & Floor — split at door-column edges to share vertices ─────────

    if s.add_ceiling:
        if _ey:   # E/W door: strip in Y
            ys = [y1] + _ey + [y2]
            for i in range(len(ys) - 1):
                _face4(bm, (x1,ys[i],zt), (x1,ys[i+1],zt), (x2,ys[i+1],zt), (x2,ys[i],zt))
        elif _ex:  # N/S door: strip in X
            xs = [x1] + _ex + [x2]
            for i in range(len(xs) - 1):
                _face4(bm, (xs[i],y1,zt), (xs[i],y2,zt), (xs[i+1],y2,zt), (xs[i+1],y1,zt))
        else:
            _face4(bm, (x1,y1,zt), (x1,y2,zt), (x2,y2,zt), (x2,y1,zt))

    if s.add_floor:
        # Winding reversed vs ceiling so normal points +Z (up = toward room interior)
        if _ey:   # E/W door: strip in Y
            ys = [y1] + _ey + [y2]
            for i in range(len(ys) - 1):
                _face4(bm, (x1,ys[i],z), (x2,ys[i],z), (x2,ys[i+1],z), (x1,ys[i+1],z))
        elif _ex:  # N/S door: strip in X
            xs = [x1] + _ex + [x2]
            for i in range(len(xs) - 1):
                _face4(bm, (xs[i],y1,z), (xs[i+1],y1,z), (xs[i+1],y2,z), (xs[i],y2,z))
        else:
            _face4(bm, (x1,y1,z), (x2,y1,z), (x2,y2,z), (x1,y2,z))

    # ── Merge all coincident vertices ──────────────────────────────────────────
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)


def _make_room_obj(name, x1, y1, x2, y2, s, door_walls=(), no_walls=(),
                   door_anchors=None, collection=None):
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    _fill_room(bm, x1, y1, x2, y2, s.z_foundation, s, door_walls, no_walls, door_anchors)
    bm.to_mesh(me)
    bm.free()
    me.update()
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.collection).objects.link(obj)
    return obj


def _rebuild_room_mesh(reg, s):
    """Regenerate the mesh of an existing room in-place from its registry entry."""
    obj_name = reg.get("obj_name", "")
    if obj_name not in bpy.data.objects:
        return
    obj = bpy.data.objects[obj_name]
    me  = obj.data
    bm  = bmesh.new()
    _fill_room(bm, reg["x1"], reg["y1"], reg["x2"], reg["y2"],
               reg.get("z", s.z_foundation), s,
               door_walls=tuple(reg.get("door_walls", [])),
               no_walls=tuple(reg.get("no_walls", [])),
               door_anchors=reg.get("door_anchors", {}),
               door_width=reg.get("door_width"),
               door_height=reg.get("door_height"),
               door_dims=reg.get("door_dims", {}))
    bm.to_mesh(me)
    bm.free()
    me.update()


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


_SNAP_DIST = 0.5   # metres


def _wall_snap_ext(pt, rooms, t_fallback):
    """
    Find the nearest OUTER wall face within _SNAP_DIST, approached from the OUTSIDE.
    Returns (snap_2d, (nx,ny), room_idx, wall_char) or None.
    """
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
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
                # Snap new room so both door-frame exteriors meet flush (each frame depth = t/2)
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


def _wall_snap_any(pt, rooms, t_fallback):
    """
    Find the nearest wall face (from any direction) within _SNAP_DIST.
    Returns (room_idx, wall_char, anchor) or None.
    anchor = cursor position clamped along the wall span.
    """
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
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
    """Return (room_idx, wall_char) of the nearest door frame, or None.
    Only walls that already have a door are considered; cursor must be within
    the door-width span along that wall."""
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
        for wc in r.get("door_walls", []):
            dd = r.get("door_dims", {}).get(wc, {})
            dw = dd.get("w", r.get("door_width", s.door_width))
            anchor = r.get("door_anchors", {}).get(
                wc, (r["y1"] + r["y2"]) * 0.5 if wc in ('E', 'W')
                    else (r["x1"] + r["x2"]) * 0.5)
            x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
            t = r.get("t", t_fallback)
            if wc == 'E':
                dist  = abs(px - x2)
                along = py
                lo, hi = anchor - dw * 0.5 - t, anchor + dw * 0.5 + t
            elif wc == 'W':
                dist  = abs(px - x1)
                along = py
                lo, hi = anchor - dw * 0.5 - t, anchor + dw * 0.5 + t
            elif wc == 'N':
                dist  = abs(py - y2)
                along = px
                lo, hi = anchor - dw * 0.5 - t, anchor + dw * 0.5 + t
            else:  # S
                dist  = abs(py - y1)
                along = px
                lo, hi = anchor - dw * 0.5 - t, anchor + dw * 0.5 + t
            if dist < best_d and lo <= along <= hi:
                best_d = dist
                best   = (i, wc)
    return best


def _clamp_anchor(anchor, reg, wall_char, door_width, margin):
    """Clamp door centre so both edges stay >= margin from the wall's corners.
    If the wall is too narrow, centre the door instead of clamping."""
    if wall_char in ('E', 'W'):
        lo = reg["y1"] + door_width * 0.5 + margin
        hi = reg["y2"] - door_width * 0.5 - margin
    else:
        lo = reg["x1"] + door_width * 0.5 + margin
        hi = reg["x2"] - door_width * 0.5 - margin
    if lo > hi:   # wall too narrow — centre the door
        return ((reg["y1"] + reg["y2"]) * 0.5 if wall_char in ('E', 'W')
                else (reg["x1"] + reg["x2"]) * 0.5)
    return max(lo, min(hi, anchor))


def _door_frame_verts(r, wall_char, z, dw, dh):
    """Return 4 CCW 3-D corners of the door opening face for GPU highlighting."""
    anchor = r.get("door_anchors", {}).get(
        wall_char,
        (r["y1"] + r["y2"]) * 0.5 if wall_char in ('E', 'W')
                                    else (r["x1"] + r["x2"]) * 0.5)
    x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
    z0, z1 = z, z + dh
    if wall_char == 'E':
        lo = max(y1, anchor - dw * 0.5)
        hi = min(y2, anchor + dw * 0.5)
        return [(x2, lo, z0), (x2, hi, z0), (x2, hi, z1), (x2, lo, z1)]
    elif wall_char == 'W':
        lo = max(y1, anchor - dw * 0.5)
        hi = min(y2, anchor + dw * 0.5)
        return [(x1, hi, z0), (x1, lo, z0), (x1, lo, z1), (x1, hi, z1)]
    elif wall_char == 'N':
        lo = max(x1, anchor - dw * 0.5)
        hi = min(x2, anchor + dw * 0.5)
        return [(lo, y2, z0), (hi, y2, z0), (hi, y2, z1), (lo, y2, z1)]
    else:  # S
        lo = max(x1, anchor - dw * 0.5)
        hi = min(x2, anchor + dw * 0.5)
        return [(hi, y1, z0), (lo, y1, z0), (lo, y1, z1), (hi, y1, z1)]


def _solid_x_overlap(nx1, nx2, r, wall_char, s):
    """Return True if [nx1,nx2] touches any SOLID section of r's N or S wall.
    Returns False only if the overlap is entirely inside the door-frame opening.
    A single-point query (nx1==nx2) is also handled correctly."""
    rx1, rx2 = r["x1"], r["x2"]
    ox1, ox2 = max(nx1, rx1), min(nx2, rx2)
    if ox1 > ox2:        # truly no X overlap at all
        return False
    if wall_char in r.get("door_walls", []) and s.add_door:
        dw     = r.get("door_dims", {}).get(wall_char, {}).get("w", s.door_width)
        anchor = r.get("door_anchors", {}).get(wall_char, (rx1 + rx2) * 0.5)
        dl = max(rx1, anchor - dw * 0.5)
        dr = min(rx2, anchor + dw * 0.5)
        if ox1 >= dl and ox2 <= dr:
            return False   # entirely within door opening — exempt
    return True


def _solid_y_overlap(ny1, ny2, r, wall_char, s):
    """Return True if [ny1,ny2] touches any SOLID section of r's E or W wall.
    Returns False only if the overlap is entirely inside the door-frame opening.
    A single-point query (ny1==ny2) is also handled correctly."""
    ry1, ry2 = r["y1"], r["y2"]
    oy1, oy2 = max(ny1, ry1), min(ny2, ry2)
    if oy1 > oy2:        # truly no Y overlap at all
        return False
    if wall_char in r.get("door_walls", []) and s.add_door:
        dw     = r.get("door_dims", {}).get(wall_char, {}).get("w", s.door_width)
        anchor = r.get("door_anchors", {}).get(wall_char, (ry1 + ry2) * 0.5)
        db = max(ry1, anchor - dw * 0.5)
        df = min(ry2, anchor + dw * 0.5)
        if oy1 >= db and oy2 <= df:
            return False   # entirely within door opening — exempt
    return True


def _clamp_y_for_rooms(new_y, fixed_y, nx1, nx2, rooms, skip_idx, s):
    """Clamp new_y so the new room keeps >= wall_thickness from existing rooms.
    Solid walls require W clearance; door-frame openings allow touching (zero gap)."""
    t = s.wall_thickness
    if new_y == fixed_y:
        return new_y
    going_south = new_y < fixed_y
    for i, r in enumerate(rooms):
        if i == skip_idx:
            continue
        rx1, ry1, rx2, ry2 = r["x1"], r["y1"], r["x2"], r["y2"]
        if nx2 <= rx1 or nx1 >= rx2:
            continue
        if going_south:
            if ry2 <= fixed_y:
                if _solid_x_overlap(nx1, nx2, r, 'N', s):
                    new_y = max(new_y, ry2 + t)   # solid wall — keep W gap
                else:
                    new_y = max(new_y, ry2)        # door frame — allow touching
        else:
            if ry1 >= fixed_y:
                if _solid_x_overlap(nx1, nx2, r, 'S', s):
                    new_y = min(new_y, ry1 - t)
                else:
                    new_y = min(new_y, ry1)
    return new_y


def _clamp_x_for_rooms(new_x, fixed_x, ny1, ny2, rooms, skip_idx, s):
    """Clamp new_x so the new room keeps >= wall_thickness from existing rooms.
    Solid walls require W clearance; door-frame openings allow touching (zero gap)."""
    t = s.wall_thickness
    if new_x == fixed_x:
        return new_x
    going_west = new_x < fixed_x
    for i, r in enumerate(rooms):
        if i == skip_idx:
            continue
        rx1, ry1, rx2, ry2 = r["x1"], r["y1"], r["x2"], r["y2"]
        if ny2 <= ry1 or ny1 >= ry2:
            continue
        if going_west:
            if rx2 <= fixed_x:
                if _solid_y_overlap(ny1, ny2, r, 'E', s):
                    new_x = max(new_x, rx2 + t)   # solid wall — keep W gap
                else:
                    new_x = max(new_x, rx2)        # door frame — allow touching
        else:
            if rx1 >= fixed_x:
                if _solid_y_overlap(ny1, ny2, r, 'W', s):
                    new_x = min(new_x, rx1 - t)
                else:
                    new_x = min(new_x, rx1)
    return new_x


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


# ═════════════════════════════════════════════════════════════════════════════
# Registry persistence helpers
# ═════════════════════════════════════════════════════════════════════════════
def _reg_to_entry(reg, entry):
    """Copy a registry dict into a ROOM_PG_registry_entry PropertyGroup item."""
    entry.x1 = reg["x1"]
    entry.y1 = reg["y1"]
    entry.x2 = reg["x2"]
    entry.y2 = reg["y2"]
    entry.t  = reg.get("t", 0.125)
    entry.z  = reg.get("z", 0.0)
    entry.door_walls   = ",".join(reg.get("door_walls", []))
    entry.no_walls     = ",".join(reg.get("no_walls", []))
    entry.door_anchors = json.dumps(reg.get("door_anchors", {}))
    entry.door_dims    = json.dumps(reg.get("door_dims", {}))
    entry.obj_name     = reg.get("obj_name", "")
    entry.door_width   = reg.get("door_width", 0.9)
    entry.door_height  = reg.get("door_height", 2.0)


def _entry_to_reg(entry):
    """Convert a ROOM_PG_registry_entry back into a registry dict."""
    return {
        "x1": entry.x1, "y1": entry.y1,
        "x2": entry.x2, "y2": entry.y2,
        "t":  entry.t,
        "z":  entry.z,
        "door_walls":   [w for w in entry.door_walls.split(",") if w],
        "no_walls":     [w for w in entry.no_walls.split(",") if w],
        "door_anchors": json.loads(entry.door_anchors) if entry.door_anchors else {},
        "door_dims":    json.loads(entry.door_dims)    if entry.door_dims    else {},
        "obj_name":     entry.obj_name,
        "door_width":   entry.door_width,
        "door_height":  entry.door_height,
    }


def _sync_to_scene(context):
    """Persist _room_list into scene.room_registry so it survives addon reloads."""
    entries = getattr(context.scene, "room_registry", None)
    if entries is None:
        return
    entries.clear()
    for reg in ROOM_OT_draw._room_list:
        _reg_to_entry(reg, entries.add())


def _sync_from_scene(context):
    """Repopulate _room_list from scene.room_registry when the list is empty
    (e.g. after addon reload or Blender file load)."""
    if ROOM_OT_draw._room_list:
        return
    entries = getattr(context.scene, "room_registry", None)
    if not entries:
        return
    for entry in entries:
        if entry.obj_name in bpy.data.objects:
            ROOM_OT_draw._room_list.append(_entry_to_reg(entry))


# ═════════════════════════════════════════════════════════════════════════════
# GPU highlight helpers
# ═════════════════════════════════════════════════════════════════════════════
def _wall_face_verts(r, wall_char, z, h, t_fallback):
    """4 CCW 3-D corners of the outer face of wall_char for room dict r."""
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

    # ── GPU draw callback ─────────────────────────────────────────────────────
    def _draw_cb(self, context):
        if not self._hovered:
            return
        _sp, _normal, room_idx, wall_char = self._hovered
        rooms = ROOM_OT_draw._room_list
        if room_idx >= len(rooms):
            return
        s     = context.scene.room_settings
        r     = rooms[room_idx]
        verts = _wall_face_verts(r, wall_char, s.z_foundation,
                                 s.wall_height, s.wall_thickness)
        idx_fill = [(0,1,2),(0,2,3)]
        idx_line = [(0,1),(1,2),(2,3),(3,0)]
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('LESS_EQUAL')
        b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
        shader.uniform_float("color", (1.0, 0.55, 0.0, 0.28))
        b.draw(shader)
        gpu.state.line_width_set(2.5)
        b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
        shader.uniform_float("color", (1.0, 0.78, 0.0, 0.95))
        b2.draw(shader)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def _add_draw_handle(self, context):
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_VIEW')

    def _remove_draw_handle(self):
        if self._draw_handle:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(
                    self._draw_handle, 'WINDOW')
            except Exception:
                pass
            self._draw_handle = None

    # ── preview helpers ───────────────────────────────────────────────────────
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
        # Ensure non-degenerate even during phase-1 (one axis may be zero)
        x2 = max(x2, x1 + 0.01)
        y2 = max(y2, y1 + 0.01)
        me = self._prev.data
        me.clear_geometry()
        dw = _preview_door_walls(self._snap_info)
        nw = _snap_no_walls(self._snap_info)
        da = _preview_door_anchors(self._snap_info)
        bm = bmesh.new()
        _fill_room(bm, x1, y1, x2, y2, s.z_foundation, s, dw, nw, da)
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

    def _reset_placement(self, context):
        """Abort current placement, return to hover phase."""
        # Revert door added to existing room during DS phase
        if self._ds_room_idx is not None:
            rooms = ROOM_OT_draw._room_list
            if 0 <= self._ds_room_idx < len(rooms):
                ex_reg = rooms[self._ds_room_idx]
                if self._ds_prev_dw is not None:
                    ex_reg["door_walls"] = self._ds_prev_dw
                ex_reg.get("door_anchors", {}).pop(self._ds_wall_char, None)
                # Restore per-door dims for this wall char (or remove if it wasn't set)
                if self._ds_prev_door_dims_wc is not None:
                    ex_reg.setdefault("door_dims", {})[self._ds_wall_char] = self._ds_prev_door_dims_wc
                else:
                    ex_reg.get("door_dims", {}).pop(self._ds_wall_char, None)
                _rebuild_room_mesh(ex_reg, context.scene.room_settings)
        self._ds_room_idx         = None
        self._ds_wall_char        = None
        self._ds_prev_dw          = None
        self._ds_prev_door_dims_wc = None
        self._delete_preview()
        self._phase       = 0
        self._phase1_end  = None
        self._start       = None
        self._end         = None
        self._snap_info   = None
        self._axis1_char  = 'X'
        self._hovered     = None
        context.area.tag_redraw()
        self._msg(context,
            "LMB – draw room  |  Hover wall to snap-connect  |  Enter/RMB – exit")

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def invoke(self, context, event):
        if context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "Must be used inside the 3D Viewport")
            return {"CANCELLED"}
        _sync_from_scene(context)
        self._start        = None
        self._end          = None
        self._prev         = None
        self._hovered      = None
        self._snap_info    = None    # (normal, wc, room_idx) locked at click 1
        self._draw_handle  = None
        self._session_rooms = []    # [(obj, reg, ex_reg, prev_dw)] for Ctrl+Z
        # multi-phase placement state
        self._phase        = 0      # 0=hover, 'DS'=door slide, 1=first side, 2=second side/depth, 3=depth
        self._axis1_char   = 'X'   # axis moved in phase 1 ('X' or 'Y')
        self._phase1_end   = None  # end Vector locked at phase 1 → 2 transition
        # DS (door-slide) phase state — only used for snapped rooms
        self._ds_room_idx          = None  # existing room index during DS
        self._ds_wall_char         = None  # wall char on existing room during DS
        self._ds_prev_dw           = None  # door_walls before DS (used for cancel/undo)
        self._ds_prev_door_dims_wc = None  # door_dims[wc] before DS (None = wasn't set)
        self._add_draw_handle(context)
        context.window_manager.modal_handler_add(self)
        self._msg(context,
            "LMB – draw room  |  Hover wall to snap-connect  |  Enter/RMB – exit")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        s = context.scene.room_settings
        z = s.z_foundation

        # ── navigation pass-through (all phases) ──────────────────────────────
        if event.type in {'MIDDLEMOUSE',
                'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3',
                'NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7',
                'NUMPAD_8','NUMPAD_9','NUMPAD_DECIMAL','NUMPAD_PERIOD',
                'F', 'TILDE'}:
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
                    dims = {"w": dp.door_width, "h": dp.door_height}
                    ex_reg = ROOM_OT_draw._room_list[self._ds_room_idx]
                    wc     = self._ds_wall_char
                    ex_reg.setdefault("door_dims", {})[wc] = dims
                    _rebuild_room_mesh(ex_reg, s)
                    partner = _find_partner_wall(
                        ROOM_OT_draw._room_list, self._ds_room_idx, wc, s.wall_thickness)
                    if partner is not None:
                        p_idx, p_wc = partner
                        ROOM_OT_draw._room_list[p_idx].setdefault("door_dims", {})[p_wc] = dims
                        _rebuild_room_mesh(ROOM_OT_draw._room_list[p_idx], s)
                    _sync_to_scene(context)
                    context.area.header_text_set(
                        f"Preset: {dp.name} ({dp.door_width:.2f}×{dp.door_height:.2f}m)  |  "
                        "Scroll to cycle · Click to confirm  |  RMB – cancel")
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        # ── Ctrl+Z — undo last placed room (only in hover phase) ──────────────
        if event.type == 'Z' and event.ctrl and event.value == 'PRESS':
            if self._phase == 0 and self._session_rooms:
                obj, reg, ex_reg, prev_dw = self._session_rooms.pop()
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
                if ex_reg is not None and prev_dw is not None:
                    ex_reg["door_walls"] = prev_dw
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

        # ── exit / cancel ──────────────────────────────────────────────────────
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

        # ── mouse move ────────────────────────────────────────────────────────
        if event.type == "MOUSEMOVE":
            pt = _ray_to_z(context, event, z)

            if self._phase == 0:
                # Hover: update wall highlight
                if pt:
                    snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list, s.wall_thickness)
                    self._hovered = snap
                    self._msg(context,
                        "Click to connect from this wall  |  Enter/RMB – exit"
                        if snap else
                        "LMB – draw room  |  Hover wall to snap-connect  |  Enter/RMB – exit")
                    context.area.tag_redraw()
                return {'PASS_THROUGH'}

            if pt is None:
                return {'RUNNING_MODAL', 'PASS_THROUGH'}

            if self._phase == 'DS':
                # Door-slide: slide anchor along the existing wall with margin clamping
                ex_reg = ROOM_OT_draw._room_list[self._ds_room_idx]
                wc     = self._ds_wall_char
                dd     = ex_reg.get("door_dims", {}).get(wc, {})
                dw     = dd.get("w", s.door_width)
                raw    = pt.y if wc in ('E', 'W') else pt.x
                anchor = _clamp_anchor(raw, ex_reg, wc, dw, s.door_margin)
                if wc in ('E', 'W'):
                    self._start.y = anchor
                else:
                    self._start.x = anchor
                ex_reg.setdefault("door_anchors", {})[wc] = anchor
                _rebuild_room_mesh(ex_reg, s)
                self._end = self._start.copy()
                context.area.tag_redraw()

            elif self._phase == 1:
                # Phase 1: one side along axis1 from the snap/start point
                # Clamp against all other rooms at the snap-wall coordinate
                rooms_p1 = ROOM_OT_draw._room_list
                sidx_p1  = self._snap_info[2] if self._snap_info else -1
                if self._axis1_char == 'X':
                    nx = _clamp_x_for_rooms(pt.x, self._start.x,
                                            self._start.y, self._start.y,
                                            rooms_p1, sidx_p1, s)
                    self._end = Vector((nx, self._start.y, z))
                else:
                    ny = _clamp_y_for_rooms(pt.y, self._start.y,
                                            self._start.x, self._start.x,
                                            rooms_p1, sidx_p1, s)
                    self._end = Vector((self._start.x, ny, z))
                self._update_preview_mesh(context)

            elif self._phase == 2:
                if self._snap_info is not None:
                    # Snapped phase 2: second side — same axis, start locked at first-side endpoint
                    # Clamp this side as well so neither edge can enter another room
                    rooms_p2 = ROOM_OT_draw._room_list
                    sidx_p2  = self._snap_info[2]
                    if self._axis1_char == 'X':
                        nx = _clamp_x_for_rooms(pt.x, self._start.x,
                                                self._start.y, self._start.y,
                                                rooms_p2, sidx_p2, s)
                        self._end = Vector((nx, self._start.y, z))
                    else:
                        ny = _clamp_y_for_rooms(pt.y, self._start.y,
                                                self._start.x, self._start.x,
                                                rooms_p2, sidx_p2, s)
                        self._end = Vector((self._start.x, ny, z))
                else:
                    # Standalone phase 2: depth — perpendicular axis, phase1_end locked
                    rooms = ROOM_OT_draw._room_list
                    if self._axis1_char == 'X':
                        nx1 = min(self._start.x, self._phase1_end.x)
                        nx2 = max(self._start.x, self._phase1_end.x)
                        ny  = _clamp_y_for_rooms(pt.y, self._start.y,
                                                 nx1, nx2, rooms, -1, s)
                        self._end = Vector((self._phase1_end.x, ny, z))
                    else:
                        ny1 = min(self._start.y, self._phase1_end.y)
                        ny2 = max(self._start.y, self._phase1_end.y)
                        nx  = _clamp_x_for_rooms(pt.x, self._start.x,
                                                 ny1, ny2, rooms, -1, s)
                        self._end = Vector((nx, self._phase1_end.y, z))
                    _apply_snap_constraint(self._end, self._start, self._snap_info)
                self._update_preview_mesh(context)

            elif self._phase == 3:
                # Snapped phase 3: depth — perpendicular axis, Y (or X) range already locked
                rooms    = ROOM_OT_draw._room_list
                snap_idx = self._snap_info[2] if self._snap_info else -1
                if self._axis1_char == 'X':
                    nx1 = min(self._start.x, self._end.x)
                    nx2 = max(self._start.x, self._end.x)
                    ny  = _clamp_y_for_rooms(pt.y, self._start.y,
                                             nx1, nx2, rooms, snap_idx, s)
                    self._end = Vector((self._end.x, ny, z))
                else:
                    ny1 = min(self._start.y, self._end.y)
                    ny2 = max(self._start.y, self._end.y)
                    nx  = _clamp_x_for_rooms(pt.x, self._start.x,
                                             ny1, ny2, rooms, snap_idx, s)
                    self._end = Vector((nx, self._end.y, z))
                _apply_snap_constraint(self._end, self._start, self._snap_info)
                self._update_preview_mesh(context)

            return {'RUNNING_MODAL', 'PASS_THROUGH'}

        # ── left mouse ────────────────────────────────────────────────────────
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            pt = _ray_to_z(context, event, z)
            if pt is None:
                return {'RUNNING_MODAL'}

            # ── Click 1: lock start point ──────────────────────────────────
            if self._phase == 0:
                snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list, s.wall_thickness)
                if snap:
                    sp, normal, room_idx, wc = snap
                    self._start      = Vector((sp.x, sp.y, z))
                    self._snap_info  = (normal, wc, room_idx)
                    self._axis1_char = 'Y' if wc in ('E', 'W') else 'X'
                    self._hovered    = None
                    # Enter DS phase: add door to existing room at snap position
                    ex_reg = ROOM_OT_draw._room_list[room_idx]
                    self._ds_room_idx          = room_idx
                    self._ds_wall_char         = wc
                    self._ds_prev_dw           = list(ex_reg.get("door_walls", []))
                    self._ds_prev_door_dims_wc = ex_reg.get("door_dims", {}).get(wc)  # None if not set
                    raw_anchor = sp.y if wc in ('E', 'W') else sp.x
                    anchor = _clamp_anchor(raw_anchor, ex_reg, wc, s.door_width, s.door_margin)
                    if wc not in ex_reg.get("door_walls", []):
                        ex_reg.setdefault("door_walls", []).append(wc)
                    ex_reg.setdefault("door_anchors", {})[wc] = anchor
                    ex_reg.setdefault("door_dims", {})[wc] = {"w": s.door_width, "h": s.door_height}
                    _rebuild_room_mesh(ex_reg, s)
                    self._end  = self._start.copy()
                    self._phase = 'DS'
                    context.area.tag_redraw()
                    self._msg(context,
                        "Slide door along wall · Click to confirm position  |  RMB – cancel")
                else:
                    # Standalone room: skip DS, go straight to phase 1
                    self._start      = Vector((pt.x, pt.y, z))
                    self._snap_info  = None
                    self._axis1_char = 'X'
                    self._end   = self._start.copy()
                    self._phase = 1
                    self._create_preview(context)
                    self._msg(context,
                        "Move to set east/west width  |  Click to confirm  |  RMB – cancel")
                return {'RUNNING_MODAL'}

            # ── Click DS: confirm door position → phase 1 ─────────────────
            if self._phase == 'DS':
                self._end   = self._start.copy()
                self._phase = 1
                self._create_preview(context)
                self._msg(context, "Move to set first side  |  Click to confirm  |  RMB – cancel")
                return {'RUNNING_MODAL'}

            # ── Click 2: lock first side / standalone width ────────────────
            if self._phase == 1:
                axis_idx = 1 if self._axis1_char == 'Y' else 0
                if abs(self._end[axis_idx] - self._start[axis_idx]) < 0.05:
                    self._msg(context, "Too small — move further first")
                    return {'RUNNING_MODAL'}
                self._phase1_end = self._end.copy()
                if self._snap_info is not None:
                    # Snapped: advance start to the first-side endpoint, go to phase 2
                    if self._axis1_char == 'Y':
                        self._start.y = self._phase1_end.y
                    else:
                        self._start.x = self._phase1_end.x
                    self._end   = self._start.copy()
                    self._phase = 2
                    self._msg(context,
                        "Move to set other side  |  Click to confirm  |  RMB – cancel")
                else:
                    self._phase = 2
                    self._msg(context,
                        "Move to set north/south depth  |  Click to place  |  RMB – cancel")
                return {'RUNNING_MODAL'}

            # ── Click 3: lock second side (snapped) OR place (standalone) ──
            if self._phase == 2:
                if self._snap_info is not None:
                    # Snapped: lock second side, set up full Y/X range for phase 3
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
                # else: standalone — fall through to place

            # ── Click 4 (snapped) / Click 3 (standalone): place room ───────
            if self._phase in (2, 3):
                x1 = min(self._start.x, self._end.x)
                y1 = min(self._start.y, self._end.y)
                x2 = max(self._start.x, self._end.x)
                y2 = max(self._start.y, self._end.y)

                if x2 - x1 > 0.05 and y2 - y1 > 0.05:
                    dw  = _preview_door_walls(self._snap_info)
                    nw  = _snap_no_walls(self._snap_info)
                    da  = _preview_door_anchors(self._snap_info)

                    # Existing room's door was already added in DS phase.
                    # Capture prev_dw from before DS for undo correctness.
                    ex_reg  = None
                    prev_dw = None
                    if self._snap_info is not None:
                        _, wc, snap_idx = self._snap_info
                        if 0 <= snap_idx < len(ROOM_OT_draw._room_list):
                            ex_reg  = ROOM_OT_draw._room_list[snap_idx]
                            prev_dw = self._ds_prev_dw if self._ds_prev_dw is not None \
                                      else list(ex_reg.get("door_walls", []))
                            # Door already set during DS; ensure it's present
                            if wc not in ex_reg.get("door_walls", []):
                                ex_reg.setdefault("door_walls", []).append(wc)
                                _rebuild_room_mesh(ex_reg, s)

                    n        = len(ROOM_OT_draw._room_list) + 1
                    room_col = _room_target_collection(context, n)
                    obj      = _make_room_obj(f"Room.{n:03}", x1, y1, x2, y2, s,
                                              dw, nw, da, room_col)
                    # Inherit DS dims for the new room's connecting wall.
                    # _preview_door_walls returns (_OPPOSITE[wc],) so the new
                    # room's door wall char is _OPPOSITE[snap_wc], not snap_wc.
                    _snap_wc   = self._snap_info[1] if self._snap_info else None
                    _snap_dims = None
                    if _snap_wc is not None and ex_reg is not None:
                        _snap_dims = ex_reg.get("door_dims", {}).get(_snap_wc)
                    _opp_snap  = _OPPOSITE.get(_snap_wc, _snap_wc) if _snap_wc else None
                    reg = {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "t":  s.wall_thickness,
                        "z":  s.z_foundation,
                        "door_walls":   list(dw),
                        "no_walls":     list(nw),
                        "door_anchors": dict(da),
                        "door_dims":    {
                            _wc: (dict(_snap_dims) if _snap_dims and _wc == _opp_snap
                                  else {"w": s.door_width, "h": s.door_height})
                            for _wc in dw
                        },
                        "obj_name":     obj.name,
                        "door_width":   s.door_width,
                        "door_height":  s.door_height,
                    }
                    ROOM_OT_draw._room_list.append(reg)
                    self._session_rooms.append((obj, reg, ex_reg, prev_dw))
                    _sync_to_scene(context)
                    for o in list(context.selected_objects):
                        o.select_set(False)
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    self.report({"INFO"}, f"Created {obj.name}")
                else:
                    self._msg(context, "Room too small — skipped")

                # Clear DS state so _reset_placement doesn't revert the placed door
                self._ds_room_idx = self._ds_wall_char = self._ds_prev_dw = None
                self._reset_placement(context)
                self._msg(context,
                    "LMB – draw another room  |  Ctrl+Z – undo  |  Enter/RMB – exit")
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        self._delete_preview()
        self._remove_draw_handle()
        self._msg(context, None)


# ── module-level helpers ───────────────────────────────────────────────────────

def _apply_snap_constraint(end, start, snap_info):
    """Force *end* to only move outward from *start* in the wall-normal direction."""
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


def _preview_door_walls(snap_info):
    """
    Standalone rooms have no doors.
    Snapped rooms get a door on their connecting wall (opposite of the wall we snapped to).
    """
    if snap_info is None:
        return ()
    _normal, wc, _idx = snap_info
    return (_OPPOSITE[wc],)


def _snap_no_walls(snap_info):
    """Both rooms keep all their walls — doors appear on both connecting faces."""
    return ()


def _preview_door_anchors(snap_info):
    """
    Return door_anchors dict for the new room's connecting wall so its door
    centre aligns exactly with the existing room's door centre on that wall.
    """
    if snap_info is None:
        return {}
    _normal, wc, room_idx = snap_info
    rooms = ROOM_OT_draw._room_list
    if room_idx < 0 or room_idx >= len(rooms):
        return {}
    ex = rooms[room_idx]
    opp = _OPPOSITE[wc]
    # Use the stored anchor if available, otherwise derive from the existing room's extent
    if wc in ('E', 'W'):
        anchor = ex.get("door_anchors", {}).get(wc, (ex["y1"] + ex["y2"]) * 0.5)
    else:
        anchor = ex.get("door_anchors", {}).get(wc, (ex["x1"] + ex["x2"]) * 0.5)
    return {opp: anchor}


@bpy.app.handlers.persistent
def _room_registry_cleanup(scene, depsgraph):
    """Remove stale registry entries for deleted room objects."""
    ROOM_OT_draw._room_list[:] = [
        r for r in ROOM_OT_draw._room_list
        if r.get("obj_name", "") in bpy.data.objects
    ]


@bpy.app.handlers.persistent
def _room_on_load(*args):
    """Clear the in-memory registry when a new file is loaded so that
    _sync_from_scene repopulates it from the freshly loaded scene data."""
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

    _DRAG_PX = 8   # pixel threshold for press → slide transition

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
            # ── Green highlight on existing hovered door ──────────────────
            ri, wc = self._hovered_door
            if ri < len(rooms):
                r  = rooms[ri]
                dd = r.get("door_dims", {}).get(wc, {})
                dw = dd.get("w", r.get("door_width",  s.door_width))
                dh = dd.get("h", r.get("door_height", s.door_height))
                verts = _door_frame_verts(r, wc, r.get("z", s.z_foundation), dw, dh)
                b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                shader.uniform_float("color", (0.05, 0.85, 0.2, 0.30))
                b.draw(shader)
                gpu.state.line_width_set(2.5)
                b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                shader.uniform_float("color", (0.1, 1.0, 0.3, 1.0))
                b2.draw(shader)

        elif self._hover_snap is not None:
            # ── Orange ghost where a new door would land ──────────────────
            ri, wc, anchor = self._hover_snap
            if ri < len(rooms):
                r  = rooms[ri]
                # Use active preset dims for the ghost
                dw = s.door_width
                dh = s.door_height
                active  = context.scene.room_active_door_preset
                presets = context.scene.room_door_presets
                if 0 <= active < len(presets):
                    dp = presets[active]
                    dw, dh = dp.door_width, dp.door_height
                # Build a temporary reg with the snapped anchor for verts
                ghost = dict(r)
                ghost["door_anchors"] = dict(r.get("door_anchors", {}))
                ghost["door_anchors"][wc] = anchor
                verts = _door_frame_verts(ghost, wc, r.get("z", s.z_foundation), dw, dh)
                b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                shader.uniform_float("color", (1.0, 0.55, 0.0, 0.18))
                b.draw(shader)
                gpu.state.line_width_set(2.0)
                b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                shader.uniform_float("color", (1.0, 0.78, 0.0, 0.7))
                b2.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def _add_draw_handle(self, context):
        self._draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_VIEW')

    def _remove_draw_handle(self):
        if self._draw_handle:
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
        """Revert preset dims saved in _preview_orig back to the rooms."""
        if self._preview_orig is None:
            return
        poi = self._preview_orig
        if poi["room_idx"] < len(rooms):
            rooms[poi["room_idx"]].setdefault("door_dims", {})[poi["wall_char"]] = poi["dims"]
            _rebuild_room_mesh(rooms[poi["room_idx"]], s)
        if poi["partner"] is not None:
            p_idx, p_wc, p_dims = poi["partner"]
            if p_idx < len(rooms):
                rooms[p_idx].setdefault("door_dims", {})[p_wc] = p_dims
                _rebuild_room_mesh(rooms[p_idx], s)
        _sync_to_scene(context)
        self._preview_orig = None

    def _remove_active_door(self, context, rooms, s, room_idx, wall_char, partner):
        """Remove door frame from room (and partner) and rebuild."""
        reg = rooms[room_idx]
        if wall_char in reg.get("door_walls", []):
            reg["door_walls"].remove(wall_char)
        reg.get("door_anchors", {}).pop(wall_char, None)
        reg.get("door_dims",    {}).pop(wall_char, None)
        _rebuild_room_mesh(reg, s)
        if partner is not None:
            p_idx, p_wc = partner
            p_reg = rooms[p_idx]
            if p_wc in p_reg.get("door_walls", []):
                p_reg["door_walls"].remove(p_wc)
            p_reg.get("door_anchors", {}).pop(p_wc, None)
            p_reg.get("door_dims",    {}).pop(p_wc, None)
            _rebuild_room_mesh(p_reg, s)
        _sync_to_scene(context)

    def _go_hover(self, context):
        """Reset to HOVER state."""
        self._phase          = 'HOVER'
        self._active_idx     = None
        self._active_wc      = None
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

    # ── lifecycle ─────────────────────────────────────────────────────────
    def invoke(self, context, event):
        if context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "Must be used inside the 3D Viewport")
            return {"CANCELLED"}
        _sync_from_scene(context)
        self._phase           = 'HOVER'
        self._hovered_door    = None   # (room_idx, wall_char) — existing door
        self._hover_snap      = None   # (room_idx, wall_char, anchor) — wall snap
        self._last_press_time = 0.0
        self._last_press_door = None   # door hovered at last LMB press (for double-click)
        self._press_screen    = None   # (x, y) screen px at press start
        self._active_idx      = None
        self._active_wc       = None
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

        # ── Scroll — cycle presets while LMB is held, navigate otherwise ───
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
            if self._phase in ('LMB_DOWN', 'SLIDING') and self._active_idx is not None:
                presets = context.scene.room_door_presets
                active  = context.scene.room_active_door_preset
                if len(presets) > 0:
                    delta      = 1 if event.type == 'WHEELUPMOUSE' else -1
                    new_active = (active + delta) % len(presets)
                    context.scene.room_active_door_preset = new_active
                    dp   = presets[new_active]
                    dims = {"w": dp.door_width, "h": dp.door_height}
                    reg  = rooms[self._active_idx]
                    # Save original dims on first scroll
                    if self._preview_orig is None:
                        orig_dims    = dict(reg.get("door_dims", {}).get(
                                           self._active_wc,
                                           {"w": s.door_width, "h": s.door_height}))
                        partner_info = None
                        if self._active_partner is not None:
                            p_idx, p_wc = self._active_partner
                            p_dims = dict(rooms[p_idx].get("door_dims", {}).get(
                                         p_wc, {"w": s.door_width, "h": s.door_height}))
                            partner_info = (p_idx, p_wc, p_dims)
                        self._preview_orig = {
                            "room_idx":  self._active_idx,
                            "wall_char": self._active_wc,
                            "dims":      orig_dims,
                            "partner":   partner_info,
                        }
                    reg.setdefault("door_dims", {})[self._active_wc] = dims
                    _rebuild_room_mesh(reg, s)
                    if self._active_partner is not None:
                        p_idx, p_wc = self._active_partner
                        rooms[p_idx].setdefault("door_dims", {})[p_wc] = dims
                        _rebuild_room_mesh(rooms[p_idx], s)
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
                # Cancel: restore dims then anchor
                self._restore_preview(context, rooms, s)
                if self._active_idx is not None and self._orig_anchor is not None:
                    reg = rooms[self._active_idx]
                    reg.setdefault("door_anchors", {})[self._active_wc] = self._orig_anchor
                    _rebuild_room_mesh(reg, s)
                    if self._active_partner is not None:
                        p_idx, p_wc = self._active_partner
                        rooms[p_idx].setdefault("door_anchors", {})[p_wc] = self._orig_anchor
                        _rebuild_room_mesh(rooms[p_idx], s)
                if self._added_in_press and self._active_idx is not None:
                    self._remove_active_door(context, rooms, s,
                                             self._active_idx, self._active_wc,
                                             self._active_partner)
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
                    self._hover_snap   = (_wall_snap_any(pt, rooms, s.wall_thickness)
                                         if self._hovered_door is None else None)
                else:
                    self._hovered_door = None
                    self._hover_snap   = None
                context.area.tag_redraw()
                return {'RUNNING_MODAL', 'PASS_THROUGH'}

            elif self._phase == 'LMB_DOWN':
                # Check drag threshold
                if (self._press_screen and
                        (abs(event.mouse_region_x - self._press_screen[0]) > self._DRAG_PX or
                         abs(event.mouse_region_y - self._press_screen[1]) > self._DRAG_PX)):
                    self._phase = 'SLIDING'
                    context.area.header_text_set(
                        "Sliding door  ·  Scroll to change preset  |  "
                        "Release to confirm  ·  RMB=cancel")
                else:
                    return {'RUNNING_MODAL'}

            if self._phase == 'SLIDING' and pt is not None and self._active_idx is not None:
                reg = rooms[self._active_idx]
                dw  = reg.get("door_dims", {}).get(self._active_wc, {}).get("w", s.door_width)
                raw = pt.y if self._active_wc in ('E', 'W') else pt.x
                anchor = _clamp_anchor(raw, reg, self._active_wc, dw, s.door_margin)
                reg.setdefault("door_anchors", {})[self._active_wc] = anchor
                _rebuild_room_mesh(reg, s)
                if self._active_partner is not None:
                    p_idx, p_wc = self._active_partner
                    rooms[p_idx].setdefault("door_anchors", {})[p_wc] = anchor
                    _rebuild_room_mesh(rooms[p_idx], s)
                context.area.tag_redraw()

            return {'RUNNING_MODAL'}

        # ── Left mouse ────────────────────────────────────────────────────
        if event.type == "LEFTMOUSE":
            # Ignore clicks outside the region
            in_region = (0 <= event.mouse_region_x < context.region.width and
                         0 <= event.mouse_region_y < context.region.height)

            if event.value == "PRESS":
                if not in_region:
                    return {'PASS_THROUGH'}

                pt  = _ray_to_z(context, event, s.z_foundation)
                now = time.time()

                # ── Double-click: remove the door that was pressed last ────
                is_double = ((now - self._last_press_time) < 0.3 and
                             self._last_press_door is not None)
                if is_double and pt is not None:
                    current_door = _door_snap(pt, rooms, s, s.wall_thickness)
                    if current_door is not None and current_door == self._last_press_door:
                        ri, wc = current_door
                        partner = _find_partner_wall(rooms, ri, wc, s.wall_thickness)
                        self._remove_active_door(context, rooms, s, ri, wc, partner)
                        self._last_press_time = 0.0
                        self._last_press_door = None
                        self._hovered_door    = None
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}

                # ── Single press ──────────────────────────────────────────
                self._last_press_time = now
                self._press_screen    = (event.mouse_region_x, event.mouse_region_y)

                if pt is None:
                    return {'PASS_THROUGH'}

                if self._hovered_door is not None:
                    # Press on existing door → ready to slide / scroll
                    ri, wc = self._hovered_door
                    self._last_press_door = (ri, wc)
                    self._active_idx      = ri
                    self._active_wc       = wc
                    self._active_partner  = _find_partner_wall(rooms, ri, wc, s.wall_thickness)
                    reg = rooms[ri]
                    self._orig_anchor     = reg.get("door_anchors", {}).get(
                        wc, (reg["y1"] + reg["y2"]) * 0.5 if wc in ('E', 'W')
                            else (reg["x1"] + reg["x2"]) * 0.5)
                    self._added_in_press  = False
                    self._preview_orig    = None
                    self._phase           = 'LMB_DOWN'
                    context.area.header_text_set(
                        "Hold+drag=slide  ·  Hold+scroll=change preset  |  "
                        "Release=confirm  ·  RMB=cancel")

                elif self._hover_snap is not None:
                    # Press on wall → add new door
                    ri, wc, raw_anchor = self._hover_snap
                    self._last_press_door = None   # new door: no double-click remove
                    if 0 <= ri < len(rooms):
                        reg = rooms[ri]
                        # Use active preset dims (or settings fallback)
                        dw = s.door_width
                        dh = s.door_height
                        active  = context.scene.room_active_door_preset
                        presets = context.scene.room_door_presets
                        if 0 <= active < len(presets):
                            dp = presets[active]
                            dw, dh = dp.door_width, dp.door_height
                        anchor = _clamp_anchor(raw_anchor, reg, wc, dw, s.door_margin)
                        if wc not in reg.get("door_walls", []):
                            reg.setdefault("door_walls", []).append(wc)
                        reg.setdefault("door_anchors", {})[wc] = anchor
                        reg.setdefault("door_dims",    {})[wc] = {"w": dw, "h": dh}
                        _rebuild_room_mesh(reg, s)
                        partner = _find_partner_wall(rooms, ri, wc, s.wall_thickness)
                        if partner is not None:
                            p_idx, p_wc = partner
                            p_reg = rooms[p_idx]
                            if p_wc not in p_reg.get("door_walls", []):
                                p_reg.setdefault("door_walls", []).append(p_wc)
                            p_reg.setdefault("door_anchors", {})[p_wc] = anchor
                            p_reg.setdefault("door_dims",    {})[p_wc] = {"w": dw, "h": dh}
                            _rebuild_room_mesh(p_reg, s)
                        self._active_idx      = ri
                        self._active_wc       = wc
                        self._active_partner  = partner
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
                    # Finalize: sync and return to hover
                    self._preview_orig = None   # dims are confirmed
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

        L.separator()
        L.operator("room.clear_registry", icon="X", text="Reset Snap Registry")


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

        # ── Unified door-edit mode button ─────────────────────────────────
        col = L.column(align=True)
        col.scale_y = 1.3
        is_editing = getattr(context.scene, "room_door_edit_active", False)
        col.operator("room.door_edit", icon="OUTLINER_OB_EMPTY",
                     text="Door Frame Edit Mode  (Ctrl+Shift+D)", depress=is_editing)
        col.separator()

        # ── Door default dimensions + margin ─────────────────────────────
        col = L.column(align=True)
        col.prop(s, "door_width")
        col.prop(s, "door_height")
        col.prop(s, "door_margin")
        col.operator("room.save_door_preset", icon="ADD", text="Save Door Preset")

        # ── Preset list ───────────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════════════════════
# Register / Unregister
# ═════════════════════════════════════════════════════════════════════════════
_classes = (
    ROOM_PG_registry_entry,
    ROOM_PG_floor,
    ROOM_PG_door_preset,
    ROOM_PG_settings,
    ROOM_OT_draw,
    ROOM_OT_clear,
    ROOM_OT_door_edit,
    ROOM_OT_add_floor,
    ROOM_OT_select_floor,
    ROOM_OT_remove_floor,
    ROOM_OT_save_door_preset,
    ROOM_OT_select_door_preset,
    ROOM_OT_remove_door_preset,
    ROOM_PT_panel,
    ROOM_PT_door_panel,
)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.room_settings           = bpy.props.PointerProperty(type=ROOM_PG_settings)
    bpy.types.Scene.room_floors             = bpy.props.CollectionProperty(type=ROOM_PG_floor)
    bpy.types.Scene.room_active_floor       = bpy.props.IntProperty(default=-1)
    bpy.types.Scene.room_door_presets       = bpy.props.CollectionProperty(type=ROOM_PG_door_preset)
    bpy.types.Scene.room_active_door_preset = bpy.props.IntProperty(default=-1)
    bpy.types.Scene.room_registry            = bpy.props.CollectionProperty(type=ROOM_PG_registry_entry)
    bpy.types.Scene.room_door_edit_active   = bpy.props.BoolProperty(default=False)

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

    if _room_registry_cleanup not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_room_registry_cleanup)
    if _room_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_room_on_load)


def unregister():
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
                 "room_registry", "room_door_edit_active"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)

    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
