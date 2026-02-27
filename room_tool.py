bl_info = {
    "name": "Room Tool",
    "version": (1, 2),
    "blender": (3, 0, 0),
    "location": "3D View ▸ N-Panel ▸ Room Tool  |  Shift+R",
    "description": "Click-drag to draw rooms with walls, ceiling and door frame",
    "category": "Mesh",
}

import bpy, bmesh, gpu
from bpy_extras import view3d_utils
from mathutils import Vector
from gpu_extras.batch import batch_for_shader

_OPPOSITE = {'S': 'N', 'N': 'S', 'E': 'W', 'W': 'E'}


# ═════════════════════════════════════════════════════════════════════════════
# Settings property group
# ═════════════════════════════════════════════════════════════════════════════
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
    add_door    : bpy.props.BoolProperty(name="Add Door Frame", default=True)
    door_width  : bpy.props.FloatProperty(
        name="Door Width",  default=0.9, min=0.1, max=10.0, unit="LENGTH")
    door_height : bpy.props.FloatProperty(
        name="Door Height", default=2.0, min=0.1, max=15.0, unit="LENGTH")


# ═════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═════════════════════════════════════════════════════════════════════════════
def _box(bm, x0, y0, z0, x1, y1, z1):
    vv = [bm.verts.new(co) for co in [
        (x0,y0,z0), (x1,y0,z0), (x1,y1,z0), (x0,y1,z0),
        (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (x0,y1,z1)]]
    for idx in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,4,7,3),(1,2,6,5)]:
        bm.faces.new([vv[i] for i in idx])


def _door_XY(bm, ax0, ay0, z, ax1, ay1, zt, dw, dh, t, axis):
    """
    Cut a door opening in a wall slab.
    axis='X': slab spans ax0..ax1 in X, constant Y band ay0..ay1 — door centred in X.
    axis='Y': slab spans ay0..ay1 in Y, constant X band ax0..ax1 — door centred in Y.
    """
    if axis == 'X':
        cx  = (ax0 + ax1) * 0.5
        dl, dr = cx - dw * 0.5, cx + dw * 0.5
        _box(bm, ax0, ay0, z,    dl,  ay1, zt)          # left column
        _box(bm, dr,  ay0, z,    ax1, ay1, zt)          # right column
        _box(bm, dl,  ay0, z+dh, dr,  ay1, zt)          # lintel
    else:   # axis == 'Y'
        cy  = (ay0 + ay1) * 0.5
        db, df = cy - dw * 0.5, cy + dw * 0.5
        _box(bm, ax0, ay0, z,    ax1, db,  zt)          # back column
        _box(bm, ax0, df,  z,    ax1, ay1, zt)          # front column
        _box(bm, ax0, db,  z+dh, ax1, df,  zt)          # lintel


def _fill_room(bm, x1, y1, x2, y2, z, s, door_walls=()):
    """
    Populate *bm* with room geometry.
    Inner floor: (x1,y1)–(x2,y2).  Wall thickness extends outward.
    door_walls: tuple of wall chars ('S','N','E','W') that get an opening.
    """
    t  = s.wall_thickness
    h  = s.wall_height
    zt = z + h
    dw = min(s.door_width,  (x2 - x1) * 0.85) if s.add_door else 0
    dh = min(s.door_height, h - t)              if s.add_door else 0

    # ── South wall (y = y1, faces –Y) ────────────────────────────────────────
    if 'S' in door_walls and s.add_door:
        _door_XY(bm, x1-t, y1-t, z, x2+t, y1, zt, dw, dh, t, 'X')
    else:
        _box(bm, x1-t, y1-t, z, x2+t, y1, zt)

    # ── North wall (y = y2, faces +Y) ────────────────────────────────────────
    if 'N' in door_walls and s.add_door:
        _door_XY(bm, x1-t, y2, z, x2+t, y2+t, zt, dw, dh, t, 'X')
    else:
        _box(bm, x1-t, y2, z, x2+t, y2+t, zt)

    # ── West wall (x = x1, faces –X) ─────────────────────────────────────────
    if 'W' in door_walls and s.add_door:
        _door_XY(bm, x1-t, y1, z, x1, y2, zt, dw, dh, t, 'Y')
    else:
        _box(bm, x1-t, y1, z, x1, y2, zt)

    # ── East wall (x = x2, faces +X) ─────────────────────────────────────────
    if 'E' in door_walls and s.add_door:
        _door_XY(bm, x2, y1, z, x2+t, y2, zt, dw, dh, t, 'Y')
    else:
        _box(bm, x2, y1, z, x2+t, y2, zt)

    # ── Ceiling ───────────────────────────────────────────────────────────────
    if s.add_ceiling:
        _box(bm, x1-t, y1-t, zt, x2+t, y2+t, zt + t)


def _make_room_obj(name, x1, y1, x2, y2, s, door_walls=(), collection=None):
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    _fill_room(bm, x1, y1, x2, y2, s.z_foundation, s, door_walls)
    bm.to_mesh(me)
    bm.free()
    me.update()
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.collection).objects.link(obj)
    return obj


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
    Find the nearest OUTER wall face within _SNAP_DIST.
    Returns (snap_2d, (nx,ny), room_idx, wall_char)  or  None.
    Uses each room's stored thickness; falls back to t_fallback.
    """
    best_d, best = _SNAP_DIST, None
    px, py = pt.x, pt.y
    for i, r in enumerate(rooms):
        x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
        t = r.get("t", t_fallback)
        # (wall_char, outer_face_coord, perp_dist, along_coord, lo, hi, normal)
        checks = [
            ('E', x2+t, abs(px-(x2+t)), py,  y1-t, y2+t, ( 1, 0)),
            ('W', x1-t, abs(px-(x1-t)), py,  y1-t, y2+t, (-1, 0)),
            ('N', y2+t, abs(py-(y2+t)), px,  x1-t, x2+t, ( 0, 1)),
            ('S', y1-t, abs(py-(y1-t)), px,  x1-t, x2+t, ( 0,-1)),
        ]
        for wc, face_pos, dist, along, lo, hi, normal in checks:
            if dist < best_d and lo <= along <= hi:
                if wc in ('E', 'W'):
                    sp = Vector((face_pos, max(y1, min(y2, py))))
                else:
                    sp = Vector((max(x1, min(x2, px)), face_pos))
                best_d = dist
                best   = (sp, normal, i, wc)
    return best


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
    else:   # S
        return [(x2+rt,y1-rt,z0),(x1-rt,y1-rt,z0),(x1-rt,y1-rt,z1),(x2+rt,y1-rt,z1)]


# ═════════════════════════════════════════════════════════════════════════════
# Modal draw operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_draw(bpy.types.Operator):
    bl_idname  = "room.draw"
    bl_label   = "Draw Room"
    bl_options = {"REGISTER", "UNDO"}

    _room_list   = []   # class-level room registry
    _addon_kmaps = []   # class-level keymap entries

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
        # semi-transparent fill
        b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
        shader.uniform_float("color", (1.0, 0.55, 0.0, 0.28))
        b.draw(shader)
        # bright border
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

    # ── preview helpers (create-once / update-in-place / delete-once) ─────────
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
        me = self._prev.data
        me.clear_geometry()
        if x2 - x1 > 0.05 and y2 - y1 > 0.05:
            dw = _preview_door_walls(self._snap_info, s)
            bm = bmesh.new()
            _fill_room(bm, x1, y1, x2, y2, s.z_foundation, s, dw)
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

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def invoke(self, context, event):
        if context.area.type != "VIEW_3D":
            self.report({"WARNING"}, "Must be used inside the 3D Viewport")
            return {"CANCELLED"}
        self._start       = None
        self._end         = None
        self._prev        = None
        self._hovered     = None   # current wall snap result for highlight
        self._snap_info   = None   # (normal_tuple, wall_char) locked at drag start
        self._draw_handle = None
        self._add_draw_handle(context)
        context.window_manager.modal_handler_add(self)
        self._msg(context,
            "LMB drag – draw room  |  Hover room wall to snap-connect  |  Enter/RMB – exit")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        s = context.scene.room_settings
        z = s.z_foundation

        # ── navigation pass-through ───────────────────────────────────────────
        if event.type in {
                'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3',
                'NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7',
                'NUMPAD_8','NUMPAD_9','NUMPAD_DECIMAL','NUMPAD_PERIOD',
                'F', 'TILDE'}:
            return {'PASS_THROUGH'}

        # ── exit ──────────────────────────────────────────────────────────────
        if event.type in {"RIGHTMOUSE", "ESC", "RET", "NUMPAD_ENTER"}:
            self._delete_preview()
            self._remove_draw_handle()
            self._msg(context, None)
            return {"FINISHED" if event.type in {"RET", "NUMPAD_ENTER"} else "CANCELLED"}

        # ── mouse move ────────────────────────────────────────────────────────
        if event.type == "MOUSEMOVE":
            pt = _ray_to_z(context, event, z)

            if self._start is None:
                # Hover — update highlight, allow orbit
                if pt:
                    snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list,
                                         s.wall_thickness)
                    self._hovered = snap
                    self._msg(context,
                        "Click to connect room from this wall  |  Enter/RMB – exit"
                        if snap else
                        "LMB drag – draw room  |  Hover wall to snap-connect  |  Enter/RMB – exit")
                    context.area.tag_redraw()
                return {'PASS_THROUGH'}
            else:
                # Drawing — constrain drag direction then refresh preview
                if pt:
                    self._end = Vector((pt.x, pt.y, z))
                    _apply_snap_constraint(self._end, self._start, self._snap_info)
                    self._update_preview_mesh(context)
                return {'RUNNING_MODAL'}

        # ── left mouse ────────────────────────────────────────────────────────
        if event.type == "LEFTMOUSE":

            if event.value == "PRESS":
                pt = _ray_to_z(context, event, z)
                if pt is None:
                    return {'RUNNING_MODAL'}
                snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list, s.wall_thickness)
                if snap:
                    sp, normal, _idx, wc = snap
                    self._start     = Vector((sp.x, sp.y, z))
                    self._snap_info = (normal, wc)
                    self._hovered   = None   # hide highlight while drawing
                    context.area.tag_redraw()
                else:
                    self._start     = Vector((pt.x, pt.y, z))
                    self._snap_info = None
                self._end = self._start.copy()
                self._create_preview(context)
                self._msg(context, "Drag to size the room  |  Release to place")
                return {'RUNNING_MODAL'}

            if event.value == "RELEASE" and self._start is not None:
                self._delete_preview()

                x1 = min(self._start.x, self._end.x)
                y1 = min(self._start.y, self._end.y)
                x2 = max(self._start.x, self._end.x)
                y2 = max(self._start.y, self._end.y)

                if x2 - x1 > 0.05 and y2 - y1 > 0.05:
                    dw  = _preview_door_walls(self._snap_info, s)
                    n   = len(ROOM_OT_draw._room_list) + 1
                    obj = _make_room_obj(f"Room.{n:03}", x1, y1, x2, y2, s,
                                         dw, context.collection)
                    ROOM_OT_draw._room_list.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "t":  s.wall_thickness,
                    })
                    for o in list(context.selected_objects):
                        o.select_set(False)
                    obj.select_set(True)
                    context.view_layer.objects.active = obj
                    self.report({"INFO"}, f"Created {obj.name}")

                self._start = self._end = self._snap_info = None
                self._msg(context,
                    "LMB drag – draw another room  |  Enter/RMB – done")
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def cancel(self, context):
        self._delete_preview()
        self._remove_draw_handle()
        self._msg(context, None)


# ── module-level helpers used by the operator ─────────────────────────────────

def _apply_snap_constraint(end, start, snap_info):
    """Force *end* to only move outward from *start* in the wall-normal direction."""
    if snap_info is None:
        return
    (nx, ny), _wc = snap_info
    if nx != 0:   # X-normal wall (East/West)
        if nx > 0:
            end.x = max(end.x, start.x + 0.05)
        else:
            end.x = min(end.x, start.x - 0.05)
    else:         # Y-normal wall (North/South)
        if ny > 0:
            end.y = max(end.y, start.y + 0.05)
        else:
            end.y = min(end.y, start.y - 0.05)


def _preview_door_walls(snap_info, s):
    """Return the door_walls tuple for the current drag."""
    if not s.add_door:
        return ()
    if snap_info is None:
        return ('S',)
    _normal, wc = snap_info
    return (wc, _OPPOSITE[wc])   # shared wall + outer wall


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

        box = L.box()
        col = box.column(align=True)
        col.label(text="Wall Dimensions:", icon="MOD_BUILD")
        col.prop(s, "wall_height")
        col.prop(s, "wall_thickness")

        box = L.box()
        col = box.column(align=True)
        col.label(text="Features:", icon="MODIFIER")
        col.prop(s, "add_ceiling")
        col.prop(s, "add_door")
        if s.add_door:
            sub = col.column(align=True)
            sub.prop(s, "door_width")
            sub.prop(s, "door_height")

        L.separator()
        L.operator("room.clear_registry", icon="X", text="Reset Snap Registry")


# ═════════════════════════════════════════════════════════════════════════════
# Register / Unregister
# ═════════════════════════════════════════════════════════════════════════════
_classes = (ROOM_PG_settings, ROOM_OT_draw, ROOM_OT_clear, ROOM_PT_panel)


def register():
    for c in _classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.room_settings = bpy.props.PointerProperty(
        type=ROOM_PG_settings)

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


def unregister():
    for km, kmi in ROOM_OT_draw._addon_kmaps:
        try: km.keymap_items.remove(kmi)
        except Exception: pass
    ROOM_OT_draw._addon_kmaps.clear()

    if hasattr(bpy.types.Scene, "room_settings"):
        del bpy.types.Scene.room_settings

    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
