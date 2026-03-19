bl_info = {
    "name": "Room Tool",
    "version": (1, 4),
    "blender": (3, 0, 0),
    "location": "3D View ▸ N-Panel ▸ Room Tool  |  Shift+R",
    "description": "Click to draw rooms. Connected rooms share aligned door openings.",
    "category": "Mesh",
}

import bpy, bmesh, gpu, json, time, math, blf
from bpy_extras import view3d_utils
from mathutils import Vector
from gpu_extras.batch import batch_for_shader

_OPPOSITE = {'S': 'N', 'N': 'S', 'E': 'W', 'W': 'E'}

# Tracks every active draw-handler token so ROOM_OT_clear_overlays can
# nuke them all even if the owning operator crashed without calling
# _remove_draw_handle.
_DRAW_HANDLES: set = bpy.app.driver_namespace.setdefault("_room_tool_draw_handles", set())


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
    mesh_object: bpy.props.PointerProperty(
        type=bpy.types.Object, name="Mesh",
        description="Optional mesh placed as a linked instance at each door opening. "
                    "Origin should be at the bottom-centre of the door. "
                    "Scaled on X (width) and Z (height) to fit the opening.")


class ROOM_PG_window_preset(bpy.types.PropertyGroup):
    """One saved window-size preset (name inherited from PropertyGroup)."""
    window_width : bpy.props.FloatProperty(
        name="Width",       default=1.0, min=0.1, max=10.0, unit="LENGTH")
    window_height: bpy.props.FloatProperty(
        name="Height",      default=1.2, min=0.1, max=15.0, unit="LENGTH")
    v_offset     : bpy.props.FloatProperty(
        name="Sill Height", default=0.9, min=0.0, max=10.0, unit="LENGTH")
    mesh_object  : bpy.props.PointerProperty(
        type=bpy.types.Object, name="Mesh",
        description="Optional mesh placed as a linked instance at each window opening. "
                    "Origin should be at the bottom-centre of the window sill. "
                    "Scaled on X (width) and Z (height) to fit the opening.")


class ROOM_PG_arch_preset(bpy.types.PropertyGroup):
    """One saved architrave preset (name inherited from PropertyGroup)."""
    width : bpy.props.FloatProperty(name="Width", default=0.07, min=0.01, max=0.5, unit="LENGTH")
    depth : bpy.props.FloatProperty(name="Depth", default=0.015, min=0.002, max=0.2, unit="LENGTH")


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
    plinth_bottom_enabled : bpy.props.BoolProperty(default=False)
    plinth_top_enabled    : bpy.props.BoolProperty(default=False)
    stairs_json           : bpy.props.StringProperty(default="[]")
    mesh_locked           : bpy.props.BoolProperty(default=False)


def _cb_mat(self, context):
    """PointerProperty update: re-apply materials — to all rooms or just the selected one."""
    s = context.scene.room_settings
    if getattr(s, 'mat_apply_mode', 'ALL') == 'SELECTED' and context.active_object:
        reg = next((r for r in ROOM_OT_draw._room_list
                    if r.get('obj_name') == context.active_object.name), None)
        if reg is not None:
            _apply_materials_one_room(reg, s)
            return
    _apply_materials_all_rooms(s)


def _cb_tiling(self, context):
    """FloatProperty update: recompute UV — on all rooms or just the selected one."""
    s = context.scene.room_settings
    if getattr(s, 'mat_apply_mode', 'ALL') == 'SELECTED' and context.active_object:
        reg = next((r for r in ROOM_OT_draw._room_list
                    if r.get('obj_name') == context.active_object.name), None)
        if reg is not None:
            _apply_uvs_one_room(reg, s)
            return
    _apply_uvs_all_rooms(s)


def _cb_stair_mat(self, context):
    """PointerProperty update: re-apply materials to all existing stairs."""
    _apply_stair_materials_all(context.scene.room_settings)


def _cb_stair_tiling(self, context):
    """FloatProperty update: recompute UV on all existing stairs."""
    _apply_stair_uvs_all(context.scene.room_settings)


def _cb_rebuild(self, context):
    """BoolProperty/FloatProperty update: full geometry rebuild on all existing rooms."""
    s = context.scene.room_settings
    for reg in ROOM_OT_draw._room_list:
        _rebuild_room_mesh(reg, s)


def _cb_threshold(self, context):
    """Update callback: sync threshold strips across all rooms when settings change."""
    _sync_all_thresholds(context.scene.room_settings)


def _cb_stair_hole(self, context):
    """FloatProperty update: rescale stair hole length along travel axis and rebuild rooms."""
    s = context.scene.room_settings
    ratio = s.stair_hole_ratio
    for rd in ROOM_OT_draw._room_list:
        changed = False
        for sh in rd.get("stair_holes", []):
            fc = sh.get("fixed_coord")
            mc = sh.get("moving_coord_full")
            if fc is None or mc is None:
                continue
            ax = sh.get("resize_axis", "y")
            if sh.get("hi_is_fixed", True):
                sh[ax + "1"] = fc - ratio * (fc - mc)
                sh[ax + "2"] = fc
            else:
                sh[ax + "1"] = fc
                sh[ax + "2"] = fc + ratio * (mc - fc)
            changed = True
        if changed:
            _rebuild_room_mesh(rd, s)


def _cb_rebuild_plinth(self, context):
    """Update callback for add_plinth_bottom / add_plinth_top scene toggles.
    Syncs the per-room flags to the new scene value then rebuilds all rooms."""
    s = context.scene.room_settings
    for reg in ROOM_OT_draw._room_list:
        reg['plinth_bottom_enabled'] = s.add_plinth_bottom
        reg['plinth_top_enabled']    = s.add_plinth_top
        _rebuild_room_mesh(reg, s)


_ARCHITRAVE_PRESETS = {
    'thin':     (0.04, 0.008),
    'standard': (0.07, 0.015),
    'wide':     (0.10, 0.020),
    'bold':     (0.13, 0.025),
}


def _cb_architrave_preset(self, context):
    """Update callback for architrave_preset — sets width/depth from preset dict."""
    s = context.scene.room_settings
    p = s.architrave_preset
    if p in _ARCHITRAVE_PRESETS:
        # Setting these fires _cb_rebuild via their own update= callbacks
        s.architrave_width = _ARCHITRAVE_PRESETS[p][0]
        s.architrave_depth = _ARCHITRAVE_PRESETS[p][1]


def _rebuild_stair_steps(context, use_depth):
    """Real-time rebuild of the active stair when the rise/depth slider changes.
    If no stair object is active the value is simply stored as a default for
    the next new stair — nothing is rebuilt."""
    obj = context.active_object
    if obj is None or "room_stair" not in obj:
        return
    sd = next((d for d in ROOM_OT_draw._stair_list
               if d.get("obj_name") == obj.name), None)
    if sd is None:
        return
    s    = context.scene.room_settings
    dz   = sd.get("z_top", 0) - sd.get("z_bot", 0)
    x_t  = sd.get("x_travel", True)
    lx1_ = min(sd.get("lx1", 0), sd.get("lx2", 0))
    lx2_ = max(sd.get("lx1", 0), sd.get("lx2", 0))
    ly1_ = min(sd.get("ly1", 0), sd.get("ly2", 0))
    ly2_ = max(sd.get("ly1", 0), sd.get("ly2", 0))
    ux1_ = min(sd.get("ux1", lx1_), sd.get("ux2", lx2_))
    ux2_ = max(sd.get("ux1", lx1_), sd.get("ux2", lx2_))
    uy1_ = min(sd.get("uy1", ly1_), sd.get("uy2", ly2_))
    uy2_ = max(sd.get("uy1", ly1_), sd.get("uy2", ly2_))
    if x_t:
        tt = abs(ux2_ - lx2_) if (ux1_+ux2_)/2 >= (lx1_+lx2_)/2 else abs(lx1_ - ux1_)
    else:
        tt = abs(uy2_ - ly2_) if (uy1_+uy2_)/2 >= (ly1_+ly2_)/2 else abs(ly1_ - uy1_)
    if use_depth:
        n = max(2, round(tt / max(0.05, s.stair_depth)))
    else:
        n = max(2, round(dz / max(0.05, s.stair_rise)))
    sd["step_rise"] = dz / n
    try:
        verts, faces, cats = _build_stair_mesh(sd, s)
        if not verts:
            return
        loc = obj.location
        verts = [(v[0] - loc.x, v[1] - loc.y, v[2] - loc.z) for v in verts]
        old_me = obj.data
        me = bpy.data.meshes.new(obj.name + "_rebuilt")
        bm = bmesh.new()
        cat_layer = bm.faces.layers.int.new("stair_cat")
        for v in verts:
            bm.verts.new(v)
        bm.verts.ensure_lookup_table()
        for fi, fidx in enumerate(faces):
            try:
                face = bm.faces.new([bm.verts[i] for i in fidx])
                face[cat_layer] = cats[fi]
            except Exception:
                pass
        bm.to_mesh(me)
        bm.free()
        me.update()
        obj.data = me
        bpy.data.meshes.remove(old_me)
        _setup_stair_materials(me, s)
        _apply_stair_uv(me, s)
        obj["room_stair"] = json.dumps({k: v for k, v in sd.items() if k != "obj_name"})
    except Exception:
        import traceback
        traceback.print_exc()


def _cb_stair_rise(self, context):
    _rebuild_stair_steps(context, use_depth=False)


def _cb_stair_depth(self, context):
    _rebuild_stair_steps(context, use_depth=True)


def _cb_stair_slab(self, context):
    """Update callback for stair_open_under / stair_slab_thick."""
    obj = context.active_object
    if obj is None or "room_stair" not in obj:
        print("[room_tool] _cb_stair_slab: no stair object active")
        return
    sd = next((d for d in ROOM_OT_draw._stair_list
               if d.get("obj_name") == obj.name), None)
    if sd is None:
        print(f"[room_tool] _cb_stair_slab: sd not found for '{obj.name}' "
              f"(list has {len(ROOM_OT_draw._stair_list)} entries)")
        return
    s = context.scene.room_settings
    sd["open_under"] = s.stair_open_under
    sd["slab_thick"] = s.stair_slab_thick
    print(f"[room_tool] _cb_stair_slab: rebuilding '{obj.name}' "
          f"open={sd['open_under']} thick={sd['slab_thick']:.3f}")
    try:
        verts, faces, cats = _build_stair_mesh(sd, s)
        print(f"[room_tool] _cb_stair_slab: built {len(verts)} verts, {len(faces)} faces")
        if not verts:
            return
        # _build_stair_mesh returns world-space coords; convert to local (obj.location is pivot)
        loc = obj.location
        verts = [(v[0] - loc.x, v[1] - loc.y, v[2] - loc.z) for v in verts]
        me = obj.data
        me.clear_geometry()
        bm = bmesh.new()
        cat_layer = bm.faces.layers.int.new("stair_cat")
        for v in verts:
            bm.verts.new(v)
        bm.verts.ensure_lookup_table()
        for fi, fidx in enumerate(faces):
            try:
                face = bm.faces.new([bm.verts[i] for i in fidx])
                face[cat_layer] = cats[fi]
            except Exception:
                pass
        bm.to_mesh(me)
        bm.free()
        me.update()
        _setup_stair_materials(me, s)
        _apply_stair_uv(me, s)
        obj["room_stair"] = json.dumps({k: v for k, v in sd.items() if k != "obj_name"})
        print(f"[room_tool] _cb_stair_slab: done, mesh has {len(me.polygons)} polys")
    except Exception:
        import traceback
        traceback.print_exc()


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
    mat_apply_mode   : bpy.props.EnumProperty(
        name="Apply To",
        description="Whether material/tiling changes apply to all rooms or only the active selected room",
        items=[
            ('ALL',      "All Rooms",      "Apply material changes to all rooms"),
            ('SELECTED', "Selected Room",  "Apply material changes only to the currently selected room"),
        ],
        default='ALL',
    )
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
        default=False, update=_cb_rebuild_plinth)
    add_plinth_top : bpy.props.BoolProperty(
        name="Top Plinth",
        description="Add a cornice / crown moulding along the top of the walls",
        default=False, update=_cb_rebuild_plinth)
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
    add_architrave : bpy.props.BoolProperty(
        name="Architrave",
        description="Add extruded door architrave (3-sided frame around each door opening)",
        default=False, update=_cb_rebuild)
    architrave_preset : bpy.props.EnumProperty(
        name="Preset",
        items=[
            ('thin',     "Thin",     "40 mm wide, 8 mm deep"),
            ('standard', "Standard", "70 mm wide, 15 mm deep"),
            ('wide',     "Wide",     "100 mm wide, 20 mm deep"),
            ('bold',     "Bold",     "130 mm wide, 25 mm deep"),
            ('custom',   "Custom",   "Use the sliders below"),
        ],
        default='standard',
        update=_cb_architrave_preset)
    architrave_width : bpy.props.FloatProperty(
        name="Width", default=0.07, min=0.01, max=0.5, unit='LENGTH',
        description="Width of the architrave strip around the door opening",
        update=_cb_rebuild)
    architrave_depth : bpy.props.FloatProperty(
        name="Depth", default=0.015, min=0.002, max=0.2, unit='LENGTH',
        description="How far the architrave protrudes from the wall surface",
        update=_cb_rebuild)
    mat_architrave        : bpy.props.PointerProperty(type=bpy.types.Material, name="Architrave", update=_cb_mat)
    mat_architrave_tiling : bpy.props.FloatProperty(name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_tiling)
    snap_to_grid : bpy.props.BoolProperty(
        name="Snap to Grid",
        description="Snap room corners to the configured grid step while drawing "
                    "(Wall snap still takes priority when near an existing wall)",
        default=False,
    )
    grid_snap_size : bpy.props.FloatProperty(
        name="Grid Size",
        description="Snap increment for room corners and dimensions "
                    "(e.g. 0.5 = 50 cm, 0.25 = 25 cm, 1.0 = 1 m)",
        default=0.5, min=0.01, max=10.0, step=25, unit='LENGTH',
    )
    pivot_mode : bpy.props.EnumProperty(
        name="Room Pivot",
        description="Where to place the origin of each room object",
        items=[
            ('WORLD_ORIGIN', "World Origin", "Origin at (0, 0, 0)"),
            ('FLOOR_CENTER', "Floor Center", "Origin at center of room floor"),
            ('GEOMETRY_CENTER', "Geometry Center", "Origin at bounding-box center of the mesh"),
        ],
        default='WORLD_ORIGIN')
    # ── Stairs ────────────────────────────────────────────────────────────────
    stair_rise  : bpy.props.FloatProperty(
        name="Step Rise",   default=0.18, min=0.05, max=0.5,  unit="LENGTH",
        description="Target height of each step — updates the selected stair in real time",
        update=_cb_stair_rise)
    stair_depth : bpy.props.FloatProperty(
        name="Step Depth",  default=0.28, min=0.05, max=1.0,  unit="LENGTH",
        description="Horizontal depth (run) of each tread — updates the selected stair in real time",
        update=_cb_stair_depth)
    stair_open_under    : bpy.props.BoolProperty(
        name="Open Under Stairs",
        description="Replace the solid triangular stringer with a thin diagonal slab, leaving the space under the stairs open",
        default=False, update=_cb_stair_slab)
    stair_slab_thick    : bpy.props.FloatProperty(
        name="Slab Thickness",
        description="Thickness of the diagonal slab under the stairs",
        default=0.15, min=0.02, max=0.5, unit="LENGTH", update=_cb_stair_slab)
    stair_hole_ratio    : bpy.props.FloatProperty(
        name="Hole Length",
        description="Length of the floor/ceiling opening as a fraction of the full stair run (0.5 = half, 1.0 = full)",
        default=1.0, min=0.5, max=1.0, update=_cb_stair_hole)
    mat_stair         : bpy.props.PointerProperty(type=bpy.types.Material, name="Stairs",
        update=_cb_stair_mat)
    mat_stair_tiling  : bpy.props.FloatProperty(
        name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_stair_tiling)
    mat_stair_step        : bpy.props.PointerProperty(type=bpy.types.Material, name="Stair Steps",
        update=_cb_stair_mat)
    mat_stair_step_tiling : bpy.props.FloatProperty(
        name="Tiling", default=1.0, min=0.001, max=1000.0, update=_cb_stair_tiling)
    stair_apply_dir    : bpy.props.EnumProperty(
        name="Direction",
        items=[('LEFT','Left',''),('RIGHT','Right','')],
        default='LEFT',
        description="Offset direction for alternating flights (relative to stair travel direction)")
    stair_apply_offset : bpy.props.FloatProperty(
        name="Offset", default=0.0, min=0.0, soft_max=10.0, unit='LENGTH',
        description="Lateral offset for alternating flights (0 = auto = stair width)")
    # ── Door thresholds ───────────────────────────────────────────────────────
    add_threshold : bpy.props.BoolProperty(
        name="Threshold",
        default=False,
        description="Place a floor threshold strip at the bottom of each door opening",
        update=_cb_threshold)
    threshold_height : bpy.props.FloatProperty(
        name="Height",
        default=0.008, min=0.002, max=0.020,
        precision=3, unit='LENGTH',
        description="Height of the threshold strip above the floor (2–20 mm)",
        update=_cb_threshold)
    threshold_depth : bpy.props.FloatProperty(
        name="Reach into room",
        default=0.025, min=0.005, max=0.15,
        precision=3, unit='LENGTH',
        description="How far the strip reaches into each room on either side of the wall",
        update=_cb_threshold)
    threshold_material : bpy.props.PointerProperty(
        type=bpy.types.Material,
        name="Material",
        description="Material applied to all threshold strip objects",
        update=_cb_threshold)
    # ── Panel section collapse state ──────────────────────────────────────────
    show_wall_dims  : bpy.props.BoolProperty(name="Wall Dimensions",  default=True)
    show_geometry   : bpy.props.BoolProperty(name="Geometry",         default=True)
    show_trims      : bpy.props.BoolProperty(name="Trims",            default=True)
    show_materials  : bpy.props.BoolProperty(name="Materials / Tiling", default=False)
    show_floors     : bpy.props.BoolProperty(name="Floors",           default=False)
    show_utilities  : bpy.props.BoolProperty(name="Utilities",        default=False)
    # ── Collection / hierarchy ────────────────────────────────────────────────
    use_hierarchy  : bpy.props.BoolProperty(
        name="Use Collections", default=True,
        description="Organise room objects into Blender collections")
    hierarchy_mode : bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('FLOORS_AND_ROOMS', "Floors + Rooms",
             "One collection per floor, sub-collection per room"),
            ('FLOORS_ONLY',      "Floors Only",
             "One collection per floor; room objects sit directly inside it"),
        ],
        default='FLOORS_AND_ROOMS')



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
    s = context.scene.room_settings
    if not s.use_hierarchy:
        return context.scene.collection        # flat — everything in scene root

    floors = context.scene.room_floors
    active = context.scene.room_active_floor
    if not (0 <= active < len(floors)):
        return context.collection

    fl        = floors[active]
    floor_col = _get_or_create_col(fl.name, context.scene.collection)

    if s.hierarchy_mode == 'FLOORS_ONLY':
        return floor_col                       # rooms go directly into floor col

    room_col = _get_or_create_col(f"Room.{room_num:03}", floor_col)
    return room_col                            # full two-level hierarchy


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
_CAT_STAIR         = 7
_CAT_ARCHITRAVE    = 8
_CAT_STAIR_STEP    = 9
_CAT_NAMES = ('walls', 'floor', 'ceiling', 'door_frames', 'window_frames', 'plinth_bottom', 'plinth_top', 'stair', 'architrave', 'stair_step')

def _face4(bm, v0, v1, v2, v3):
    bm.faces.new([bm.verts.new(co) for co in (v0, v1, v2, v3)])


def _fill_room(bm, x1, y1, x2, y2, z, s, doors=(), no_walls=(), windows=(),
               add_plinth_bottom=None, add_plinth_top=None, stair_holes=(),
               add_architrave=None):
    """
    Populate *bm* with room geometry as an interior shell.
    doors:       list of {"wc": str, "anchor": float, "w": float, "h": float}
                 Multiple doors per wall are supported.
    windows:     list of {"wc": str, "anchor": float, "v_offset": float, "w": float, "h": float}
    no_walls:    wall chars to skip entirely.
    add_plinth_bottom/top: if not None, overrides the global s.add_plinth_* flag for this room.
    stair_holes: list of {"x1","y1","x2","y2"} openings cut from ceiling and floor.
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
                _cur_cat[0] = _CAT_WALL; _cur_tiling[0] = s.mat_walls_tiling
                _f4((dr_i,y1,z+dh_i),(dl_i,y1,z+dh_i),(dl_i,y1,zt),(dr_i,y1,zt))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((dl_i,y1,ll[k]),(dl_i,y1-fd,ll[k]),(dl_i,y1-fd,ll[k+1]),(dl_i,y1,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((dr_i,y1-fd,rl[k]),(dr_i,y1,rl[k]),(dr_i,y1,rl[k+1]),(dr_i,y1-fd,rl[k+1]))
                _f4((dr_i,y1,z+dh_i),(dl_i,y1,z+dh_i),(dl_i,y1-fd,z+dh_i),(dr_i,y1-fd,z+dh_i))
                _cur_cat[0] = _CAT_FLOOR; _cur_tiling[0] = s.mat_floor_tiling
                _f4((dl_i,y1-fd,z),(dr_i,y1-fd,z),(dr_i,y1,z),(dl_i,y1,z))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
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
                _cur_cat[0] = _CAT_WALL; _cur_tiling[0] = s.mat_walls_tiling
                _f4((dl_i,y2,z+dh_i),(dr_i,y2,z+dh_i),(dr_i,y2,zt),(dl_i,y2,zt))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((dl_i,y2+fd,ll[k]),(dl_i,y2,ll[k]),(dl_i,y2,ll[k+1]),(dl_i,y2+fd,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((dr_i,y2,rl[k]),(dr_i,y2+fd,rl[k]),(dr_i,y2+fd,rl[k+1]),(dr_i,y2,rl[k+1]))
                _f4((dl_i,y2,z+dh_i),(dr_i,y2,z+dh_i),(dr_i,y2+fd,z+dh_i),(dl_i,y2+fd,z+dh_i))
                _cur_cat[0] = _CAT_FLOOR; _cur_tiling[0] = s.mat_floor_tiling
                _f4((dl_i,y2,z),(dr_i,y2,z),(dr_i,y2+fd,z),(dl_i,y2+fd,z))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
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
                _cur_cat[0] = _CAT_WALL; _cur_tiling[0] = s.mat_walls_tiling
                _f4((x1,db_i,z+dh_i),(x1,df_i,z+dh_i),(x1,df_i,zt),(x1,db_i,zt))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((x1,db_i,ll[k]),(x1-fd,db_i,ll[k]),(x1-fd,db_i,ll[k+1]),(x1,db_i,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((x1-fd,df_i,rl[k]),(x1,df_i,rl[k]),(x1,df_i,rl[k+1]),(x1-fd,df_i,rl[k+1]))
                _f4((x1,db_i,z+dh_i),(x1-fd,db_i,z+dh_i),(x1-fd,df_i,z+dh_i),(x1,df_i,z+dh_i))
                _cur_cat[0] = _CAT_FLOOR; _cur_tiling[0] = s.mat_floor_tiling
                _f4((x1-fd,db_i,z),(x1,db_i,z),(x1,df_i,z),(x1-fd,df_i,z))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
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
                _cur_cat[0] = _CAT_WALL; _cur_tiling[0] = s.mat_walls_tiling
                _f4((x2,df_i,z+dh_i),(x2,db_i,z+dh_i),(x2,db_i,zt),(x2,df_i,zt))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
                ll = sorted([z] + lev_l + [z + dh_i])
                for k in range(len(ll) - 1):
                    _f4((x2+fd,db_i,ll[k]),(x2,db_i,ll[k]),(x2,db_i,ll[k+1]),(x2+fd,db_i,ll[k+1]))
                rl = sorted([z] + lev_r + [z + dh_i])
                for k in range(len(rl) - 1):
                    _f4((x2,df_i,rl[k]),(x2+fd,df_i,rl[k]),(x2+fd,df_i,rl[k+1]),(x2,df_i,rl[k+1]))
                _f4((x2+fd,db_i,z+dh_i),(x2,db_i,z+dh_i),(x2,df_i,z+dh_i),(x2+fd,df_i,z+dh_i))
                _cur_cat[0] = _CAT_FLOOR; _cur_tiling[0] = s.mat_floor_tiling
                _f4((x2,db_i,z),(x2+fd,db_i,z),(x2+fd,df_i,z),(x2,df_i,z))
                _cur_cat[0] = _CAT_DOOR_FRAME; _cur_tiling[0] = s.mat_door_frame_tiling
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
    # Build augmented split lists that include stair hole boundaries so the
    # grid subdivides exactly at hole edges (no T-joints) and hole cells
    # can be skipped cleanly.
    # Each hole dict may have a "cut" key: "ceiling", "floor", or absent/other = both.
    _sh_xs = sorted(set(_ex + [h["x1"] for h in stair_holes] + [h["x2"] for h in stair_holes]))
    _sh_ys = sorted(set(_ey + [h["y1"] for h in stair_holes] + [h["y2"] for h in stair_holes]))

    def _in_stair_hole(cx, cy, surface):
        for sh in stair_holes:
            cut = sh.get("cut", "both")
            if cut != "both" and cut != surface:
                continue
            if sh["x1"] <= cx <= sh["x2"] and sh["y1"] <= cy <= sh["y2"]:
                return True
        return False

    if s.add_ceiling:
        _cur_cat[0] = _CAT_CEILING
        _cur_tiling[0] = s.mat_ceiling_tiling
        if _sh_xs or _sh_ys or stair_holes:
            xs = [x1] + [v for v in _sh_xs if x1 < v < x2] + [x2]
            ys = [y1] + [v for v in _sh_ys if y1 < v < y2] + [y2]
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    if _in_stair_hole((xs[i]+xs[i+1])*0.5, (ys[j]+ys[j+1])*0.5, "ceiling"):
                        continue
                    _f4((xs[i],ys[j],zt),(xs[i],ys[j+1],zt),
                               (xs[i+1],ys[j+1],zt),(xs[i+1],ys[j],zt))
        else:
            _f4((x1,y1,zt),(x1,y2,zt),(x2,y2,zt),(x2,y1,zt))

    if s.add_floor:
        _cur_cat[0] = _CAT_FLOOR
        _cur_tiling[0] = s.mat_floor_tiling
        if _sh_xs or _sh_ys or stair_holes:
            xs = [x1] + [v for v in _sh_xs if x1 < v < x2] + [x2]
            ys = [y1] + [v for v in _sh_ys if y1 < v < y2] + [y2]
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    if _in_stair_hole((xs[i]+xs[i+1])*0.5, (ys[j]+ys[j+1])*0.5, "floor"):
                        continue
                    _f4((xs[i],ys[j],z),(xs[i+1],ys[j],z),
                               (xs[i+1],ys[j+1],z),(xs[i],ys[j+1],z))
        else:
            _f4((x1,y1,z),(x2,y1,z),(x2,y2,z),(x1,y2,z))

    # ── Stair hole shaft reveals ──────────────────────────────────────────────
    # 4 vertical panels around each stair hole perimeter.
    # Each room extends INTO the slab by half the slab thickness so the two
    # panels snap at the midpoint — same technique as window jambs.
    # "ceiling" cut: panels go UP   from zt into the slab (toward upper floor)
    # "floor"   cut: panels go DOWN from z  into the slab (toward lower ceiling)
    for sh in stair_holes:
        sx1, sy1, sx2, sy2 = sh["x1"], sh["y1"], sh["x2"], sh["y2"]
        cut = sh.get("cut", "both")
        _cur_cat[0]    = _CAT_WALL
        _cur_tiling[0] = s.mat_walls_tiling
        if s.add_ceiling and cut in ("ceiling", "both"):
            slab_top = sh.get("slab_z", zt)
            d2 = max(0.0, slab_top - zt) / 2       # half slab thickness
            if d2 > 1e-6:
                # panels go UP from zt to zt+d2; inward normals (face tunnel centre)
                _f4((sx2,sy1,zt+d2),(sx2,sy1,zt),(sx1,sy1,zt),(sx1,sy1,zt+d2))  # S (+Y)
                _f4((sx1,sy2,zt),(sx2,sy2,zt),(sx2,sy2,zt+d2),(sx1,sy2,zt+d2))  # N (−Y)
                _f4((sx1,sy1,zt),(sx1,sy2,zt),(sx1,sy2,zt+d2),(sx1,sy1,zt+d2))  # W (+X)
                _f4((sx2,sy2,zt),(sx2,sy1,zt),(sx2,sy1,zt+d2),(sx2,sy2,zt+d2))  # E (−X)
        if s.add_floor and cut in ("floor", "both"):
            slab_bot = sh.get("slab_z", z)
            d2 = max(0.0, z - slab_bot) / 2        # half slab thickness
            if d2 > 1e-6:
                # panels go DOWN from z to z-d2; inward normals (face tunnel centre)
                _f4((sx1,sy1,z-d2),(sx1,sy1,z),(sx2,sy1,z),(sx2,sy1,z-d2))      # S (+Y)
                _f4((sx2,sy2,z),(sx1,sy2,z),(sx1,sy2,z-d2),(sx2,sy2,z-d2))      # N (−Y)
                _f4((sx1,sy2,z),(sx1,sy1,z),(sx1,sy1,z-d2),(sx1,sy2,z-d2))      # W (+X)
                _f4((sx2,sy1,z),(sx2,sy2,z),(sx2,sy2,z-d2),(sx2,sy1,z-d2))      # E (−X)

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

    # ── Architrave ─────────────────────────────────────────────────────────────
    _add_arch = add_architrave if add_architrave is not None else getattr(s, 'add_architrave', False)
    if _add_arch:
        _aw_raw = max(getattr(s, 'architrave_width', 0.07), 0.001)
        _ad     = max(getattr(s, 'architrave_depth', 0.015), 0.001)
        _cur_cat[0] = _CAT_ARCHITRAVE

        # South wall — inner face at y=y1, protrudes +Y
        if 'S' not in no_walls:
            for _dl, _dr, _dh in _wall_door_segs('S', x1, x2, x2 - x1):
                _aw = min(_aw_raw, (_dr - _dl) * 0.5 - 1e-4)
                if _aw <= 0:
                    continue
                # Left jamb front (+Y face, angled top for miter)
                _f4((_dl,      y1+_ad, z),        (_dl-_aw, y1+_ad, z),        (_dl-_aw, y1+_ad, z+_dh+_aw), (_dl,      y1+_ad, z+_dh))
                # Left jamb inner (+X face, faces door opening)
                _f4((_dl, y1, z),       (_dl, y1+_ad, z),       (_dl, y1+_ad, z+_dh),       (_dl, y1, z+_dh))
                # Left jamb outer (-X face)
                _f4((_dl-_aw, y1+_ad, z), (_dl-_aw, y1, z), (_dl-_aw, y1, z+_dh+_aw), (_dl-_aw, y1+_ad, z+_dh+_aw))
                # Left miter face (45° diagonal)
                _f4((_dl, y1, z+_dh), (_dl, y1+_ad, z+_dh), (_dl-_aw, y1+_ad, z+_dh+_aw), (_dl-_aw, y1, z+_dh+_aw))
                # Head front (+Y face, trapezoid)
                _f4((_dr,      y1+_ad, z+_dh),    (_dl,      y1+_ad, z+_dh),    (_dl-_aw, y1+_ad, z+_dh+_aw), (_dr+_aw, y1+_ad, z+_dh+_aw))
                # Head inner (-Z face, visible from inside door opening looking up)
                _f4((_dr, y1, z+_dh), (_dl, y1, z+_dh), (_dl, y1+_ad, z+_dh), (_dr, y1+_ad, z+_dh))
                # Head top (+Z face)
                _f4((_dl-_aw, y1, z+_dh+_aw), (_dr+_aw, y1, z+_dh+_aw), (_dr+_aw, y1+_ad, z+_dh+_aw), (_dl-_aw, y1+_ad, z+_dh+_aw))
                # Right miter face (45° diagonal)
                _f4((_dr+_aw, y1, z+_dh+_aw), (_dr, y1, z+_dh), (_dr, y1+_ad, z+_dh), (_dr+_aw, y1+_ad, z+_dh+_aw))
                # Right jamb front (+Y face, angled top)
                _f4((_dr+_aw, y1+_ad, z),     (_dr,      y1+_ad, z),     (_dr,      y1+_ad, z+_dh),     (_dr+_aw, y1+_ad, z+_dh+_aw))
                # Right jamb inner (-X face, faces door opening)
                _f4((_dr, y1+_ad, z+_dh), (_dr, y1+_ad, z), (_dr, y1, z), (_dr, y1, z+_dh))
                # Right jamb outer (+X face)
                _f4((_dr+_aw, y1, z), (_dr+_aw, y1+_ad, z), (_dr+_aw, y1+_ad, z+_dh+_aw), (_dr+_aw, y1, z+_dh+_aw))

        # North wall — inner face at y=y2, protrudes -Y  (reverse along-axis for correct normals)
        if 'N' not in no_walls:
            for _dl, _dr, _dh in _wall_door_segs('N', x1, x2, x2 - x1):
                _aw = min(_aw_raw, (_dr - _dl) * 0.5 - 1e-4)
                if _aw <= 0:
                    continue
                # Left jamb front (-Y face, angled top)
                _f4((_dl-_aw, y2-_ad, z),        (_dl,      y2-_ad, z),        (_dl,      y2-_ad, z+_dh),        (_dl-_aw, y2-_ad, z+_dh+_aw))
                # Left jamb inner (-X face)
                _f4((_dl, y2-_ad, z), (_dl, y2, z), (_dl, y2, z+_dh), (_dl, y2-_ad, z+_dh))
                # Left jamb outer (+X face... wait -X for N wall left jamb outer)
                _f4((_dl-_aw, y2, z), (_dl-_aw, y2-_ad, z), (_dl-_aw, y2-_ad, z+_dh+_aw), (_dl-_aw, y2, z+_dh+_aw))
                # Left miter
                _f4((_dl-_aw, y2, z+_dh+_aw), (_dl-_aw, y2-_ad, z+_dh+_aw), (_dl, y2-_ad, z+_dh), (_dl, y2, z+_dh))
                # Head front (-Y face, trapezoid)
                _f4((_dr+_aw, y2-_ad, z+_dh+_aw), (_dl-_aw, y2-_ad, z+_dh+_aw), (_dl,      y2-_ad, z+_dh),    (_dr,      y2-_ad, z+_dh))
                # Head inner (-Z face)
                _f4((_dl, y2-_ad, z+_dh), (_dr, y2-_ad, z+_dh), (_dr, y2, z+_dh), (_dl, y2, z+_dh))
                # Head top (+Z face)
                _f4((_dr+_aw, y2-_ad, z+_dh+_aw), (_dl-_aw, y2-_ad, z+_dh+_aw), (_dl-_aw, y2, z+_dh+_aw), (_dr+_aw, y2, z+_dh+_aw))
                # Right miter
                _f4((_dr+_aw, y2-_ad, z+_dh+_aw), (_dr+_aw, y2, z+_dh+_aw), (_dr, y2, z+_dh), (_dr, y2-_ad, z+_dh))
                # Right jamb front (-Y face, angled top)
                _f4((_dr,      y2-_ad, z+_dh),     (_dr+_aw, y2-_ad, z+_dh+_aw), (_dr+_aw, y2-_ad, z),     (_dr,      y2-_ad, z))
                # Right jamb inner (+X face)
                _f4((_dr, y2, z+_dh), (_dr, y2, z), (_dr, y2-_ad, z), (_dr, y2-_ad, z+_dh))
                # Right jamb outer (-X... +X for N right)
                _f4((_dr+_aw, y2-_ad, z), (_dr+_aw, y2, z), (_dr+_aw, y2, z+_dh+_aw), (_dr+_aw, y2-_ad, z+_dh+_aw))

        # East wall — inner face at x=x2, protrudes -X  (door anchor on Y axis)
        if 'E' not in no_walls:
            for _dl, _dr, _dh in _wall_door_segs('E', y1, y2, y2 - y1):
                _aw = min(_aw_raw, (_dr - _dl) * 0.5 - 1e-4)
                if _aw <= 0:
                    continue
                # Left jamb front (-X face, angled top)
                _f4((x2-_ad, _dl-_aw, z),        (x2-_ad, _dl,      z),        (x2-_ad, _dl,      z+_dh),        (x2-_ad, _dl-_aw, z+_dh+_aw))
                # Left jamb inner (-Y face, faces door)
                _f4((x2-_ad, _dl, z), (x2, _dl, z), (x2, _dl, z+_dh), (x2-_ad, _dl, z+_dh))
                # Left jamb outer (+Y face)
                _f4((x2, _dl-_aw, z), (x2-_ad, _dl-_aw, z), (x2-_ad, _dl-_aw, z+_dh+_aw), (x2, _dl-_aw, z+_dh+_aw))
                # Left miter
                _f4((x2, _dl-_aw, z+_dh+_aw), (x2-_ad, _dl-_aw, z+_dh+_aw), (x2-_ad, _dl, z+_dh), (x2, _dl, z+_dh))
                # Head front (-X face, trapezoid)
                _f4((x2-_ad, _dr+_aw, z+_dh+_aw), (x2-_ad, _dl-_aw, z+_dh+_aw), (x2-_ad, _dl,      z+_dh),    (x2-_ad, _dr,      z+_dh))
                # Head inner (-Z face)
                _f4((x2-_ad, _dl, z+_dh), (x2-_ad, _dr, z+_dh), (x2, _dr, z+_dh), (x2, _dl, z+_dh))
                # Head top (+Z face)
                _f4((x2-_ad, _dr+_aw, z+_dh+_aw), (x2, _dr+_aw, z+_dh+_aw), (x2, _dl-_aw, z+_dh+_aw), (x2-_ad, _dl-_aw, z+_dh+_aw))
                # Right miter
                _f4((x2-_ad, _dr+_aw, z+_dh+_aw), (x2-_ad, _dr, z+_dh), (x2, _dr, z+_dh), (x2, _dr+_aw, z+_dh+_aw))
                # Right jamb front (-X face, angled top)
                _f4((x2-_ad, _dr,      z),        (x2-_ad, _dr+_aw, z),        (x2-_ad, _dr+_aw, z+_dh+_aw), (x2-_ad, _dr,      z+_dh))
                # Right jamb inner (+Y face)
                _f4((x2, _dr, z+_dh), (x2, _dr, z), (x2-_ad, _dr, z), (x2-_ad, _dr, z+_dh))
                # Right jamb outer (-Y face)
                _f4((x2-_ad, _dr+_aw, z+_dh+_aw), (x2-_ad, _dr+_aw, z), (x2, _dr+_aw, z), (x2, _dr+_aw, z+_dh+_aw))

        # West wall — inner face at x=x1, protrudes +X  (door anchor on Y axis)
        if 'W' not in no_walls:
            for _dl, _dr, _dh in _wall_door_segs('W', y1, y2, y2 - y1):
                _aw = min(_aw_raw, (_dr - _dl) * 0.5 - 1e-4)
                if _aw <= 0:
                    continue
                # Left jamb front (+X face, angled top)
                _f4((x1+_ad, _dr,      z),        (x1+_ad, _dr+_aw, z),        (x1+_ad, _dr+_aw, z+_dh+_aw), (x1+_ad, _dr,      z+_dh))
                # Left jamb inner (+Y face, faces door)
                _f4((x1+_ad, _dr, z), (x1, _dr, z), (x1, _dr, z+_dh), (x1+_ad, _dr, z+_dh))
                # Left jamb outer (-Y face)
                _f4((x1, _dr+_aw, z), (x1+_ad, _dr+_aw, z), (x1+_ad, _dr+_aw, z+_dh+_aw), (x1, _dr+_aw, z+_dh+_aw))
                # Left miter
                _f4((x1, _dr+_aw, z+_dh+_aw), (x1+_ad, _dr+_aw, z+_dh+_aw), (x1+_ad, _dr, z+_dh), (x1, _dr, z+_dh))
                # Head front (+X face, trapezoid)
                _f4((x1+_ad, _dl-_aw, z+_dh+_aw), (x1+_ad, _dr+_aw, z+_dh+_aw), (x1+_ad, _dr,      z+_dh),    (x1+_ad, _dl,      z+_dh))
                # Head inner (-Z face)
                _f4((x1+_ad, _dr, z+_dh), (x1+_ad, _dl, z+_dh), (x1, _dl, z+_dh), (x1, _dr, z+_dh))
                # Head top (+Z face)
                _f4((x1, _dr+_aw, z+_dh+_aw), (x1+_ad, _dr+_aw, z+_dh+_aw), (x1+_ad, _dl-_aw, z+_dh+_aw), (x1, _dl-_aw, z+_dh+_aw))
                # Right miter
                _f4((x1+_ad, _dl-_aw, z+_dh+_aw), (x1, _dl-_aw, z+_dh+_aw), (x1, _dl, z+_dh), (x1+_ad, _dl, z+_dh))
                # Right jamb front (+X face, angled top)
                _f4((x1+_ad, _dl,      z+_dh),     (x1+_ad, _dl-_aw, z+_dh+_aw), (x1+_ad, _dl-_aw, z),     (x1+_ad, _dl,      z))
                # Right jamb inner (-Y face)
                _f4((x1, _dl, z+_dh), (x1, _dl, z), (x1+_ad, _dl, z), (x1+_ad, _dl, z+_dh))
                # Right jamb outer (+Y face)
                _f4((x1+_ad, _dl-_aw, z), (x1, _dl-_aw, z), (x1, _dl-_aw, z+_dh+_aw), (x1+_ad, _dl-_aw, z+_dh+_aw))

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


def _apply_cube_uv_to_mesh(me, s, obj=None):
    """Apply cube-projection UV using local vertex coordinates (post-pivot).
    Matches bpy.ops.uv.cube_project(cube_size=1/tiling) per material category.

    When *obj* is supplied the world-space normal is used to infer the tiling
    category for any face whose room_cat attribute is 0 (the default Blender
    assigns to newly-created geometry such as extruded faces).  This means
    extruded floor/ceiling quads automatically pick up floor/ceiling tiling
    instead of being incorrectly bucketed as walls.
    """
    uv_layer = me.uv_layers.new(name="UVMap")
    if uv_layer is None:
        return  # mesh has no polygons — nothing to UV-map
    cat_attr = me.attributes.get("room_cat") if hasattr(me, 'attributes') else None
    tiling_map = [
        s.mat_walls_tiling,                            # 0 = _CAT_WALL
        s.mat_floor_tiling,                            # 1 = _CAT_FLOOR
        s.mat_ceiling_tiling,                          # 2 = _CAT_CEILING
        s.mat_door_frame_tiling,                       # 3 = _CAT_DOOR_FRAME
        s.mat_window_frame_tiling,                     # 4 = _CAT_WIN_FRAME
        getattr(s, 'mat_plinth_bottom_tiling', 1.0),  # 5 = _CAT_PLINTH_BOTTOM
        getattr(s, 'mat_plinth_top_tiling',    1.0),  # 6 = _CAT_PLINTH_TOP
        1.0,                                           # 7 = _CAT_STAIR (uses walls mat in room mesh)
        getattr(s, 'mat_architrave_tiling',    1.0),  # 8 = _CAT_ARCHITRAVE
    ]
    # World-rotation matrix so we can classify extruded faces by world normal.
    rot3 = obj.matrix_world.to_3x3().normalized() if obj is not None else None

    verts  = me.vertices
    loops  = me.loops
    for poly in me.polygons:
        cat = cat_attr.data[poly.index].value if cat_attr else 0

        # For newly-created geometry (room_cat == 0) use the world-space normal
        # to distinguish wall faces from horizontal (floor / ceiling) faces.
        # Original parametric floor/ceiling faces have room_cat == 1 or 2 and
        # are unaffected by this branch.
        if cat == 0 and rot3 is not None:
            wn_z = (rot3 @ poly.normal).z
            if wn_z > 0.7:          # mostly up  → floor
                cat = 1             # _CAT_FLOOR
            elif wn_z < -0.7:       # mostly down → ceiling
                cat = 2             # _CAT_CEILING
            # else: stays 0 = _CAT_WALL

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
        mat_pb   = getattr(s, 'mat_plinth_bottom', None) or mat_w
        mat_pt   = getattr(s, 'mat_plinth_top',    None) or mat_w
        mat_arch = getattr(s, 'mat_architrave',    None) or mat_w
        mat_list, slot_map = [], {}
        # Index must match _CAT_* constants (7=stair→wall fallback, 8=architrave)
        for cat, mat in enumerate((mat_w, mat_f, mat_c, mat_d, mat_wf, mat_pb, mat_pt, mat_w, mat_arch)):
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
        obj = bpy.data.objects[obj_name]
        me  = obj.data
        uv = me.uv_layers.get("UVMap")
        if uv:
            me.uv_layers.remove(uv)
        _apply_cube_uv_to_mesh(me, s, obj=obj)


def _apply_materials_one_room(reg, s):
    """Update material slots + face material_index for a single room registry entry."""
    obj_name = reg.get("obj_name", "")
    if obj_name not in bpy.data.objects:
        return
    me = bpy.data.objects[obj_name].data
    while me.materials:
        me.materials.pop(index=0)
    mat_w  = getattr(s, 'mat_walls',        None)
    mat_f  = getattr(s, 'mat_floor',        None)
    mat_c  = getattr(s, 'mat_ceiling',      None)
    mat_d  = getattr(s, 'mat_door_frame',   None) or mat_w
    mat_wf = getattr(s, 'mat_window_frame', None) or mat_w
    mat_pb   = getattr(s, 'mat_plinth_bottom', None) or mat_w
    mat_pt   = getattr(s, 'mat_plinth_top',    None) or mat_w
    mat_arch = getattr(s, 'mat_architrave',    None) or mat_w
    mat_list, slot_map = [], {}
    for cat, mat in enumerate((mat_w, mat_f, mat_c, mat_d, mat_wf, mat_pb, mat_pt, mat_w, mat_arch)):
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


def _apply_uvs_one_room(reg, s):
    """Recompute the UVMap for a single room registry entry."""
    obj_name = reg.get("obj_name", "")
    if obj_name not in bpy.data.objects:
        return
    obj = bpy.data.objects[obj_name]
    me  = obj.data
    uv = me.uv_layers.get("UVMap")
    if uv:
        me.uv_layers.remove(uv)
    _apply_cube_uv_to_mesh(me, s, obj=obj)


def _build_stair_mesh(sd, s):
    """Straight solid staircase fitting a rectangular footprint.

    The long axis of the rectangle is the travel direction.
    Steps ascend from z_bot at one short edge to z_top at the other.

    Geometry (fully enclosed):
      - treads (top face of each step, +Z normal)
      - risers (front vertical face of each step)
      - flat soffit at z_bot (-Z normal)
      - back wall at the high end of the run
      - left / right solid stringer walls (fan-triangulated staircase profile)
      - optional railing panels on each side

    sd keys: lx1,ly1,lx2,ly2 (lower hole footprint / rect width)
             ux1,uy1,ux2,uy2 (upper hole footprint — may be offset along travel axis)
             z_bot, z_top
             step_rise, add_railing, railing_height, railing_thick
             x_travel (bool, travel along X axis vs Y axis)
    """
    lx1 = min(sd["lx1"], sd["lx2"]); lx2 = max(sd["lx1"], sd["lx2"])
    ly1 = min(sd["ly1"], sd["ly2"]); ly2 = max(sd["ly1"], sd["ly2"])
    # Upper rect — fall back to lower rect if not stored (old/undo state)
    ux1 = min(sd.get("ux1", lx1), sd.get("ux2", lx2))
    ux2 = max(sd.get("ux1", lx1), sd.get("ux2", lx2))
    uy1 = min(sd.get("uy1", ly1), sd.get("uy2", ly2))
    uy2 = max(sd.get("uy1", ly1), sd.get("uy2", ly2))
    z_bot  = sd["z_bot"]
    z_top  = sd["z_top"]
    rise_t = max(sd.get("step_rise",  getattr(s, "stair_rise", 0.18)), 0.05)

    dz = z_top - z_bot
    if dz < 0.01:
        return [], [], []

    n = max(2, round(dz / rise_t))
    actual_rise = dz / n

    # Determine travel axis: use offset direction if stored, else rect dimensions
    rw = lx2 - lx1
    rl = ly2 - ly1
    x_travel = sd.get("x_travel", rw >= rl)

    # Travel extent: always use the max edge of each opening rect.
    # This ensures that mirrored flights span the same range as the original,
    # with no gap (landing distance) between adjacent flights.
    if x_travel:
        t_start, t_end = lx2, ux2
    else:
        t_start, t_end = ly2, uy2

    td = 1 if t_end >= t_start else -1
    total_travel = abs(t_end - t_start)
    if total_travel < 0.05:
        return [], [], []
    step_d = total_travel / n   # horizontal depth per step

    verts_out, faces_out, cats_out = [], [], []

    def _quad(a, b, c, d_, cat=_CAT_STAIR):
        base = len(verts_out)
        verts_out.extend([tuple(a), tuple(b), tuple(c), tuple(d_)])
        faces_out.append((base, base + 1, base + 2, base + 3))
        cats_out.append(cat)

    def _tri(a, b, c, cat=_CAT_STAIR):
        base = len(verts_out)
        verts_out.extend([tuple(a), tuple(b), tuple(c)])
        faces_out.append((base, base + 1, base + 2))
        cats_out.append(cat)

    def _step_profile(t_lo, t_hi, n_steps, rise_per_step):
        """Build (t,z) polygon profile for one stringer side."""
        d = (t_hi - t_lo) / n_steps
        steps_lr = []
        for i in range(n_steps):
            t_f = t_lo + i * d
            t_b = t_f + d
            zh  = z_bot + (i + 1) * rise_per_step
            steps_lr.append((min(t_f, t_b), max(t_f, t_b), zh))
        steps_lr.sort()
        profile = [(t_lo, z_bot)]
        for (tl, tr, zh) in steps_lr:
            profile.append((tl, zh))
            profile.append((tr, zh))
        profile.append((t_hi, z_bot))
        return profile, steps_lr

    # Helpers: interpolate perpendicular (width) bounds linearly from lower to upper rect.
    # frac=0 → lower rect perpendicular; frac=1 → upper rect perpendicular.
    _span = (t_end - t_start) if abs(t_end - t_start) > 1e-6 else 1.0
    def _frac(t):
        return max(0.0, min(1.0, (t - t_start) / _span))

    open_under  = sd.get("open_under",  False)
    slab_thick  = sd.get("slab_thick",  0.15)

    if x_travel:
        # Perpendicular axis is Y; interpolate between ly1/ly2 (lower) and uy1/uy2 (upper)
        def _ya(t): return ly1 + (uy1 - ly1) * _frac(t)
        def _yb(t): return ly2 + (uy2 - ly2) * _frac(t)
        # td already set above from t_start/t_end direction

        for i in range(n):
            xf = t_start + td * i * step_d
            xb = t_start + td * (i + 1) * step_d
            zl = z_bot + i * actual_rise
            zh = zl + actual_rise
            yaf = _ya(xf); ybf = _yb(xf)
            yab = _ya(xb); ybb = _yb(xb)
            if td > 0:
                _quad((xf, yaf, zh), (xb, yab, zh), (xb, ybb, zh), (xf, ybf, zh), cat=_CAT_STAIR_STEP)  # tread
                _quad((xf, ybf, zl), (xf, yaf, zl), (xf, yaf, zh), (xf, ybf, zh))  # riser
            else:
                _quad((xb, yab, zh), (xf, yaf, zh), (xf, ybf, zh), (xb, ybb, zh), cat=_CAT_STAIR_STEP)
                _quad((xf, yaf, zl), (xf, ybf, zl), (xf, ybf, zh), (xf, yaf, zh))

        z_slab_top = z_bot + dz   # height of the slab at t_end

        if open_under:
            # ── Thin diagonal slab (open-under mode) ─────────────────────────
            # Bottom face runs parallel to top face at uniform thickness slab_thick.
            # The bottom face starts at the floor level (z_bot), offset inward by
            # (slab_thick / dz) * total_travel, so thickness is always uniform.
            ya0 = _ya(t_start); ya1 = _ya(t_end)
            yb0 = _yb(t_start); yb1 = _yb(t_end)
            # Top-face corners
            A  = (t_start, ya0, z_bot);      B  = (t_end, ya1, z_slab_top)
            C  = (t_end,   yb1, z_slab_top); D  = (t_start, yb0, z_bot)
            # Bottom-face lower start: offset along travel so z = z_bot at that x
            bt_offset = (td * slab_thick * total_travel / dz) if dz > 1e-6 else 0.0
            bt_start  = t_start + bt_offset
            # Bottom-face corners: parallel slab, uniform thickness
            A2 = (bt_start, _ya(bt_start), z_bot)
            D2 = (bt_start, _yb(bt_start), z_bot)
            B2 = (t_end, ya1, z_slab_top - slab_thick)
            C2 = (t_end, yb1, z_slab_top - slab_thick)
            _quad(A, B, C, D)       # top inclined face
            _quad(D2, C2, B2, A2)   # bottom inclined face (uniform thickness)
            if td > 0:
                _quad(A, A2, B2, B)     # ya side
                _quad(D, C, C2, D2)     # yb side
                _quad(A, D, D2, A2)     # lower floor-level end cap
                _quad(B, B2, C2, C)     # upper end cap
            else:
                _quad(A, B, B2, A2)     # ya side (td<0)
                _quad(D2, C2, C, D)     # yb side (td<0)
                _quad(A2, D2, D, A)     # lower floor-level end cap (td<0)
                _quad(B, C, C2, B2)     # upper end cap (td<0)
            # Close the triangular gap between each step riser and the slab.
            # For step i: V_top=(xf,y,zh), V_bot=(xf,y,zl), V_back=(xb,y,zh).
            # Slab top surface passes exactly through V_bot and V_back, so the
            # triangle fills the open wedge under each tread.
            for i in range(n):
                xf = t_start + td * i * step_d
                xb = t_start + td * (i + 1) * step_d
                zl = z_bot + i * actual_rise
                zh = zl + actual_rise
                # ya side triangle (normal −Y for td>0, reversed for td<0)
                base = len(verts_out)
                verts_out += [(xf, _ya(xf), zh), (xf, _ya(xf), zl), (xb, _ya(xb), zh)]
                if td > 0:
                    faces_out.append((base, base+1, base+2))
                else:
                    faces_out.append((base, base+2, base+1))
                cats_out.append(_CAT_STAIR)
                # yb side triangle (normal +Y — reversed winding)
                base = len(verts_out)
                verts_out += [(xf, _yb(xf), zh), (xf, _yb(xf), zl), (xb, _yb(xb), zh)]
                if td > 0:
                    faces_out.append((base, base+2, base+1))
                else:
                    faces_out.append((base, base+1, base+2))
                cats_out.append(_CAT_STAIR)
        else:
            # ── Solid stringer (default) ──────────────────────────────────────
            # Soffit — horizontal quad at z_bot
            _quad((t_start, _yb(t_start), z_bot), (t_end, _yb(t_end), z_bot),
                  (t_end,   _ya(t_end),   z_bot), (t_start, _ya(t_start), z_bot))

            # Back wall at the high end
            xw = t_end; ya_w = _ya(xw); yb_w = _yb(xw)
            if td > 0:
                _quad((xw, ya_w, z_bot), (xw, yb_w, z_bot), (xw, yb_w, z_top), (xw, ya_w, z_top))
            else:
                _quad((xw, yb_w, z_bot), (xw, ya_w, z_bot), (xw, ya_w, z_top), (xw, yb_w, z_top))

            # Solid N-gon stringer on each side
            _, steps_lr = _step_profile(t_start, t_end, n, actual_rise)

            def _x_profile(side_fn):
                pts = [(t_start, side_fn(t_start), z_bot)]
                if td > 0:
                    for (tl, tr, zh) in steps_lr:
                        pts.append((tl, side_fn(tl), zh))
                        pts.append((tr, side_fn(tr), zh))
                else:
                    for (tl, tr, zh) in reversed(steps_lr):
                        pts.append((tr, side_fn(tr), zh))
                        pts.append((tl, side_fn(tl), zh))
                pts.append((t_end, side_fn(t_end), z_bot))
                return pts

            def _stringer_ngon(side_fn, outward_neg_y):
                pts = _x_profile(side_fn)
                if outward_neg_y == (td > 0):
                    pts = pts[::-1]
                base = len(verts_out)
                verts_out.extend(pts)
                faces_out.append(tuple(range(base, base + len(pts))))
                cats_out.append(_CAT_STAIR)

            _stringer_ngon(_ya, outward_neg_y=True)
            _stringer_ngon(_yb, outward_neg_y=False)

    else:   # Y travel — perpendicular axis is X; interpolate lx1/lx2 → ux1/ux2
        def _xa(t): return lx1 + (ux1 - lx1) * _frac(t)
        def _xb(t): return lx2 + (ux2 - lx2) * _frac(t)
        # td already set above from t_start/t_end direction

        for i in range(n):
            yf  = t_start + td * i * step_d
            yb_ = t_start + td * (i + 1) * step_d
            zl  = z_bot + i * actual_rise
            zh  = zl + actual_rise
            xaf = _xa(yf); xbf = _xb(yf)
            xab = _xa(yb_); xbb = _xb(yb_)
            if td > 0:
                _quad((xaf, yf, zh), (xab, yb_, zh), (xbb, yb_, zh), (xbf, yf, zh), cat=_CAT_STAIR_STEP)  # tread
                _quad((xbf, yf, zl), (xaf, yf, zl), (xaf, yf, zh), (xbf, yf, zh))    # riser
            else:
                _quad((xab, yb_, zh), (xaf, yf, zh), (xbf, yf, zh), (xbb, yb_, zh), cat=_CAT_STAIR_STEP)
                _quad((xaf, yf, zl), (xbf, yf, zl), (xbf, yf, zh), (xaf, yf, zh))

        z_slab_top = z_bot + dz

        if open_under:
            # ── Thin diagonal slab (open-under mode, y_travel) ───────────────
            # Bottom face runs parallel to top at uniform thickness slab_thick,
            # starting at floor level (y offset so z = z_bot at bottom start).
            xa0 = _xa(t_start); xa1 = _xa(t_end)
            xb0 = _xb(t_start); xb1 = _xb(t_end)
            A  = (xa0, t_start, z_bot);      B  = (xa1, t_end, z_slab_top)
            C  = (xb1, t_end,   z_slab_top); D  = (xb0, t_start, z_bot)
            bt_offset = (td * slab_thick * total_travel / dz) if dz > 1e-6 else 0.0
            bt_start  = t_start + bt_offset
            A2 = (_xa(bt_start), bt_start, z_bot)
            D2 = (_xb(bt_start), bt_start, z_bot)
            B2 = (xa1, t_end, z_slab_top - slab_thick)
            C2 = (xb1, t_end, z_slab_top - slab_thick)
            _quad(A, B, C, D)       # top inclined face
            _quad(D2, C2, B2, A2)   # bottom inclined face (uniform thickness)
            if td > 0:
                _quad(A, A2, B2, B)     # xa side
                _quad(D, C, C2, D2)     # xb side
                _quad(A, D, D2, A2)     # lower floor-level end cap
                _quad(B, B2, C2, C)     # upper end cap
            else:
                _quad(A, B, B2, A2)     # xa side (td<0)
                _quad(D2, C2, C, D)     # xb side (td<0)
                _quad(A2, D2, D, A)     # lower floor-level end cap (td<0)
                _quad(B, C, C2, B2)     # upper end cap (td<0)
            # Close triangular gap under each step (y_travel version)
            for i in range(n):
                yf = t_start + td * i * step_d
                yb_ = t_start + td * (i + 1) * step_d
                zl  = z_bot + i * actual_rise
                zh  = zl + actual_rise
                # xa side triangle (normal −X for td>0)
                base = len(verts_out)
                verts_out += [(_xa(yf), yf, zh), (_xa(yf), yf, zl), (_xa(yb_), yb_, zh)]
                if td > 0:
                    faces_out.append((base, base+1, base+2))
                else:
                    faces_out.append((base, base+2, base+1))
                cats_out.append(_CAT_STAIR)
                # xb side triangle (normal +X — reversed winding)
                base = len(verts_out)
                verts_out += [(_xb(yf), yf, zh), (_xb(yf), yf, zl), (_xb(yb_), yb_, zh)]
                if td > 0:
                    faces_out.append((base, base+2, base+1))
                else:
                    faces_out.append((base, base+1, base+2))
                cats_out.append(_CAT_STAIR)
        else:
            # ── Solid stringer (default, y_travel) ───────────────────────────
            _quad((_xb(t_start), t_start, z_bot), (_xb(t_end), t_end, z_bot),
                  (_xa(t_end),   t_end,   z_bot), (_xa(t_start), t_start, z_bot))

            yw = t_end; xa_w = _xa(yw); xb_w = _xb(yw)
            if td > 0:
                _quad((xb_w, yw, z_bot), (xa_w, yw, z_bot), (xa_w, yw, z_top), (xb_w, yw, z_top))
            else:
                _quad((xa_w, yw, z_bot), (xb_w, yw, z_bot), (xb_w, yw, z_top), (xa_w, yw, z_top))

            _, steps_lr = _step_profile(t_start, t_end, n, actual_rise)

            def _y_profile(side_fn):
                pts = [(side_fn(t_start), t_start, z_bot)]
                if td > 0:
                    for (tl, tr, zh) in steps_lr:
                        pts.append((side_fn(tl), tl, zh))
                        pts.append((side_fn(tr), tr, zh))
                else:
                    for (tl, tr, zh) in reversed(steps_lr):
                        pts.append((side_fn(tr), tr, zh))
                        pts.append((side_fn(tl), tl, zh))
                pts.append((side_fn(t_end), t_end, z_bot))
                return pts

            def _stringer_y_ngon(side_fn, outward_neg_x):
                pts = _y_profile(side_fn)
                if outward_neg_x == (td < 0):
                    pts = pts[::-1]
                base = len(verts_out)
                verts_out.extend(pts)
                faces_out.append(tuple(range(base, base + len(pts))))
                cats_out.append(_CAT_STAIR)

            _stringer_y_ngon(_xa, outward_neg_x=True)
            _stringer_y_ngon(_xb, outward_neg_x=False)

    return verts_out, faces_out, cats_out


def _stair_pivot_xy(sd):
    """Return (px, py) — the top-corner vertex used as the stair object's origin.
    Mirrors _pivot_point() on ROOM_OT_stair_move but works on a raw sd dict."""
    lmx = (sd["lx1"] + sd["lx2"]) / 2
    umx = (sd.get("ux1", sd["lx1"]) + sd.get("ux2", sd["lx2"])) / 2
    lmy = (sd["ly1"] + sd["ly2"]) / 2
    umy = (sd.get("uy1", sd["ly1"]) + sd.get("uy2", sd["ly2"])) / 2
    ux1 = sd.get("ux1", sd["lx1"]); ux2 = sd.get("ux2", sd["lx2"])
    uy1 = sd.get("uy1", sd["ly1"]); uy2 = sd.get("uy2", sd["ly2"])
    if sd.get("x_travel", True):
        return (ux2, uy1) if umx >= lmx else (ux1, uy2)
    else:
        return (ux1, uy2) if umy >= lmy else (ux2, uy1)


def _snap_stair_pt(x, y, room, grid=0.1, wall_snap=0.25):
    """Snap (x, y) to the nearest grid point, then to an inner wall face if close enough."""
    t   = room.get("t", 0.125)
    x   = round(x / grid) * grid
    y   = round(y / grid) * grid
    wx1 = room["x1"] + t;  wx2 = room["x2"] - t
    wy1 = room["y1"] + t;  wy2 = room["y2"] - t
    if   abs(x - wx1) < wall_snap: x = wx1
    elif abs(x - wx2) < wall_snap: x = wx2
    if   abs(y - wy1) < wall_snap: y = wy1
    elif abs(y - wy2) < wall_snap: y = wy2
    return x, y


def _make_stair_obj(name, sd, s, collection=None):
    """Create a Blender object for a staircase from a stair dict."""
    verts, faces, cats = _build_stair_mesh(sd, s)
    if not verts:
        return None
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    cat_layer = bm.faces.layers.int.new("stair_cat")
    for v in verts:
        bm.verts.new(v)
    bm.verts.ensure_lookup_table()
    for fi, fidx in enumerate(faces):
        try:
            face = bm.faces.new([bm.verts[i] for i in fidx])
            face[cat_layer] = cats[fi]
        except Exception:
            pass
    bm.to_mesh(me)
    bm.free()
    me.update()
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.collection).objects.link(obj)
    _setup_stair_materials(me, s)
    _apply_stair_uv(me, s)   # UV computed while verts are still in world space
    # Set object origin to centroid of all vertices
    pivot = Vector((
        sum(v[0] for v in verts) / len(verts),
        sum(v[1] for v in verts) / len(verts),
        sum(v[2] for v in verts) / len(verts),
    ))
    for v in me.vertices:
        v.co -= pivot
    obj.location = pivot
    me.update()
    # Mark as stair so panels/operators can identify it
    obj["room_stair"] = json.dumps({k: v for k, v in sd.items() if k != "obj_name"})
    return obj


def _setup_stair_materials(me, s):
    """Assign stair/railing materials to a stair mesh."""
    while me.materials:
        me.materials.pop(index=0)
    mat_s    = getattr(s, 'mat_stair',      None)
    mat_step = getattr(s, 'mat_stair_step', None)
    # _CAT_STAIR_STEP falls back to mat_s when no separate material assigned
    mat_list, slot_map = [], {}
    for cat, mat in ((_CAT_STAIR, mat_s), (_CAT_STAIR_STEP, mat_step or mat_s)):
        if mat is None:
            slot_map[cat] = None
        elif mat in mat_list:
            slot_map[cat] = mat_list.index(mat)
        else:
            slot_map[cat] = len(mat_list)
            mat_list.append(mat)
    for mat in mat_list:
        me.materials.append(mat)
    cat_attr = me.attributes.get("stair_cat") if hasattr(me, 'attributes') else None
    if cat_attr and any(v is not None for v in slot_map.values()):
        for poly in me.polygons:
            si = slot_map.get(cat_attr.data[poly.index].value)
            if si is not None:
                poly.material_index = si
    me.update()


def _apply_stair_materials_all(s):
    """Re-apply stair materials to all existing stair meshes without geometry rebuild."""
    for sd in ROOM_OT_draw._stair_list:
        obj_name = sd.get("obj_name")
        if not obj_name:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.data:
            _setup_stair_materials(obj.data, s)


def _apply_stair_uvs_all(s):
    """Recompute stair UVs on all existing stair meshes without geometry rebuild."""
    for sd in ROOM_OT_draw._stair_list:
        obj_name = sd.get("obj_name")
        if not obj_name:
            continue
        obj = bpy.data.objects.get(obj_name)
        if obj and obj.data:
            me = obj.data
            # Remove existing UV layers before reapplying
            while me.uv_layers:
                me.uv_layers.remove(me.uv_layers[0])
            _apply_stair_uv(me, s)


def _apply_stair_uv(me, s):
    """Cube-project UV for stair mesh, with separate tiling for treads vs sides."""
    uv_layer = me.uv_layers.new(name="UVMap")
    sc_side = getattr(s, 'mat_stair_tiling',      1.0)
    sc_step = getattr(s, 'mat_stair_step_tiling', 1.0)
    cat_attr = me.attributes.get("stair_cat") if hasattr(me, 'attributes') else None
    verts = me.vertices
    loops = me.loops
    for poly in me.polygons:
        cat = cat_attr.data[poly.index].value if cat_attr else _CAT_STAIR
        sc = sc_step if cat == _CAT_STAIR_STEP else sc_side
        n = poly.normal
        ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
        for li in range(poly.loop_start, poly.loop_start + poly.loop_total):
            co = verts[loops[li].vertex_index].co
            if az >= ax and az >= ay:
                uv_layer.data[li].uv = (co.x * sc, co.y * sc)
            elif ax >= ay:
                uv_layer.data[li].uv = (co.y * sc, co.z * sc)
            else:
                uv_layer.data[li].uv = (co.x * sc, co.z * sc)


def _get_or_create_stair_col(context):
    """Return the 'Stairs' collection, creating it under the scene collection if needed."""
    return _get_or_create_col("Stairs", context.scene.collection)




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
    _apply_cube_uv_to_mesh(me, s, obj=obj)
    return obj


def _rebuild_room_mesh(reg, s):
    """Regenerate the mesh of an existing room in-place from its registry entry.

    When reg["mesh_locked"] is True the geometry is NOT regenerated — only
    opening mesh instances are repositioned.  This lets users manually edit the
    room in Edit Mode and still use the door/window/stair tools without losing
    their changes.
    """
    obj_name = reg.get("obj_name", "")
    if obj_name not in bpy.data.objects:
        return
    obj = bpy.data.objects[obj_name]

    if reg.get("mesh_locked", False):
        # Locked: skip geometry rebuild, only reposition opening instances.
        # Refresh ew refs first so doors/windows follow any mesh edits (extrusions etc.)
        _refresh_ew_refs(reg, s)
        col = next(iter(obj.users_collection), None)
        _sync_opening_meshes(reg, collection=col, s=s)
        # Still update plinth dims/materials when settings change
        if reg.get('plinth_bottom_enabled', False) or reg.get('plinth_top_enabled', False):
            _recalculate_plinth_for_obj(reg, obj, s)
        return

    obj.location = (0.0, 0.0, 0.0)   # reset before regenerating at world coords
    me  = obj.data
    bm  = bmesh.new()
    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    z   = reg.get("z", s.z_foundation)
    cat_vert_sets = _fill_room(bm, x1, y1, x2, y2, z, s,
               doors=reg.get("doors", []),
               no_walls=tuple(reg.get("no_walls", [])),
               windows=reg.get("windows", []),
               add_plinth_bottom=reg.get('plinth_bottom_enabled', False),
               add_plinth_top   =reg.get('plinth_top_enabled',    False),
               stair_holes      =reg.get('stair_holes', []),
               add_architrave   =getattr(s, 'add_architrave', False))
    _setup_room_materials(me, s, bm)
    bm.to_mesh(me)
    bm.free()
    me.update()
    _setup_room_vertex_groups(obj, cat_vert_sets)
    _apply_room_pivot(obj, x1, y1, x2, y2, z, s.pivot_mode)
    _apply_cube_uv_to_mesh(me, s, obj=obj)
    # Reposition any assigned door/window mesh instances
    col = next(iter(obj.users_collection), None)
    _sync_opening_meshes(reg, collection=col, s=s)


# ── Opening mesh helpers ──────────────────────────────────────────────────────

def _opening_world_pos(opening, reg, is_window=False, src=None):
    """Return (cx, cy, cz, rot_z) — the object-origin world position for a door/window mesh.

    When *src* is supplied the bounding box is used to place the mesh so that:
      • It is centred horizontally inside the opening (anchor ± w/2)
      • Its front face (bb_min_y) is flush with the wall face — door/window
        extends fully into the room from the shared wall boundary
      • Its bottom sits on the floor     (z or z+v_offset for windows)
    This works regardless of where the mesh origin sits relative to the geometry.

    When *src* is None the legacy behaviour is used (origin assumed at bottom-centre
    of the mesh at the wall face).
    """
    wc     = opening["wc"]
    anchor = opening["anchor"]
    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    z      = reg.get("z", 0.0)
    cz_base = z + (opening.get("v_offset", 0.0) if is_window else 0.0)

    import math
    rot_map = {'S': 0.0, 'N': math.pi, 'W': math.pi * 0.5, 'E': -math.pi * 0.5}
    rot_z   = rot_map[wc]

    t  = reg.get("t", 0.2)
    fd = t * 0.5
    ew = opening.get("ew")

    if src is None:
        # Legacy: origin at bottom-centre of mesh, placed flush with the wall face.
        if ew:
            wcp = ew["wall_coord"]
            wall_xy = {'S': (anchor, wcp), 'N': (anchor, wcp),
                       'W': (wcp, anchor), 'E': (wcp, anchor)}
        else:
            wall_xy = {'S': (anchor, y1), 'N': (anchor, y2),
                       'W': (x1, anchor), 'E': (x2, anchor)}
        wx, wy = wall_xy[wc]
        return wx, wy, cz_base, rot_z

    # ── Bounding-box-aware placement ─────────────────────────────────────────
    # Use raw local (pre-scale) mesh data — the instance shares the raw mesh
    # and gets its own scale, so the source object's scale (os) is irrelevant.
    bb     = src.bound_box          # 8 corners in raw local space
    xs_w   = [v[0] for v in bb]
    ys_w   = [v[1] for v in bb]
    zs_w   = [v[2] for v in bb]

    bb_mid_x = (min(xs_w) + max(xs_w)) / 2.0   # local X centre (along-opening axis)
    bb_min_z = min(zs_w)                        # local Z bottom

    # Instance scale: X→opening width, Y unchanged, Z→opening height.
    src_dx = max(xs_w) - min(xs_w)
    src_dz = max(zs_w) - min(zs_w)
    sx = (opening.get("w", opening.get("window_width",  1.0)) /
          max(src_dx, 1e-4)) if src_dx > 1e-4 else 1.0
    sz = (opening.get("h", opening.get("window_height", 1.0)) /
          max(src_dz, 1e-4)) if src_dz > 1e-4 else 1.0

    # Align bb_mid_y to the external doorframe edge.
    # For extra walls use the wall_coord directly as the frame centre.
    if ew:
        wcp = ew["wall_coord"]
        if wc in ('N', 'S'):
            outer_y = wcp + fd if wc == 'N' else wcp - fd
            frame_outer = {wc: (anchor, outer_y)}
        else:
            outer_x = wcp + fd if wc == 'E' else wcp - fd
            frame_outer = {wc: (outer_x, anchor)}
    else:
        frame_outer = {'S': (anchor, y1 - fd),
                       'N': (anchor, y2 + fd),
                       'W': (x1 - fd, anchor),
                       'E': (x2 + fd, anchor)}
    target_wx, target_wy = frame_outer[wc]

    cr, sr = math.cos(rot_z), math.sin(rot_z)
    bb_mid_y = (min(ys_w) + max(ys_w)) / 2.0
    lx, ly, lz = bb_mid_x * sx, bb_mid_y, bb_min_z * sz
    cx = target_wx - lx * cr + ly * sr
    cy = target_wy - lx * sr - ly * cr
    cz = cz_base   - lz

    return cx, cy, cz, rot_z


def _place_opening_mesh(opening, source_name, w, h, reg, collection=None):
    """Create (or update) a linked mesh instance for a door or window.
    opening  – the door/window dict (modified in-place to store mesh_obj_name)
    source_name – bpy.data.objects key of the source mesh
    w, h     – opening width and height in metres
    reg      – room registry dict (provides x1/y1/x2/y2/z)
    collection – Blender collection to link the instance into"""
    src = bpy.data.objects.get(source_name)
    if src is None:
        return

    is_win = "v_offset" in opening
    cx, cy, cz, rot_z = _opening_world_pos(opening, reg, is_window=is_win, src=src)

    # Re-use existing instance if possible, else create a new one
    inst_name = opening.get("mesh_obj_name", "")
    inst = bpy.data.objects.get(inst_name)
    if inst is None or inst.data != src.data:
        # Remove stale instance if it exists under a different mesh
        if inst is not None:
            for col in list(inst.users_collection):
                col.objects.unlink(inst)
            bpy.data.objects.remove(inst)
        inst = bpy.data.objects.new(f"Opening.{src.name}", src.data)
        target_col = collection or bpy.context.collection
        target_col.objects.link(inst)
        opening["mesh_obj_name"] = inst.name
        opening["mesh_source"]   = source_name

    # Scale X = opening width, Z = opening height; Y (depth) unchanged.
    # Use raw local mesh data — the instance shares raw mesh data and gets its
    # own scale, so the source object's os is irrelevant here.
    bb     = src.bound_box
    xs_w   = [v[0] for v in bb]
    ys_w   = [v[1] for v in bb]
    src_dx = max(xs_w) - min(xs_w)
    src_dz = max(v[2] for v in bb) - min(v[2] for v in bb)
    sx = w / max(src_dx, 1e-4) if src_dx > 1e-4 else 1.0
    sz = h / max(src_dz, 1e-4) if src_dz > 1e-4 else 1.0

    flip_lr = opening.get("flip_lr", False)
    flip_io = opening.get("flip_io", False)

    # When a flip negates a scale axis the bounding-box centre moves to the
    # opposite side of the origin.  Compensate so the mesh stays inside the
    # opening by shifting cx/cy by 2× the displaced centre (in world space).
    if flip_lr or flip_io:
        import math
        bb_mid_x = (min(xs_w) + max(xs_w)) / 2.0
        bb_mid_y = (min(ys_w) + max(ys_w)) / 2.0
        cr = math.cos(rot_z)
        sr = math.sin(rot_z)
        if flip_lr:
            # Negating sx shifts the local-X centre by -2*bb_mid_x*sx.
            # In world space (rotated by rot_z): dx = -shift*cr, dy = -shift*sr
            shift_x = 2.0 * bb_mid_x * sx
            cx += shift_x * cr
            cy += shift_x * sr
        if flip_io:
            # Negating sy (=1→-1) shifts the local-Y centre by -2*bb_mid_y.
            # In world space: dx = -(-2*bb_mid_y)*(-sr) = -2*bb_mid_y*sr,
            #                 dy = -(-2*bb_mid_y)*( cr) =  2*bb_mid_y*cr
            cy_shift = 2.0 * bb_mid_y
            cx -= cy_shift * sr
            cy += cy_shift * cr

    # Position / rotate
    inst.location = (cx, cy, cz)
    inst.rotation_euler = (0.0, 0.0, rot_z)
    inst.scale = (-sx if flip_lr else sx, -1.0 if flip_io else 1.0, sz)


def _place_threshold(door, door_idx, reg, collection, s):
    """Create or update a threshold strip mesh object for a single door.

    The strip is a flat box spanning the full wall thickness (depth),
    threshold_width wide (capped at door width), and threshold_height tall,
    placed on the floor at the door's anchor position.
    """
    if s is None or not getattr(s, 'add_threshold', False):
        _remove_threshold(door)
        return

    obj_name = reg.get("obj_name", "")
    # Reuse stored name so the object survives door-index shifts
    thr_name = door.get("threshold_obj_name") or f"Threshold_{obj_name}_{door_idx}"

    th_h   = getattr(s, 'threshold_height', 0.008)
    t      = reg.get("t", 0.125)
    # th_d = per-side reach; clamp so strip never exceeds the wall boundary
    th_d   = min(getattr(s, 'threshold_depth', 0.025), t * 0.5)
    door_w = door.get("w", getattr(s, 'door_width', 0.9))
    half_w = door_w * 0.5    # matches the door opening width exactly
    half_d = th_d            # th_d is already per-side reach

    wc      = door.get("wc", "S")
    anchor  = door.get("anchor", 0.0)
    z_floor = reg.get("z", 0.0)

    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    ew  = door.get("ew")
    wcp = (ew["wall_coord"] if ew else
           {"S": y1, "N": y2, "W": x1, "E": x2}[wc])
    # Shift from inner face to physical wall centre so the strip straddles
    # the wall evenly on both sides rather than sitting half inside the wall.
    wall_centre = wcp + (t * 0.5 if wc in ('N', 'E') else -t * 0.5)

    # Build mesh in LOCAL space centred at origin so the object pivot
    # sits at the geometry centre.  The object is then translated to the
    # correct world position via thr_obj.location below.
    half_h = th_h * 0.5
    p = [
        (-half_w, -half_d, -half_h),
        ( half_w, -half_d, -half_h),
        ( half_w,  half_d, -half_h),
        (-half_w,  half_d, -half_h),
        (-half_w, -half_d,  half_h),
        ( half_w, -half_d,  half_h),
        ( half_w,  half_d,  half_h),
        (-half_w,  half_d,  half_h),
    ]
    # World-space centre of the strip
    if wc in ('N', 'S'):
        obj_x, obj_y = anchor, wall_centre
    else:
        # E/W wall: strip local X = depth axis, local Y = door-width axis
        obj_x, obj_y = wall_centre, anchor

    mesh_name = thr_name + "_mesh"
    mesh = bpy.data.meshes.get(mesh_name)
    if mesh is None:
        mesh = bpy.data.meshes.new(mesh_name)

    bm = bmesh.new()
    vs = [bm.verts.new(co) for co in p]
    bm.faces.new([vs[0], vs[3], vs[2], vs[1]])   # bottom
    bm.faces.new([vs[4], vs[5], vs[6], vs[7]])   # top
    bm.faces.new([vs[0], vs[1], vs[5], vs[4]])   # side A
    bm.faces.new([vs[2], vs[3], vs[7], vs[6]])   # side B
    bm.faces.new([vs[0], vs[4], vs[7], vs[3]])   # end 1
    bm.faces.new([vs[1], vs[2], vs[6], vs[5]])   # end 2
    bm.to_mesh(mesh)
    bm.free()

    thr_obj = bpy.data.objects.get(thr_name)
    if thr_obj is None:
        thr_obj = bpy.data.objects.new(thr_name, mesh)
        if collection is not None:
            collection.objects.link(thr_obj)
        elif bpy.context.scene is not None:
            bpy.context.scene.collection.objects.link(thr_obj)
    else:
        thr_obj.data = mesh

    import math as _math
    thr_obj.location       = (obj_x, obj_y, z_floor + half_h)
    # E/W walls: rotate 90° around Z so local X (depth) aligns with world X
    thr_obj.rotation_euler = (0.0, 0.0, _math.pi * 0.5 if wc in ('E', 'W') else 0.0)
    thr_obj.scale          = (1.0, 1.0, 1.0)

    mat = getattr(s, 'threshold_material', None)
    if mat is not None:
        if thr_obj.data.materials:
            thr_obj.data.materials[0] = mat
        else:
            thr_obj.data.materials.append(mat)

    door["threshold_obj_name"] = thr_name


def _remove_threshold(door):
    """Remove the threshold mesh object for this door dict (if any)."""
    thr_name = door.pop("threshold_obj_name", None)
    if not thr_name:
        return
    thr_obj = bpy.data.objects.get(thr_name)
    if thr_obj is None:
        return
    for col in list(thr_obj.users_collection):
        col.objects.unlink(thr_obj)
    mesh = thr_obj.data
    bpy.data.objects.remove(thr_obj)
    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def _sync_all_thresholds(s):
    """Sync threshold strips across ALL rooms — exactly one strip per door opening.
    For paired (shared-wall) doors the room with the lower list index owns the
    object; the other room stores a reference to the same object name."""
    if s is None:
        return
    rooms = ROOM_OT_draw._room_list
    t_default = getattr(s, 'wall_thickness', 0.125)
    # Auto-enable threshold when any two connected rooms share a door pair
    if not getattr(s, 'add_threshold', False):
        for ri_c, reg_c in enumerate(rooms):
            t_c = reg_c.get("t", t_default)
            for di_c in range(len(reg_c.get("doors", []))):
                if _find_partner_door(rooms, ri_c, di_c, t_c) is not None:
                    s['add_threshold'] = True   # bypass update cb, avoid recursion
                    break
            if getattr(s, 'add_threshold', False):
                break
    for ri, reg in enumerate(rooms):
        obj = bpy.data.objects.get(reg.get("obj_name", ""))
        col = next(iter(obj.users_collection), None) if obj else None
        t   = reg.get("t", t_default)
        for di, door in enumerate(reg.get("doors", [])):
            if not getattr(s, 'add_threshold', False):
                _remove_threshold(door)
                continue
            partner = _find_partner_door(rooms, ri, di, t)
            if partner is not None:
                p_ri, p_di = partner
                if p_ri < ri or (p_ri == ri and p_di < di):
                    # Partner already processed and owns the strip.
                    # Only remove a *separate* object this door may have created
                    # previously; never remove the shared object owned by the partner.
                    p_doors = rooms[p_ri].get("doors", [])
                    owner_name = (p_doors[p_di].get("threshold_obj_name")
                                  if p_di < len(p_doors) else None)
                    self_name = door.get("threshold_obj_name")
                    if self_name and self_name != owner_name:
                        _remove_threshold(door)
                    else:
                        door.pop("threshold_obj_name", None)
                    if owner_name:
                        door["threshold_obj_name"] = owner_name
                    continue
            _place_threshold(door, di, reg, col, s)


def _remove_opening_mesh(opening):
    """Remove the mesh instance linked to a door/window dict (if any)."""
    inst_name = opening.pop("mesh_obj_name", None)
    opening.pop("mesh_source", None)
    if inst_name:
        inst = bpy.data.objects.get(inst_name)
        if inst is not None:
            for col in list(inst.users_collection):
                col.objects.unlink(inst)
            bpy.data.objects.remove(inst)


def _sync_opening_meshes(reg, collection=None, s=None):
    """Reposition / recreate all door and window mesh instances for a room.
    Called after _rebuild_room_mesh so instances stay in sync with geometry."""
    wall_h = getattr(s, 'wall_height', None) if s is not None else None
    for opening in reg.get("doors", []) + reg.get("windows", []):
        src_name = opening.get("mesh_source")
        if not src_name:
            continue
        w = opening.get("w", opening.get("window_width", 1.0))
        h = opening.get("h", opening.get("window_height", 1.0))
        # Clamp window sill height so the mesh instance never floats above the room.
        # This corrects v_offset values that were stored incorrectly (e.g. from
        # sampling a mesh placed high in the scene).
        if "v_offset" in opening and wall_h is not None:
            t = reg.get("t", 0.125)
            max_voff = max(0.0, wall_h - h - t)
            if opening["v_offset"] > max_voff:
                opening["v_offset"] = max_voff
        _place_opening_mesh(opening, src_name, w, h, reg, collection)
    # Sync threshold strips globally so paired doors share one strip
    _sync_all_thresholds(s)


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
    mat_pb   = getattr(s, 'mat_plinth_bottom', None) or mat_w
    mat_pt   = getattr(s, 'mat_plinth_top',    None) or mat_w
    mat_arch = getattr(s, 'mat_architrave',    None) or mat_w
    # Build deduplicated slot list so shared materials use one slot
    mat_list  = []
    slot_map  = {}
    # Index must match the _CAT_* constants (0=wall … 6=plinth_top, 7=stair, 8=architrave)
    for cat, mat in enumerate((mat_w, mat_f, mat_c, mat_d, mat_wf, mat_pb, mat_pt, mat_w, mat_arch)):
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


def _apply_grid_snap(pt, context, s):
    """Quantize pt.x / pt.y to the configured grid step if snap_to_grid is enabled."""
    if not getattr(s, 'snap_to_grid', False):
        return pt
    step = getattr(s, 'grid_snap_size', 0.5)
    if step < 1e-6:
        return pt
    return Vector((
        round(pt.x / step) * step,
        round(pt.y / step) * step,
        pt.z,
    ))


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
    Returns (snap_2d, (nx,ny), room_idx, wall_char, ew_or_None) or None.
    ew_or_None is a dict {"wc", "wall_coord", "lo", "hi"} for extra (extruded) walls,
    or None for the four original room walls.
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
        # (wall_char, outer_face_pos, perp_dist, along, span_lo, span_hi, normal)
        checks = [
            ('E', x2+t, abs(px-(x2+t)), py,  y1, y2, ( 1, 0)),
            ('W', x1-t, abs(px-(x1-t)), py,  y1, y2, (-1, 0)),
            ('N', y2+t, abs(py-(y2+t)), px,  x1, x2, ( 0, 1)),
            ('S', y1-t, abs(py-(y1-t)), px,  x1, x2, ( 0,-1)),
        ]
        for wc, face_pos, perp, along, lo, hi, normal in checks:
            # Direction guard: cursor must be outside the room on this wall's side
            if wc == 'E' and px < x2:  continue
            if wc == 'W' and px > x1:  continue
            if wc == 'N' and py < y2:  continue
            if wc == 'S' and py > y1:  continue
            # True 2D distance — penalises cursor past the wall endpoints
            excess = max(0.0, lo - along, along - hi)
            dist   = (perp * perp + excess * excess) ** 0.5
            if dist < best_d:
                if wc == 'E':
                    sp = Vector((x2 + t, max(y1, min(y2, py))))
                elif wc == 'W':
                    sp = Vector((x1 - t, max(y1, min(y2, py))))
                elif wc == 'N':
                    sp = Vector((max(x1, min(x2, px)), y2 + t))
                else:
                    sp = Vector((max(x1, min(x2, px)), y1 - t))
                best_d = dist
                best   = (sp, normal, i, wc, None)

        # ── Extra walls (locked rooms only) ───────────────────────────────────
        for ew in _detect_extra_walls(r, t):
            ewc  = ew["wc"]
            ewp  = ew["wall_coord"]
            lo   = ew["lo"]
            hi   = ew["hi"]
            _nmap = {'N': (0, 1), 'S': (0,-1), 'E': (1, 0), 'W': (-1, 0)}
            if ewc in ('N', 'S'):
                perp   = abs(py - ewp)
                along  = px
                excess = max(0.0, lo - px, px - hi)
                sp_fn  = lambda: Vector((max(lo, min(hi, px)),
                                         ewp + t if ewc == 'N' else ewp - t))
            else:
                perp   = abs(px - ewp)
                along  = py
                excess = max(0.0, lo - py, py - hi)
                sp_fn  = lambda: Vector((ewp + t if ewc == 'E' else ewp - t,
                                         max(lo, min(hi, py))))
            dist = (perp * perp + excess * excess) ** 0.5
            if dist < best_d:
                best_d = dist
                best   = (sp_fn(), _nmap[ewc], i, ewc, ew)

    return best


def _detect_extra_walls(reg, t):
    """Scan a locked room's mesh for vertical axis-aligned faces that are either
    outside the original four wall planes (x1/y1/x2/y2) OR at the original wall
    planes but with a span that extends beyond the original room footprint.

    Returns a list of dicts::
        {"wc": "N"|"S"|"E"|"W", "wall_coord": float, "lo": float, "hi": float}
    Only called (and only useful) when reg["mesh_locked"] is True.
    """
    if not reg.get("mesh_locked", False):
        return []
    obj = bpy.data.objects.get(reg.get("obj_name", ""))
    if obj is None:
        return []

    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    tol = max(t * 0.35, 0.025)
    known_y = [y1 - t, y1, y2, y2 + t]
    known_x = [x1 - t, x1, x2, x2 + t]

    me = obj.data
    mw = obj.matrix_world
    rot = mw.to_quaternion()

    wall_groups = {}   # (wc, bucket) → {wc, wall_coord, lo, hi}

    for poly in me.polygons:
        n = poly.normal.copy()
        n.rotate(rot)
        if abs(n.z) > 0.3:
            continue  # skip horizontal faces

        wverts = [mw @ me.vertices[vi].co for vi in poly.vertices]

        # ── Y-normal face (N or S wall) ──────────────────────────────────────
        if abs(n.y) > 0.7 and abs(n.x) < 0.5:
            y_pos = sum(v.y for v in wverts) / len(wverts)
            x_min = min(v.x for v in wverts)
            x_max = max(v.x for v in wverts)
            at_known_y = any(abs(y_pos - yp) < tol for yp in known_y)
            if at_known_y:
                # Face is at an original wall plane. Only include it when its
                # X span extends beyond the original room bounds — this means
                # the wall was extended outward (the new outer section can't
                # be snapped via the original wall's fixed [x1,x2] span).
                if not (x_min < x1 - tol or x_max > x2 + tol):
                    continue
            else:
                # Face at a new Y position. Reject full-room-width interior
                # partition artifacts; allow short L-shape step walls.
                if y1 + tol < y_pos < y2 - tol and (x_max - x_min) > (x2 - x1) + t:
                    continue
            # Room meshes have inward normals — use position relative to room
            # centre to determine wall character, not the face normal direction.
            wc     = 'N' if y_pos > (y1 + y2) * 0.5 else 'S'
            bucket = (wc, round(y_pos / tol))
            if bucket in wall_groups:
                g = wall_groups[bucket]
                g["lo"] = min(g["lo"], x_min)
                g["hi"] = max(g["hi"], x_max)
            else:
                wall_groups[bucket] = {"wc": wc, "wall_coord": y_pos,
                                       "lo": x_min, "hi": x_max}

        # ── X-normal face (E or W wall) ──────────────────────────────────────
        elif abs(n.x) > 0.7 and abs(n.y) < 0.5:
            x_pos = sum(v.x for v in wverts) / len(wverts)
            y_min = min(v.y for v in wverts)
            y_max = max(v.y for v in wverts)
            at_known_x = any(abs(x_pos - xp) < tol for xp in known_x)
            if at_known_x:
                # Face is at an original wall plane. Only include when its
                # Y span extends beyond the original room bounds.
                if not (y_min < y1 - tol or y_max > y2 + tol):
                    continue
            else:
                # Face at a new X position. Reject full-room-height interior
                # partition artifacts; allow short L-shape step walls.
                if x1 + tol < x_pos < x2 - tol and (y_max - y_min) > (y2 - y1) + t:
                    continue
            # Room meshes have inward normals — use position relative to room
            # centre to determine wall character, not the face normal direction.
            wc     = 'E' if x_pos > (x1 + x2) * 0.5 else 'W'
            bucket = (wc, round(x_pos / tol))
            if bucket in wall_groups:
                g = wall_groups[bucket]
                g["lo"] = min(g["lo"], y_min)
                g["hi"] = max(g["hi"], y_max)
            else:
                wall_groups[bucket] = {"wc": wc, "wall_coord": x_pos,
                                       "lo": y_min, "hi": y_max}

    return list(wall_groups.values())


def _refresh_ew_refs(reg, s):
    """Re-detect extra walls from the locked mesh and update every door/window
    ew dict in-place so overlays and mesh placement use current positions.

    Must be called after any mesh modification (extrusion, pull, etc.) on a
    locked room so that stale wall_coord / lo / hi values are corrected.
    """
    if not reg.get("mesh_locked", False):
        return
    t = reg.get("t", getattr(s, "wall_thickness", 0.125))
    extra_walls = _detect_extra_walls(reg, t)

    # Group current extra walls by wc for fast lookup
    ew_by_wc: dict = {}
    for ew in extra_walls:
        ew_by_wc.setdefault(ew["wc"], []).append(ew)

    for opening in reg.get("doors", []) + reg.get("windows", []):
        stored_ew = opening.get("ew")
        if stored_ew is None:
            continue
        wc = opening.get("wc") or stored_ew.get("wc", "N")
        candidates = ew_by_wc.get(wc, [])
        if not candidates:
            # The extra wall no longer exists — fall back to the original wall
            opening.pop("ew", None)
            continue
        # Pick the candidate whose wall_coord is closest to the stored one
        best = min(candidates, key=lambda e: abs(e["wall_coord"] - stored_ew["wall_coord"]))
        stored_ew["wall_coord"] = best["wall_coord"]
        stored_ew["lo"]         = best["lo"]
        stored_ew["hi"]         = best["hi"]


def _wall_snap_any(pt, rooms, t_fallback, current_z=None):
    """
    Find the nearest wall face (from any direction) within _SNAP_DIST.
    Returns (room_idx, wall_char, anchor, ew_or_None) or None.
    anchor = cursor position clamped along the wall span.
    current_z: when set, only considers rooms on the same floor (within 1 mm).

    Distance is the true 2-D distance from the cursor to the nearest point on
    the wall segment (not just the perpendicular component).  This prevents
    snapping when the cursor is past the end of a wall — e.g. in the corner
    junction area or on the floor slab between two rooms.
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
        # (wall_char, perp_dist, along_coord, span_lo, span_hi, clamped_anchor)
        checks = [
            ('S', abs(py - y1), px, x1, x2, max(x1, min(x2, px))),
            ('N', abs(py - y2), px, x1, x2, max(x1, min(x2, px))),
            ('W', abs(px - x1), py, y1, y2, max(y1, min(y2, py))),
            ('E', abs(px - x2), py, y1, y2, max(y1, min(y2, py))),
        ]
        for wc, perp, along, lo, hi, anchor in checks:
            # How far the cursor is PAST the wall's endpoint (0 when inside span)
            excess = max(0.0, lo - along, along - hi)
            dist   = (perp * perp + excess * excess) ** 0.5
            if dist < best_d:
                best_d = dist
                best   = (i, wc, anchor, None)

        # ── Extra walls (locked rooms only) ───────────────────────────────────
        for ew in _detect_extra_walls(r, t):
            ewc = ew["wc"]
            ewp = ew["wall_coord"]
            lo  = ew["lo"]
            hi  = ew["hi"]
            if ewc in ('N', 'S'):
                perp   = abs(py - ewp)
                anchor = max(lo, min(hi, px))
                excess = max(0.0, lo - px, px - hi)
            else:
                perp   = abs(px - ewp)
                anchor = max(lo, min(hi, py))
                excess = max(0.0, lo - py, py - hi)
            dist = (perp * perp + excess * excess) ** 0.5
            if dist < best_d:
                best_d = dist
                best   = (i, ewc, anchor, ew)

    return best


def _snap_original_wall_superseded(reg, wc, sv, t):
    """Return True if a wall face exists FURTHER OUTWARD than the recorded
    original plane — meaning the original plane is now interior (the wall was
    extruded in edit mode and the room extended beyond x1/y1/x2/y2).

    Fast path uses _detect_extra_walls (locked rooms).
    Fallback scans the mesh directly so it also works for unlocked rooms.
    """
    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    tol = max(t * 0.4, 0.03)

    # ── Fast path (locked rooms) ───────────────────────────────────────────
    for ew in _detect_extra_walls(reg, t):
        if ew["wc"] != wc:
            continue
        ewp = ew["wall_coord"]
        further = (
            (wc == 'E' and ewp > x2 + tol) or
            (wc == 'W' and ewp < x1 - tol) or
            (wc == 'N' and ewp > y2 + tol) or
            (wc == 'S' and ewp < y1 - tol)
        )
        if further and ew["lo"] - tol <= sv <= ew["hi"] + tol:
            return True

    # ── Fallback: direct mesh scan (works for unlocked rooms too) ──────────
    obj = bpy.data.objects.get(reg.get("obj_name", ""))
    if obj is None:
        return False
    me  = obj.data
    mw  = obj.matrix_world
    rot = mw.to_quaternion()
    for poly in me.polygons:
        n = poly.normal.copy()
        n.rotate(rot)
        if abs(n.z) > 0.3:
            continue
        wverts = [mw @ me.vertices[vi].co for vi in poly.vertices]
        if abs(n.y) > 0.7 and abs(n.x) < 0.5 and wc in ('N', 'S'):
            y_pos = sum(v.y for v in wverts) / len(wverts)
            x_lo  = min(v.x for v in wverts)
            x_hi  = max(v.x for v in wverts)
            further = (wc == 'N' and y_pos > y2 + tol) or (wc == 'S' and y_pos < y1 - tol)
            if further and x_lo - tol <= sv <= x_hi + tol:
                return True
        elif abs(n.x) > 0.7 and abs(n.y) < 0.5 and wc in ('E', 'W'):
            x_pos = sum(v.x for v in wverts) / len(wverts)
            y_lo  = min(v.y for v in wverts)
            y_hi  = max(v.y for v in wverts)
            further = (wc == 'E' and x_pos > x2 + tol) or (wc == 'W' and x_pos < x1 - tol)
            if further and y_lo - tol <= sv <= y_hi + tol:
                return True
    return False


def _snap_wall_is_shared(reg, wc, sv, rooms, t_fallback):
    """Return True if wall *wc* at along-position *sv* is shared (open) with
    an adjacent room.

    Uses the same outer-face proximity test as _find_partner_wall / _wall_adj_zones:
    two rooms are adjacent when their facing outer-wall planes are within
    2×max(t_a, t_b) of each other AND their perpendicular spans overlap at *sv*.
    """
    x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    t_a = reg.get("t", t_fallback)
    if   wc == 'E': a_pos = x2 + t_a
    elif wc == 'W': a_pos = x1 - t_a
    elif wc == 'N': a_pos = y2 + t_a
    else:           a_pos = y1 - t_a

    for other in rooms:
        if other is reg:
            continue
        t_b  = other.get("t", t_fallback)
        thr  = 2.0 * max(t_a, t_b)
        ox1, oy1, ox2, oy2 = other["x1"], other["y1"], other["x2"], other["y2"]
        # Outer face of the opposing wall on the other room
        if   wc == 'E': b_pos = ox1 - t_b
        elif wc == 'W': b_pos = ox2 + t_b
        elif wc == 'N': b_pos = oy1 - t_b
        else:           b_pos = oy2 + t_b
        if abs(a_pos - b_pos) > thr:
            continue
        # Perpendicular overlap span
        if wc in ('E', 'W'):
            lo, hi = max(y1, oy1), min(y2, oy2)
        else:
            lo, hi = max(x1, ox1), min(x2, ox2)
        if lo < hi and lo <= sv <= hi:
            return True
    return False


def _snap_wall_from_ray(context, event, rooms, t_fallback, current_z=None):
    """Intersect the mouse ray with each room's wall planes and snap to the
    nearest valid hit.

    Returns (room_idx, wall_char, anchor, ew_or_None) — same signature as
    _wall_snap_any — or None if the ray misses every wall.

    Uses analytic plane-intersection rather than mesh ray-casting so it works
    correctly regardless of face normal direction (i.e. whether the camera is
    inside or outside the room).  A hit is only accepted if the Z coordinate of
    the intersection falls within the wall's vertical extent, which prevents the
    floor and ceiling from triggering false wall snaps.
    """
    from mathutils.geometry import intersect_line_plane

    r, rv3d   = context.region, context.region_data
    co        = event.mouse_region_x, event.mouse_region_y
    origin    = view3d_utils.region_2d_to_origin_3d(r, rv3d, co)
    direction = view3d_utils.region_2d_to_vector_3d(r, rv3d, co).normalized()
    end       = origin + direction * 500.0

    best_dist = float('inf')
    best      = None

    for ri, reg in enumerate(rooms):
        if not _room_is_usable(reg):
            continue
        if current_z is not None and abs(reg.get("z", 0.0) - current_z) > 0.001:
            continue

        x1, y1, x2, y2 = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
        t       = reg.get("t", t_fallback)
        z_floor = reg.get("z", 0.0)
        try:
            wh = context.scene.room_settings.wall_height
        except Exception:
            wh = 2.4

        # ── Original 4 walls ────────────────────────────────────────────────
        # Each entry: (wc, plane_point, plane_normal, span_lo, span_hi, axis)
        # axis='x' → the "along" coordinate is pt.x; axis='y' → pt.y
        wall_planes = [
            ('S', Vector((0.0, y1, 0.0)), Vector((0.0, 1.0, 0.0)), x1, x2, 'x'),
            ('N', Vector((0.0, y2, 0.0)), Vector((0.0, 1.0, 0.0)), x1, x2, 'x'),
            ('W', Vector((x1, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), y1, y2, 'y'),
            ('E', Vector((x2, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), y1, y2, 'y'),
        ]

        for wc, pco, pno, lo, hi, ax in wall_planes:
            pt = intersect_line_plane(origin, end, pco, pno)
            if pt is None:
                continue
            # Reject hits behind the camera
            if (pt - origin).dot(direction) < 0:
                continue
            # Reject floor- and ceiling-aimed rays: hit must be within wall height
            if not (z_floor + 0.05 <= pt.z <= z_floor + wh + 0.05):
                continue
            # Reject if the hit is outside the wall's horizontal span (with tolerance)
            sv = pt.x if ax == 'x' else pt.y
            if not (lo - t <= sv <= hi + t):
                continue
            # Reject if this wall segment is shared (open) with an adjacent room
            if _snap_wall_is_shared(reg, wc, sv, rooms, t):
                continue
            # Reject if this wall plane is now interior (the room was extruded outward)
            if _snap_original_wall_superseded(reg, wc, sv, t):
                continue
            d = (pt - origin).length
            if d < best_dist:
                best_dist = d
                anchor    = max(lo, min(hi, sv))
                best      = (ri, wc, anchor, None)

        # ── Extra walls (L-shaped rooms / locked-mesh additions) ────────────
        for ew_c in _detect_extra_walls(reg, t):
            ewc = ew_c["wc"]
            ewp = ew_c["wall_coord"]
            lo  = ew_c["lo"]
            hi  = ew_c["hi"]
            if ewc in ('N', 'S'):
                pco, pno, ax = Vector((0.0, ewp, 0.0)), Vector((0.0, 1.0, 0.0)), 'x'
            else:
                pco, pno, ax = Vector((ewp, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), 'y'

            pt = intersect_line_plane(origin, end, pco, pno)
            if pt is None:
                continue
            if (pt - origin).dot(direction) < 0:
                continue
            if not (z_floor + 0.05 <= pt.z <= z_floor + wh + 0.05):
                continue
            sv = pt.x if ax == 'x' else pt.y
            if not (lo - t <= sv <= hi + t):
                continue
            d = (pt - origin).length
            if d < best_dist:
                best_dist = d
                anchor    = max(lo, min(hi, sv))
                best      = (ri, ewc, anchor, ew_c)
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
            ew     = door.get("ew")
            if wc == 'E':
                dist  = abs(px - (ew["wall_coord"] if ew else x2))
                along = py
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            elif wc == 'W':
                dist  = abs(px - (ew["wall_coord"] if ew else x1))
                along = py
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            elif wc == 'N':
                dist  = abs(py - (ew["wall_coord"] if ew else y2))
                along = px
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            else:  # S
                dist  = abs(py - (ew["wall_coord"] if ew else y1))
                along = px
                lo    = anchor - dw * 0.5 - t
                hi    = anchor + dw * 0.5 + t
            if dist < best_d and lo <= along <= hi:
                best_d = dist
                best   = (i, di)
    return best


def _wall_span(reg, wall_char, wall_span=None):
    """Return (lo, hi) span along wall.  wall_span overrides reg bounds (extra walls)."""
    if wall_span is not None:
        return wall_span
    if wall_char in ('E', 'W'):
        return reg["y1"], reg["y2"]
    return reg["x1"], reg["x2"]


def _clamp_anchor(anchor, reg, wall_char, door_width, margin, zones=(), wall_span=None):
    """Clamp door centre to wall-edge clearance + zone boundaries (soft clamp, no rejection).
    zones: [(lo, hi)] blocked intervals from adjacent rooms (same as _wall_adj_zones output).
    wall_span: (lo, hi) override for extra (extruded) walls."""
    t      = reg.get("t", 0.125)
    margin = max(margin, t)          # never closer to corner than one wall-thickness
    span_lo, span_hi = _wall_span(reg, wall_char, wall_span)
    lo = span_lo + door_width * 0.5 + margin
    hi = span_hi - door_width * 0.5 - margin
    if lo > hi:
        return (span_lo + span_hi) * 0.5
    anchor = max(lo, min(hi, anchor))
    if zones:
        sub = _valid_spans(lo, hi, zones, door_width, margin)
        if sub:
            anchor = min((abs(max(sl, min(sr, anchor)) - anchor),
                          max(sl, min(sr, anchor))) for sl, sr in sub)[1]
    return anchor


def _valid_anchor(anchor, reg, wall_char, door_width, margin, skip_idx=None, zones=(),
                  wall_span=None):
    """Return clamped anchor if position is valid, None if it must be rejected.

    Rejects when:
    - Wall is too narrow to fit the door (with margin on both sides), OR
    - The (clamped) position would straddle a shared-wall boundary (zones), OR
    - The (clamped) position would be closer than *margin* to any existing
      door on the same wall (gap between door edges < margin).

    skip_idx: door index in reg['doors'] to ignore (used when sliding).
    zones: [(lo, hi)] blocked intervals from adjacent rooms.
    wall_span: (lo, hi) override for extra (extruded) walls.
    """
    t      = reg.get("t", 0.125)
    margin = max(margin, t)          # never closer to corner than one wall-thickness
    span_lo, span_hi = _wall_span(reg, wall_char, wall_span)
    lo = span_lo + door_width * 0.5 + margin
    hi = span_hi - door_width * 0.5 - margin
    if lo > hi:
        return None   # wall too narrow
    anchor = max(lo, min(hi, anchor))   # clamp to wall edges
    if zones:
        sub = _valid_spans(lo, hi, zones, door_width, margin)
        if not sub:
            return None
        anchor = min((abs(max(sl, min(sr, anchor)) - anchor),
                      max(sl, min(sr, anchor))) for sl, sr in sub)[1]
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


def _max_array_count(reg, wc, ww, gap, margin, wall_span=None):
    """Maximum windows that physically fit on the wall interior."""
    span_lo, span_hi = _wall_span(reg, wc, wall_span)
    available = span_hi - span_lo - 2 * margin
    if available <= 0 or ww <= 0:
        return 0
    return max(0, int((available + gap) / (ww + gap)))


def _array_anchors(center, count, ww, gap, reg, wc, margin, wall_span=None):
    """Return list of `count` anchor positions for a window array centered at `center`.
    Clamps the center so the whole array stays within wall span + margin.
    """
    total = count * ww + (count - 1) * gap
    span_lo, span_hi = _wall_span(reg, wc, wall_span)
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


def _clamp_window_anchor(anchor, reg, wall_char, win_width, margin, zones=(), wall_span=None):
    """Clamp window centre to wall-edge clearance only (soft clamp, no rejection)."""
    t      = reg.get("t", 0.125)
    margin = max(margin, t)          # never closer to corner than one wall-thickness
    span_lo, span_hi = _wall_span(reg, wall_char, wall_span)
    lo = span_lo + win_width * 0.5 + margin
    hi = span_hi - win_width * 0.5 - margin
    if lo > hi:
        return (span_lo + span_hi) * 0.5
    anchor = max(lo, min(hi, anchor))
    if zones:
        sub = _valid_spans(lo, hi, zones, win_width, margin)
        if sub:
            anchor = min((abs(max(sl, min(sr, anchor)) - anchor),
                          max(sl, min(sr, anchor))) for sl, sr in sub)[1]
    return anchor


def _valid_window_anchor(anchor, reg, wall_char, win_width, margin,
                         skip_idx=None, zones=(), wall_span=None):
    """Return clamped anchor if valid, None if wall too narrow or overlapping.
    skip_idx: window index in reg['windows'] to ignore (used when sliding).
    zones: [(lo, hi)] blocked intervals from adjacent rooms.
    wall_span: (lo, hi) override for extra (extruded) walls.
    """
    t      = reg.get("t", 0.125)
    margin = max(margin, t)          # never closer to corner than one wall-thickness
    span_lo, span_hi = _wall_span(reg, wall_char, wall_span)
    lo = span_lo + win_width * 0.5 + margin
    hi = span_hi - win_width * 0.5 - margin
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
    z0 = room_z + win.get("v_offset", 0.9)
    z1 = z0 + wh
    ew = win.get("ew")
    if ew:
        wcp  = ew["wall_coord"]
        slo  = ew["lo"]
        shi  = ew["hi"]
        lo   = max(slo, anchor - ww * 0.5)
        hi   = min(shi, anchor + ww * 0.5)
        if wc == 'E':
            return [(wcp, lo, z0), (wcp, hi, z0), (wcp, hi, z1), (wcp, lo, z1)]
        elif wc == 'W':
            return [(wcp, hi, z0), (wcp, lo, z0), (wcp, lo, z1), (wcp, hi, z1)]
        elif wc == 'N':
            return [(lo, wcp, z0), (hi, wcp, z0), (hi, wcp, z1), (lo, wcp, z1)]
        else:
            return [(hi, wcp, z0), (lo, wcp, z0), (lo, wcp, z1), (hi, wcp, z1)]
    x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
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
            ew     = win.get("ew")
            if wc == 'E':
                dist  = abs(px - (ew["wall_coord"] if ew else x2));  along = py
            elif wc == 'W':
                dist  = abs(px - (ew["wall_coord"] if ew else x1));  along = py
            elif wc == 'N':
                dist  = abs(py - (ew["wall_coord"] if ew else y2));  along = px
            else:
                dist  = abs(py - (ew["wall_coord"] if ew else y1));  along = px
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
    z0, z1 = z, z + dh
    ew = door.get("ew")
    if ew:
        wcp = ew["wall_coord"]
        slo = ew["lo"]
        shi = ew["hi"]
        lo  = max(slo, anchor - dw * 0.5)
        hi  = min(shi, anchor + dw * 0.5)
        if wc == 'E':
            return [(wcp, lo, z0), (wcp, hi, z0), (wcp, hi, z1), (wcp, lo, z1)]
        elif wc == 'W':
            return [(wcp, hi, z0), (wcp, lo, z0), (wcp, lo, z1), (wcp, hi, z1)]
        elif wc == 'N':
            return [(lo, wcp, z0), (hi, wcp, z0), (hi, wcp, z1), (lo, wcp, z1)]
        else:
            return [(hi, wcp, z0), (lo, wcp, z0), (lo, wcp, z1), (hi, wcp, z1)]
    x1, y1, x2, y2 = r["x1"], r["y1"], r["x2"], r["y2"]
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


def _mesh_ghost_tris(src, opening, reg, is_window=False):
    """Return a flat list of world-space (x, y, z) tuples, 3 per triangle, for
    drawing a semi-transparent GPU preview of *src* placed at *opening*.
    Uses the same scale / rotation / translation as _place_opening_mesh."""
    import math
    cx, cy, cz, rot_z = _opening_world_pos(opening, reg, is_window=is_window, src=src)

    bb     = src.bound_box
    xs_w   = [v[0] for v in bb]
    zs_w   = [v[2] for v in bb]
    src_dx = max(xs_w) - min(xs_w)
    src_dz = max(zs_w) - min(zs_w)
    w  = opening.get("w",  opening.get("window_width",  1.0))
    h  = opening.get("h",  opening.get("window_height", 1.0))
    sx = w / max(src_dx, 1e-4) if src_dx > 1e-4 else 1.0
    sz = h / max(src_dz, 1e-4) if src_dz > 1e-4 else 1.0

    cr, sr = math.cos(rot_z), math.sin(rot_z)
    mverts = src.data.vertices
    out    = []
    for poly in src.data.polygons:
        vidx = list(poly.vertices)
        for i in range(1, len(vidx) - 1):
            for vi in (vidx[0], vidx[i], vidx[i + 1]):
                co = mverts[vi].co
                lx, ly, lz = co.x * sx, co.y, co.z * sz
                out.append((cx + lx * cr - ly * sr,
                             cy + lx * sr + ly * cr,
                             cz + lz))
    return out


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
    entry.doors_json   = json.dumps(reg.get("doors",       []))
    entry.windows_json = json.dumps(reg.get("windows",     []))
    entry.stairs_json  = json.dumps(reg.get("stair_holes", []))
    entry.no_walls     = ",".join(reg.get("no_walls", []))
    entry.obj_name   = reg.get("obj_name", "")
    entry.plinth_bottom_enabled = reg.get('plinth_bottom_enabled', False)
    entry.plinth_top_enabled    = reg.get('plinth_top_enabled',    False)
    entry.mesh_locked           = reg.get('mesh_locked',           False)
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
    try:
        stair_holes = json.loads(getattr(entry, "stairs_json", None) or "[]")
    except Exception:
        stair_holes = []
    return {
        "x1": entry.x1, "y1": entry.y1,
        "x2": entry.x2, "y2": entry.y2,
        "t":  entry.t,
        "z":  entry.z,
        "doors":       doors,
        "windows":     windows,
        "stair_holes": stair_holes,
        "no_walls": [w for w in entry.no_walls.split(",") if w],
        "obj_name": entry.obj_name,
        "plinth_bottom_enabled": getattr(entry, 'plinth_bottom_enabled', False),
        "plinth_top_enabled":    getattr(entry, 'plinth_top_enabled',    False),
        "mesh_locked":           getattr(entry, 'mesh_locked',           False),
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


def _door_swing_arc(wc, anchor, dw, z, wall_pos, flip_lr=False, flip_io=False, n=16):
    """Return (hinge_pt, closed_tip, arc_pts) for a door-swing arc at floor height z.
    All points are (x, y, z) tuples.  The arc sweeps 90° from the closed door
    position (along the wall) to the open position (into/out of the room)."""
    import math as _math
    from mathutils import Vector as _V
    hw = dw * 0.5
    if wc in ('N', 'S'):
        along  = _V((1, 0, 0))
        inward = _V((0, -1 if wc == 'N' else 1, 0))
        hinge_base = _V((anchor, wall_pos, z))
    else:
        along  = _V((0, 1, 0))
        inward = _V((-1 if wc == 'E' else 1, 0, 0))
        hinge_base = _V((wall_pos, anchor, z))
    if flip_lr:
        hinge  = hinge_base + along * hw
        c_tip  = hinge_base - along * hw
    else:
        hinge  = hinge_base - along * hw
        c_tip  = hinge_base + along * hw
    if flip_io:
        inward = -inward
    close_dir = (c_tip - hinge).normalized()
    cross_z   = close_dir.x * inward.y - close_dir.y * inward.x
    sweep     = _math.pi * 0.5 * (1 if cross_z >= 0 else -1)
    base_a    = _math.atan2(close_dir.y, close_dir.x)
    arc = []
    for i in range(n + 1):
        a = base_a + sweep * (i / n)
        arc.append((hinge.x + dw * _math.cos(a),
                    hinge.y + dw * _math.sin(a), z))
    return (hinge.x, hinge.y, z), (c_tip.x, c_tip.y, z), arc


def _draw_hint_bar(context, text, color=(0.88, 0.88, 0.88, 1.0)):
    """Draw a small one-line key hint at the bottom centre of the 3D viewport.
    Call this from any modal draw callback (POST_VIEW or POST_PIXEL)."""
    try:
        if context.region is None:
            return
        font_id = 0
        blf.size(font_id, 12)
        w, h = blf.dimensions(font_id, text)
        x = (context.region.width - w) * 0.5
        y = 8
        # dark background pill
        gpu.state.blend_set('ALPHA')
        pad_x, pad_y = 10, 5
        bx1, bx2 = x - pad_x, x + w + pad_x
        by1, by2 = y - pad_y, y + h + pad_y
        bg_verts = [(bx1,by1),(bx2,by1),(bx2,by2),(bx1,by2)]
        bg_tris  = [(0,1,2),(0,2,3)]
        _sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        _b  = batch_for_shader(_sh, 'TRIS', {"pos": bg_verts}, indices=bg_tris)
        _sh.uniform_float("color", (0.0, 0.0, 0.0, 0.55))
        _b.draw(_sh)
        gpu.state.blend_set('NONE')
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 3, 0, 0, 0, 0.8)
        blf.shadow_offset(font_id, 0, -1)
        blf.color(font_id, *color)
        blf.position(font_id, x, y, 0)
        blf.draw(font_id, text)
        blf.disable(font_id, blf.SHADOW)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Modal draw operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_draw(bpy.types.Operator):
    bl_idname  = "room.draw"
    bl_label   = "Draw Room"
    bl_options = {"REGISTER", "UNDO"}

    _room_list   = []
    _stair_list  = []   # list of stair runtime dicts
    _addon_kmaps = []

    # ── GPU draw callback ──────────────────────────────────────────────────
    def _draw_cb(self, context):
        s      = context.scene.room_settings
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')

        # ── Orange rectangle overlay while the user is dragging ────────────
        if self._start is not None and self._end is not None:
            z  = s.z_foundation
            x1 = min(self._start.x, self._end.x)
            y1 = min(self._start.y, self._end.y)
            x2 = max(self._start.x, self._end.x)
            y2 = max(self._start.y, self._end.y)
            tris = [
                (x1, y1, z), (x2, y1, z), (x2, y2, z),
                (x1, y1, z), (x2, y2, z), (x1, y2, z),
            ]
            bf = batch_for_shader(shader, 'TRIS', {"pos": tris})
            shader.uniform_float("color", (1.0, 0.55, 0.0, 0.12))
            bf.draw(shader)
            corners = [(x1,y1,z),(x2,y1,z),(x2,y2,z),(x1,y2,z)]
            lines   = [corners[i] for pair in ((0,1),(1,2),(2,3),(3,0)) for i in pair]
            bl = batch_for_shader(shader, 'LINES', {"pos": lines})
            shader.uniform_float("color", (1.0, 0.65, 0.0, 1.0))
            gpu.state.line_width_set(2.5)
            bl.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

        if not self._hovered:
            return
        sp, _normal, room_idx, wall_char, hov_ew, *_ = (*self._hovered, None)
        rooms = ROOM_OT_draw._room_list
        if room_idx >= len(rooms):
            return
        r     = rooms[room_idx]
        # Guard: clear hover if the room object was deleted or hidden
        _obj = bpy.data.objects.get(r.get("obj_name", ""))
        if _obj is None or not _obj.visible_get():
            self._hovered = None
            return
        if hov_ew:
            # Extra wall highlight: use ew bounds
            wcp = hov_ew["wall_coord"]
            lo  = hov_ew["lo"]
            hi  = hov_ew["hi"]
            z0  = s.z_foundation
            z1  = z0 + s.wall_height
            if wall_char in ('N', 'S'):
                verts = [(lo, wcp, z0), (hi, wcp, z0), (hi, wcp, z1), (lo, wcp, z1)]
            else:
                verts = [(wcp, lo, z0), (wcp, hi, z0), (wcp, hi, z1), (wcp, lo, z1)]
        else:
            verts = _wall_face_verts(r, wall_char, s.z_foundation,
                                     s.wall_height, s.wall_thickness)
        idx_fill = [(0,1,2),(0,2,3)]
        idx_line = [(0,1),(1,2),(2,3),(3,0)]
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
            _t_dc     = r.get("t", s.wall_thickness)
            _zones_dc = _wall_adj_zones(r, wall_char, rooms, _t_dc)
            anchor      = _valid_anchor(raw_anchor, r, wall_char, dw, s.door_margin, zones=_zones_dc)
            is_blocked  = (anchor is None)
            draw_anchor = _clamp_anchor(raw_anchor, r, wall_char, dw, s.door_margin, zones=_zones_dc)
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
        _draw_hint_bar(context, getattr(self, '_hint_text', ''))

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
        self._hint_text = text

    def _idle_msg(self, context):
        s = context.scene.room_settings
        door_state = "ON" if s.add_door else "OFF"
        grid_txt   = f"  ·  GRID {s.grid_snap_size * 100:.4g} cm" if s.snap_to_grid else ""
        self._msg(context,
            f"Door [{door_state}]  ·  LMB – draw room  |  Hover wall to snap-connect"
            f"  ·  D – toggle door{grid_txt}  |  Enter/RMB – exit")

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
        self._hint_text    = "LMB – click to start drawing  ·  hover a wall to snap-connect  ·  Enter/RMB – exit"
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

        # Tab / Shift+Tab — cycle door presets during DS phase
        if event.type == 'TAB' and event.value == 'PRESS':
            if self._phase == 'DS':
                presets = context.scene.room_door_presets
                active  = context.scene.room_active_door_preset
                if len(presets) > 1:
                    delta      = -1 if event.shift else 1
                    new_active = (active + delta) % len(presets)
                    context.scene.room_active_door_preset = new_active
                    dp   = presets[new_active]
                    ex_reg = ROOM_OT_draw._room_list[self._ds_room_idx]
                    di     = self._ds_door_idx
                    if di is not None and di < len(ex_reg.get("doors", [])):
                        new_src = dp.mesh_object.name if dp.mesh_object else None
                        ex_reg["doors"][di]["w"]           = dp.door_width
                        ex_reg["doors"][di]["h"]           = dp.door_height
                        ex_reg["doors"][di]["mesh_source"] = new_src
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
                                    pd["w"]           = dp.door_width
                                    pd["h"]           = dp.door_height
                                    pd["mesh_source"] = new_src
                                    break
                            _rebuild_room_mesh(p_reg, s)
                        _sync_to_scene(context)
                        context.area.header_text_set(
                            f"Preset: {dp.name} ({dp.door_width:.2f}×{dp.door_height:.2f}m)  |  "
                            "Tab/Shift+Tab to cycle · Click to confirm  |  RMB – cancel")
                        context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # Scroll — navigation only (presets via Tab/Shift+Tab)
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
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
            if pt is not None:
                pt = _apply_grid_snap(pt, context, s)

            if self._phase == 0:
                if pt:
                    snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list, s.wall_thickness, current_z=z)
                    self._hovered = snap
                    if snap:
                        if s.add_door:
                            sp, _n, ri, wc, *_ = snap
                            dw = s.door_width
                            active  = context.scene.room_active_door_preset
                            presets = context.scene.room_door_presets
                            if 0 <= active < len(presets):
                                dw = presets[active].door_width
                            raw = sp.y if wc in ('E', 'W') else sp.x
                            _r_hov   = ROOM_OT_draw._room_list[ri]
                            _t_hov   = _r_hov.get("t", s.wall_thickness)
                            _zones_hov = _wall_adj_zones(_r_hov, wc, ROOM_OT_draw._room_list, _t_hov)
                            if _valid_anchor(raw, _r_hov, wc, dw, s.door_margin,
                                             zones=_zones_hov) is None:
                                self._msg(context,
                                    "Cannot place door here — blocked  ·  "
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
                    _ew_ds   = doors[di].get("ew")
                    _t_ds    = ex_reg.get("t", s.wall_thickness)
                    _zones_ds = ([] if _ew_ds else
                                 _wall_adj_zones(ex_reg, wc, ROOM_OT_draw._room_list, _t_ds))
                    _span_ds  = (_ew_ds["lo"], _ew_ds["hi"]) if _ew_ds else None
                    anchor = _valid_anchor(raw, ex_reg, wc, dw, s.door_margin, skip_idx=di,
                                          zones=_zones_ds, wall_span=_span_ds)
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
            pt = _apply_grid_snap(pt, context, s)

            # ── Click 1: lock start point ──────────────────────────────
            if self._phase == 0:
                snap = _wall_snap_ext(pt, ROOM_OT_draw._room_list, s.wall_thickness, current_z=z)
                if snap:
                    sp, normal, room_idx, wc, snap_ew = snap
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
                        _t_dsc    = ex_reg.get("t", s.wall_thickness)
                        _zones_dsc = ([] if snap_ew else
                                      _wall_adj_zones(ex_reg, wc, ROOM_OT_draw._room_list, _t_dsc))
                        _span_dsc  = (snap_ew["lo"], snap_ew["hi"]) if snap_ew else None
                        anchor = _valid_anchor(raw_anchor, ex_reg, wc, ds_dw, s.door_margin,
                                               zones=_zones_dsc, wall_span=_span_dsc)
                        if anchor is None:
                            # Blocked (shared wall boundary, window, or wall too narrow) — skip door
                            self._ds_room_idx = self._ds_wall_char = self._ds_door_idx = None
                            self._end   = self._start.copy()
                            self._phase = 1
                            self._create_preview(context)
                            self._msg(context,
                                "No room for a door here — window is blocking  |  RMB – cancel")
                        else:
                            new_door = {"wc": wc, "anchor": anchor,
                                        "w": ds_dw, "h": ds_dh}
                            # Attach mesh from active preset (same as door edit mode)
                            _dp2 = context.scene.room_door_presets
                            _da2 = context.scene.room_active_door_preset
                            if 0 <= _da2 < len(_dp2) and _dp2[_da2].mesh_object:
                                new_door["mesh_source"] = _dp2[_da2].mesh_object.name
                            if snap_ew:
                                new_door["ew"] = snap_ew
                            ex_reg.setdefault("doors", []).append(new_door)
                            self._ds_door_idx = len(ex_reg["doors"]) - 1
                            _rebuild_room_mesh(ex_reg, s)
                            self._end  = self._start.copy()
                            self._phase = 'DS'
                            context.area.tag_redraw()
                            self._msg(context,
                                "Slide door along wall · Tab/Shift+Tab=preset  |  Click to confirm  |  RMB – cancel")
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
                        "plinth_bottom_enabled": s.add_plinth_bottom,
                        "plinth_top_enabled":    s.add_plinth_top,
                        "mesh_locked":           False,
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


_hint_cb_state = {"prev_mode": None, "prev_obj": None}   # module-level state


def _draw_room_hint_cb():
    """Persistent POST_PIXEL handler: draws hint text at the bottom of the 3D
    viewport when the user is in Edit Mode on a room mesh.  Also detects
    edit-mode exit and auto-recalculates plinth if the room has plinth enabled."""
    try:
        ctx = bpy.context
        if ctx.area is None or ctx.region is None:
            return
        active = ctx.active_object
        # ── Detect edit-mode exit and auto-recalculate plinth ─────────────
        cur_mode = ctx.mode
        prev_mode = _hint_cb_state["prev_mode"]
        prev_obj  = _hint_cb_state["prev_obj"]
        _hint_cb_state["prev_mode"] = cur_mode
        _hint_cb_state["prev_obj"]  = active.name if active else None
        if (prev_mode == 'EDIT_MESH' and cur_mode != 'EDIT_MESH' and
                prev_obj is not None):
            exited_reg = next((r for r in ROOM_OT_draw._room_list
                               if r.get('obj_name') == prev_obj), None)
            if exited_reg is not None:
                ex_obj = bpy.data.objects.get(prev_obj)
                s_ec   = getattr(ctx.scene, 'room_settings', None)
                if ex_obj and s_ec and (
                        exited_reg.get('plinth_bottom_enabled', False) or
                        exited_reg.get('plinth_top_enabled', False)):
                    try:
                        _recalculate_plinth_for_obj(exited_reg, ex_obj, s_ec)
                    except Exception:
                        pass
        if active is None:
            return
        reg = next((r for r in ROOM_OT_draw._room_list
                    if r.get('obj_name') == active.name), None)
        if reg is None:
            return
        if ctx.mode != 'EDIT_MESH':
            return
        text = "Exit Edit Mode  ›  Save Edits  to preserve your changes"
        font_id = 0
        blf.size(font_id, 13)
        w, _ = blf.dimensions(font_id, text)
        x = (ctx.region.width - w) * 0.5
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 5, 0, 0, 0, 0.9)
        blf.shadow_offset(font_id, 0, -1)
        blf.color(font_id, 1.0, 0.85, 0.2, 1.0)
        blf.position(font_id, x, 18, 0)
        blf.draw(font_id, text)
        blf.disable(font_id, blf.SHADOW)
    except Exception:
        pass


@bpy.app.handlers.persistent
def _room_registry_cleanup(scene, depsgraph):
    """Remove rooms whose objects no longer exist (deleted or not yet restored by undo).
    Also detect deleted stair objects, strip their holes from rooms, and schedule a
    deferred mesh rebuild (timer avoids infinite depsgraph recursion)."""
    ROOM_OT_draw._room_list[:] = [
        r for r in ROOM_OT_draw._room_list
        if r.get("obj_name", "") in bpy.data.objects
    ]

    # Detect stair objects that were deleted outside the dedicated operator (e.g. X key)
    deleted_stairs = [
        sd for sd in ROOM_OT_draw._stair_list
        if sd.get("obj_name", "") not in bpy.data.objects
    ]
    if not deleted_stairs:
        return

    ROOM_OT_draw._stair_list[:] = [
        sd for sd in ROOM_OT_draw._stair_list
        if sd.get("obj_name", "") in bpy.data.objects
    ]

    # Strip holes from rooms — track which ones need rebuilding
    rooms_to_rebuild = []
    for sd in deleted_stairs:
        name = sd.get("obj_name", "")
        for reg in ROOM_OT_draw._room_list:
            old = reg.get("stair_holes", [])
            new = [h for h in old if h.get("stair_obj") != name]
            if len(new) != len(old):
                reg["stair_holes"] = new
                if reg not in rooms_to_rebuild:
                    rooms_to_rebuild.append(reg)

    if not rooms_to_rebuild:
        return

    # Defer the mesh rebuild to the next frame so we don't recurse into the depsgraph
    def _deferred():
        try:
            s = bpy.context.scene.room_settings
            for reg in rooms_to_rebuild:
                _rebuild_room_mesh(reg, s)
            _sync_to_scene(bpy.context)
        except Exception:
            pass
        return None  # don't repeat

    bpy.app.timers.register(_deferred, first_interval=0.0)


@bpy.app.handlers.persistent
def _room_on_load(*args):
    ROOM_OT_draw._room_list.clear()


def _stair_cut_holes(sd, li, ui, rooms, s):
    """Compute and add ceiling/floor holes for a staircase into the given rooms.

    sd must contain: lx1,ly1,lx2,ly2, ux1,uy1,ux2,uy2, x_travel, z_bot, z_top, obj_name.
    li / ui are the room-list indices for the lower / upper rooms respectively.
    Appends hole dicts and rebuilds affected room meshes (does NOT sync scene).
    """
    lx1, ly1 = sd["lx1"], sd["ly1"]
    lx2, ly2 = sd["lx2"], sd["ly2"]
    ux1, uy1 = sd["ux1"], sd["uy1"]
    ux2, uy2 = sd["ux2"], sd["uy2"]
    x_travel = sd.get("x_travel", True)
    obj_name = sd.get("obj_name", "")
    z_bot    = sd["z_bot"]
    z_top    = sd["z_top"]

    if x_travel:
        if (ux1 + ux2) / 2 >= (lx1 + lx2) / 2:  # upper to the right
            hx1, hx2 = lx2, ux2
            hole_resize = {"resize_axis": "x", "hi_is_fixed": True,
                           "fixed_coord": ux2, "moving_coord_full": lx2}
        else:                                       # upper to the left
            hx1, hx2 = ux2, lx2
            hole_resize = {"resize_axis": "x", "hi_is_fixed": False,
                           "fixed_coord": ux2, "moving_coord_full": lx2}
        hy1, hy2 = min(ly1, uy1), max(ly2, uy2)
    else:
        if (uy1 + uy2) / 2 >= (ly1 + ly2) / 2:  # upper above
            hy1, hy2 = ly2, uy2
            hole_resize = {"resize_axis": "y", "hi_is_fixed": True,
                           "fixed_coord": uy2, "moving_coord_full": ly2}
        else:                                      # upper below
            hy1, hy2 = uy2, ly2
            hole_resize = {"resize_axis": "y", "hi_is_fixed": False,
                           "fixed_coord": uy2, "moving_coord_full": ly2}
        hx1, hx2 = min(lx1, ux1), max(lx2, ux2)

    zt_lower = z_bot + s.wall_height   # lower room ceiling = bottom of slab
    lower_hole = {"x1": hx1, "y1": hy1, "x2": hx2, "y2": hy2,
                  "cut": "ceiling", "stair_obj": obj_name,
                  "slab_z": z_top, **hole_resize}
    upper_hole = {"x1": hx1, "y1": hy1, "x2": hx2, "y2": hy2,
                  "cut": "floor",   "stair_obj": obj_name,
                  "slab_z": zt_lower, **hole_resize}
    rooms[li].setdefault("stair_holes", []).append(lower_hole)
    _rebuild_room_mesh(rooms[li], s)
    if ui != li:
        rooms[ui].setdefault("stair_holes", []).append(upper_hole)
        _rebuild_room_mesh(rooms[ui], s)


def _resync_stair_holes(context):
    """Re-read stair_holes from the scene registry for every room in _room_list.
    Must be called on stair operator invoke so that Ctrl+Z state is properly
    reflected before we start a new placement."""
    entries = getattr(context.scene, "room_registry", None)
    if not entries:
        return
    entry_map = {e.obj_name: e for e in entries}
    for reg in ROOM_OT_draw._room_list:
        e = entry_map.get(reg.get("obj_name", ""))
        if e:
            try:
                reg["stair_holes"] = json.loads(getattr(e, "stairs_json", None) or "[]")
            except Exception:
                reg["stair_holes"] = []


@bpy.app.handlers.persistent
def _room_undo_post(*args):
    """After any undo, resync ALL room data (doors, windows, stair_holes) from
    the scene registry so the Python _room_list stays consistent with Blender's
    undo state.  Without this, undone doors/windows come back on the next mesh
    rebuild and 'phantom' openings persist after Ctrl+Z."""
    try:
        scene = bpy.context.scene
        if not scene:
            return
        entries = getattr(scene, "room_registry", None)
        if not entries:
            return
        entry_map = {e.obj_name: e for e in entries}
        for reg in ROOM_OT_draw._room_list:
            e = entry_map.get(reg.get("obj_name", ""))
            if e:
                try:
                    reg["doors"]       = json.loads(getattr(e, "doors_json",   None) or "[]")
                    reg["windows"]     = json.loads(getattr(e, "windows_json", None) or "[]")
                    reg["stair_holes"] = json.loads(getattr(e, "stairs_json",  None) or "[]")
                    reg["mesh_locked"] = getattr(e, "mesh_locked", False)
                except Exception:
                    pass
    except Exception:
        pass


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
    bl_description = ("Remove stuck GPU overlays AND orphaned door/window mesh instances.\n"
                      "Use this if orange/green highlights or floating door/window objects remain.")

    def execute(self, context):
        # ── 1. Remove stuck GPU draw handlers ─────────────────────────────
        removed_handles = 0
        for handle in list(_DRAW_HANDLES):
            try:
                bpy.types.SpaceView3D.draw_handler_remove(handle, 'WINDOW')
                removed_handles += 1
            except Exception:
                pass
        _DRAW_HANDLES.clear()

        # ── 2. Remove orphaned Opening.* mesh instances ────────────────────
        # Collect every mesh_obj_name that IS legitimately tracked
        tracked = set()
        for reg in ROOM_OT_draw._room_list:
            for opening in reg.get("doors", []) + reg.get("windows", []):
                name = opening.get("mesh_obj_name")
                if name:
                    tracked.add(name)

        # Any "Opening.*" object NOT in tracked is a phantom — delete it
        removed_objects = 0
        for obj in list(bpy.data.objects):
            if obj.name.startswith("Opening.") and obj.name not in tracked:
                for col in list(obj.users_collection):
                    col.objects.unlink(obj)
                bpy.data.objects.remove(obj)
                removed_objects += 1

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        self.report({'INFO'},
            f"Cleared {removed_handles} overlay handle(s), "
            f"{removed_objects} phantom mesh instance(s)")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════════
# Unified door-frame edit-mode operator
# LMB click        → add door (no drag)
# LMB hold + drag  → slide door along wall
# Double-click     → remove door
# Tab / Shift+Tab  → cycle preset live
# RMB / Esc        → cancel / exit
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_door_edit(bpy.types.Operator):
    bl_idname  = "room.door_edit"
    bl_label   = "Door Frame Edit Mode"
    bl_description = ("Edit door frames.  "
                      "LMB=add  ·  LMB+drag=slide  ·  Tab/Shift+Tab=preset  ·  "
                      "Double-click=flip/remove  |  RMB/Esc=exit")
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
                # Refresh stale ew refs so overlay draws at the current wall position
                if r.get("mesh_locked", False):
                    _refresh_ew_refs(r, s)
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
            ri, wc, anchor, snap_ew = self._hover_snap
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
                if snap_ew:
                    ghost_door["ew"] = snap_ew
                # Check whether this position is blocked (shared wall boundary, window, or too narrow)
                _t_de     = r.get("t", s.wall_thickness)
                _zones_de = ([] if snap_ew else _wall_adj_zones(r, wc, rooms, _t_de))
                _span_de  = (snap_ew["lo"], snap_ew["hi"]) if snap_ew else None
                is_blocked = (_valid_anchor(anchor, r, wc, dw, s.door_margin,
                                            zones=_zones_de, wall_span=_span_de) is None)
                # ── Mesh ghost (semi-transparent solid) ─────────────────────
                _mesh_src_de = None
                if 0 <= active < len(presets) and presets[active].mesh_object:
                    _mesh_src_de = presets[active].mesh_object
                if _mesh_src_de is not None:
                    try:
                        _gtris = _mesh_ghost_tris(_mesh_src_de, ghost_door, r)
                        if _gtris:
                            _gb = batch_for_shader(shader, 'TRIS', {"pos": _gtris})
                            shader.uniform_float("color",
                                (0.55, 0.82, 1.0, 0.40) if not is_blocked
                                else (1.0, 0.40, 0.30, 0.35))
                            _gb.draw(shader)
                    except Exception:
                        pass
                # ── Rectangle outline ────────────────────────────────────────
                verts = _door_frame_verts(ghost, ghost_door,
                                          r.get("z", s.z_foundation), dw, dh)
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
                # ── Swing arc ────────────────────────────────────────────
                if not is_blocked:
                    try:
                        z_arc = r.get("z", s.z_foundation) + 0.02
                        wp = (snap_ew["wall_coord"] if snap_ew else
                              {"S": r["y1"], "N": r["y2"],
                               "W": r["x1"], "E": r["x2"]}[wc])
                        _h, _ct, _arc = _door_swing_arc(wc, anchor, dw, z_arc, wp)
                        arc_lines = []
                        for i in range(len(_arc) - 1):
                            arc_lines.extend([_arc[i], _arc[i + 1]])
                        arc_lines.extend([_h, _ct])
                        ba = batch_for_shader(shader, 'LINES', {"pos": arc_lines})
                        shader.uniform_float("color", (0.1, 1.0, 0.3, 0.5))
                        gpu.state.line_width_set(1.5)
                        ba.draw(shader)
                    except Exception:
                        pass

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        # ── W×H cursor readout ───────────────────────────────────────────
        if self._hover_snap is not None:
            try:
                dw = s.door_width;  dh = s.door_height
                active  = context.scene.room_active_door_preset
                presets = context.scene.room_door_presets
                if 0 <= active < len(presets):
                    dw, dh = presets[active].door_width, presets[active].door_height
                label = f"W {dw*100:.0f}  ×  H {dh*100:.0f} cm"
                font_id = 0
                blf.size(font_id, 12)
                blf.color(font_id, 1.0, 1.0, 1.0, 0.9)
                mx = getattr(self, '_mouse_rx', 0) + 14
                my = getattr(self, '_mouse_ry', 0) + 14
                blf.position(font_id, mx, my, 0)
                blf.draw(font_id, label)
            except Exception:
                pass
        _draw_hint_bar(context, getattr(self, '_hint_text', ''))

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
    def _header(self, context, text=None):
        if text is None:
            text = ("LMB – add  ·  LMB+drag – slide  ·  Tab/Shift+Tab – preset  ·  "
                    "Double-click – flip/remove  ·  RMB/Esc – exit")
        context.area.header_text_set(text)
        self._hint_text = text

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
            _remove_threshold(reg["doors"][di])
            _remove_opening_mesh(reg["doors"][di])
            reg["doors"].pop(di)
        _rebuild_room_mesh(reg, s)
        if partner is not None:
            p_ri, p_di = partner
            p_reg = rooms[p_ri]
            if p_di < len(p_reg.get("doors", [])):
                _remove_threshold(p_reg["doors"][p_di])
                _remove_opening_mesh(p_reg["doors"][p_di])
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
        self._hint_text       = ""
        self._mouse_rx        = 0
        self._mouse_ry        = 0
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

        # ── Tab / Shift+Tab — cycle door presets ──────────────────────────────
        if event.type == 'TAB' and event.value == 'PRESS':
            presets = context.scene.room_door_presets
            if len(presets) > 1:
                active     = context.scene.room_active_door_preset
                delta      = -1 if event.shift else 1
                new_active = (active + delta) % len(presets)
                context.scene.room_active_door_preset = new_active
                dp = presets[new_active]
                if (self._active_idx is not None and self._active_door_idx is not None):
                    # A door is actively being held/slid — update its dims + mesh
                    reg   = rooms[self._active_idx]
                    di    = self._active_door_idx
                    doors = reg.get("doors", [])
                    if di < len(doors):
                        # Save original dims on first cycle (for RMB cancel)
                        if self._preview_orig is None:
                            orig_dims    = {"w": doors[di]["w"], "h": doors[di]["h"]}
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
                        new_src = dp.mesh_object.name if dp.mesh_object else None
                        doors[di]["w"]           = dp.door_width
                        doors[di]["h"]           = dp.door_height
                        doors[di]["mesh_source"] = new_src
                        _rebuild_room_mesh(reg, s)
                        _sync_opening_meshes(reg, s=s)
                        if self._active_partner is not None:
                            p_ri, p_di = self._active_partner
                            p_doors = rooms[p_ri].get("doors", [])
                            if p_di < len(p_doors):
                                p_doors[p_di]["w"]           = dp.door_width
                                p_doors[p_di]["h"]           = dp.door_height
                                p_doors[p_di]["mesh_source"] = new_src
                            _rebuild_room_mesh(rooms[p_ri], s)
                            _sync_opening_meshes(rooms[p_ri], s=s)
                        _sync_to_scene(context)
                        context.area.header_text_set(
                            f"Preset: {dp.name} ({dp.door_width:.2f}×{dp.door_height:.2f}m)  |  "
                            "Tab/Shift+Tab to cycle  ·  Release LMB to confirm  |  RMB=cancel")
                else:
                    # No door held — just cycle the active preset for next placement
                    context.area.header_text_set(
                        f"Preset: {dp.name} ({dp.door_width:.2f}×{dp.door_height:.2f}m)  |  "
                        "Tab/Shift+Tab to cycle")
                context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        # ── Scroll — navigation only (presets via Tab/Shift+Tab) ──────────
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.value == 'PRESS':
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
            self._mouse_rx = event.mouse_region_x
            self._mouse_ry = event.mouse_region_y
            pt = _ray_to_z(context, event, s.z_foundation)

            if self._phase == 'HOVER':
                # Existing door frame detection still uses floor projection.
                # New-placement snap uses mesh raycast so it only fires when the
                # cursor is actually pointing at a wall surface.
                self._hovered_door = _door_snap(pt, rooms, s, s.wall_thickness) if pt else None
                self._hover_snap   = (
                    _snap_wall_from_ray(context, event, rooms, s.wall_thickness,
                                        current_z=s.z_foundation)
                    if self._hovered_door is None else None
                )
                # Show a warning header when the ghost position is blocked
                if self._hover_snap is not None:
                    ri, wc, anchor, snap_ew = self._hover_snap
                    if 0 <= ri < len(rooms):
                        dw = s.door_width
                        active  = context.scene.room_active_door_preset
                        presets = context.scene.room_door_presets
                        if 0 <= active < len(presets):
                            dw = presets[active].door_width
                        _t_hv    = rooms[ri].get("t", s.wall_thickness)
                        _zones_hv = ([] if snap_ew else _wall_adj_zones(rooms[ri], wc, rooms, _t_hv))
                        _span_hv  = (snap_ew["lo"], snap_ew["hi"]) if snap_ew else None
                        if _valid_anchor(anchor, rooms[ri], wc, dw, s.door_margin,
                                         zones=_zones_hv, wall_span=_span_hv) is None:
                            context.area.header_text_set("Cannot place door here — blocked")
                    self._hint_text = "Cannot place door here — blocked"
                        else:
                            self._header(context)
                    else:
                        self._header(context)
                else:
                    self._header(context)
                context.area.tag_redraw()
                return {'RUNNING_MODAL', 'PASS_THROUGH'}

            elif self._phase == 'LMB_DOWN':
                if (self._press_screen and
                        (abs(event.mouse_region_x - self._press_screen[0]) > self._DRAG_PX or
                         abs(event.mouse_region_y - self._press_screen[1]) > self._DRAG_PX)):
                    self._phase = 'SLIDING'
                    _sl_txt = ("Sliding door  ·  Tab/Shift+Tab – preset  ·  "
                               "Release to confirm  ·  RMB – cancel")
                    context.area.header_text_set(_sl_txt)
                    self._hint_text = _sl_txt
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
                    _t_sl    = reg.get("t", s.wall_thickness)
                    _zones_sl = _wall_adj_zones(reg, self._active_wc, rooms, _t_sl)
                    anchor = _valid_anchor(raw, reg, self._active_wc, dw,
                                          s.door_margin, skip_idx=di, zones=_zones_sl)
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

                # ── Double-click: open flip / remove dialog ────────────
                is_double = ((now - self._last_press_time) < 0.3 and
                             self._last_press_door is not None)
                if is_double and pt is not None:
                    current_door = _door_snap(pt, rooms, s, s.wall_thickness)
                    if current_door is not None and current_door == self._last_press_door:
                        ri, di = current_door
                        self._last_press_time = 0.0
                        self._last_press_door = None
                        self._hovered_door    = None
                        self._go_hover(context)
                        bpy.ops.room.door_flip('INVOKE_DEFAULT', room_idx=ri, door_idx=di)
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
                        "Hold+drag=slide  ·  Tab/Shift+Tab=preset  |  "
                        "Release=confirm  ·  RMB=cancel")

                elif self._hover_snap is not None:
                    # Press on wall → add new door
                    ri, wc, raw_anchor, snap_ew = self._hover_snap
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
                        _t_add    = reg.get("t", s.wall_thickness)
                        _zones_add = ([] if snap_ew else _wall_adj_zones(reg, wc, rooms, _t_add))
                        _span_add  = (snap_ew["lo"], snap_ew["hi"]) if snap_ew else None
                        anchor = _valid_anchor(raw_anchor, reg, wc, dw, s.door_margin,
                                               zones=_zones_add, wall_span=_span_add)
                        if anchor is None:
                            context.area.header_text_set(
                                "Cannot place door here — try a different spot")
                            self._last_press_door = None
                            return {'RUNNING_MODAL'}
                        new_door = {"wc": wc, "anchor": anchor, "w": dw, "h": dh}
                        # Attach mesh from active preset if one is assigned
                        _dp = context.scene.room_door_presets
                        _da = context.scene.room_active_door_preset
                        if 0 <= _da < len(_dp) and _dp[_da].mesh_object:
                            new_door["mesh_source"] = _dp[_da].mesh_object.name
                        if snap_ew:
                            new_door["ew"] = snap_ew
                        reg.setdefault("doors", []).append(new_door)
                        new_di = len(reg["doors"]) - 1
                        _rebuild_room_mesh(reg, s)
                        partner = _find_partner_wall(rooms, ri, wc, s.wall_thickness)
                        partner_door_idx = None
                        if partner is not None:
                            p_ri, p_wc = partner
                            p_reg = rooms[p_ri]
                            p_door = {"wc": p_wc, "anchor": anchor, "w": dw, "h": dh}
                            if new_door.get("mesh_source"):
                                p_door["mesh_source"] = new_door["mesh_source"]
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
                            "Hold+drag=slide  ·  Tab/Shift+Tab=preset  |  "
                            "Release=confirm  ·  RMB=cancel")
                else:
                    self._last_press_door = None
                    return {'PASS_THROUGH'}

                return {'RUNNING_MODAL'}

            elif event.value == "RELEASE":
                if self._phase in ('LMB_DOWN', 'SLIDING'):
                    self._preview_orig = None   # dims confirmed
                    _sync_to_scene(context)
                    bpy.ops.ed.undo_push(message="Place Door")
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
# Door flip / remove popup operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_door_flip(bpy.types.Operator):
    bl_idname  = "room.door_flip"
    bl_label   = "Flip / Remove Door"
    bl_options = {'REGISTER', 'UNDO'}

    room_idx  : bpy.props.IntProperty(options={'HIDDEN'})
    door_idx  : bpy.props.IntProperty(options={'HIDDEN'})
    flip_lr   : bpy.props.BoolProperty(name="Flip Left \u2194 Right",     default=False)
    flip_io   : bpy.props.BoolProperty(name="Flip Inside \u2194 Outside", default=False)
    do_remove : bpy.props.BoolProperty(name="Remove Door",                default=False)

    def invoke(self, context, event):
        rooms = ROOM_OT_draw._room_list
        ri, di = self.room_idx, self.door_idx
        if ri < len(rooms):
            doors = rooms[ri].get("doors", [])
            if di < len(doors):
                self.flip_lr = doors[di].get("flip_lr", False)
                self.flip_io = doors[di].get("flip_io", False)
        return context.window_manager.invoke_props_dialog(self, width=240)

    def draw(self, context):
        col = self.layout.column(align=True)
        col.prop(self, "flip_lr",   toggle=True, icon="ARROW_LEFTRIGHT")
        col.prop(self, "flip_io",   toggle=True, icon="UV_SYNC_SELECT")
        col.separator()
        col.prop(self, "do_remove", toggle=True, icon="TRASH")

    def execute(self, context):
        s     = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list
        ri, di = self.room_idx, self.door_idx
        if ri >= len(rooms):
            return {'CANCELLED'}
        doors = rooms[ri].get("doors", [])
        if di >= len(doors):
            return {'CANCELLED'}
        if self.do_remove:
            partner = _find_partner_door(rooms, ri, di, s.wall_thickness)
            _remove_opening_mesh(doors[di])
            doors.pop(di)
            _rebuild_room_mesh(rooms[ri], s)
            if partner:
                p_ri, p_di = partner
                p_doors = rooms[p_ri].get("doors", [])
                if p_di < len(p_doors):
                    _remove_opening_mesh(p_doors[p_di])
                    p_doors.pop(p_di)
                _rebuild_room_mesh(rooms[p_ri], s)
        else:
            doors[di]["flip_lr"] = self.flip_lr
            doors[di]["flip_io"] = self.flip_io
            if doors[di].get("mesh_source"):
                _place_opening_mesh(doors[di], doors[di]["mesh_source"],
                                    doors[di]["w"], doors[di]["h"], rooms[ri])
        _sync_to_scene(context)
        bpy.ops.ed.undo_push(message="Flip Door")
        return {'FINISHED'}


class ROOM_OT_door_flip_all(bpy.types.Operator):
    bl_idname  = "room.door_flip_all"
    bl_label   = "Flip All Doors"
    bl_options = {'REGISTER', 'UNDO'}

    flip_axis : bpy.props.EnumProperty(
        items=[('LR', "Left/Right",     "Mirror all doors left\u2194right"),
               ('IO', "Inside/Outside", "Mirror all doors inside\u2194outside")],
        default='LR')

    def execute(self, context):
        s = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list
        for reg in rooms:
            for door in reg.get("doors", []):
                if self.flip_axis == 'LR':
                    door["flip_lr"] = not door.get("flip_lr", False)
                else:
                    door["flip_io"] = not door.get("flip_io", False)
                if door.get("mesh_source"):
                    _place_opening_mesh(door, door["mesh_source"],
                                        door["w"], door["h"], reg)
        _sync_to_scene(context)
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════════
# Window-frame edit operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_window_edit(bpy.types.Operator):
    bl_idname  = "room.window_edit"
    bl_label   = "Window Frame Edit Mode"
    bl_description = ("Edit window frames.  "
                      "LMB=place (click H then V)  ·  LMB+drag=slide  ·  "
                      "Tab/Shift+Tab=preset  ·  Double-click=remove  |  RMB/Esc=exit")
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
                        # Clamp v_offset for display in case stored value is wrong
                        _voff_clamped = max(0.0, min(win.get("v_offset", 0.9),
                                                     max(0.0, s.wall_height - wh - s.wall_thickness)))
                        _win_draw = dict(win, v_offset=_voff_clamped)
                        verts  = _window_frame_verts(r, _win_draw, r.get("z", s.z_foundation), ww, wh)
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
                # Refresh stale ew refs so overlay draws at the current wall position
                if r.get("mesh_locked", False):
                    _refresh_ew_refs(r, s)
                wins = r.get("windows", [])
                if wi < len(wins):
                    win  = wins[wi]
                    ww, wh = win["w"], win["h"]
                    _voff_h = max(0.0, min(win.get("v_offset", 0.9),
                                           max(0.0, s.wall_height - wh - s.wall_thickness)))
                    verts = _window_frame_verts(r, dict(win, v_offset=_voff_h),
                                                r.get("z", s.z_foundation), ww, wh)
                    b = batch_for_shader(shader, 'TRIS', {"pos": verts}, indices=idx_fill)
                    shader.uniform_float("color", (0.05, 0.85, 0.2, 0.30))
                    b.draw(shader)
                    gpu.state.line_width_set(2.5)
                    b2 = batch_for_shader(shader, 'LINES', {"pos": verts}, indices=idx_line)
                    shader.uniform_float("color", (0.1, 1.0, 0.3, 1.0))
                    b2.draw(shader)

        elif self._hover_snap is not None:
            ri, wc, anchor, snap_ew = self._hover_snap
            if ri < len(rooms):
                r  = rooms[ri]
                # Guard: skip if room object deleted or hidden
                _obj = bpy.data.objects.get(r.get("obj_name", ""))
                if _obj is None or not _obj.visible_get():
                    self._hover_snap = None
                    gpu.state.blend_set('NONE')
                    gpu.state.depth_test_set('NONE')
                    return
                active  = context.scene.room_active_window_preset
                presets = context.scene.room_window_presets
                if 0 <= active < len(presets):
                    wp = presets[active]
                    ww, wh, voff = wp.window_width, wp.window_height, wp.v_offset
                else:
                    ww, wh = s.window_width, s.window_height
                    voff = max(0.0, (s.wall_height - wh) / 2.0)  # centre in wall
                # Clamp sill height so window fits inside the room walls
                voff = max(0.0, min(voff, max(0.0, s.wall_height - wh - s.wall_thickness)))
                count   = s.window_array_count
                gap     = s.window_array_gap
                _span_we = (snap_ew["lo"], snap_ew["hi"]) if snap_ew else None
                count   = min(count, max(1, _max_array_count(rooms[ri], wc, ww, gap, s.door_margin,
                                                              wall_span=_span_we)))
                anchors = _array_anchors(anchor, count, ww, gap, rooms[ri], wc, s.door_margin,
                                         wall_span=_span_we)
                # Green when every anchor is valid, red if any is blocked.
                # Draw at zone-clamped positions so the ghost shows exactly where
                # the window will land (snapped away from the adjacent-room zone).
                _t_cb = r.get("t", s.wall_thickness)
                _zones_cb = ([] if snap_ew else _wall_adj_zones(r, wc, rooms, _t_cb))
                display_anchors = [_clamp_window_anchor(anch, r, wc, ww, s.door_margin,
                                                        zones=_zones_cb, wall_span=_span_we)
                                   for anch in anchors]
                is_blocked = any(
                    _valid_window_anchor(anch, r, wc, ww, s.door_margin,
                                         zones=_zones_cb, wall_span=_span_we) is None
                    for anch in anchors
                )
                # Resolve mesh source for ghost preview
                _mesh_src_we = None
                if 0 <= active < len(presets) and presets[active].mesh_object:
                    _mesh_src_we = presets[active].mesh_object
                for anch in display_anchors:
                    ghost_win = {"wc": wc, "anchor": anch, "v_offset": voff, "w": ww, "h": wh}
                    if snap_ew:
                        ghost_win["ew"] = snap_ew
                    # ── Mesh ghost (semi-transparent solid) ──────────────────
                    if _mesh_src_we is not None:
                        try:
                            _gtris = _mesh_ghost_tris(_mesh_src_we, ghost_win, r,
                                                      is_window=True)
                            if _gtris:
                                _gb = batch_for_shader(shader, 'TRIS', {"pos": _gtris})
                                shader.uniform_float("color",
                                    (0.55, 0.82, 1.0, 0.40) if not is_blocked
                                    else (1.0, 0.40, 0.30, 0.35))
                                _gb.draw(shader)
                        except Exception:
                            pass
                    # ── Rectangle outline ─────────────────────────────────────
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
        # ── W×H cursor readout ───────────────────────────────────────────
        if self._hover_snap is not None:
            try:
                s_r = context.scene.room_settings
                ww, wh, _ = self._resolve_dims(s_r)
                label = f"W {ww*100:.0f}  ×  H {wh*100:.0f} cm"
                font_id = 0
                blf.size(font_id, 12)
                blf.color(font_id, 1.0, 1.0, 1.0, 0.9)
                blf.position(font_id, getattr(self, '_mouse_rx', 0) + 14,
                             getattr(self, '_mouse_ry', 0) + 14, 0)
                blf.draw(font_id, label)
            except Exception:
                pass
        _draw_hint_bar(context, getattr(self, '_hint_text', ''))

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
    def _header(self, context, text=None):
        if text is None:
            text = ("LMB – place  ·  LMB+drag – slide  ·  Tab/Shift+Tab – preset  ·  "
                    "Scroll – count  ·  Double-click – remove  ·  RMB/Esc – exit")
        context.area.header_text_set(text)
        self._hint_text = text

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
                _remove_opening_mesh(wins[wi + k])
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
                    _remove_opening_mesh(p_wins[p_wi + k])
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
        active  = context.scene.room_active_window_preset
        presets = context.scene.room_window_presets
        if 0 <= active < len(presets):
            wp = presets[active]
            ww, wh, voff = wp.window_width, wp.window_height, wp.v_offset
        else:
            ww, wh = s.window_width, s.window_height
            # Default: centre window vertically in the wall
            voff = max(0.0, (s.wall_height - wh) / 2.0)
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
        # Repair any windows whose stored v_offset is out of bounds (e.g. from
        # sampling a mesh placed high in the scene).  Only repositions instances.
        s_inv = context.scene.room_settings
        for _reg in ROOM_OT_draw._room_list:
            _sync_opening_meshes(_reg, s=s_inv)
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
        self._hint_text       = ""
        self._mouse_rx        = 0
        self._mouse_ry        = 0
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

        # ── Tab / Shift+Tab — cycle window presets ────────────────────────────
        if event.type == 'TAB' and event.value == 'PRESS':
            presets = context.scene.room_window_presets
            if len(presets) > 0:
                active     = context.scene.room_active_window_preset
                delta      = -1 if event.shift else 1
                new_active = (active + delta) % len(presets)
                context.scene.room_active_window_preset = new_active
                wp = presets[new_active]
                # If there is an active window being placed/moved, update its dims
                if (self._phase in ('LMB_DOWN', 'SLIDING', 'H_POSITION', 'V_POSITION') and
                        self._active_idx is not None and self._active_win_idx is not None):
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
                    context.area.header_text_set(
                        f"Preset: {wp.name} ({wp.window_width:.2f}×{wp.window_height:.2f}m)  |  "
                        "Tab/Shift+Tab to cycle  |  RMB – cancel")
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
                           f"|  Tab/Shift+Tab=preset  |  Click confirms H  |  RMB – cancel")
                    context.area.header_text_set(hdr)
                    context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            # All other phases: navigation
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
            self._mouse_rx = event.mouse_region_x
            self._mouse_ry = event.mouse_region_y
            pt = _ray_to_z(context, event, s.z_foundation)

            if self._phase == 'HOVER':
                # Existing window frame detection uses floor projection.
                # New-placement snap uses mesh raycast.
                self._hovered_win = _window_snap(pt, rooms, s, s.wall_thickness) if pt else None
                self._hover_snap  = (
                    _snap_wall_from_ray(context, event, rooms, s.wall_thickness,
                                        current_z=s.z_foundation)
                    if self._hovered_win is None else None
                )
                # Warning header when hover position is blocked
                if self._hover_snap is not None:
                    ri, wc, anchor, snap_ew = self._hover_snap
                    if 0 <= ri < len(rooms):
                        ww     = s.window_width
                        active  = context.scene.room_active_window_preset
                        presets = context.scene.room_window_presets
                        if 0 <= active < len(presets):
                            ww = presets[active].window_width
                        count   = s.window_array_count
                        gap     = s.window_array_gap
                        _span_hv = (snap_ew["lo"], snap_ew["hi"]) if snap_ew else None
                        anchors = _array_anchors(anchor, count, ww, gap,
                                                 rooms[ri], wc, s.door_margin,
                                                 wall_span=_span_hv)
                        _t_hv = rooms[ri].get("t", s.wall_thickness)
                        _zones_hv = ([] if snap_ew else _wall_adj_zones(rooms[ri], wc, rooms, _t_hv))
                        blocked = any(
                            _valid_window_anchor(anch, rooms[ri], wc, ww, s.door_margin,
                                                 zones=_zones_hv, wall_span=_span_hv) is None
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
                        "Sliding window  ·  Tab/Shift+Tab=preset  |  "
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
                        # Preset owns the sill height — skip V_POSITION, done
                        _sync_to_scene(context)
                        bpy.ops.ed.undo_push(message="Place Window")
                        self._go_hover(context)
                    else:
                        # No preset — let user drag to set sill height
                        self._phase = 'V_POSITION'
                        context.area.header_text_set(
                            "Move up/down to set sill height  |  Click to confirm  |  RMB – cancel")
                    return {'RUNNING_MODAL'}

                # ── V_POSITION: second click confirms vertical placement ────
                if self._phase == 'V_POSITION':
                    # Silently remember sill height for the next freehand placement
                    if (self._active_idx is not None and
                            self._active_win_idx is not None and
                            self._active_idx < len(rooms)):
                        _wins = rooms[self._active_idx].get("windows", [])
                        if self._active_win_idx < len(_wins):
                            s.window_v_offset = _wins[self._active_win_idx].get(
                                "v_offset", s.window_v_offset)
                    _sync_to_scene(context)
                    bpy.ops.ed.undo_push(message="Place Window")
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
                        bpy.ops.ed.undo_push(message="Remove Window")
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
                        "Hold+drag=slide  ·  Tab/Shift+Tab=preset  |  "
                        "Release=confirm  ·  RMB=cancel")

                elif self._hover_snap is not None:
                    # Press on wall → place N windows as array, enter H_POSITION
                    ri, wc, raw_anchor, snap_ew = self._hover_snap
                    self._last_press_win = None
                    if 0 <= ri < len(rooms):
                        reg          = rooms[ri]
                        ww, wh, voff = self._get_preset_dims(context)
                        # Clamp sill height so window fits inside the room walls
                        voff = max(0.0, min(voff, max(0.0, s.wall_height - wh - s.wall_thickness)))
                        count        = s.window_array_count
                        gap          = s.window_array_gap
                        _span_lmb    = (snap_ew["lo"], snap_ew["hi"]) if snap_ew else None
                        count        = min(count, max(1, _max_array_count(reg, wc, ww, gap, s.door_margin,
                                                                           wall_span=_span_lmb)))
                        anchors      = _array_anchors(raw_anchor, count, ww, gap,
                                                      reg, wc, s.door_margin,
                                                      wall_span=_span_lmb)
                        _t_lmb = reg.get("t", s.wall_thickness)
                        _zones_lmb = ([] if snap_ew else _wall_adj_zones(reg, wc, rooms, _t_lmb))
                        # Capture zone-snapped anchors so stored positions never
                        # land inside the adjacent-room blocked zone.
                        clamped_lmb = [_valid_window_anchor(anch, reg, wc, ww, s.door_margin,
                                                            zones=_zones_lmb, wall_span=_span_lmb)
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
                        # Attach mesh from active window preset if one is assigned
                        _wp  = context.scene.room_window_presets
                        _wa  = context.scene.room_active_window_preset
                        _win_src = (_wp[_wa].mesh_object.name
                                    if 0 <= _wa < len(_wp) and _wp[_wa].mesh_object
                                    else None)
                        first_wi     = len(reg.setdefault("windows", []))
                        for k, anch in enumerate(anchors):
                            wd = {"wc": wc, "anchor": anch,
                                  "v_offset": voff, "w": ww, "h": wh,
                                  "array_n": count, "array_idx": k}
                            if _win_src:
                                wd["mesh_source"] = _win_src
                            if snap_ew:
                                wd["ew"] = snap_ew
                            reg["windows"].append(wd)
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
                                pd = {"wc": p_wc, "anchor": anch,
                                      "v_offset": voff, "w": ww, "h": wh,
                                      "array_n": count, "array_idx": k}
                                if _win_src:
                                    pd["mesh_source"] = _win_src
                                p_wins.append(pd)
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
                            f"|  Tab/Shift+Tab=preset  |  Click confirms H  |  RMB – cancel")
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
# Stair edit operator
# ═════════════════════════════════════════════════════════════════════════════
def _find_room_at(pt2d, z, rooms):
    """Return the index of the first room whose floor Z matches and contains pt2d, or None."""
    for i, r in enumerate(rooms):
        if abs(r.get("z", 0.0) - z) > 0.01:
            continue
        if r["x1"] <= pt2d.x <= r["x2"] and r["y1"] <= pt2d.y <= r["y2"]:
            return i
    return None


def _rect_fits_in_room(rx1, ry1, rx2, ry2, z, rooms, eps=1e-4):
    """Return the room index if the rectangle fits within the room's inner floor area
    at the given Z, else None.  x1/y1/x2/y2 are the INNER wall faces (walls extend
    outward), so the valid floor area is exactly x1..x2, y1..y2.
    eps: tolerance so a rect snapped flush to the inner wall face is accepted."""
    for i, r in enumerate(rooms):
        if abs(r.get("z", 0.0) - z) > 0.01:
            continue
        if (rx1 >= r["x1"] - eps and ry1 >= r["y1"] - eps and
                rx2 <= r["x2"] + eps and ry2 <= r["y2"] + eps):
            return i
    return None


def _rect_overlaps_stair_hole(rx1, ry1, rx2, ry2, z, rooms, eps=1e-4):
    """Return True if the rectangle overlaps any existing stair hole in rooms at
    the given Z level.  Prevents a second stair from being placed on top of an
    existing floor/ceiling opening."""
    for r in rooms:
        if abs(r.get("z", 0.0) - z) > 0.01:
            continue
        for hole in r.get("stair_holes", []):
            hx1 = hole.get("x1", 0); hy1 = hole.get("y1", 0)
            hx2 = hole.get("x2", 0); hy2 = hole.get("y2", 0)
            if (rx1 < hx2 - eps and rx2 > hx1 + eps and
                    ry1 < hy2 - eps and ry2 > hy1 + eps):
                return True
    return False


class ROOM_OT_stair_edit(bpy.types.Operator):
    bl_idname  = "room.stair_edit"
    bl_label   = "Stair Edit Mode"
    bl_description = ("Draw stairs between two floor levels.  "
                      "Step 1: LMB-drag on lower floor to draw the stair footprint.  "
                      "Step 2: slide mouse along the travel axis to offset upper opening "
                      "(green = valid), LMB to confirm.  RMB/Esc = cancel")
    bl_options = {"REGISTER", "UNDO"}

    # ── GPU draw callback ──────────────────────────────────────────────────
    def _draw_cb(self, context):
        rooms  = ROOM_OT_draw._room_list
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')

        COL_GREEN_FILL = (0.05, 0.85, 0.2,  0.15)
        COL_GREEN_LINE = (0.1,  1.0,  0.3,  1.0)
        COL_RED_FILL   = (1.0,  0.08, 0.05, 0.20)
        COL_RED_LINE   = (1.0,  0.12, 0.05, 1.0)
        COL_OK_DONE    = (1.0,  0.55, 0.0,  1.0)   # orange = lower rect locked in

        def _rect_fill(x0, y0, x1, y1, z, col_fill, col_line):
            mn_x, mx_x = min(x0, x1), max(x0, x1)
            mn_y, mx_y = min(y0, y1), max(y0, y1)
            tris = [(mn_x, mn_y, z), (mx_x, mn_y, z), (mx_x, mx_y, z),
                    (mn_x, mn_y, z), (mx_x, mx_y, z), (mn_x, mx_y, z)]
            bf = batch_for_shader(shader, 'TRIS', {"pos": tris})
            shader.uniform_float("color", col_fill)
            bf.draw(shader)
            corners = [(mn_x,mn_y,z),(mx_x,mn_y,z),(mx_x,mx_y,z),(mn_x,mx_y,z)]
            lines   = [corners[i] for pair in ((0,1),(1,2),(2,3),(3,0)) for i in pair]
            bl = batch_for_shader(shader, 'LINES', {"pos": lines})
            shader.uniform_float("color", col_line)
            gpu.state.line_width_set(2.5)
            bl.draw(shader)

        # ── Phase LOWER_DRAG: live rectangle on lower floor ────────────────
        if self._phase == 'LOWER_DRAG' and self._lower_c1 is not None:
            cx, cy = self._cursor.x, self._cursor.y
            _rx1 = min(self._lower_c1.x, cx); _ry1 = min(self._lower_c1.y, cy)
            _rx2 = max(self._lower_c1.x, cx); _ry2 = max(self._lower_c1.y, cy)
            fits = (_rect_fits_in_room(_rx1, _ry1, _rx2, _ry2,
                                       self._z_lower, rooms) is not None and
                    not _rect_overlaps_stair_hole(_rx1, _ry1, _rx2, _ry2,
                                                  self._z_lower, rooms))
            cf = COL_GREEN_FILL if fits else COL_RED_FILL
            cl = COL_GREEN_LINE if fits else COL_RED_LINE
            _rect_fill(self._lower_c1.x, self._lower_c1.y, cx, cy,
                       self._z_lower, cf, cl)

        # ── Once lower rect is locked: keep it shown in orange ─────────────
        if self._lower_c2 is not None:
            lx1 = min(self._lower_c1.x, self._lower_c2.x)
            ly1 = min(self._lower_c1.y, self._lower_c2.y)
            lx2 = max(self._lower_c1.x, self._lower_c2.x)
            ly2 = max(self._lower_c1.y, self._lower_c2.y)
            _rect_fill(lx1, ly1, lx2, ly2,
                       self._z_lower, (1.0, 0.55, 0.0, 0.12), COL_OK_DONE)

            # ── Phase UPPER_SLIDE: upper rect slides along travel axis ───────
            if self._phase == 'UPPER_SLIDE' and self._upper_rect is not None:
                ux1, uy1, ux2, uy2 = self._upper_rect
                fits = (_rect_fits_in_room(ux1, uy1, ux2, uy2,
                                           self._z_upper, rooms) is not None and
                        not _rect_overlaps_stair_hole(ux1, uy1, ux2, uy2,
                                                      self._z_upper, rooms))
                cf = COL_GREEN_FILL if fits else COL_RED_FILL
                cl = COL_GREEN_LINE if fits else COL_RED_LINE
                _rect_fill(ux1, uy1, ux2, uy2, self._z_upper, cf, cl)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')
        _draw_hint_bar(context, getattr(self, '_hint_text', ''))

    def _add_draw_handle(self, context):
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_VIEW')
        _DRAW_HANDLES.add(self._handle)

    def _remove_draw_handle(self):
        if self._handle:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            except Exception:
                pass
            _DRAW_HANDLES.discard(self._handle)
            self._handle = None

    # ── Finalise ───────────────────────────────────────────────────────────
    def _finalise(self, context):
        s     = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list

        # Lower rect (drawn by user)
        lx1 = min(self._lower_c1.x, self._lower_c2.x)
        ly1 = min(self._lower_c1.y, self._lower_c2.y)
        lx2 = max(self._lower_c1.x, self._lower_c2.x)
        ly2 = max(self._lower_c1.y, self._lower_c2.y)

        if (lx2 - lx1) < 0.05 or (ly2 - ly1) < 0.05:
            self.report({'WARNING'}, "Rectangle too small — draw a larger footprint")
            return False

        if (self._z_upper - self._z_lower) < 0.05:
            self.report({'WARNING'}, "Floor height difference too small")
            return False

        # Validate: lower rect must be fully inside a lower-floor room
        li = _rect_fits_in_room(lx1, ly1, lx2, ly2, self._z_lower, rooms)
        if li is None:
            self.report({'WARNING'}, "Lower footprint is outside room bounds")
            return False

        # Upper rect from the slide position
        if self._upper_rect is None:
            ux1, uy1, ux2, uy2 = lx1, ly1, lx2, ly2   # no offset (fallback)
        else:
            ux1, uy1, ux2, uy2 = self._upper_rect

        ui = _rect_fits_in_room(ux1, uy1, ux2, uy2, self._z_upper, rooms)
        if ui is None:
            self.report({'WARNING'}, "Upper footprint is outside room bounds — "
                                     "slide to align with the upper room (green = valid)")
            return False
        if _rect_overlaps_stair_hole(ux1, uy1, ux2, uy2, self._z_upper, rooms):
            self.report({'WARNING'}, "Upper footprint overlaps an existing stair opening")
            return False

        # Determine travel axis from actual offset direction between rects
        rw = lx2 - lx1; rl = ly2 - ly1
        x_off = abs((ux1 + ux2) / 2 - (lx1 + lx2) / 2)
        y_off = abs((uy1 + uy2) / 2 - (ly1 + ly2) / 2)
        x_travel = (x_off >= y_off) if (x_off > 1e-6 or y_off > 1e-6) else (rw >= rl)
        # Travel = distance between nearest facing edges (same logic as _build_stair_mesh)
        if x_travel:
            if (ux1 + ux2) / 2 >= (lx1 + lx2) / 2:
                total_travel = abs(ux1 - lx2)
            else:
                total_travel = abs(lx1 - ux2)
        else:
            if (uy1 + uy2) / 2 >= (ly1 + ly2) / 2:
                total_travel = abs(uy1 - ly2)
            else:
                total_travel = abs(ly1 - uy2)
        if total_travel < 0.05:
            self.report({'WARNING'}, "Total stair travel too short — "
                                     "slide the upper opening further away")
            return False

        sd = {
            "lx1": lx1, "ly1": ly1, "lx2": lx2, "ly2": ly2,
            "ux1": ux1, "uy1": uy1, "ux2": ux2, "uy2": uy2,
            "z_bot":           self._z_lower,
            "z_top":           self._z_upper,
            "step_rise":       s.stair_rise,
            "x_travel":        x_travel,
            "open_under":      s.stair_open_under,
            "slab_thick":      s.stair_slab_thick,
        }

        # ── Create stair mesh ───────────────────────────────────────────
        col = _get_or_create_stair_col(context)
        num = len(ROOM_OT_draw._stair_list) + 1
        obj_name = f"Stair.{num:03}"
        while obj_name in bpy.data.objects:
            num += 1
            obj_name = f"Stair.{num:03}"
        sd["obj_name"] = obj_name
        stair_obj = _make_stair_obj(obj_name, sd, s, collection=col)
        if stair_obj is None:
            self.report({'WARNING'}, "Could not build stair geometry — check floor heights")
            return False

        ROOM_OT_draw._stair_list.append(sd)

        # ── Cut holes: ceiling of lower room + floor of upper room ─────────
        _stair_cut_holes(sd, li, ui, rooms, s)

        _sync_to_scene(context)
        bpy.ops.ed.undo_push(message="Place Stair")
        self.report({'INFO'}, f"Created {obj_name}")
        return True

    # ── Modal ──────────────────────────────────────────────────────────────
    def invoke(self, context, event):
        # Resync runtime state from scene (handles post-undo stale data)
        _sync_from_scene(context)
        _resync_stair_holes(context)

        s      = context.scene.room_settings
        floors = context.scene.room_floors

        # Build sorted list of all z levels that have rooms
        self._all_zs = sorted({round(r["z"], 4) for r in ROOM_OT_draw._room_list})
        if len(self._all_zs) < 2:
            # Fall back to floor definitions
            zs_fl = sorted({f.z_offset for f in floors})
            if len(zs_fl) >= 2:
                self._all_zs = zs_fl
            else:
                self.report({'ERROR'}, "Need rooms on at least 2 different floors")
                return {'CANCELLED'}
        # Default lower/upper — overridden automatically when user clicks
        self._z_lower = self._all_zs[0]
        self._z_upper = self._all_zs[1]

        # Phases: LOWER_FIRST → LOWER_DRAG → UPPER_SLIDE
        self._phase           = 'LOWER_FIRST'
        self._lower_c1        = None
        self._lower_c2        = None
        self._upper_rect      = None   # (ux1, uy1, ux2, uy2) updated on mousemove
        self._cursor          = Vector((0, 0, 0))
        self._handle          = None
        self._last_lmb_time   = 0.0    # for double-click-to-exit detection
        self._hint_text       = "LMB – set lower rect corner  ·  RMB/Esc – cancel"

        context.scene.room_stair_edit_active = True

        # Auto X-ray: stairs span two floors so X-ray makes placement easier.
        # Save previous state and enable; restored on finish/cancel.
        spc = context.space_data
        self._xray_was_on = getattr(getattr(spc, 'shading', None), 'show_xray', False)
        if hasattr(spc, 'shading'):
            spc.shading.show_xray = True

        self.report({'INFO'}, "X-Ray enabled for stair placement — disabled on confirm/cancel")
        self._add_draw_handle(context)
        context.window_manager.modal_handler_add(self)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        context.area.tag_redraw()

        # Let viewport navigation (pan/orbit/zoom/numpad) pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3',
                'NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7',
                'NUMPAD_8','NUMPAD_9','NUMPAD_DECIMAL','NUMPAD_PERIOD',
                'F','TILDE'}:
            return {'PASS_THROUGH'}

        if event.type in ('ESC', 'RIGHTMOUSE') and event.value == 'PRESS':
            self._remove_draw_handle()
            context.scene.room_stair_edit_active = False
            self._restore_xray(context)
            return {'CANCELLED'}

        # Enter / Numpad-Enter: clean exit from stair mode (when not mid-draw)
        if event.type in ('RET', 'NUMPAD_ENTER') and event.value == 'PRESS':
            if self._phase == 'LOWER_FIRST':
                self._remove_draw_handle()
                context.scene.room_stair_edit_active = False
                self._restore_xray(context)
                return {'FINISHED'}
            # Mid-draw: cancel current placement, return to idle
            self._lower_c1   = None
            self._lower_c2   = None
            self._upper_rect = None
            self._phase      = 'LOWER_FIRST'
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        s_draw = context.scene.room_settings
        _stair_grid = s_draw.grid_snap_size if s_draw.snap_to_grid else 0.1

        z_plane = self._z_upper if self._phase == 'UPPER_SLIDE' else self._z_lower
        if event.type == 'MOUSEMOVE':
            pt = _ray_to_z(context, event, z_plane)
            if pt is not None:
                # Grid snap (user-configured) then wall snap during lower rect phases
                if self._phase in ('LOWER_FIRST', 'LOWER_DRAG'):
                    pt = _apply_grid_snap(pt, context, s_draw)
                if self._phase == 'LOWER_DRAG':
                    ri = _rect_fits_in_room(pt.x, pt.y, pt.x, pt.y,
                                            self._z_lower, ROOM_OT_draw._room_list)
                    if ri is not None:
                        pt.x, pt.y = _snap_stair_pt(
                            pt.x, pt.y, ROOM_OT_draw._room_list[ri],
                            grid=_stair_grid)
                self._cursor = pt
                # During UPPER_SLIDE: dominant-axis detection —
                # whichever axis the cursor has moved further from the lower
                # rect centre, that axis slides; the other stays locked.
                if self._phase == 'UPPER_SLIDE' and self._lower_c2 is not None:
                    lx1 = min(self._lower_c1.x, self._lower_c2.x)
                    ly1 = min(self._lower_c1.y, self._lower_c2.y)
                    lx2 = max(self._lower_c1.x, self._lower_c2.x)
                    ly2 = max(self._lower_c1.y, self._lower_c2.y)
                    dx = pt.x - (lx1 + lx2) * 0.5
                    dy = pt.y - (ly1 + ly2) * 0.5
                    rooms = ROOM_OT_draw._room_list
                    if abs(dx) >= abs(dy):   # X dominant → slide X, lock Y
                        ux1t = lx1 + dx;  ux2t = lx2 + dx
                        ri = _rect_fits_in_room(ux1t, ly1, ux2t, ly2,
                                                self._z_upper, rooms)
                        if ri is not None:
                            r = rooms[ri]; tw = r.get("t", 0.125)
                            wx1 = r["x1"] + tw;  wx2 = r["x2"] - tw
                            for edge, wall in ((ux1t, wx1), (ux2t, wx2)):
                                if abs(edge - wall) < 0.25:
                                    ux1t += wall - edge
                                    ux2t += wall - edge
                                    break
                        self._upper_rect = (ux1t, ly1, ux2t, ly2)
                    else:                    # Y dominant → slide Y, lock X
                        uy1t = ly1 + dy;  uy2t = ly2 + dy
                        ri = _rect_fits_in_room(lx1, uy1t, lx2, uy2t,
                                                self._z_upper, rooms)
                        if ri is not None:
                            r = rooms[ri]; tw = r.get("t", 0.125)
                            wy1 = r["y1"] + tw;  wy2 = r["y2"] - tw
                            for edge, wall in ((uy1t, wy1), (uy2t, wy2)):
                                if abs(edge - wall) < 0.25:
                                    uy1t += wall - edge
                                    uy2t += wall - edge
                                    break
                        self._upper_rect = (lx1, uy1t, lx2, uy2t)

        if event.type == 'LEFTMOUSE':
            pt = _ray_to_z(context, event, z_plane)
            if pt is None:
                return {'RUNNING_MODAL'}

            if event.value == 'PRESS':
                if self._phase == 'LOWER_FIRST':
                    import time as _time
                    _now = _time.time()
                    # Double-click in idle phase → exit stair mode cleanly
                    if _now - self._last_lmb_time < 0.3:
                        self._remove_draw_handle()
                        context.scene.room_stair_edit_active = False
                        self._restore_xray(context)
                        return {'FINISHED'}
                    self._last_lmb_time = _now
                    # Auto-detect which floor the user clicked on via ray cast
                    # against actual room mesh objects (reliable for any view angle)
                    room_names = {r.get("obj_name", "") for r in ROOM_OT_draw._room_list}
                    from bpy_extras import view3d_utils as v3u
                    co = (event.mouse_region_x, event.mouse_region_y)
                    rv3d = context.space_data.region_3d
                    ray_org = v3u.region_2d_to_origin_3d(context.region, rv3d, co)
                    ray_dir = v3u.region_2d_to_vector_3d(context.region, rv3d, co)
                    ray_dir.normalize()
                    hit, _loc, _nor, _idx, hit_obj, _mx = context.scene.ray_cast(
                        context.evaluated_depsgraph_get(), ray_org, ray_dir)
                    if hit and hit_obj and hit_obj.name in room_names:
                        for rd in ROOM_OT_draw._room_list:
                            if rd.get("obj_name") == hit_obj.name:
                                rz = round(rd["z"], 4)
                                abv = [z for z in self._all_zs if z > rz + 0.01]
                                if abv:
                                    self._z_lower, self._z_upper = rz, abv[0]
                                    npt = _ray_to_z(context, event, rz)
                                    if npt:
                                        pt = npt
                                break
                    pt = _apply_grid_snap(pt, context, s_draw)
                    ri = _rect_fits_in_room(pt.x, pt.y, pt.x, pt.y,
                                            self._z_lower, ROOM_OT_draw._room_list)
                    if ri is not None:
                        pt.x, pt.y = _snap_stair_pt(
                            pt.x, pt.y, ROOM_OT_draw._room_list[ri],
                            grid=_stair_grid)
                    self._lower_c1  = pt.copy()
                    self._phase     = 'LOWER_DRAG'
                    self._hint_text = "LMB release – set opposite corner of lower footprint  ·  RMB – cancel"

            elif event.value == 'RELEASE':
                if self._phase == 'LOWER_DRAG':
                    # Apply snap to the release point before size check
                    pt = _apply_grid_snap(pt, context, s_draw)
                    ri = _rect_fits_in_room(pt.x, pt.y, pt.x, pt.y,
                                            self._z_lower, ROOM_OT_draw._room_list)
                    if ri is not None:
                        pt.x, pt.y = _snap_stair_pt(
                            pt.x, pt.y, ROOM_OT_draw._room_list[ri],
                            grid=_stair_grid)
                    if abs(pt.x - self._lower_c1.x) < 0.05 and abs(pt.y - self._lower_c1.y) < 0.05:
                        self.report({'WARNING'}, "Too small — drag further to define footprint")
                        self._phase = 'LOWER_FIRST'
                        return {'RUNNING_MODAL'}
                    rx1 = min(self._lower_c1.x, pt.x); rx2 = max(self._lower_c1.x, pt.x)
                    ry1 = min(self._lower_c1.y, pt.y); ry2 = max(self._lower_c1.y, pt.y)
                    if _rect_fits_in_room(rx1, ry1, rx2, ry2,
                                          self._z_lower, ROOM_OT_draw._room_list) is None:
                        self.report({'WARNING'}, "Footprint must be fully inside a lower-floor room")
                        self._phase = 'LOWER_FIRST'
                        return {'RUNNING_MODAL'}
                    if _rect_overlaps_stair_hole(rx1, ry1, rx2, ry2,
                                                  self._z_lower, ROOM_OT_draw._room_list):
                        self.report({'WARNING'}, "Footprint overlaps an existing stair opening")
                        self._phase = 'LOWER_FIRST'
                        return {'RUNNING_MODAL'}
                    self._lower_c2 = pt.copy()
                    # Initialise upper rect at same XY as lower (zero offset)
                    lx1 = rx1; ly1 = ry1; lx2 = rx2; ly2 = ry2
                    self._upper_rect = (lx1, ly1, lx2, ly2)
                    self._phase     = 'UPPER_SLIDE'
                    self._hint_text = "Move to position upper rect  ·  LMB – confirm  ·  RMB – cancel"
                    self.report({'INFO'},
                                "Footprint set — slide along travel axis to position upper "
                                "opening (green = valid, red = outside room), LMB to confirm")

                elif self._phase == 'UPPER_SLIDE':
                    # Re-compute upper_rect from click with wall snap on leading edge
                    lx1 = min(self._lower_c1.x, self._lower_c2.x)
                    ly1 = min(self._lower_c1.y, self._lower_c2.y)
                    lx2 = max(self._lower_c1.x, self._lower_c2.x)
                    ly2 = max(self._lower_c1.y, self._lower_c2.y)
                    dx = pt.x - (lx1 + lx2) * 0.5
                    dy = pt.y - (ly1 + ly2) * 0.5
                    rooms = ROOM_OT_draw._room_list
                    if abs(dx) >= abs(dy):  # X dominant → slide X, lock Y
                        ux1t = lx1 + dx;  ux2t = lx2 + dx
                        ri = _rect_fits_in_room(ux1t, ly1, ux2t, ly2, self._z_upper, rooms)
                        if ri is not None:
                            r = rooms[ri]; tw = r.get("t", 0.125)
                            wx1 = r["x1"] + tw;  wx2 = r["x2"] - tw
                            for edge, wall in ((ux1t, wx1), (ux2t, wx2)):
                                if abs(edge - wall) < 0.25:
                                    ux1t += wall - edge;  ux2t += wall - edge;  break
                        self._upper_rect = (ux1t, ly1, ux2t, ly2)
                    else:                   # Y dominant → slide Y, lock X
                        uy1t = ly1 + dy;  uy2t = ly2 + dy
                        ri = _rect_fits_in_room(lx1, uy1t, lx2, uy2t, self._z_upper, rooms)
                        if ri is not None:
                            r = rooms[ri]; tw = r.get("t", 0.125)
                            wy1 = r["y1"] + tw;  wy2 = r["y2"] - tw
                            for edge, wall in ((uy1t, wy1), (uy2t, wy2)):
                                if abs(edge - wall) < 0.25:
                                    uy1t += wall - edge;  uy2t += wall - edge;  break
                        self._upper_rect = (lx1, uy1t, lx2, uy2t)
                    if self._finalise(context):
                        # Stay in mode — reset to idle so user can place another stair.
                        # Exit with Enter or double-click.
                        self._lower_c1      = None
                        self._lower_c2      = None
                        self._upper_rect    = None
                        self._last_lmb_time = 0.0
                        self._phase         = 'LOWER_FIRST'
                        context.area.tag_redraw()
                        return {'RUNNING_MODAL'}
                    # Invalid — keep running so user can adjust

        return {'RUNNING_MODAL'}

    def _restore_xray(self, context):
        spc = context.space_data
        if hasattr(spc, 'shading'):
            spc.shading.show_xray = getattr(self, '_xray_was_on', False)

    def cancel(self, context):
        self._remove_draw_handle()
        context.scene.room_stair_edit_active = False
        self._restore_xray(context)


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


# ═════════════════════════════════════════════════════════════════════════════
# Architrave-preset operators
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_save_arch_preset(bpy.types.Operator):
    bl_idname      = "room.save_arch_preset"
    bl_label       = "Save Architrave Preset"
    bl_description = "Save current architrave width and depth as a named preset"

    def execute(self, context):
        s       = context.scene.room_settings
        presets = context.scene.room_arch_presets
        n       = len(presets) + 1
        ap       = presets.add()
        ap.name  = f"Arch.{n:02}"
        ap.width = s.architrave_width
        ap.depth = s.architrave_depth
        context.scene.room_active_arch_preset = len(presets) - 1
        return {"FINISHED"}


class ROOM_OT_select_arch_preset(bpy.types.Operator):
    bl_idname      = "room.select_arch_preset"
    bl_label       = "Select Architrave Preset"
    bl_description = "Apply this preset's width and depth to the settings"

    preset_index: bpy.props.IntProperty()

    def execute(self, context):
        presets = context.scene.room_arch_presets
        idx     = self.preset_index
        if 0 <= idx < len(presets):
            if context.scene.room_active_arch_preset == idx:
                context.scene.room_active_arch_preset = -1
            else:
                context.scene.room_active_arch_preset = idx
                ap = presets[idx]
                context.scene.room_settings.architrave_preset = 'custom'
                context.scene.room_settings.architrave_width  = ap.width
                context.scene.room_settings.architrave_depth  = ap.depth
        return {"FINISHED"}


class ROOM_OT_remove_arch_preset(bpy.types.Operator):
    bl_idname      = "room.remove_arch_preset"
    bl_label       = "Remove Architrave Preset"
    bl_description = "Remove this architrave preset"

    preset_index: bpy.props.IntProperty()

    def execute(self, context):
        presets = context.scene.room_arch_presets
        idx     = self.preset_index
        if 0 <= idx < len(presets):
            presets.remove(idx)
            active = context.scene.room_active_arch_preset
            context.scene.room_active_arch_preset = max(-1, min(active, len(presets) - 1))
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
    sampled_obj_name: bpy.props.StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context):
        L = self.layout
        L.prop(self, "apply_as", expand=True)
        col = L.column(align=True)
        col.label(text=f"Width:  {self.sampled_width:.3f} m")
        col.label(text=f"Height: {self.sampled_height:.3f} m")
        if self.apply_as == 'WINDOW':
            s = context.scene.room_settings
            col.label(text=f"Sill:   {s.window_v_offset:.3f} m  (keep current)")

    def execute(self, context):
        s   = context.scene.room_settings
        src = bpy.data.objects.get(self.sampled_obj_name)
        if self.apply_as == 'DOOR':
            s.door_width  = self.sampled_width
            s.door_height = self.sampled_height
            # Always create a new preset — never overwrite an existing one
            presets  = context.scene.room_door_presets
            base     = src.name if src is not None else "Door"
            # Generate a unique name
            existing = {p.name for p in presets}
            name     = base
            counter  = 1
            while name in existing:
                name = f"{base}.{counter:03d}"
                counter += 1
            dp             = presets.add()
            dp.name        = name
            dp.door_width  = self.sampled_width
            dp.door_height = self.sampled_height
            if src is not None:
                dp.mesh_object = src
            context.scene.room_active_door_preset = len(presets) - 1
        else:
            s.window_width    = self.sampled_width
            s.window_height   = self.sampled_height
            # Keep window_v_offset at its current value — sill height is set
            # manually and should not be overridden by where the mesh happens
            # to be placed in the scene (which could be anywhere).
            # Always create a new preset — never overwrite an existing one
            presets  = context.scene.room_window_presets
            base     = src.name if src is not None else "Window"
            existing = {p.name for p in presets}
            name     = base
            counter  = 1
            while name in existing:
                name = f"{base}.{counter:03d}"
                counter += 1
            wp             = presets.add()
            wp.name        = name
            wp.window_width  = self.sampled_width
            wp.window_height = self.sampled_height
            wp.v_offset      = s.window_v_offset   # inherit current sill height
            if src is not None:
                wp.mesh_object = src
            context.scene.room_active_window_preset = len(presets) - 1
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
                sampled_z_bottom=round(z_bot, 4),
                sampled_obj_name=obj.name)
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
        reg[key] = not reg.get(key, False)
        _rebuild_room_mesh(reg, s)
        _sync_to_scene(context)
        return {'FINISHED'}


class ROOM_OT_toggle_mesh_lock(bpy.types.Operator):
    """Lock or unlock the room mesh.

    Locked — geometry is preserved even when doors/windows/stairs are added or
    global settings change.  Opening mesh instances are still repositioned.
    Unlock & Rebuild — clears the lock and regenerates from stored parameters
    (manual edits will be lost)."""
    bl_idname  = "room.toggle_mesh_lock"
    bl_label   = "Toggle Mesh Lock"
    bl_options = {"UNDO"}  # No REGISTER — prevents redo-panel re-execution toggling the flag back

    # Explicit target state instead of a toggle so Blender redo/re-execution is idempotent.
    lock:    bpy.props.BoolProperty(default=True)
    rebuild: bpy.props.BoolProperty(default=False)

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
        if self.rebuild:
            reg['mesh_locked'] = False
            _rebuild_room_mesh(reg, s)
        else:
            reg['mesh_locked'] = self.lock   # explicit set, not a toggle
        _sync_to_scene(context)
        return {'FINISHED'}


def _recalculate_plinth_for_obj(reg, obj, s):
    """Rebuild plinth geometry in-place on *obj* using current mesh wall-base edges.

    Works on locked rooms without touching the rest of the mesh.
    Returns the number of unique base edges processed (0 if no edges found).
    """
    pb = reg.get('plinth_bottom_enabled', False)
    pt = reg.get('plinth_top_enabled',    False)
    if not pb and not pt:
        return 0

    ph_b   = getattr(s, 'plinth_bottom_height',    0.10)
    pt_b   = getattr(s, 'plinth_bottom_thickness', 0.02)
    ph_t   = getattr(s, 'plinth_top_height',       0.10)
    pt_t   = getattr(s, 'plinth_top_thickness',    0.02)
    wall_h = s.wall_height
    z_floor = reg.get("z", 0.0)

    me     = obj.data
    bm     = bmesh.new()
    bm.from_mesh(me)
    mw     = obj.matrix_world
    mw_inv = mw.inverted()

    # ── Remove existing plinth faces ─────────────────────────────────────────
    for grp_name in ('plinth_bottom', 'plinth_top'):
        vg = obj.vertex_groups.get(grp_name)
        if vg is None:
            continue
        idx = vg.index
        bm.verts.ensure_lookup_table()
        plinth_verts = set()
        deform = bm.verts.layers.deform.active
        for v in bm.verts:
            try:
                if deform and idx in v[deform]:
                    plinth_verts.add(v)
            except Exception:
                pass
        if plinth_verts:
            bmesh.ops.delete(bm, geom=list(plinth_verts), context='VERTS')

    # ── Find wall-base edges (floor level) and wall-top edges (ceiling level) ──
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    _TOL = 0.05
    z_ceil = z_floor + wall_h
    plinth_edges     = []   # floor-level → bottom plinth source
    plinth_edges_top = []   # ceiling-level → top plinth source (spans door area)
    for face in bm.faces:
        fn = face.normal.copy()
        if abs(fn.z) > 0.3:
            continue
        for edge in face.edges:
            va_z = (mw @ edge.verts[0].co).z
            vb_z = (mw @ edge.verts[1].co).z
            if abs(va_z - z_floor) < _TOL and abs(vb_z - z_floor) < _TOL:
                va_w = mw @ edge.verts[0].co
                vb_w = mw @ edge.verts[1].co
                plinth_edges.append((va_w.copy(), vb_w.copy(), fn.copy()))
            if abs(va_z - z_ceil) < _TOL and abs(vb_z - z_ceil) < _TOL:
                va_w = mw @ edge.verts[0].co
                vb_w = mw @ edge.verts[1].co
                plinth_edges_top.append((va_w.copy(), vb_w.copy(), fn.copy()))

    if not plinth_edges:
        bm.to_mesh(me)
        bm.free()
        return 0

    seen = set()
    unique_edges = []
    for va, vb, fn in plinth_edges:
        key = (round(min(va.x, vb.x), 3), round(min(va.y, vb.y), 3),
               round(max(va.x, vb.x), 3), round(max(va.y, vb.y), 3))
        if key not in seen:
            seen.add(key)
            unique_edges.append((va, vb, fn))

    # Top plinth uses ceiling-level edges — these span the full wall width
    # including above doors, so the top strip is continuous for standard doors.
    seen_top = set()
    unique_edges_top = []
    for va, vb, fn in plinth_edges_top:
        key = (round(min(va.x, vb.x), 3), round(min(va.y, vb.y), 3),
               round(max(va.x, vb.x), 3), round(max(va.y, vb.y), 3))
        if key not in seen_top:
            seen_top.add(key)
            unique_edges_top.append((va, vb, fn))
    # Fall back to floor-level if no ceiling edges found (older mesh format)
    if not unique_edges_top:
        unique_edges_top = unique_edges

    # ── Door-gap helpers ──────────────────────────────────────────────────────
    x1r, y1r, x2r, y2r = reg["x1"], reg["y1"], reg["x2"], reg["y2"]
    t_r = reg.get("t", 0.125)
    _pos_tol = t_r * 2.5 + 0.05   # tolerance for matching edge to wall plane

    def _edge_wc_and_axis(va_e, vb_e, fn_e):
        """Wall character and along-axis inferred from edge position + face normal.

        Room meshes have inward-facing normals, so we use position relative to
        the room mid-point (not the normal direction) to decide N/S/E/W.
        The normal is only used to determine whether this is a Y-normal (N/S)
        or X-normal (E/W) wall.
        """
        if abs(fn_e.y) >= abs(fn_e.x):
            # Y-normal face: N or S determined by position vs room mid-Y
            edge_y = (va_e.y + vb_e.y) * 0.5
            return ('N' if edge_y > (y1r + y2r) * 0.5 else 'S'), 'x'
        else:
            # X-normal face: E or W determined by position vs room mid-X
            edge_x = (va_e.x + vb_e.x) * 0.5
            return ('E' if edge_x > (x1r + x2r) * 0.5 else 'W'), 'y'

    def _wall_ref_coord(wc_e):
        """Original wall-plane coordinate for a given wall character."""
        return {  'N': y2r, 'S': y1r, 'E': x2r, 'W': x1r }[wc_e]

    def _door_gaps_for_edge(va_e, vb_e, fn_e, z_plinth_lo=None):
        """Return list of (gap_lo, gap_hi) in the edge's along-coordinate space.

        z_plinth_lo: if given, doors whose tops do not reach this Z level are
        excluded — used to suppress gaps in the top plinth strip for doors that
        are shorter than the wall.
        """
        wc_e, ax = _edge_wc_and_axis(va_e, vb_e, fn_e)
        if ax == 'x':
            edge_pos = (va_e.y + vb_e.y) * 0.5    # wall-normal coordinate
            e_lo = min(va_e.x, vb_e.x)
            e_hi = max(va_e.x, vb_e.x)
        else:
            edge_pos = (va_e.x + vb_e.x) * 0.5
            e_lo = min(va_e.y, vb_e.y)
            e_hi = max(va_e.y, vb_e.y)

        # Determine what wall position this edge represents:
        # Could be an original wall plane or an extra-wall plane.
        ref_pos = _wall_ref_coord(wc_e)

        # Half wall-thickness tolerance — allows the gap to be cut even when
        # the door span only barely overlaps the plinth edge (floating-point
        # boundary case: door placed at the very end of a shortened wall).
        _bt = t_r * 0.5

        gaps = []
        for door in reg.get("doors", []):
            if door.get("wc") != wc_e:
                continue
            dw = door.get("w", s.door_width)
            dh = door.get("h", s.door_height)
            anc = door.get("anchor", 0.0)
            # Find which wall plane this door belongs to
            ew = door.get("ew")
            if ew:
                door_wall_pos = ew.get("wall_coord", ref_pos)
            else:
                door_wall_pos = ref_pos
            # Only process doors whose wall position matches this edge
            if abs(edge_pos - door_wall_pos) > _pos_tol:
                continue
            # Skip doors that don't reach the plinth strip (top plinth only)
            if z_plinth_lo is not None and z_floor + dh <= z_plinth_lo + 0.05:
                continue
            # Extend door span by half wall-thickness so doors at the very
            # edge of a plinth segment still produce a valid gap.
            dl = max(e_lo, anc - dw * 0.5 - _bt)
            dr = min(e_hi, anc + dw * 0.5 + _bt)
            if dr > dl + 1e-4:
                gaps.append((dl, dr))

        # Merge overlapping gaps
        if not gaps:
            return []
        gaps.sort()
        merged = [gaps[0]]
        for lo, hi in gaps[1:]:
            if lo <= merged[-1][1] + 1e-4:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def _edge_segments(va_e, vb_e, fn_e, z_plinth_lo=None):
        """Split edge into sub-segments after removing door gaps.
        Returns list of (va_seg, vb_seg) world-Vector pairs."""
        _wc_e, ax = _edge_wc_and_axis(va_e, vb_e, fn_e)
        gaps = _door_gaps_for_edge(va_e, vb_e, fn_e, z_plinth_lo=z_plinth_lo)
        if not gaps:
            return [(va_e, vb_e)]
        # Project along-axis onto a 1-D parametric line
        if ax == 'x':
            lo_full = min(va_e.x, vb_e.x)
            hi_full = max(va_e.x, vb_e.x)
        else:
            lo_full = min(va_e.y, vb_e.y)
            hi_full = max(va_e.y, vb_e.y)
        if hi_full - lo_full < 1e-4:
            return []
        # Direction vector along the edge
        edge_dir = (vb_e - va_e).normalized() if (vb_e - va_e).length > 1e-6 else Vector((1, 0, 0))

        def _interp(t_val):
            """World-space point at parametric position t_val (0=va, 1=vb)."""
            return va_e + edge_dir * ((vb_e - va_e).length * t_val)

        def _coord_to_t(coord):
            total = hi_full - lo_full
            return (coord - lo_full) / total if total > 1e-6 else 0.0

        segments = []
        cursor = lo_full
        for gl, gr in gaps:
            gl = max(gl, lo_full)
            gr = min(gr, hi_full)
            if gl > cursor + 1e-4:
                segments.append((_interp(_coord_to_t(cursor)),
                                  _interp(_coord_to_t(gl))))
            cursor = gr
        if cursor < hi_full - 1e-4:
            segments.append((_interp(_coord_to_t(cursor)),
                              _interp(_coord_to_t(hi_full))))
        return segments

    # ── Emit plinth quads ─────────────────────────────────────────────────────
    def add_plinth_quad(va, vb, fn, height, thickness, z_base, top=False):
        inward = -fn.copy(); inward.z = 0; inward.normalize()
        z0_w = z_base + (wall_h - height if top else 0.0)
        z1_w = z0_w + height
        p0_w = va.copy(); p0_w.z = z0_w
        p1_w = vb.copy(); p1_w.z = z0_w
        p2_w = p1_w + inward * thickness; p2_w.z = z0_w
        p3_w = p0_w + inward * thickness; p3_w.z = z0_w
        p4_w = va.copy(); p4_w.z = z1_w
        p5_w = vb.copy(); p5_w.z = z1_w
        p6_w = p5_w + inward * thickness; p6_w.z = z1_w
        p7_w = p4_w + inward * thickness; p7_w.z = z1_w
        pts_l = [mw_inv @ p for p in [p0_w, p1_w, p2_w, p3_w, p4_w, p5_w, p6_w, p7_w]]
        v = [bm.verts.new(co) for co in pts_l]
        bm.faces.new([v[0], v[1], v[5], v[4]])
        bm.faces.new([v[4], v[5], v[6], v[7]])
        bm.faces.new([v[0], v[3], v[7], v[4]])
        bm.faces.new([v[1], v[5], v[6], v[2]])
        bm.faces.new([v[2], v[6], v[7], v[3]])
        return v

    # Z level where the top plinth strip begins — doors shorter than this don't
    # need a gap cut in the top plinth.
    z_top_plinth_lo = z_floor + wall_h - ph_t

    new_verts_bottom = []
    new_verts_top    = []
    if pb:
        for va, vb, fn in unique_edges:
            for seg_va, seg_vb in _edge_segments(va, vb, fn):
                if (seg_vb - seg_va).length < 1e-4:
                    continue
                new_verts_bottom.extend(
                    add_plinth_quad(seg_va, seg_vb, fn, ph_b, pt_b, z_floor))
    if pt:
        # Top plinth uses ceiling-level source edges so it spans the door area;
        # _edge_segments with z_plinth_lo still cuts gaps for tall doors.
        for va, vb, fn in unique_edges_top:
            for seg_va, seg_vb in _edge_segments(va, vb, fn, z_plinth_lo=z_top_plinth_lo):
                if (seg_vb - seg_va).length < 1e-4:
                    continue
                new_verts_top.extend(
                    add_plinth_quad(seg_va, seg_vb, fn, ph_t, pt_t, z_floor, top=True))

    # Collect indices before freeing bmesh
    bm.verts.index_update()
    bottom_indices = [v.index for v in new_verts_bottom]
    top_indices    = [v.index for v in new_verts_top]

    bm.to_mesh(me)
    bm.free()
    me.update()

    # ── Vertex groups ─────────────────────────────────────────────────────────
    def assign_vg(name, indices):
        vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        vg.add(indices, 1.0, 'REPLACE')

    _setup_room_vertex_groups(obj, {})
    if bottom_indices:
        assign_vg('plinth_bottom', bottom_indices)
    if top_indices:
        assign_vg('plinth_top', top_indices)

    _apply_cube_uv_to_mesh(me, s, obj=obj)
    return len(unique_edges)


class ROOM_OT_recalculate_plinth(bpy.types.Operator):
    """Regenerate plinth geometry from the current (possibly edited) mesh shape.
    Works on locked rooms — scans wall base edges and rebuilds the plinth strips
    in-place without touching the rest of the mesh."""
    bl_idname  = "room.recalculate_plinth"
    bl_label   = "Recalculate Plinth"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None:
            return False
        return any(r.get('obj_name') == obj.name for r in ROOM_OT_draw._room_list)

    def execute(self, context):
        obj = context.active_object
        s   = context.scene.room_settings
        reg = next((r for r in ROOM_OT_draw._room_list
                    if r.get('obj_name') == obj.name), None)
        if reg is None:
            return {'CANCELLED'}
        if not reg.get('plinth_bottom_enabled', False) and not reg.get('plinth_top_enabled', False):
            self.report({'INFO'}, "No plinth enabled on this room")
            return {'CANCELLED'}
        n = _recalculate_plinth_for_obj(reg, obj, s)
        _sync_to_scene(context)
        msg = f"Plinth recalculated along {n} base edges" if n else "No wall-base edges found at floor level"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class ROOM_OT_recalculate_uv(bpy.types.Operator):
    """Re-project the UV map for the selected room from its current geometry.

    Use this after manually extruding or reshaping a locked room in Edit Mode
    to fix stretched or incorrectly-tiled UVs on the new faces.

    The projection is axis-aligned cube mapping (same as the automatic UV
    applied when a room is first created), with one extra step: extruded faces
    that have no room_cat data (Blender assigns 0 by default) are classified
    as floor, ceiling, or wall automatically from their world-space normal,
    so they pick up the correct tiling value instead of always getting wall
    tiling."""
    bl_idname  = "room.recalculate_uv"
    bl_label   = "Recalculate UV"
    bl_options = {"UNDO"}

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
        _apply_uvs_one_room(reg, s)
        self.report({'INFO'}, f"UV recalculated for {obj.name}")
        return {'FINISHED'}


class ROOM_OT_sync_thresholds(bpy.types.Operator):
    """Recreate threshold strips on all doors using the current settings.
    Run this after changing threshold settings or loading a file."""
    bl_idname  = "room.sync_thresholds"
    bl_label   = "Apply Thresholds"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.room_settings
        _sync_all_thresholds(s)
        total = sum(len(r.get("doors", [])) for r in ROOM_OT_draw._room_list)
        self.report({'INFO'}, f"Thresholds synced across {total} door(s)")
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

        # ── Quick Start hint (shown only when no rooms exist) ─────────────────
        if not ROOM_OT_draw._room_list:
            box = L.box()
            col = box.column(align=True)
            col.label(text="Getting started:", icon="INFO")
            col.label(text="Press Shift+R or click above")
            col.label(text="then click + drag to draw a room.")

        # ── Selected Room ─────────────────────────────────────────────────────
        _sel_obj = context.active_object
        if _sel_obj:
            _sel_reg = next((r for r in ROOM_OT_draw._room_list
                             if r.get('obj_name') == _sel_obj.name), None)
            if _sel_reg is not None:
                box = L.box()
                col = box.column(align=True)
                col.label(text=f"Selected: {_sel_obj.name}", icon="MESH_CUBE")
                # Room dimensions
                _rx1 = _sel_reg.get('x1', 0.0); _rx2 = _sel_reg.get('x2', 0.0)
                _ry1 = _sel_reg.get('y1', 0.0); _ry2 = _sel_reg.get('y2', 0.0)
                _rw  = abs(_rx2 - _rx1);        _rd  = abs(_ry2 - _ry1)
                _rh  = _sel_reg.get('h', s.wall_height)
                col.label(text=f"{_rw*100:.0f} cm \u00d7 {_rd*100:.0f} cm \u00d7 {_rh*100:.0f} cm  (W\u00d7D\u00d7H)",
                          icon="NONE")
                col.separator()
                _locked = _sel_reg.get('mesh_locked', False)
                _save_row = col.row(align=True)
                _save_row.enabled = not _locked
                op = _save_row.operator("room.toggle_mesh_lock",
                                        text="Save Edits" if not _locked else "Edits Saved",
                                        icon="CHECKMARK" if _locked else "FILE_TICK",
                                        depress=_locked)
                op.lock    = True
                op.rebuild = False
                op = col.operator("room.toggle_mesh_lock",
                                  text="Restore Original Room",
                                  icon="LOOP_BACK")
                op.lock    = False
                op.rebuild = True
                col.separator()
                col.operator("room.recalculate_uv",    text="Recalculate UV",    icon="UV")
                col.operator("room.recalculate_plinth", text="Recalculate Plinth", icon="MOD_SOLIDIFY")

        # ── Pivot + Organisation ──────────────────────────────────────────────
        L.prop(s, "pivot_mode", text="Pivot")
        row = L.row(align=True)
        row.prop(s, "use_hierarchy", toggle=True)
        if s.use_hierarchy:
            L.row().prop(s, "hierarchy_mode", expand=True)

        # ── Floors ────────────────────────────────────────────────────────────
        box = L.box()
        row = box.row()
        row.prop(s, "show_floors",
                 icon='TRIA_DOWN' if s.show_floors else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Floors", icon="RENDERLAYERS")
        if s.show_floors:
            col = box.column(align=True)
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

        # ── Wall Dimensions ───────────────────────────────────────────────────
        box = L.box()
        row = box.row()
        row.prop(s, "show_wall_dims",
                 icon='TRIA_DOWN' if s.show_wall_dims else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Wall Dimensions", icon="MOD_BUILD")
        if s.show_wall_dims:
            col = box.column(align=True)
            col.prop(s, "wall_height")
            col.prop(s, "wall_thickness")
            col.separator()
            row = col.row(align=True)
            row.prop(s, "snap_to_grid", toggle=True, icon="SNAP_GRID")
            if s.snap_to_grid:
                row.prop(s, "grid_snap_size", text="")

        # ── Geometry ──────────────────────────────────────────────────────────
        box = L.box()
        row = box.row()
        row.prop(s, "show_geometry",
                 icon='TRIA_DOWN' if s.show_geometry else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Geometry", icon="MESH_GRID")
        if s.show_geometry:
            col = box.column(align=True)
            col.prop(s, "add_ceiling")
            col.prop(s, "add_floor")
            col.prop(s, "add_door")
            col.prop(s, "add_architrave")
            col.prop(s, "add_threshold")

        # ── Trims ─────────────────────────────────────────────────────────────
        box = L.box()
        row = box.row()
        row.prop(s, "show_trims",
                 icon='TRIA_DOWN' if s.show_trims else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Trims", icon="MOD_SOLIDIFY")
        if s.show_trims:
            col = box.column(align=True)
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
            col.label(text="Architrave:")
            row = col.row(align=True)
            row.prop(s, "add_architrave", toggle=True, icon="MOD_SOLIDIFY")
            if s.add_architrave:
                sub = col.column(align=True)
                sub.prop(s, "architrave_preset")
                sub = col.column(align=True)
                sub.enabled = (s.architrave_preset == 'custom')
                sub.prop(s, "architrave_width")
                sub.prop(s, "architrave_depth")
                col.operator("room.save_arch_preset", icon="ADD", text="Save as Preset")
                arch_presets = context.scene.room_arch_presets
                arch_active  = context.scene.room_active_arch_preset
                if arch_presets:
                    col.separator()
                    for i, ap in enumerate(arch_presets):
                        is_active = (i == arch_active)
                        row = col.row(align=True)
                        op = row.operator(
                            "room.select_arch_preset",
                            text=f"{ap.name}  {ap.width*100:.0f}mm \u00d7 {ap.depth*10:.1f}mm",
                            icon="LAYER_ACTIVE" if is_active else "LAYER_USED",
                            depress=is_active,
                        )
                        op.preset_index = i
                        rem = row.operator("room.remove_arch_preset", text="", icon="X")
                        rem.preset_index = i
                        if is_active:
                            col.prop(ap, "name", text="Rename")

            col.separator()
            col.label(text="Threshold:")
            row = col.row(align=True)
            row.prop(s, "add_threshold", toggle=True, icon="MESH_PLANE")
            if s.add_threshold:
                sub = col.column(align=True)
                sub.prop(s, "threshold_height")
                sub.prop(s, "threshold_depth")
                sub.prop(s, "threshold_material", text="Material", icon="MATERIAL")
                sub.operator("room.sync_thresholds", icon="FILE_REFRESH",
                             text="Apply to All Doors")

        # ── Materials / Tiling ────────────────────────────────────────────────
        box = L.box()
        row = box.row()
        row.prop(s, "show_materials",
                 icon='TRIA_DOWN' if s.show_materials else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Materials / Tiling", icon="MATERIAL")
        if s.show_materials:
            col = box.column(align=True)
            row = col.row(align=True)
            row.prop(s, "mat_apply_mode", expand=True)
            col.separator()

            col.label(text="Surfaces:")
            for mat_attr, tile_attr, label in (
                ("mat_walls",   "mat_walls_tiling",   "Walls"),
                ("mat_floor",   "mat_floor_tiling",   "Floor"),
                ("mat_ceiling", "mat_ceiling_tiling", "Ceiling"),
            ):
                row = col.row(align=True)
                split = row.split(factor=0.65, align=True)
                split.prop(s, mat_attr, text=label)
                split.prop(s, tile_attr, text="")

            col.separator()
            col.label(text="Openings:")
            for mat_attr, tile_attr, label in (
                ("mat_door_frame",   "mat_door_frame_tiling",   "Door (∅=Walls)"),
                ("mat_window_frame", "mat_window_frame_tiling", "Win (∅=Walls)"),
            ):
                row = col.row(align=True)
                split = row.split(factor=0.65, align=True)
                split.prop(s, mat_attr, text=label)
                split.prop(s, tile_attr, text="")

            col.separator()
            col.label(text="Trims:")
            for mat_attr, tile_attr, label in (
                ("mat_plinth_bottom", "mat_plinth_bottom_tiling", "Bot Plinth (∅=Walls)"),
                ("mat_plinth_top",    "mat_plinth_top_tiling",    "Top Plinth (∅=Walls)"),
                ("mat_architrave",    "mat_architrave_tiling",    "Architrave (∅=Walls)"),
            ):
                row = col.row(align=True)
                split = row.split(factor=0.65, align=True)
                split.prop(s, mat_attr, text=label)
                split.prop(s, tile_attr, text="")

            col.separator()
            col.label(text="Stairs:")
            for mat_attr, tile_attr, label in (
                ("mat_stair",      "mat_stair_tiling",      "Sides"),
                ("mat_stair_step", "mat_stair_step_tiling", "Treads"),
            ):
                row = col.row(align=True)
                split = row.split(factor=0.65, align=True)
                split.prop(s, mat_attr, text=label)
                split.prop(s, tile_attr, text="")

        # ── Utilities ─────────────────────────────────────────────────────────
        box = L.box()
        row = box.row()
        row.prop(s, "show_utilities",
                 icon='TRIA_DOWN' if s.show_utilities else 'TRIA_RIGHT',
                 icon_only=True, emboss=False)
        row.label(text="Utilities", icon="TOOL_SETTINGS")
        if s.show_utilities:
            col = box.column(align=True)
            col.operator("room.clear_registry",  icon="X",              text="Reset Snap Registry")
            col.operator("room.clear_overlays",  icon="GHOST_DISABLED", text="Clear Phantom Overlays")


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
        col.operator("room.sample_dims", icon="EYEDROPPER", text="Sample from Object  (Ctrl+Shift+E)")
        col.operator("room.save_door_preset", icon="ADD", text="Save Door Preset")
        col.separator()
        col.label(text="Batch Flip All Doors:")
        row = col.row(align=True)
        row.operator("room.door_flip_all", text="Flip L/R",  icon="ARROW_LEFTRIGHT").flip_axis = 'LR'
        row.operator("room.door_flip_all", text="Flip I/O",  icon="UV_SYNC_SELECT").flip_axis  = 'IO'

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
                    col.prop(dp, "door_width")
                    col.prop(dp, "door_height")
                    col.prop(dp, "mesh_object", text="Mesh", icon="MESH_DATA")


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
        col.operator("room.sample_dims", icon="EYEDROPPER", text="Sample from Object  (Ctrl+Shift+E)")
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
                    col.prop(wp, "mesh_object", text="Mesh", icon="MESH_DATA")


class ROOM_PT_stair_panel(bpy.types.Panel):
    bl_label       = "Stairs"
    bl_idname      = "ROOM_PT_stair_panel"
    bl_parent_id   = "ROOM_PT_panel"
    bl_space_type  = "VIEW_3D"
    bl_region_type = "UI"
    bl_category    = "Room Tool"
    bl_options     = {"DEFAULT_CLOSED"}

    def draw(self, context):
        L = self.layout
        s = context.scene.room_settings
        floors = context.scene.room_floors

        col = L.column(align=True)
        col.scale_y = 1.3
        is_editing = getattr(context.scene, "room_stair_edit_active", False)
        col.operator("room.stair_edit", icon="OUTLINER_OB_EMPTY",
                     text="Stair Edit Mode  (Ctrl+Shift+S)", depress=is_editing)
        col.separator()

        col = L.column(align=True)

        col.separator()
        col.prop(s, "stair_rise")
        col.prop(s, "stair_depth")
        col.separator()
        col.prop(s, "stair_open_under")
        if s.stair_open_under:
            col.prop(s, "stair_slab_thick")
        col.prop(s, "stair_hole_ratio", slider=True)
        col.separator()
        col.label(text="Apply to Other Floors:", icon="DUPLICATE")
        row = col.row(align=True)
        row.prop(s, "stair_apply_dir",    text="Offset Side", expand=True)
        col.prop(s, "stair_apply_offset", text="Offset (0=auto)")
        col.operator("room.stair_apply_floors",
                     text="Apply to Other Floors", icon="DUPLICATE")
        col.label(text="Active=stair, also select target rooms", icon="INFO")

        # ── Per-stair step editing ─────────────────────────────────────────
        obj = context.active_object
        if obj:
            sd = next((d for d in ROOM_OT_draw._stair_list
                       if d.get("obj_name") == obj.name), None)
            if sd is not None:
                box = L.box()
                col2 = box.column(align=True)
                col2.label(text=f"Edit: {obj.name}", icon="OUTLINER_OB_MESH")
                # Compute current step stats for display
                dz    = sd.get("z_top", 0) - sd.get("z_bot", 0)
                x_t   = sd.get("x_travel", True)
                lx1_  = min(sd.get("lx1",0), sd.get("lx2",0))
                lx2_  = max(sd.get("lx1",0), sd.get("lx2",0))
                ly1_  = min(sd.get("ly1",0), sd.get("ly2",0))
                ly2_  = max(sd.get("ly1",0), sd.get("ly2",0))
                ux1_  = min(sd.get("ux1",lx1_), sd.get("ux2",lx2_))
                ux2_  = max(sd.get("ux1",lx1_), sd.get("ux2",lx2_))
                uy1_  = min(sd.get("uy1",ly1_), sd.get("uy2",ly2_))
                uy2_  = max(sd.get("uy1",ly1_), sd.get("uy2",ly2_))
                if x_t:
                    if (ux1_+ux2_)/2 >= (lx1_+lx2_)/2:
                        tt = abs(ux2_ - lx2_)
                    else:
                        tt = abs(lx1_ - ux1_)
                else:
                    if (uy1_+uy2_)/2 >= (ly1_+ly2_)/2:
                        tt = abs(uy2_ - ly2_)
                    else:
                        tt = abs(ly1_ - uy1_)
                cur_rise = sd.get("step_rise", s.stair_rise)
                n_cur    = max(2, round(dz / cur_rise))
                col2.label(text=f"Steps: {n_cur}  |  Depth: {tt/n_cur:.3f} m",
                           icon="INFO")
                col2.separator()
                col2.prop(s, "stair_rise",  text="Rise")
                col2.prop(s, "stair_depth", text="Depth")
                col2.separator()
                col2.operator("room.stair_move", text="Move / Rotate",
                              icon="OBJECT_ORIGIN")
                col2.operator("room.stair_delete", text="Delete Stair",
                              icon="TRASH")


class ROOM_OT_stair_rebuild(bpy.types.Operator):
    """Rebuild the selected stair with a new step rise or depth,
    keeping the total run and height unchanged."""
    bl_idname = "room.stair_rebuild"
    bl_label  = "Rebuild Stair Steps"
    bl_options = {"REGISTER", "UNDO"}

    use_depth: bpy.props.BoolProperty(
        name="Use Depth as Driver",
        default=False,
        description="Derive step count from depth instead of rise",
    )

    def execute(self, context):
        obj = context.active_object
        if obj is None or "room_stair" not in obj:
            self.report({"WARNING"}, "No stair object selected")
            return {"CANCELLED"}

        sd = next((d for d in ROOM_OT_draw._stair_list
                   if d.get("obj_name") == obj.name), None)
        if sd is None:
            self.report({"WARNING"}, f"Stair data not found for {obj.name}")
            return {"CANCELLED"}

        s    = context.scene.room_settings
        dz   = sd.get("z_top", 0) - sd.get("z_bot", 0)
        x_t  = sd.get("x_travel", True)
        lx1_ = min(sd.get("lx1",0), sd.get("lx2",0))
        lx2_ = max(sd.get("lx1",0), sd.get("lx2",0))
        ly1_ = min(sd.get("ly1",0), sd.get("ly2",0))
        ly2_ = max(sd.get("ly1",0), sd.get("ly2",0))
        ux1_ = min(sd.get("ux1",lx1_), sd.get("ux2",lx2_))
        ux2_ = max(sd.get("ux1",lx1_), sd.get("ux2",lx2_))
        uy1_ = min(sd.get("uy1",ly1_), sd.get("uy2",ly2_))
        uy2_ = max(sd.get("uy1",ly1_), sd.get("uy2",ly2_))
        if x_t:
            if (ux1_+ux2_)/2 >= (lx1_+lx2_)/2:
                total_travel = abs(ux2_ - lx2_)
            else:
                total_travel = abs(lx1_ - ux1_)
        else:
            if (uy1_+uy2_)/2 >= (ly1_+ly2_)/2:
                total_travel = abs(uy2_ - ly2_)
            else:
                total_travel = abs(ly1_ - uy1_)

        if self.use_depth:
            new_depth = max(0.05, s.stair_depth)
            n = max(2, round(total_travel / new_depth))
        else:
            new_rise = max(0.05, s.stair_rise)
            n = max(2, round(dz / new_rise))

        # Store the effective rise (height / n) so the mesh is exact
        sd["step_rise"] = dz / n

        # Rebuild mesh in-place
        old_me = obj.data
        verts, faces, cats = _build_stair_mesh(sd, s)
        if not verts:
            self.report({"WARNING"}, "Rebuild produced no geometry")
            return {"CANCELLED"}

        # _build_stair_mesh returns world-space coords; convert to local space
        loc = obj.location
        verts = [(v[0] - loc.x, v[1] - loc.y, v[2] - loc.z) for v in verts]

        me = bpy.data.meshes.new(obj.name + "_rebuilt")
        bm = bmesh.new()
        cat_layer = bm.faces.layers.int.new("stair_cat")
        for v in verts:
            bm.verts.new(v)
        bm.verts.ensure_lookup_table()
        for fi, fidx in enumerate(faces):
            try:
                face = bm.faces.new([bm.verts[i] for i in fidx])
                face[cat_layer] = cats[fi]
            except Exception:
                pass
        bm.to_mesh(me)
        bm.free()
        me.update()
        obj.data = me
        bpy.data.meshes.remove(old_me)
        _setup_stair_materials(me, s)
        _apply_stair_uv(me, s)
        # Update the stored JSON marker
        obj["room_stair"] = json.dumps({k: v for k, v in sd.items() if k != "obj_name"})

        step_d = total_travel / n
        self.report({"INFO"},
                    f"{obj.name}: {n} steps, rise={sd['step_rise']:.3f} m, depth={step_d:.3f} m")
        return {"FINISHED"}


# ═════════════════════════════════════════════════════════════════════════════
# Stair delete operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_stair_delete(bpy.types.Operator):
    """Delete the selected stair and remove its floor/ceiling holes from all rooms."""
    bl_idname  = "room.stair_delete"
    bl_label   = "Delete Stair"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and "room_stair" in obj

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = context.active_object
        if obj is None or "room_stair" not in obj:
            self.report({"WARNING"}, "No stair object selected")
            return {"CANCELLED"}

        s    = context.scene.room_settings
        name = obj.name

        # Remove from stair list
        ROOM_OT_draw._stair_list[:] = [
            d for d in ROOM_OT_draw._stair_list
            if d.get("obj_name") != name
        ]

        # Remove holes referencing this stair from every room, rebuild affected ones
        for reg in ROOM_OT_draw._room_list:
            old_holes = reg.get("stair_holes", [])
            new_holes = [h for h in old_holes if h.get("stair_obj") != name]
            if len(new_holes) != len(old_holes):
                reg["stair_holes"] = new_holes
                _rebuild_room_mesh(reg, s)

        # Delete the Blender object
        bpy.data.objects.remove(obj)

        _sync_to_scene(context)
        return {"FINISHED"}


# ═════════════════════════════════════════════════════════════════════════════
# Stair move / rotate operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_stair_move(bpy.types.Operator):
    """Move or rotate the selected stair together with its ceiling/floor holes.
    Mouse → translate  |  R → rotate 90°  |  LMB → confirm (green)  |  RMB/Esc → cancel"""
    bl_idname  = "room.stair_move"
    bl_label   = "Move / Rotate Stair"
    bl_options = {"REGISTER", "UNDO"}

    # ── constants ─────────────────────────────────────────────────────────────
    _COL_GREEN_FILL = (0.05, 0.85, 0.2,  0.15)
    _COL_GREEN_LINE = (0.1,  1.0,  0.3,  1.0)
    _COL_RED_FILL   = (1.0,  0.08, 0.05, 0.20)
    _COL_RED_LINE   = (1.0,  0.12, 0.05, 1.0)
    _COL_ROOM_LINE  = (0.3,  0.8,  1.0,  0.55)   # cyan room bounds hint

    # ── invoke ────────────────────────────────────────────────────────────────
    def invoke(self, context, event):
        obj = context.active_object
        if obj is None or "room_stair" not in obj:
            # Pass through so double-click doesn't swallow other interactions
            return {'PASS_THROUGH'}

        sd = next((d for d in ROOM_OT_draw._stair_list
                   if d.get("obj_name") == obj.name), None)
        if sd is None:
            self.report({'WARNING'}, "Stair data not found in runtime list")
            return {'CANCELLED'}

        s     = context.scene.room_settings
        rooms = ROOM_OT_draw._room_list

        # Find lower / upper room indices using the existing helper
        lx1 = min(sd["lx1"], sd["lx2"]); lx2 = max(sd["lx1"], sd["lx2"])
        ly1 = min(sd["ly1"], sd["ly2"]); ly2 = max(sd["ly1"], sd["ly2"])
        ux1 = min(sd.get("ux1", lx1), sd.get("ux2", lx2))
        ux2 = max(sd.get("ux1", lx1), sd.get("ux2", lx2))
        uy1 = min(sd.get("uy1", ly1), sd.get("uy2", ly2))
        uy2 = max(sd.get("uy1", ly1), sd.get("uy2", ly2))
        li = _rect_fits_in_room(lx1, ly1, lx2, ly2, sd["z_bot"], rooms)
        ui = _rect_fits_in_room(ux1, uy1, ux2, uy2, sd["z_top"], rooms)
        if li is None or ui is None:
            # Fall back to z-only match
            li = next((i for i, r in enumerate(rooms)
                       if abs(r.get("z", s.z_foundation) - sd["z_bot"]) < 0.05), None)
            ui = next((i for i, r in enumerate(rooms)
                       if abs(r.get("z", s.z_foundation) - sd["z_top"]) < 0.05), None)
        if li is None or ui is None:
            self.report({'WARNING'}, "Could not find rooms for this stair")
            return {'CANCELLED'}

        self._sd       = sd
        self._obj_name = obj.name
        self._rooms    = rooms
        self._s        = s
        self._li       = li
        self._ui       = ui
        self._handle   = None

        # Base rect positions (updated each time R is pressed)
        self._base = dict(lx1=lx1, ly1=ly1, lx2=lx2, ly2=ly2,
                          ux1=ux1, uy1=uy1, ux2=ux2, uy2=uy2,
                          x_travel=sd.get("x_travel", True))
        self._preview = dict(self._base)
        self._valid   = True   # starts valid (currently placed)

        # LMB drag-rotation state
        self._lmb_down       = False
        self._lmb_press_xy   = None   # screen (x, y) where LMB was pressed
        self._rot_base       = None   # snapshot of _preview at LMB press
        self._is_drag_rotate = False  # True once drag threshold exceeded
        self._cur_rot_n      = 0      # last snapped rotation index (0–3)

        # Axis lock: None = free XY,  'X' = X only,  'Y' = Y only
        self._axis_lock = None

        # Warp cursor to pivot point so movement starts relative to the stair
        _px, _py = self._pivot_point(self._base)
        _vp_region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
        _rv3d      = context.area.spaces.active.region_3d
        if _vp_region and _rv3d:
            from mathutils import Vector as _V
            _piv2d = view3d_utils.location_3d_to_region_2d(
                _vp_region, _rv3d, _V((_px, _py, sd["z_top"])))
            if _piv2d:
                context.window.cursor_warp(
                    int(_piv2d.x) + _vp_region.x,
                    int(_piv2d.y) + _vp_region.y)
        # Set mouse_start = pivot 3D so delta is zero on first move
        self._mouse_start = (_px, _py)
        self._base_pivot  = (_px, _py)   # pivot of the base state for grid snap

        self._add_draw_handle(context)
        context.window_manager.modal_handler_add(self)
        context.window.cursor_set('MOVE_X')
        return {'RUNNING_MODAL'}

    # ── helpers ───────────────────────────────────────────────────────────────
    def _update_preview(self, raw_dx, raw_dy):
        b         = self._base
        GRID      = 0.1    # base snap resolution in metres
        WALL_SNAP = 0.35   # pull toward a wall face when this close
        EPS       = 1e-4   # float tolerance

        # Axis lock
        if self._axis_lock == 'X':
            raw_dy = 0.0
        elif self._axis_lock == 'Y':
            raw_dx = 0.0

        # ── Gather inner bounds for both rooms ────────────────────────────────
        rooms = self._rooms
        INF = 1e9
        if 0 <= self._li < len(rooms):
            rl = rooms[self._li]; tl = rl.get("t", 0.125)
            l_x1, l_x2 = rl["x1"] + tl, rl["x2"] - tl
            l_y1, l_y2 = rl["y1"] + tl, rl["y2"] - tl
        else:
            l_x1, l_x2, l_y1, l_y2 = -INF, INF, -INF, INF

        if 0 <= self._ui < len(rooms):
            ru = rooms[self._ui]; tu = ru.get("t", 0.125)
            u_x1, u_x2 = ru["x1"] + tu, ru["x2"] - tu
            u_y1, u_y2 = ru["y1"] + tu, ru["y2"] - tu
        else:
            u_x1, u_x2, u_y1, u_y2 = -INF, INF, -INF, INF

        # Valid translation range that keeps BOTH rects inside their rooms
        dx_min = max(l_x1 - b["lx1"], u_x1 - b["ux1"]) - EPS
        dx_max = min(l_x2 - b["lx2"], u_x2 - b["ux2"]) + EPS
        dy_min = max(l_y1 - b["ly1"], u_y1 - b["uy1"]) - EPS
        dy_max = min(l_y2 - b["ly2"], u_y2 - b["uy2"]) + EPS

        # ── Step 1: pivot-based grid snap ────────────────────────────────────
        px0, py0 = self._base_pivot
        dx = round((px0 + raw_dx) / GRID) * GRID - px0
        dy = round((py0 + raw_dy) / GRID) * GRID - py0

        # ── Step 2: wall-snap override ────────────────────────────────────────
        # Check every edge of both rects against every wall of both rooms.
        # If the raw-mouse position would place an edge within WALL_SNAP of a
        # wall, override the grid snap for that axis.
        best_x = WALL_SNAP
        best_y = WALL_SNAP
        snap_dx = None
        snap_dy = None

        for edge_x, edge_y in (
            (b["lx1"], b["ly1"]), (b["lx2"], b["ly1"]),
            (b["lx1"], b["ly2"]), (b["lx2"], b["ly2"]),
            (b["ux1"], b["uy1"]), (b["ux2"], b["uy1"]),
            (b["ux1"], b["uy2"]), (b["ux2"], b["uy2"]),
        ):
            for wall_x in (l_x1, l_x2, u_x1, u_x2):
                d = abs((edge_x + raw_dx) - wall_x)
                if d < best_x:
                    best_x = d
                    snap_dx = wall_x - edge_x

            for wall_y in (l_y1, l_y2, u_y1, u_y2):
                d = abs((edge_y + raw_dy) - wall_y)
                if d < best_y:
                    best_y = d
                    snap_dy = wall_y - edge_y

        if snap_dx is not None:
            dx = snap_dx
        if snap_dy is not None:
            dy = snap_dy

        # ── Step 3: clamp to valid range so neither rect leaves its room ──────
        # This replaces the old validate-then-reject logic: instead of refusing
        # a wall snap that would push the other rect out, we clamp as far as
        # both rooms permit — so the stair always reaches the limiting wall.
        dx = max(dx_min, min(dx_max, dx))
        dy = max(dy_min, min(dy_max, dy))

        # ── Apply translation ─────────────────────────────────────────────────
        self._preview = dict(
            lx1=b["lx1"]+dx, ly1=b["ly1"]+dy,
            lx2=b["lx2"]+dx, ly2=b["ly2"]+dy,
            ux1=b["ux1"]+dx, uy1=b["uy1"]+dy,
            ux2=b["ux2"]+dx, uy2=b["uy2"]+dy,
            x_travel=b["x_travel"])
        p = self._preview
        self._valid = (
            _rect_fits_in_room(p["lx1"],p["ly1"],p["lx2"],p["ly2"],
                               self._sd["z_bot"], self._rooms) is not None and
            _rect_fits_in_room(p["ux1"],p["uy1"],p["ux2"],p["uy2"],
                               self._sd["z_top"], self._rooms) is not None)

    def _pivot_point(self, p):
        """Return (px, py) — far corner of the upper rect at the travel-axis end.
        This is the 'top corner vertex' shown at the top of the stair structure."""
        lmx = (p["lx1"] + p["lx2"]) / 2
        umx = (p["ux1"] + p["ux2"]) / 2
        lmy = (p["ly1"] + p["ly2"]) / 2
        umy = (p["uy1"] + p["uy2"]) / 2
        if p["x_travel"]:
            if umx >= lmx:   # upper to the right → far corner = (ux2, uy1)
                return p["ux2"], p["uy1"]
            else:            # upper to the left  → far corner = (ux1, uy2)
                return p["ux1"], p["uy2"]
        else:
            if umy >= lmy:   # upper above        → far corner = (ux1, uy2)
                return p["ux1"], p["uy2"]
            else:            # upper below        → far corner = (ux2, uy1)
                return p["ux2"], p["uy1"]

    def _rotate_90(self, cur_xy):
        """Rotate preview 90° CW around the pivot (far corner of upper rect)."""
        p  = self._preview
        cx, cy = self._pivot_point(p)

        def rot(x, y):           # 90° CW
            return cx + (y - cy), cy - (x - cx)

        ll1 = rot(p["lx1"], p["ly1"]); ll2 = rot(p["lx2"], p["ly2"])
        ul1 = rot(p["ux1"], p["uy1"]); ul2 = rot(p["ux2"], p["uy2"])

        nb = dict(
            lx1=min(ll1[0],ll2[0]), ly1=min(ll1[1],ll2[1]),
            lx2=max(ll1[0],ll2[0]), ly2=max(ll1[1],ll2[1]),
            ux1=min(ul1[0],ul2[0]), uy1=min(ul1[1],ul2[1]),
            ux2=max(ul1[0],ul2[0]), uy2=max(ul1[1],ul2[1]),
            x_travel=not p["x_travel"])

        self._base        = nb
        self._preview     = dict(nb)
        self._base_pivot  = self._pivot_point(nb)  # keep snap anchor on new pivot
        self._mouse_start = cur_xy   # reset delta to zero
        self._valid = (
            _rect_fits_in_room(nb["lx1"],nb["ly1"],nb["lx2"],nb["ly2"],
                               self._sd["z_bot"], self._rooms) is not None and
            _rect_fits_in_room(nb["ux1"],nb["uy1"],nb["ux2"],nb["uy2"],
                               self._sd["z_top"], self._rooms) is not None)

    def _apply(self):
        sd   = self._sd
        p    = self._preview
        name = self._obj_name
        rooms = self._rooms
        s     = self._s
        li, ui = self._li, self._ui

        # Update sd in-place
        sd.update(lx1=p["lx1"], ly1=p["ly1"], lx2=p["lx2"], ly2=p["ly2"],
                  ux1=p["ux1"], uy1=p["uy1"], ux2=p["ux2"], uy2=p["uy2"],
                  x_travel=p["x_travel"])

        # Remove old holes for this stair from every room
        for r in rooms:
            r["stair_holes"] = [h for h in r.get("stair_holes", [])
                                if h.get("stair_obj") != name]

        # Recompute hole bounds (same logic as _finalise)
        lx1 = min(sd["lx1"],sd["lx2"]); lx2 = max(sd["lx1"],sd["lx2"])
        ly1 = min(sd["ly1"],sd["ly2"]); ly2 = max(sd["ly1"],sd["ly2"])
        ux1 = min(sd["ux1"],sd["ux2"]); ux2 = max(sd["ux1"],sd["ux2"])
        uy1 = min(sd["uy1"],sd["uy2"]); uy2 = max(sd["uy1"],sd["uy2"])
        x_t = sd["x_travel"]
        if x_t:
            if (ux1+ux2)/2 >= (lx1+lx2)/2:
                hx1, hx2 = lx2, ux2
            else:
                hx1, hx2 = ux1, lx1
            hy1, hy2 = min(ly1,uy1), max(ly2,uy2)
        else:
            if (uy1+uy2)/2 >= (ly1+ly2)/2:
                hy1, hy2 = ly2, uy2
            else:
                hy1, hy2 = uy1, ly1
            hx1, hx2 = min(lx1,ux1), max(lx2,ux2)

        z_upper  = rooms[ui].get("z", s.z_foundation)
        zt_lower = rooms[li].get("z", s.z_foundation) + s.wall_height
        lower_hole = {"x1":hx1,"y1":hy1,"x2":hx2,"y2":hy2,
                      "cut":"ceiling","stair_obj":name,"slab_z":z_upper}
        upper_hole = {"x1":hx1,"y1":hy1,"x2":hx2,"y2":hy2,
                      "cut":"floor","stair_obj":name,"slab_z":zt_lower}

        rooms[li].setdefault("stair_holes", []).append(lower_hole)
        _rebuild_room_mesh(rooms[li], s)
        if ui != li:
            rooms[ui].setdefault("stair_holes", []).append(upper_hole)
            _rebuild_room_mesh(rooms[ui], s)

        # Rebuild stair mesh in-place
        obj = bpy.data.objects.get(name)
        if obj:
            old_me = obj.data
            verts, faces, cats = _build_stair_mesh(sd, s)
            if verts:
                me  = bpy.data.meshes.new(name)
                bm2 = bmesh.new()
                cl  = bm2.faces.layers.int.new("stair_cat")
                for v in verts: bm2.verts.new(v)
                bm2.verts.ensure_lookup_table()
                for fi, fidx in enumerate(faces):
                    try:
                        face = bm2.faces.new([bm2.verts[i] for i in fidx])
                        face[cl] = cats[fi]
                    except Exception: pass
                bm2.to_mesh(me); bm2.free(); me.update()
                obj.data = me
                bpy.data.meshes.remove(old_me)
                _setup_stair_materials(me, s)
                _apply_stair_uv(me, s)   # UV while verts still in world space
                # Re-apply pivot origin
                px, py = _stair_pivot_xy(sd)
                pz = sd.get("z_top", 0.0)
                pivot = Vector((px, py, pz))
                for v in me.vertices:
                    v.co -= pivot
                obj.location = pivot
                me.update()
                obj["room_stair"] = json.dumps(
                    {k: v for k, v in sd.items() if k != "obj_name"})

    # ── GPU draw callback ─────────────────────────────────────────────────────
    def _draw_cb(self, context):
        p = getattr(self, "_preview", None)
        if p is None:
            return

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('ALWAYS')
        shader.bind()

        valid = getattr(self, "_valid", False)
        cf = self._COL_GREEN_FILL if valid else self._COL_RED_FILL
        cl = self._COL_GREEN_LINE if valid else self._COL_RED_LINE

        def draw_rect(x1, y1, x2, y2, z, fc, lc):
            tris = [(x1,y1,z),(x2,y1,z),(x2,y2,z),
                    (x1,y1,z),(x2,y2,z),(x1,y2,z)]
            bf = batch_for_shader(shader, 'TRIS', {"pos": tris})
            shader.uniform_float("color", fc)
            bf.draw(shader)
            corners = [(x1,y1,z),(x2,y1,z),(x2,y2,z),(x1,y2,z)]
            lines   = [corners[i] for pair in ((0,1),(1,2),(2,3),(3,0)) for i in pair]
            bl = batch_for_shader(shader, 'LINES', {"pos": lines})
            shader.uniform_float("color", lc)
            gpu.state.line_width_set(2.5)
            bl.draw(shader)

        z_lo = self._sd["z_bot"]
        z_hi = self._sd["z_top"]

        # Lower footprint rect (at lower floor level)
        draw_rect(p["lx1"],p["ly1"],p["lx2"],p["ly2"], z_lo, cf, cl)
        # Upper footprint rect (at upper floor level)
        draw_rect(p["ux1"],p["uy1"],p["ux2"],p["uy2"], z_hi, cf, cl)

        # Room interior bounds — cyan outline to show placement constraints
        rooms = self._rooms
        s = context.scene.room_settings
        for ri, z in ((self._li, z_lo), (self._ui, z_hi)):
            if 0 <= ri < len(rooms):
                r  = rooms[ri]
                t_ = r.get("t", s.wall_thickness)
                draw_rect(r["x1"]+t_, r["y1"]+t_,
                          r["x2"]-t_, r["y2"]-t_,
                          z, (0,0,0,0), self._COL_ROOM_LINE)

        # ── Gizmo at pivot point ──────────────────────────────────────────
        px, py = self._pivot_point(p)
        gz  = self._sd["z_top"]   # draw gizmo at upper floor level
        AL  = 0.40                # arrow shaft length
        AH  = 0.07                # arrowhead size
        CR  = 0.18                # rotation arc radius
        N   = 20                  # arc segments

        # White cross — exact pivot marker
        gpu.state.line_width_set(2.0)
        cs = 0.05
        cross_pts = [(px-cs,py,gz),(px+cs,py,gz),
                     (px,py-cs,gz),(px,py+cs,gz)]
        bc = batch_for_shader(shader,'LINES',{"pos":cross_pts})
        shader.uniform_float("color",(1,1,1,1)); bc.draw(shader)

        # Red +X arrow
        gpu.state.line_width_set(3.0)
        x_ln = [(px,py,gz),(px+AL,py,gz)]
        bx = batch_for_shader(shader,'LINES',{"pos":x_ln})
        shader.uniform_float("color",(1,0.15,0.15,1)); bx.draw(shader)
        x_hd = [(px+AL,py,gz),(px+AL-AH,py+AH*0.5,gz),
                (px+AL,py,gz),(px+AL-AH,py-AH*0.5,gz)]
        bxh = batch_for_shader(shader,'LINES',{"pos":x_hd})
        shader.uniform_float("color",(1,0.15,0.15,1)); bxh.draw(shader)

        # Green +Y arrow
        y_ln = [(px,py,gz),(px,py+AL,gz)]
        by_ = batch_for_shader(shader,'LINES',{"pos":y_ln})
        shader.uniform_float("color",(0.15,1,0.15,1)); by_.draw(shader)
        y_hd = [(px,py+AL,gz),(px+AH*0.5,py+AL-AH,gz),
                (px,py+AL,gz),(px-AH*0.5,py+AL-AH,gz)]
        byh = batch_for_shader(shader,'LINES',{"pos":y_hd})
        shader.uniform_float("color",(0.15,1,0.15,1)); byh.draw(shader)

        # Blue rotation arc (270° = 3/4 circle, suggests R key rotates)
        arc_pts = [
            (px + CR*math.cos(math.pi*0.5 + math.pi*2*(i/N)*0.75),
             py + CR*math.sin(math.pi*0.5 + math.pi*2*(i/N)*0.75), gz)
            for i in range(N+1)]
        arc_segs = [pt for i in range(len(arc_pts)-1)
                    for pt in (arc_pts[i], arc_pts[i+1])]
        barc = batch_for_shader(shader,'LINES',{"pos":arc_segs})
        shader.uniform_float("color",(0.3,0.6,1.0,1)); barc.draw(shader)

        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def _add_draw_handle(self, context):
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            self._draw_cb, (context,), 'WINDOW', 'POST_VIEW')
        _DRAW_HANDLES.add(self._handle)

    def _remove_draw_handle(self):
        if self._handle:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
            except Exception: pass
            _DRAW_HANDLES.discard(self._handle)
            self._handle = None

    # ── modal ─────────────────────────────────────────────────────────────────
    def modal(self, context, event):
        if context.area:
            context.area.tag_redraw()

        # Let viewport navigation (pan/orbit/zoom/numpad) pass through
        if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE',
                'NUMPAD_0','NUMPAD_1','NUMPAD_2','NUMPAD_3',
                'NUMPAD_4','NUMPAD_5','NUMPAD_6','NUMPAD_7',
                'NUMPAD_8','NUMPAD_9','NUMPAD_DECIMAL','NUMPAD_PERIOD',
                'F','TILDE'}:
            return {'PASS_THROUGH'}

        if event.type == 'MOUSEMOVE':
            if self._lmb_down:
                # ── LMB held: drag-to-rotate ──────────────────────────────
                dxs  = event.mouse_x - self._lmb_press_xy[0]
                dys  = event.mouse_y - self._lmb_press_xy[1]
                dist = math.sqrt(dxs * dxs + dys * dys)
                if dist > 15:            # threshold to enter rotation mode
                    self._is_drag_rotate = True
                if self._is_drag_rotate:
                    # Map drag angle → nearest 90° increment (0/1/2/3)
                    angle = math.atan2(dys, dxs)
                    deg   = math.degrees(angle) % 360
                    n     = int((deg + 45) / 90) % 4
                    if n != self._cur_rot_n:
                        # Restore pre-drag snapshot then apply n × 90° CW
                        self._base    = dict(self._rot_base)
                        self._preview = dict(self._rot_base)
                        for _ in range(n):
                            self._rotate_90(self._mouse_start)
                        self._cur_rot_n = n
            else:
                # ── Normal translate mode ─────────────────────────────────
                pt = _ray_to_z(context, event, self._sd["z_bot"])
                if pt and self._mouse_start:
                    self._update_preview(pt.x - self._mouse_start[0],
                                         pt.y - self._mouse_start[1])

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            # Start drag-rotation tracking (confirm happens on RELEASE)
            self._lmb_down       = True
            self._lmb_press_xy   = (event.mouse_x, event.mouse_y)
            self._rot_base       = dict(self._preview)   # snapshot
            self._is_drag_rotate = False
            self._cur_rot_n      = 0
            context.window.cursor_set('SCROLL_XY')

        elif event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self._is_drag_rotate:
                # Rotation drag ended — lock rotated state, stay in move mode
                self._base       = dict(self._preview)
                self._base_pivot = self._pivot_point(self._preview)  # new pivot
                pt = _ray_to_z(context, event, self._sd["z_bot"])
                if pt:
                    self._mouse_start = (pt.x, pt.y)   # reset delta to zero
                self._is_drag_rotate = False
                self._lmb_down = False
                self._axis_lock = None
                context.window.cursor_set('MOVE_X')
            elif self._lmb_down:
                # Short click (no drag) → confirm placement if valid
                self._lmb_down = False
                if self._valid:
                    self._apply()
                    self._remove_draw_handle()
                    context.window.cursor_set('DEFAULT')
                    return {'FINISHED'}
                # Invalid position — stay running, show red feedback
                context.window.cursor_set('MOVE_X')

        elif event.type == 'R' and event.value == 'PRESS':
            # R key: single-step 90° rotation
            pt = _ray_to_z(context, event, self._sd["z_bot"])
            cur_xy = (pt.x, pt.y) if pt else self._mouse_start
            self._rotate_90(cur_xy)

        elif event.type == 'X' and event.value == 'PRESS':
            # Toggle X-axis lock (press again to free)
            self._axis_lock = None if self._axis_lock == 'X' else 'X'
            context.window.cursor_set('MOVE_X' if self._axis_lock == 'X' else 'MOVE_X')

        elif event.type == 'Y' and event.value == 'PRESS':
            # Toggle Y-axis lock (press again to free)
            self._axis_lock = None if self._axis_lock == 'Y' else 'Y'
            context.window.cursor_set('MOVE_Y' if self._axis_lock == 'Y' else 'MOVE_X')

        elif event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self._remove_draw_handle()
            context.window.cursor_set('DEFAULT')
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


# ═════════════════════════════════════════════════════════════════════════════
# Apply stair to other floors operator
# ═════════════════════════════════════════════════════════════════════════════
class ROOM_OT_stair_apply_floors(bpy.types.Operator):
    """Duplicate the active stair flight to selected rooms on other floors as a switchback staircase"""
    bl_idname = "room.stair_apply_floors"
    bl_label  = "Apply to Other Floors"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.room_settings

        # 1. Identify active stair
        stair_obj = context.active_object
        if stair_obj is None or "room_stair" not in stair_obj:
            self.report({'ERROR'}, "Active object must be a stair")
            return {'CANCELLED'}

        sd = next((d for d in ROOM_OT_draw._stair_list
                   if d.get("obj_name") == stair_obj.name), None)
        if sd is None:
            self.report({'ERROR'}, f"Stair data not found for '{stair_obj.name}'")
            return {'CANCELLED'}

        # 2. Find target rooms from selected objects
        room_name_set = {r.get("obj_name", "") for r in ROOM_OT_draw._room_list}
        target_objs = [o for o in context.selected_objects
                       if o != stair_obj and o.name in room_name_set]
        if not target_objs:
            self.report({'ERROR'}, "Select room objects on the target floors")
            return {'CANCELLED'}

        # 3. Collect unique z levels of target rooms
        target_data = []
        for o in target_objs:
            for r in ROOM_OT_draw._room_list:
                if r.get("obj_name") == o.name:
                    target_data.append(r)
                    break
        target_zs = sorted({round(r["z"], 4) for r in target_data})

        # 4. Determine direction
        z_bot_orig = sd["z_bot"]
        z_top_orig = sd["z_top"]
        rooms_above = [z for z in target_zs if z > z_top_orig + 0.1]
        rooms_below = [z for z in target_zs if z < z_bot_orig - 0.1]

        if rooms_above and rooms_below:
            self.report({'ERROR'}, "Selected rooms must be all above or all below the existing stair")
            return {'CANCELLED'}
        if not rooms_above and not rooms_below:
            self.report({'ERROR'}, "No target rooms found above or below the existing stair")
            return {'CANCELLED'}

        going_up    = bool(rooms_above)
        sorted_zs   = sorted(rooms_above if going_up else rooms_below,
                             reverse=not going_up)

        # 5. Original footprint components
        lx1_o = sd["lx1"]; ly1_o = sd["ly1"]; lx2_o = sd["lx2"]; ly2_o = sd["ly2"]
        ux1_o = sd["ux1"]; uy1_o = sd["uy1"]; ux2_o = sd["ux2"]; uy2_o = sd["uy2"]
        # Detect travel axis from how upper opening is offset from lower opening
        _dx = abs((ux1_o + ux2_o) - (lx1_o + lx2_o))
        _dy = abs((uy1_o + uy2_o) - (ly1_o + ly2_o))
        x_travel = sd.get("x_travel", _dx >= _dy)

        # 6. Compute lateral offset for alternating (mirrored) flights
        rooms_all = ROOM_OT_draw._room_list
        # z levels for the first new flight (used to test if B fits in the room)
        first_z_lo = z_top_orig if going_up else sorted_zs[0]
        first_z_hi = sorted_zs[0] if going_up else z_bot_orig

        def _B_fits(sign_test):
            if x_travel:
                tlo = (ux1_o, ly1_o + sign_test*off_amount, ux2_o, ly2_o + sign_test*off_amount)
                thi = (lx1_o, ly1_o + sign_test*off_amount, lx2_o, ly2_o + sign_test*off_amount)
            else:
                tlo = (lx1_o + sign_test*off_amount, uy1_o, lx2_o + sign_test*off_amount, uy2_o)
                thi = (lx1_o + sign_test*off_amount, ly1_o, lx2_o + sign_test*off_amount, ly2_o)
            return (
                _rect_fits_in_room(tlo[0], tlo[1], tlo[2], tlo[3], first_z_lo, rooms_all) is not None and
                _rect_fits_in_room(thi[0], thi[1], thi[2], thi[3], first_z_hi, rooms_all) is not None
            )

        if x_travel:
            stair_width = ly2_o - ly1_o
            off_amount  = s.stair_apply_offset if s.stair_apply_offset > 1e-4 else stair_width
            traveling_pos = (ux1_o + ux2_o) > (lx1_o + lx2_o)
            sign = 1.0 if traveling_pos else -1.0
            if s.stair_apply_dir == 'RIGHT':
                sign = -sign
            ox, oy = 0.0, sign * off_amount
        else:
            stair_width = lx2_o - lx1_o
            off_amount  = s.stair_apply_offset if s.stair_apply_offset > 1e-4 else stair_width
            traveling_pos = (uy1_o + uy2_o) > (ly1_o + ly2_o)
            sign = -1.0 if traveling_pos else 1.0
            if s.stair_apply_dir == 'RIGHT':
                sign = -sign
            ox, oy = sign * off_amount, 0.0

        # Auto-correct: if B doesn't fit in the room with the chosen direction, try the opposite
        if not _B_fits(sign if x_travel else sign) and _B_fits(-sign if x_travel else -sign):
            sign = -sign
            if x_travel:
                ox, oy = 0.0, sign * off_amount
            else:
                ox, oy = sign * off_amount, 0.0

        # Template A = original footprint (even flights)
        # Template B = offset + mirrored (odd flights): lower↔upper swapped, shifted.
        # Use ly (not uy) for x_travel and lx (not ux) for y_travel so B's perpendicular
        # extent is consistent with A's and the two flights sit flush side-by-side.
        if x_travel:
            B_lo = (ux1_o, ly1_o + oy, ux2_o, ly2_o + oy)
            B_hi = (lx1_o, ly1_o + oy, lx2_o, ly2_o + oy)
        else:
            B_lo = (lx1_o + ox, uy1_o, lx2_o + ox, uy2_o)
            B_hi = (lx1_o + ox, ly1_o, lx2_o + ox, ly2_o)

        stair_col = _get_or_create_stair_col(context)
        prev_z    = z_top_orig if going_up else z_bot_orig
        n_created = 0

        for i, new_z in enumerate(sorted_zs):
            mirrored = (i % 2 == 0)  # flight 1 (i=0) mirrors original flight 0

            if going_up:
                z_bot_new, z_top_new = prev_z, new_z
            else:
                z_bot_new, z_top_new = new_z, prev_z

            if mirrored:
                lx1, ly1, lx2, ly2 = B_lo
                ux1, uy1, ux2, uy2 = B_hi
            else:
                lx1, ly1, lx2, ly2 = lx1_o, ly1_o, lx2_o, ly2_o
                ux1, uy1, ux2, uy2 = ux1_o, uy1_o, ux2_o, uy2_o

            # 7. Validate footprints fit in rooms
            if _rect_fits_in_room(lx1, ly1, lx2, ly2, z_bot_new, rooms_all) is None:
                self.report({'ERROR'},
                    f"{'Mirrored' if mirrored else 'Original'} lower footprint does not fit "
                    f"in a room at Z={z_bot_new:.2f} (flight {i+1})")
                return {'CANCELLED'}
            if _rect_fits_in_room(ux1, uy1, ux2, uy2, z_top_new, rooms_all) is None:
                self.report({'ERROR'},
                    f"{'Mirrored' if mirrored else 'Original'} upper footprint does not fit "
                    f"in a room at Z={z_top_new:.2f} (flight {i+1})")
                return {'CANCELLED'}

            # 8. Build new stair object
            new_sd = dict(sd)
            new_sd.update(lx1=lx1, ly1=ly1, lx2=lx2, ly2=ly2,
                          ux1=ux1, uy1=uy1, ux2=ux2, uy2=uy2,
                          z_bot=z_bot_new, z_top=z_top_new)
            idx = len(ROOM_OT_draw._stair_list) + 1
            obj = _make_stair_obj(f"Stair.{idx:03d}", new_sd, s, stair_col)
            if obj is None:
                self.report({'ERROR'},
                    f"Could not build stair geometry for flight {i+1} "
                    f"(Z {z_bot_new:.2f}→{z_top_new:.2f}).")
                return {'CANCELLED'}
            new_sd["obj_name"] = obj.name
            ROOM_OT_draw._stair_list.append(new_sd)

            # Cut floor/ceiling holes in both rooms
            li_new = _rect_fits_in_room(lx1, ly1, lx2, ly2, z_bot_new, rooms_all)
            ui_new = _rect_fits_in_room(ux1, uy1, ux2, uy2, z_top_new, rooms_all)
            if li_new is not None and ui_new is not None:
                _stair_cut_holes(new_sd, li_new, ui_new, rooms_all, s)
            elif li_new is not None:
                _stair_cut_holes(new_sd, li_new, li_new, rooms_all, s)

            prev_z = z_top_new if going_up else z_bot_new
            n_created += 1

        _sync_to_scene(context)
        self.report({'INFO'}, f"Created {n_created} stair flight(s)")
        return {'FINISHED'}


# ═════════════════════════════════════════════════════════════════════════════
# Register / Unregister
# ═════════════════════════════════════════════════════════════════════════════
_classes = (
    ROOM_PG_registry_entry,
    ROOM_PG_floor,
    ROOM_PG_door_preset,
    ROOM_PG_window_preset,
    ROOM_PG_arch_preset,
    ROOM_PG_settings,
    ROOM_OT_draw,
    ROOM_OT_clear,
    ROOM_OT_clear_overlays,
    ROOM_OT_door_edit,
    ROOM_OT_door_flip,
    ROOM_OT_door_flip_all,
    ROOM_OT_window_edit,
    ROOM_OT_stair_edit,
    ROOM_OT_stair_rebuild,
    ROOM_OT_stair_delete,
    ROOM_OT_stair_move,
    ROOM_OT_stair_apply_floors,
    ROOM_OT_add_floor,
    ROOM_OT_select_floor,
    ROOM_OT_remove_floor,
    ROOM_OT_save_door_preset,
    ROOM_OT_select_door_preset,
    ROOM_OT_remove_door_preset,
    ROOM_OT_save_window_preset,
    ROOM_OT_select_window_preset,
    ROOM_OT_remove_window_preset,
    ROOM_OT_save_arch_preset,
    ROOM_OT_select_arch_preset,
    ROOM_OT_remove_arch_preset,
    ROOM_OT_sample_dims_apply,
    ROOM_OT_sample_dims,
    ROOM_OT_toggle_room_plinth,
    ROOM_OT_toggle_mesh_lock,
    ROOM_OT_recalculate_plinth,
    ROOM_OT_recalculate_uv,
    ROOM_OT_sync_thresholds,
    ROOM_PT_panel,
    ROOM_PT_door_panel,
    ROOM_PT_window_panel,
    ROOM_PT_stair_panel,
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
    bpy.types.Scene.room_arch_presets         = bpy.props.CollectionProperty(type=ROOM_PG_arch_preset)
    bpy.types.Scene.room_active_arch_preset   = bpy.props.IntProperty(default=-1)
    bpy.types.Scene.room_registry             = bpy.props.CollectionProperty(type=ROOM_PG_registry_entry)
    bpy.types.Scene.room_door_edit_active     = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.room_window_edit_active   = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.room_stair_edit_active    = bpy.props.BoolProperty(default=False)

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
        kmi4 = km.keymap_items.new("room.stair_edit", "S", "PRESS", shift=True, ctrl=True)
        ROOM_OT_draw._addon_kmaps.append((km, kmi4))
        # Double-click on a stair object → open Move / Rotate immediately
        kmi5 = km.keymap_items.new("room.stair_move", "LEFTMOUSE", "DOUBLE_CLICK")
        ROOM_OT_draw._addon_kmaps.append((km, kmi5))
        # Ctrl+Shift+E → sample object dimensions (doors & windows)
        kmi6 = km.keymap_items.new("room.sample_dims", "E", "PRESS", shift=True, ctrl=True)
        ROOM_OT_draw._addon_kmaps.append((km, kmi6))
        # Delete → delete stair + remove holes (poll ensures this only fires for stair objects;
        #     for all other objects Blender falls through to its default Delete behaviour)
        kmi7 = km.keymap_items.new("room.stair_delete", "DEL", "PRESS")
        ROOM_OT_draw._addon_kmaps.append((km, kmi7))

    # Persistent edit-mode hint at bottom of viewport
    _h = bpy.types.SpaceView3D.draw_handler_add(
        _draw_room_hint_cb, (), 'WINDOW', 'POST_PIXEL')
    _DRAW_HANDLES.add(_h)

    if _room_registry_cleanup not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_room_registry_cleanup)
    if _room_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_room_on_load)
    if _room_undo_post not in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.append(_room_undo_post)


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
    if _room_undo_post in bpy.app.handlers.undo_post:
        bpy.app.handlers.undo_post.remove(_room_undo_post)

    for km, kmi in ROOM_OT_draw._addon_kmaps:
        try: km.keymap_items.remove(kmi)
        except Exception: pass
    ROOM_OT_draw._addon_kmaps.clear()

    for attr in ("room_settings", "room_floors", "room_active_floor",
                 "room_door_presets", "room_active_door_preset",
                 "room_window_presets", "room_active_window_preset",
                 "room_arch_presets", "room_active_arch_preset",
                 "room_registry", "room_door_edit_active", "room_window_edit_active",
                 "room_stair_edit_active"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)

    for c in reversed(_classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
