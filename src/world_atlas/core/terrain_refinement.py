"""Deterministic subgrid ground synthesis constrained by the saved terrain.

This adds plausible modelled relief, not recovered measurements.  A bounded
coordinate deformation refines the *whole* ground, preserving continuous
land/water topology.  Height detail preserves its sign and therefore cannot
create detached lakes or islands.  Original ground samples and river anchors
are fixed.  Coasts, elevation fills and contours must query this same field.
"""

import math

import numpy as np
import shapely
from scipy import ndimage
from scipy.optimize import elementwise
from scipy.spatial import cKDTree

from world_atlas.core.continuous_terrain import PhysicalTerrainField


_PATCH_FRACTION = .25
_PATCH_DERIVATIVE = 96/(25*math.sqrt(5))


def _smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _bilinear(values, x, y):
    height, width = values.shape
    column, row = np.floor(x - .5).astype(np.int64), np.floor(y - .5).astype(np.int64)
    tx, ty = x - .5 - column, y - .5 - row
    first, last = np.clip(row, 0, height - 1), np.clip(row + 1, 0, height - 1)
    return ((1 - ty) * ((1 - tx) * values[first, column % width]
                       + tx * values[first, (column + 1) % width])
            + ty * ((1 - tx) * values[last, column % width]
                    + tx * values[last, (column + 1) % width]))


class _DrainageRelief:
    """Continuous valley profiles around the inherited tributary graph.

    Flow accumulation is relative catchment support, not cubic metres/second.
    It controls kilometre-scale valley breadth; saved ground differences
    control strength. No random height spectrum supplies unrelated pits.
    """

    def __init__(self, base, flow_to, discharge, radius_km):
        self.height, self.width = base.height, base.width
        self.radius_km = radius_km
        flow, support = np.asarray(flow_to), np.asarray(discharge, dtype=float)
        if (flow.shape != base.native_m.shape or support.shape != flow.shape
                or not np.issubdtype(flow.dtype, np.integer)
                or np.any(flow < -1) or np.any(flow >= flow.size)
                or not np.all(np.isfinite(support)) or np.any(support < 0)):
            raise ValueError("terrain refinement requires the saved drainage graph and accumulation")
        ids = np.flatnonzero((flow.ravel() >= 0) & (support.ravel() >= 5)
                             & (base.native_m.ravel() > 0))
        target = flow.ravel()[ids]
        drop = base.native_m.ravel()[ids] - base.native_m.ravel()[target]
        keep = (drop > 0) & (base.native_m.ravel()[target] > 0)
        ids, target, drop = ids[keep], target[keep], drop[keep]
        start = np.column_stack((ids % self.width + .5, ids // self.width + .5))
        end = np.column_stack((target % self.width + .5, target // self.width + .5))
        end[:, 0] = start[:, 0] + ((end[:, 0]-start[:, 0]+self.width/2) % self.width-self.width/2)
        direction = end-start
        following = flow.ravel()[target]
        next_id = np.where(following >= 0, following, target)
        next_point = np.column_stack((next_id % self.width + .5, next_id // self.width + .5))
        next_point[:, 0] = end[:, 0] + ((next_point[:, 0]-end[:, 0]+self.width/2) % self.width-self.width/2)
        outgoing = next_point-end
        outgoing = np.where((following >= 0)[:, None], outgoing, direction)
        # The graph's downstream direction bends tributaries into their parent
        # valley. Cubic Hermite arcs pass through the actual native nodes.
        tangent = (direction+outgoing)/2
        t = np.linspace(0, 1, 5)[None, :, None]
        curves = ((2*t**3-3*t**2+1)*start[:, None, :]
                  + (t**3-2*t**2+t)*direction[:, None, :]
                  + (-2*t**3+3*t**2)*end[:, None, :]
                  + (t**3-t**2)*tangent[:, None, :])
        self.curves = np.concatenate((curves-[self.width, 0], curves, curves+[self.width, 0]))
        self._tree = shapely.STRtree(shapely.linestrings(self.curves))
        q = support.ravel()[ids]
        breadth = np.minimum(8., 2.5+1.15*np.log2(q))
        strength = _smoothstep(drop/500) * (.3+.7*_smoothstep(np.log2(q)/8))
        self.breadth = np.tile(breadth, 3)
        self.strength = np.tile(strength, 3)
        self.cell_y_km = math.pi*radius_km/self.height
        self.cell_x_km = 2*math.pi*radius_km/self.width
        self.maximum_breadth = float(np.max(breadth)) if len(breadth) else 0.
        self.edge_count = len(ids)
        native = np.empty(base.native_m.shape)
        x = np.arange(self.width)[None, :]+.5
        for row in range(0, self.height, 32):
            y = np.arange(row, min(row+32, self.height))[:, None]+.5
            native[row:row+len(y)] = self.sample(x, y)
        self.pchip = PhysicalTerrainField(native, land_mask=native > 0,
            sea_level_m=0, elevation_scale_m=1, elevation_exponent=1)

    def sample(self, x, y):
        x, y = np.broadcast_arrays(x, y)
        result = np.zeros(x.size)
        if not self.edge_count:
            return result.reshape(x.shape)
        xx, yy = x.ravel(), y.ravel()
        east_km = self.cell_x_km*np.maximum(np.cos(math.pi/2-yy*math.pi/self.height), .08)
        pairs = self._tree.query(shapely.box(xx-self.maximum_breadth/east_km,
            yy-self.maximum_breadth/self.cell_y_km,
            xx+self.maximum_breadth/east_km, yy+self.maximum_breadth/self.cell_y_km))
        if pairs.shape[1]:
            point, edge = pairs
            curve = self.curves[edge]
            distance2 = np.full(len(point), np.inf)
            for index in range(4):
                ax = (curve[:, index, 0]-xx[point])*east_km[point]
                ay = (curve[:, index, 1]-yy[point])*self.cell_y_km
                dx = (curve[:, index+1, 0]-curve[:, index, 0])*east_km[point]
                dy = (curve[:, index+1, 1]-curve[:, index, 1])*self.cell_y_km
                along = np.clip(-(ax*dx+ay*dy)/np.maximum(dx*dx+dy*dy, 1e-15), 0, 1)
                distance2 = np.minimum(distance2, (ax+along*dx)**2+(ay+along*dy)**2)
            fraction = np.sqrt(distance2)/self.breadth[edge]
            inside = fraction < 1
            # Compact C2 valley profile: adjoining drainage branches combine
            # continuously; nearest-feature ownership cannot create seams.
            t = fraction[inside]
            contribution = self.strength[edge[inside]]*(1-t)**4*(1+4*t)
            np.add.at(result, point[inside], contribution)
        return (-np.tanh(result)).reshape(x.shape)


class _CoastalLandforms:
    """Disjoint, compact C2 ground deformations around inherited shore forms.

    Each disk excludes saved centres and rivers. A displacement <= R/4 has
    derivative norm <= .43, so the map is one-to-one. Disjoint support gives
    the same bound for their union and requires no global protection cap.
    """

    def __init__(self, base, drainage, *, seed, rivers, anchors):
        height, width = base.height, base.width
        dry = base.native_m > 0
        east = np.roll(dry, -1, axis=1)
        mixed = ((dry[:-1] | east[:-1] | dry[1:] | east[1:])
                 & ~(dry[:-1] & east[:-1] & dry[1:] & east[1:]))
        rows, columns = np.nonzero(mixed)
        rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
        fraction = rng.uniform(.15, .85, len(rows))
        y = rows+.5+fraction
        x0, x1 = columns+.5, columns+1.5
        z0, z1 = base.sample_points(x0, y), base.sample_points(x1, y)
        horizontal = (z0 > 0) != (z1 > 0)
        vertical_x = columns+.5+fraction
        yy0, yy1 = rows+.5, rows+1.5
        zz0, zz1 = base.sample_points(vertical_x, yy0), base.sample_points(vertical_x, yy1)
        keep = horizontal | ((zz0 > 0) != (zz1 > 0))
        first = np.column_stack((np.where(horizontal, x0, vertical_x),
                                 np.where(horizontal, y, yy0)))[keep]
        last = np.column_stack((np.where(horizontal, x1, vertical_x),
                                np.where(horizontal, y, yy1)))[keep]
        first_sign = np.where(horizontal, z0, zz0)[keep] > 0
        lower, upper = np.zeros(len(first)), np.ones(len(first))
        for _ in range(24):
            t = (lower+upper)/2
            points = first+(last-first)*t[:, None]
            same = (base.sample_points(points[:, 0], points[:, 1]) > 0) == first_sign
            lower, upper = np.where(same, t, lower), np.where(same, upper, t)
        centres = first+(last-first)*((lower+upper)/2)[:, None]
        centres[:, 0] %= width
        native_offset = (centres-.5)-np.round(centres-.5)
        radius = .85*np.linalg.norm(native_offset, axis=1)
        radius = np.minimum(radius, .85*np.minimum(centres[:, 1], height-centres[:, 1]))
        if rivers is not None and len(centres):
            pairs, distance = rivers.query_nearest(shapely.points(centres),
                                                   all_matches=False, return_distance=True)
            ordered = np.empty(len(centres)); ordered[pairs[0]] = distance
            radius = np.minimum(radius, .85*ordered)
        if anchors is not None:
            distance, _ = anchors.query(centres)
            radius = np.minimum(radius, .85*distance)
        x, y = centres[:, 0], centres[:, 1]
        epsilon = 1/32
        dx = (base.sample_points(x+epsilon, y)-base.sample_points(x-epsilon, y))/(2*epsilon)
        dy = (base.sample_points(x, y+epsilon)-base.sample_points(x, y-epsilon))/(2*epsilon)
        gradient = np.hypot(dx, dy)
        normal = np.column_stack((dx, dy))/np.maximum(gradient[:, None], 1e-15)
        tangent = np.column_stack((-normal[:, 1], normal[:, 0]))
        before, after = centres-tangent*radius[:, None], centres+tangent*radius[:, None]
        curvature = (base.sample_points(before[:, 0], before[:, 1])
                     + base.sample_points(after[:, 0], after[:, 1]))
        valley = drainage.sample(x, y)-drainage.pchip.sample_points(x, y)
        structure = .85*valley-.5*np.tanh(curvature/(gradient*radius+30))
        # Amplify inherited concavity/convexity into subgrid landforms. This
        # changes neither their sign nor their world-coordinate locations;
        # a planar coast with no tributary influence receives no deformation.
        strength = np.tanh(15*structure)
        # A patch smaller than one shoreline sample cannot be resolved by
        # the shared 16x extraction. Preserve that source feature unchanged.
        keep = (radius >= 1/16) & (np.abs(strength) > .05)
        centres, radius, normal, strength = centres[keep], radius[keep], normal[keep], strength[keep]
        if len(radius):
            copies = np.concatenate((centres-[width, 0], centres, centres+[width, 0]))
            tree = cKDTree(copies)
            blocked = np.zeros(len(centres), dtype=bool)
            selected = []
            for index in np.argsort(-np.abs(strength)*radius, kind="stable"):
                if blocked[index]:
                    continue
                selected.append(index)
                neighbors = np.asarray(tree.query_ball_point(centres[index], radius[index]+radius.max()), dtype=int)
                offset = copies[neighbors]-centres[index]
                neighbor_ids = neighbors % len(centres)
                overlap = np.linalg.norm(offset, axis=1) <= radius[index]+radius[neighbor_ids]
                blocked[neighbor_ids[overlap]] = True
            selected = np.asarray(selected)
            centres, radius, normal, strength = centres[selected], radius[selected], normal[selected], strength[selected]
        self.count = len(radius)
        self.centres = np.concatenate((centres-[width, 0], centres, centres+[width, 0]))
        self.radius = np.tile(radius, 3)
        displacement = normal*(_PATCH_FRACTION*radius*strength)[:, None]
        self.displacement = np.tile(displacement, (3, 1))
        self._tree = shapely.STRtree(shapely.box(self.centres[:, 0]-self.radius,
            self.centres[:, 1]-self.radius, self.centres[:, 0]+self.radius,
            self.centres[:, 1]+self.radius))
        self.maximum_displacement = float(np.max(np.linalg.norm(displacement, axis=1))) if self.count else 0.

    def sample(self, x, y):
        warped_x, warped_y = x.copy(), y.copy()
        if not self.count:
            return warped_x, warped_y
        pairs = self._tree.query(shapely.points(x.ravel(), y.ravel()))
        if pairs.shape[1]:
            point, patch = pairs
            xx, yy = x.ravel(), y.ravel()
            fraction2 = ((xx[point]-self.centres[patch, 0])**2
                         + (yy[point]-self.centres[patch, 1])**2)/self.radius[patch]**2
            keep = fraction2 < 1
            point, patch, fraction2 = point[keep], patch[keep], fraction2[keep]
            influence = (1-fraction2)**3
            warped_x.ravel()[point] += self.displacement[patch, 0]*influence
            warped_y.ravel()[point] += self.displacement[patch, 1]*influence
        return warped_x, warped_y


class RefinedTerrainField:
    """One finest continuous ground, independently sampled at any map scale.

    The native-centre constraint is an exact anchoring operation. It never
    modifies a saved array. Each local ground deformation has a Lipschitz
    bound below one, and disjoint support makes their union one-to-one. Its zero contour is
    the transported source coast. Positive vertical gain changes relief but
    cannot alter the land/water domain. Rasterised/vectorised output still
    requires component and native classification validation before publishing.
    """

    def __init__(self, base: PhysicalTerrainField, *, seed, radius_km,
                 flow_to, discharge, river_segments, river_anchors):
        if not math.isfinite(radius_km) or radius_km <= 0:
            raise ValueError("terrain refinement requires a positive planet radius")
        segments = np.asarray(river_segments, dtype=float)
        anchors = np.asarray(river_anchors, dtype=float)
        if (segments.ndim != 3 or segments.shape[1:] != (2, 2) or not np.all(np.isfinite(segments))
                or np.any(segments[:, :, 1] < 0) or np.any(segments[:, :, 1] > base.height)
                or anchors.ndim != 2 or anchors.shape[1] != 2
                or not np.all(np.isfinite(anchors)) or np.any(anchors[:, 1] < 0)
                or np.any(anchors[:, 1] > base.height)):
            raise ValueError("terrain refinement requires matching rivers and finite native anchors")
        self.base = base
        self.native_m = base.native_m
        self.height, self.width = base.height, base.width
        self.sea_level_m = base.sea_level_m
        self.elevation_scale_m = base.elevation_scale_m
        self.elevation_exponent = base.elevation_exponent
        self.seed, self.radius_km = int(seed), float(radius_km)
        permitted = np.ones(base.native_m.shape)
        # Deformation fixes the displayed latitude frame as well as longitude
        # periodicity, so it cannot move ground outside the source hemisphere.
        permitted[:2], permitted[-2:] = 0, 0
        dx = (np.roll(base.native_m, -1, axis=1) - np.roll(base.native_m, 1, axis=1))/2
        dy = (np.gradient(base.native_m, axis=0) if self.height > 1
              else np.zeros(base.native_m.shape))
        local_relief = np.hypot(dx, dy)
        self._height = _DrainageRelief(base, flow_to, discharge, radius_km)
        self._amplitude = permitted * (12 + .8*local_relief
                                      + 450*_smoothstep(np.maximum(base.native_m, 0)/3500))
        self._amplitude.flags.writeable = False
        if len(segments):
            segments = segments.copy()
            segments[:, 0, 0] %= self.width
            segments[:, 1, 0] = segments[:, 0, 0] + ((segments[:, 1, 0]-segments[:, 0, 0]
                                                     + self.width/2) % self.width-self.width/2)
            periodic = np.concatenate((segments - [self.width, 0], segments,
                                       segments + [self.width, 0]))
            self._rivers = shapely.STRtree(shapely.linestrings(periodic))
            support = np.zeros(base.native_m.shape, dtype=bool)
            # Coarse support only excludes points certainly outside the .5
            # cell valley buffer. Actual weights use GEOS segment distances.
            for segment in segments:
                west, north = np.floor(segment.min(axis=0)).astype(int)
                east, south = np.floor(segment.max(axis=0)).astype(int)
                support[np.clip(np.arange(north, south+1), 0, self.height-1)[:, None],
                        (np.arange(west, east+1) % self.width)[None, :]] = True
            self._river_support_distance = ndimage.distance_transform_edt(
                ~np.tile(support, (1, 3)))[:, self.width:2*self.width]
        else:
            self._rivers = None
            self._river_support_distance = None
        if len(anchors):
            anchors = anchors.copy()
            anchors[:, 0] %= self.width
            self._anchors = cKDTree(np.concatenate((anchors - [self.width, 0], anchors,
                                                    anchors + [self.width, 0])))
        else:
            self._anchors = None
        self._landforms = _CoastalLandforms(base, self._height, seed=seed,
            rivers=self._rivers, anchors=self._anchors)
        self.diagnostics = {"schema":"procedural-terrain-refinement-v1", "seed":self.seed,
            "radiusKm":self.radius_km, "reliefModel":"inherited-drainage-valley-profiles",
            "tributaryEdges":self._height.edge_count, "coastalLandforms":self._landforms.count,
            "coordinateJacobianBound":_PATCH_FRACTION*_PATCH_DERIVATIVE,
            "maximumCoordinateDisplacementCells":self._landforms.maximum_displacement,
            "interpretation":"Procedural subgrid relief conditioned by saved model ground; no recovered measurements."}

    def palette_elevation(self, relative_m):
        return self.base.palette_elevation(relative_m)

    def contour_height_m(self, palette_height):
        return self.base.contour_height_m(palette_height)

    def _anchor_weight(self, x, y):
        if self._anchors is None:
            return 1.
        distance, _ = self._anchors.query(np.column_stack((x.ravel(), y.ravel())))
        return _smoothstep((distance.reshape(x.shape)-.1)/.4)

    def _river_weight(self, x, y):
        weight = np.ones(x.shape)
        if self._rivers is None:
            return weight
        nearby = _bilinear(self._river_support_distance, x, y) < 2.5
        if np.any(nearby):
            pairs, distance = self._rivers.query_nearest(shapely.points(x[nearby], y[nearby]),
                                                        all_matches=False, return_distance=True)
            ordered = np.empty(np.count_nonzero(nearby))
            ordered[pairs[0]] = distance
            weight[nearby] = _smoothstep((ordered-.18)/.32)
        return weight

    def forward_ground_coordinates(self, x, y):
        """Apply the existing ground deformation in an unwrapped longitude.

        The latitude frame and saved centres are fixed by the original compact
        patches. Keeping the longitude's period here lets a branch cross the
        map cut without introducing a second, independently sampled shoreline.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(y < 0) or np.any(y > self.height)):
            raise ValueError("ground coordinates require finite map coordinates")
        longitude = np.asarray(x % self.width)
        period = x-longitude
        warped_x, warped_y = self._landforms.sample(longitude, y)
        return warped_x+period, warped_y

    def inverse_ground_coordinates(self, x, y):
        """Invert the unchanged, disjoint C2 ground patches.

        A patch fixes its disk boundary and has displacement derivative norm
        below one. Its image is therefore the same disk, and the inverse is
        unique. Inside that disk the displacement has a fixed direction: one
        bracketed scalar root determines its influence instead of rebuilding
        or sampling an inverse coordinate field.
        """
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(y < 0) or np.any(y > self.height)):
            raise ValueError("inverse ground coordinates require finite map coordinates")
        shape = x.shape
        longitude = x % self.width
        period = x-longitude
        points = np.column_stack((longitude.ravel(), y.ravel()))
        result = points.copy()
        patches = self._landforms
        pairs = patches._tree.query(shapely.points(points))
        if pairs.shape[1]:
            point, patch = pairs
            offset = points[point]-patches.centres[patch]
            radius = patches.radius[patch]
            keep = np.sum(offset*offset, axis=1) < radius*radius
            point, patch, offset, radius = point[keep], patch[keep], offset[keep], radius[keep]
            if len(np.unique(point)) != len(point):
                raise ValueError("ground deformation disks must be disjoint")
            displacement = patches.displacement[patch]

            def residual(influence, offset_x, offset_y, dx, dy, radius):
                fraction2 = ((offset_x-dx*influence)**2+(offset_y-dy*influence)**2)/radius**2
                return influence-np.maximum(1-fraction2, 0.)**3

            root = elementwise.find_root(residual, (np.zeros(len(point)), np.ones(len(point))),
                args=(offset[:, 0], offset[:, 1], displacement[:, 0], displacement[:, 1], radius),
                tolerances={"xatol": 1e-14, "xrtol": 4*np.finfo(float).eps,
                            "fatol": 0., "frtol": 0.}, maxiter=100)
            if not np.all(root.success):
                raise ValueError("inverse ground deformation roots must converge")
            result[point] -= displacement*root.x[:, None]
        return (result[:, 0].reshape(shape)+period, result[:, 1].reshape(shape))

    def sample_points(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(y < 0) or np.any(y > self.height)):
            raise ValueError("terrain refinement requires finite native map coordinates")
        x = x % self.width
        native = (x - .5 == np.floor(x - .5)) & (y - .5 == np.floor(y - .5))
        if bool(native.all()):
            return self.native_m[(y-.5).astype(int), (x-.5).astype(int)].copy()
        protection = self._anchor_weight(x, y)*self._river_weight(x, y)
        warped_x, warped_y = self.forward_ground_coordinates(x, y)
        ground = self.base.sample_points(warped_x, warped_y)
        detail = self._height.sample(x, y) - self._height.pchip.sample_points(x, y)
        amplitude = _bilinear(self._amplitude, x, y)*protection
        # Gain is strictly positive (>=.55). The shore is determined by the
        # transported saved ground alone; valleys cannot become new sea/lakes.
        gain = 1 + .45*np.tanh(amplitude/(np.abs(ground) + 30))*detail
        result = ground*gain
        if np.any(native):
            result = np.asarray(result).copy()
            result[native] = self.native_m[(y[native]-.5).astype(int), (x[native]-.5).astype(int)]
        return result

    def sample_rect(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (x.ndim != 1 or y.ndim != 1 or not x.size or not y.size
                or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
            raise ValueError("terrain refinement requires increasing map axes")
        return self.sample_points(x[None, :], y[:, None])


def terrain_from_source(grid, source):
    """Build the one published ground and its protection anchors from inputs.

    Rendering and independent output checks must use this factory, so they
    cannot disagree about the physical datum or duplicate mouth locations.
    Source sign validation is strict; no older palette field is substituted.
    """
    diagnostics = source.diagnostics
    base = PhysicalTerrainField(source.relative_elevation_m, land_mask=grid.water == 0,
        sea_level_m=diagnostics["seaLevelMeters"],
        elevation_scale_m=diagnostics["elevationScaleMeters"],
        elevation_exponent=diagnostics["elevationExponent"])
    from .river_network import hydrologic_outlet_targets

    flow = grid.flow_to.ravel().copy()
    for terminal, target in hydrologic_outlet_targets(grid, source.relative_elevation_m).items():
        flow[terminal] = target
    ids = np.flatnonzero((grid.water.ravel() == 0)
                        & (grid.river_order.ravel() > 0) & (flow >= 0))
    targets = flow[ids]
    start = np.column_stack((ids % base.width+.5, ids // base.width+.5))
    end = np.column_stack((targets % base.width+.5, targets // base.width+.5))
    end[:, 0] = start[:, 0] + ((end[:, 0]-start[:, 0]+base.width/2) % base.width-base.width/2)
    segments = np.stack((start, end), axis=1)
    mouth = grid.water.ravel()[targets] != 0
    first, last = start[mouth], end[mouth]
    lower, upper = np.zeros(len(first)), np.ones(len(first))
    for _ in range(36):
        t = (lower+upper)/2
        point = first+(last-first)*t[:, None]
        dry = base.sample_points(point[:, 0], point[:, 1]) > 0
        lower, upper = np.where(dry, t, lower), np.where(dry, upper, t)
    anchors = first+(last-first)*((lower+upper)/2)[:, None]
    return RefinedTerrainField(base, seed=diagnostics["seed"],
        radius_km=grid.metadata["planet"]["radiusKm"],
        flow_to=flow.reshape(grid.shape), discharge=grid.discharge,
        river_segments=segments, river_anchors=anchors)
