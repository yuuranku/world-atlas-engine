"""Paint supported landform areas and slope directions in the loaded map blocks.

Texture is a cartographic convention within an observed model class. Reeds,
short hatches and white-blue snow paint do not assert additional terrain sites.
"""
from dataclasses import replace

import numpy as np
import shapely

from .cartographic_features import filled_features
from .cartographic_tiles import TileFeature, feature_definitions, feature_markup
from .continuous_scalar import scalar_band_paths
from .raster_topology import categorical_coverage
from .wetlands import WETLAND_SUPPORT_THRESHOLD


def mask_faces(mask):
    """Use the common ownership reconstruction for categorical land cover."""
    values=np.asarray(mask,dtype=bool)
    if values.ndim!=2 or min(values.shape)<1:
        raise ValueError("landform areas require a native two-dimensional support")
    if not values.any():
        return []
    faces,labels=categorical_coverage(values.astype(np.int32),np.ones(values.shape,dtype=bool),category_count=2)
    # Polygon records are a common representation, independent of SVG ink.
    result=[]
    for polygon,label in zip(faces,labels,strict=True):
        if label!=1:continue
        rings=[np.asarray(ring.coords) for ring in (polygon.exterior,*polygon.interiors)]
        codes=[]
        for ring in rings:
            current=np.full(len(ring),2,dtype=np.uint8);current[0]=1;current[-1]=79
            codes.append(current)
        result.append((np.vstack(rings),np.concatenate(codes)))
    return result


def landform_features(grid,inventory):
    """Return reusable geometry records; themes contain no copied texture."""
    from .cartographic_symbols import AREA_PATTERNS,LANDFORM_STYLES,pattern_markup
    features=[]
    colors={"wetland":"#6a999b","snow-mountain":"#edf5f6","arid-upland":"#928069"}
    for kind in AREA_PATTERNS:
        mask=np.asarray(inventory.masks[kind],dtype=bool)
        if mask.shape!=grid.shape or np.any(mask & (grid.water!=0)):
            raise ValueError("landform texture must have verified native land support")
        if not mask.any():continue
        if kind=="wetland":
            support=np.asarray(inventory.wetland_support,dtype=float)
            if (support.shape!=grid.shape or not np.all(np.isfinite(support))
                    or np.any((support<0)|(support>1))
                    or not np.array_equal(support>=WETLAND_SUPPORT_THRESHOLD,mask)):
                raise ValueError("wetland paint must use the continuous hydrologic support field")
            paths=scalar_band_paths(support,grid.water==0,[WETLAND_SUPPORT_THRESHOLD])[1]
        else:
            paths=mask_faces(mask)
        common={"data-landform-type":kind,"data-landform-support":"hydrologic-field" if kind=="wetland" else "indicator-field"}
        # A restrained regional tint remains readable when the world overview
        # omits detailed glyphs. There is no outlined mask or native-cell grid.
        features.extend(filled_features(paths,"geographic-textures",colors[kind],section="ink",clip="land",
            opacity=.12 if kind=="wetland" else .16 if kind=="snow-mountain" else .055,attributes=common))
        identifier=f"landform-pattern-{kind}"
        definition=pattern_markup(kind,identifier)
        minimum=LANDFORM_STYLES[kind]["min_scale"] if kind in LANDFORM_STYLES else 4
        maximum=LANDFORM_STYLES[kind]["max_scale"] if kind in LANDFORM_STYLES else 8193
        textured=filled_features(paths,"geographic-textures",f"url(#{identifier})",section="ink",clip="land",
            opacity=.48,attributes={**common,"data-landform-texture":"true","data-min-scale":str(minimum),
                                   "data-max-scale":str(maximum)})
        features.extend(replace(feature,definitions={identifier:definition}) for feature in textured)
    steep=np.asarray(inventory.masks["steep-slope"],dtype=bool)
    east,north=np.asarray(inventory.direction_east),np.asarray(inventory.direction_north)
    if steep.shape!=grid.shape or east.shape!=grid.shape or north.shape!=grid.shape:
        raise ValueError("slope hachures require aligned, measured physical directions")
    rows,columns=np.nonzero(steep & (grid.water==0))
    if len(rows):
        extents=grid.metadata["extents"]
        latitude=extents["north"]-(rows+.5)/grid.shape[0]*(extents["north"]-extents["south"])
        axis_ratio=(extents["north"]-extents["south"])/grid.shape[0]/(
            (extents["east"]-extents["west"])/grid.shape[1])
        dx=east[rows,columns]*axis_ratio/np.maximum(np.cos(np.radians(latitude)),.025)
        dy=-north[rows,columns]
        length=np.abs(dx)+np.abs(dy)
        valid=np.isfinite(length)&(length>0)
        dx,dy=.16*dx[valid]/length[valid],.16*dy[valid]/length[valid]
        centers=np.column_stack((columns[valid]+.5,rows[valid]+.5))
        direction=np.column_stack((dx,dy))
        lines=np.stack((centers-direction*.5,centers+direction*.5),axis=1)
        if len(lines):
            style=LANDFORM_STYLES["steep-slope"]
            attributes={"fill":"none","stroke":style["color"],"stroke-width":".38","stroke-opacity":".28",
                 "vector-effect":"non-scaling-stroke","stroke-linecap":"round","clip":"land",
                 "data-landform-type":"steep-slope","data-landform-texture":"true",
                 "data-landform-hachure":"downslope","data-min-scale":str(style["min_scale"]),
                 "data-max-scale":str(style["max_scale"])}
            # Spatial chunks keep the tile writer's tree useful: one worldwide
            # MultiLineString would repeat every slope intersection per block.
            keys=rows[valid]//32*((grid.shape[1]+31)//32)+columns[valid]//32
            order=np.argsort(keys,kind="stable")
            groups=np.split(order,np.flatnonzero(np.diff(keys[order]))+1)
            features.extend(TileFeature(shapely.multilinestrings(shapely.linestrings(lines[group])),
                attributes,"geographic-textures",section="ink") for group in groups)
    return tuple(features)


def landform_svg_body(features):
    """An optional static export; the flat runtime uses these same tile records."""
    markup='<g data-tile-layer="geographic-textures">'+''.join(feature_markup(feature)for feature in features)+'</g>'
    return '<defs>'+feature_definitions(features,markup)+'</defs>'+markup
