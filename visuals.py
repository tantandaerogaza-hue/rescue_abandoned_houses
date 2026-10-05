"""Render original supplied PNGs as tinted map glyphs; source assets stay unchanged."""
import base64
from functools import lru_cache
from pathlib import Path
import pydeck as pdk

GROUPS={
 '교통':('#01A8AF',['bus','subway','parking']),
 '생활편의':('#58B368',['store','mart','pharmacy']),
 '안전':('#E65373',['cctv','police','fire']),
 '대학·학습·체육':('#8A63D2',['library','sports','university']),
 '산업·보육·교육':('#D89B24',['industry','daycare','school']),
 '관광':('#FA6C41',['attraction'])}
COLORS={kind:color for color,kinds in GROUPS.values() for kind in kinds}

@lru_cache(maxsize=32)
def icon_url(kind):
    png=base64.b64encode((Path(__file__).parent/'assets'/f'{kind}.png').read_bytes()).decode()
    color=COLORS[kind]
    # UI-only SVG filter: remove white background, retain original alpha, tint dark ink.
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
    <defs><filter id="ink" color-interpolation-filters="sRGB">
    <feColorMatrix in="SourceGraphic" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  -.2126 -.7152 -.0722 0 1" result="dark"/>
    <feComposite in="dark" in2="SourceAlpha" operator="in" result="shape"/>
    <feFlood flood-color="{color}"/><feComposite in2="shape" operator="in"/>
    </filter></defs><circle cx="32" cy="32" r="30" fill="white" stroke="{color}" stroke-width="3"/>
    <image x="13" y="13" width="38" height="38" href="data:image/png;base64,{png}" filter="url(#ink)"/>
    </svg>'''
    return 'data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()

def icon_layers(frame):
    layers=[]
    for kind in frame['kind'].unique():
        data=frame.loc[frame.kind==kind].copy()
        data['icon']=[{'url':icon_url(kind),'width':64,'height':64,'anchorX':32,'anchorY':32,'mask':False} for _ in range(len(data))]
        layers.append(pdk.Layer('IconLayer',id='facility_'+kind,data=data,
            get_position='[longitude,latitude]',get_icon='icon',get_size=30,size_units="'pixels'",
            size_min_pixels=30,size_max_pixels=30,pickable=True))
    return layers

def legend_html():
    return '<div style="display:flex;flex-wrap:wrap;gap:14px;margin:10px 0">'+''.join(
        f'<span style="color:#FAFAFA"><span style="color:{c}">●</span> {name}</span>' for name,(c,_) in GROUPS.items())+'</div>'
