const qs = (s)=>document.querySelector(s);

let map = L.map('map', {zoomControl:true}).setView([42.4259,25.6346], 18);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom:20,
  attribution:'&copy; OpenStreetMap contributors'
}).addTo(map);

function makeIcon(occ){
  const cls = occ ? "bad" : "good";
  return L.divIcon({
    className:"",
    html:`<div class="dot ${cls}" style="width:14px;height:14px;border-radius:4px;border:1px solid rgba(0,0,0,.35)"></div>`,
    iconSize:[16,16],
    iconAnchor:[8,8]
  });
}

const markers = new Map();

function upsert(camId, s){
  if(s.lat == null || s.lng == null) return;
  const key = camId + ":" + s.id;
  const icon = makeIcon(!!s.occupied);
  if(!markers.has(key)){
    const m = L.marker([s.lat, s.lng], {icon}).addTo(map);
    m.bindTooltip(`${camId}/${s.id} ${s.occupied ? "заето" : "свободно"}`);
    markers.set(key, m);
  }else{
    const m = markers.get(key);
    m.setIcon(icon);
    m.setLatLng([s.lat, s.lng]);
  }
}

async function refresh(){
  const url = (window.API_BASE || "").replace(/\/$/, "") + "/api/public/state";
  const res = await fetch(url);
  if(!res.ok) throw new Error("Public API failed");
  const data = await res.json();
  let centered = false;
  for(const cam of (data.cameras || [])){
    if(!centered && cam.map && cam.map.center){
      map.setView([cam.map.center[0], cam.map.center[1]], cam.map.zoom || 18);
      centered = true;
    }
    for(const s of (cam.spots || [])) upsert(cam.camera_id, s);
  }
  qs("#msg").textContent = "Updated: " + new Date().toLocaleTimeString();
}

qs("#refreshBtn").onclick = ()=>refresh().catch(e => qs("#msg").textContent = "❌ " + e.message);
refresh().catch(e => qs("#msg").textContent = "❌ " + e.message);
setInterval(()=>refresh().catch(()=>{}), 3000);
